"""Calibrate and replay quote sensitivity without simulating fills or PnL.

Run from the repository root. CSVs can be collected with collect_avellaneda_gate.py.
The resulting config always remains in preview mode and contains no credentials.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import replace
from decimal import Decimal as D
from pathlib import Path

import yaml

# Direct script execution also works outside an installed Hummingbot environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hummingbot.strategy_v2.utils.avellaneda_perpetual import (  # noqa: E402
    AvellanedaEngine,
    AvellanedaSettings,
    Snapshot,
    quote_proposal,
    round_step,
)
from hummingbot.strategy_v2.utils.avellaneda_perpetual_gamma import GammaCalibration, GammaController  # noqa: E402


def load_capture(path, config):
    snapshots = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["trading_pair"] != config.trading_pair:
                raise ValueError("CSV trading pair does not match config")
            s = Snapshot(float(row["timestamp"]), D(row["bid"]), D(row["ask"]), D("0"),
                         D(row["available"]), D(row["equity"]), D(row["price_tick"]),
                         D(row["amount_step"]), D(row["min_amount"]), D(row["min_notional"]),
                         book_age=float(row["book_age"]))
            if not s.valid or not 0 <= s.book_age <= config.stale_seconds:
                raise ValueError("Capture contains invalid/stale observations; recapture")
            if snapshots and s.timestamp <= snapshots[-1].timestamp:
                raise ValueError("Capture timestamps must increase")
            snapshots.append(s)
    if not snapshots:
        raise ValueError("Empty capture")
    return snapshots


def calibrate_capture(snapshots, config):
    config = AvellanedaSettings(**dict(config.model_dump(), dry_run=True, gamma_mode="adaptive", gamma_calibration=None))
    engine = AvellanedaEngine(config)
    for s in snapshots:
        engine.step(s)
        if engine.gamma.profile:
            return engine.gamma.profile
    raise ValueError(engine.calibration_reason)


def replay_quotes(snapshots, config, profile):
    """Use only observations after calibration, with explicit hypothetical inventory.

    Every mode gets the same observed book, variance, fee floor, quantity and risk
    limits. The CSV contains proposals, never claims about exchange fills.
    """
    values = dict(config.model_dump(), gamma_mode="adaptive", gamma_calibration=profile, dry_run=True)
    adaptive_config = AvellanedaSettings(**values)
    estimator = AvellanedaEngine(AvellanedaSettings(**dict(values, gamma_mode="fixed", risk_factor=1)))
    controllers = {name: GammaController(adaptive_config) for name in ("flat", "long_first", "short_first", "long_three")}
    rows = []
    for s in snapshots:
        if not s.valid or not s.ready or not 0 <= s.book_age <= config.stale_seconds:
            continue
        estimator.observe(s)
        if s.timestamp <= profile.created_at or len(estimator.samples) < config.warmup_samples:
            continue
        error = controllers["flat"].compatibility_error(s)
        if error:
            raise ValueError(error)
        max_base = round_step(config.max_position_quote / s.mid, s.amount_step)
        positions = dict(flat=D("0"), long_first=min(profile.reference_amount, max_base),
                         short_first=-min(profile.reference_amount, max_base),
                         long_three=min(profile.reference_amount * 3, max_base))
        for scenario, position in positions.items():
            inventory = float(position * s.mid / config.max_position_quote)
            controller = controllers[scenario]
            controller.update(s.timestamp, inventory, estimator.variance)
            for mode, gamma in (("fixed_1", 1), ("calibrated_fixed", profile.gamma_base),
                                ("adaptive", controller.current)):
                quotes, reservation, spread, model_spread = quote_proposal(
                    config, replace(s, position=position), estimator.variance, gamma)
                prices = {q.side: q.price for q in quotes}
                rows.append(dict(timestamp=s.timestamp, scenario=scenario, mode=mode, gamma=gamma,
                                 inventory_ratio=inventory, variance=estimator.variance,
                                 raw_skew_ticks=float((reservation - s.mid) / s.price_tick),
                                 spread=str(spread), spread_floor_active=model_spread <= max(config.min_spread, config.fee_floor),
                                 bid=str(prices.get("buy", "")), ask=str(prices.get("sell", "")),
                                 buy_amount=str(next((q.amount for q in quotes if q.side == "buy"), "")),
                                 sell_amount=str(next((q.amount for q in quotes if q.side == "sell"), ""))))
    if not rows:
        raise ValueError("Capture must continue beyond calibration to compare subsequent quotes")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["mode"])].append(row)
    summary = [dict(scenario=scenario, mode=mode, observations=len(group),
                    mean_abs_raw_skew_ticks=sum(abs(r["raw_skew_ticks"]) for r in group) / len(group),
                    gamma_min=min(r["gamma"] for r in group), gamma_max=max(r["gamma"] for r in group),
                    spread_floor_fraction=sum(r["spread_floor_active"] for r in group) / len(group))
               for (scenario, mode), group in grouped.items()]
    return rows, summary


def preview_config(config, profile):
    return dict(config.model_dump(mode="json"), dry_run=True, gamma_mode="adaptive",
                gamma_calibration=profile.model_dump(mode="json"))


def write_new(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("examples/avellaneda_perpetual_gate.yml"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture", type=Path)
    source.add_argument("--calibration", type=Path, help="Existing script-exported profile or replay report")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--quotes", type=Path)
    parser.add_argument("--write-preview-config", type=Path)
    args = parser.parse_args()
    config = AvellanedaSettings(**yaml.safe_load(args.config.read_text()))
    if not config.dry_run:
        parser.error("Use a preview config for calibration")
    try:
        if args.capture:
            snapshots = load_capture(args.capture, config)
            profile = calibrate_capture(snapshots, config)
            rows, summary = replay_quotes(snapshots, config, profile)
            report = dict(calibration=profile.model_dump(mode="json"), replay_summary=summary,
                          calibration_checks=dict(passed=True, reason="both rounded inventory directions are visible",
                                                  reference_horizon_variance=profile.variance_per_second * profile.horizon_seconds,
                                                  reference_horizon_sigma=math.sqrt(profile.variance_per_second * profile.horizon_seconds),
                                                  gamma_min=profile.gamma_base * config.gamma_min_ratio,
                                                  gamma_max=min(10000, profile.gamma_base * config.gamma_max_ratio),
                                                  reference_long_raw_skew_ticks=-profile.target_ticks),
                          capture_samples=len(snapshots), replay_after_calibration=True,
                          balances="hypothetical CSV balances; no authenticated account",
                          interpretation="quote sensitivity with hypothetical inventory; no fills or PnL simulated")
            if args.quotes:
                args.quotes.parent.mkdir(parents=True, exist_ok=True)
                with args.quotes.open("x", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)
        else:
            data = json.loads(args.calibration.read_text())
            profile = GammaCalibration.model_validate(data.get("calibration", data))
            # Compare config settings here; live startup additionally verifies
            # exchange rules and profile age before any account mutations.
            candidate = AvellanedaSettings(**preview_config(config, profile))
            reference = Snapshot(profile.created_at, profile.reference_mid - profile.price_tick / 2,
                                 profile.reference_mid + profile.price_tick / 2, D("0"), D("100"), D("100"),
                                 profile.price_tick, profile.amount_step, profile.min_amount, profile.min_notional)
            error = GammaController(candidate).compatibility_error(reference)
            if error:
                raise ValueError(error)
            report = profile.model_dump(mode="json")
        if args.report:
            write_new(args.report, json.dumps(report, indent=2) + "\n")
        if args.write_preview_config:
            write_new(args.write_preview_config, yaml.safe_dump(preview_config(config, profile), sort_keys=False))
        print(json.dumps(report, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Calibration/replay failed: {exc}\n")


if __name__ == "__main__":
    main()
