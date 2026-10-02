"""Bounded JSONL audit and observed fill markouts; never invents executions."""
import json
import math
import os
import uuid
from dataclasses import asdict, fields, is_dataclass
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .account_risk import mode_path
from .adaptive import AdaptiveSettings, quote_plan
from .core import Market, Settings
from .microstructure import MicroSignal


class TelemetrySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    path: str = "logs/gate_avellaneda_quality.jsonl"
    sample_seconds: float = Field(default=1, ge=0.1, le=10)
    markout_wait_seconds: float = Field(default=2, gt=0, le=10)
    max_bytes: int = Field(default=20_000_000, ge=10000)
    backups: int = Field(default=5, ge=1, le=20)


def plain(value):
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Nonfinite audit value")
        return str(value)
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, BaseModel):
        return plain(value.model_dump())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(v) for v in value]
    return value


def fee_in_usdt(fee, amount, price):
    """Conservative estimate only; official cash ledger remains authoritative."""
    if fee is None or not hasattr(fee, "percent") or not hasattr(fee, "flat_fees"):
        return None
    try:
        percent = Decimal(str(fee.percent))
        if not percent.is_finite() or (percent != 0 and getattr(fee, "percent_token", None) not in (None, "USDT")):
            return None
        total = amount * price * percent
        for item in fee.flat_fees:
            if item.token != "USDT":
                return None
            total += Decimal(str(item.amount))
        return total if total.is_finite() else None
    except (ArithmeticError, TypeError, AttributeError, ValueError):
        return None


class QualityRecorder:
    horizons = (1, 5, 30)

    def __init__(self, settings, dry_run, on_markout=None):
        self.settings = settings
        self.path = mode_path(settings.path, dry_run)
        self.session = uuid.uuid4().hex
        self.dry_run = dry_run
        self.error = ""
        self.next_sample = 0
        self.pending = []
        self.on_markout = on_markout

    def record(self, kind, now, **values):
        if not self.settings.enabled:
            return True
        try:
            record = plain(dict(schema=1, session=self.session, dry_run=self.dry_run,
                                kind=kind, at=now, **values))
            raw = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size + len(raw.encode()) > self.settings.max_bytes:
                for index in range(self.settings.backups, 0, -1):
                    source = self.path if index == 1 else Path(str(self.path) + f".{index - 1}")
                    target = Path(str(self.path) + f".{index}")
                    if source.exists():
                        os.replace(source, target)
            with open(self.path, "a", encoding="utf-8") as stream:
                os.chmod(self.path, 0o600)
                stream.write(raw)
            self.error = ""
            return True
        except (OSError, ValueError, TypeError) as exc:
            self.error = f"quality log unavailable: {exc}"
            return False

    def fill(self, now, pair, order_id, trade_id, amount, price, buy, close, market,
             timestamp, anchor_market=None, fee=None, exit_reason="", regime="calm"):
        if not self.settings.enabled:
            return
        timestamp = timestamp if math.isfinite(timestamp) and 0 <= timestamp <= now else now
        anchor = (anchor_market.mid if anchor_market is not None
                  and anchor_market.observed_at <= timestamp
                  and timestamp - anchor_market.observed_at <= self.settings.markout_wait_seconds else None)
        fill_id = uuid.uuid4().hex
        fee_estimate = fee_in_usdt(fee, amount, price)
        fill = dict(fill_id=fill_id, pair=pair, order_id=order_id, trade_id=trade_id,
                    amount=amount, price=price, buy=buy, close=close, market=market,
                    filled_at=timestamp, anchor_mid=anchor, fee_estimate_usdt=fee_estimate, exit_reason=exit_reason, regime=regime)
        self.record("fill", now, **fill)
        self.pending.append(dict(**fill, remaining=set(self.horizons)))

    def sample(self, now, markets, max_age):
        if not self.settings.enabled:
            return
        if now >= self.next_sample:
            self.record("markets", now, mids={p: dict(mid=m.mid, observed_at=m.observed_at)
                                              for p, m in markets.items() if m.ready and 0 <= now - m.observed_at <= max_age})
            self.next_sample = now + self.settings.sample_seconds
        for fill in self.pending:
            market = markets.get(fill["pair"])
            for horizon in sorted(fill["remaining"]):
                target = fill["filled_at"] + horizon
                if now < target:
                    continue
                fresh = (market is not None and market.ready and 0 <= now - market.observed_at <= max_age
                         and target <= market.observed_at <= target + self.settings.markout_wait_seconds)
                if not fresh and now <= target + self.settings.markout_wait_seconds:
                    continue
                values = dict(fill_id=fill["fill_id"], pair=fill["pair"], horizon=horizon, observed=bool(fresh))
                if fresh:
                    side = Decimal(1 if fill["buy"] else -1)
                    amount, price, anchor = fill["amount"], fill["price"], fill["anchor_mid"]
                    gross = side * amount * (market.mid - price)
                    values.update(observed_at=market.observed_at, mid=market.mid,
                                  gross_usdt=gross, bps=gross / (amount * price) * 10000,
                                  entry_edge_usdt=side * amount * (anchor - price) if anchor is not None else None,
                                  post_fill_move_usdt=side * amount * (market.mid - anchor) if anchor is not None else None,
                                  fee_adjusted_estimate_usdt=gross - fill["fee_estimate_usdt"]
                                  if fill["fee_estimate_usdt"] is not None else None)
                else:
                    values["reason"] = "no fresh observation within horizon window"
                if self.record("markout", now, **values):
                    if self.on_markout is not None:
                        self.on_markout(fill, values)
                    fill["remaining"].remove(horizon)
        self.pending = [fill for fill in self.pending if fill["remaining"]]

    def decision(self, now, market, position, age, risk, adaptive, signal, allow_open, blocked_side, gamma, plan,
                 quality=None, exit_cost_bps=Decimal(0)):
        return self.record("decision", now, market=market, position=position, age=age, risk=risk,
                           adaptive=adaptive, signal=signal, allow_open=allow_open, quality=quality, exit_cost_bps=exit_cost_bps,
                           blocked_side=blocked_side, gamma=gamma, expected=plan)


def dataclass_from_json(cls, data):
    return cls(**{field.name: Decimal(data[field.name]) if field.type is Decimal else data[field.name]
                  for field in fields(cls) if field.name in data})


def replay_decision(record):
    if record["kind"] != "decision" or record["schema"] != 1:
        raise ValueError("Unsupported decision record")
    market = dataclass_from_json(Market, record["market"])
    signal = dataclass_from_json(MicroSignal, record["signal"]) if record["signal"] is not None else None
    plan = quote_plan(market, Decimal(record["position"]), record["age"],
                      dataclass_from_json(Settings, record["risk"]),
                      dataclass_from_json(AdaptiveSettings, record["adaptive"]), record["at"], signal,
                      allow_open=record["allow_open"], blocked_side=record["blocked_side"], gamma=Decimal(record["gamma"]),
                      quality=record.get("quality"), exit_cost_bps=Decimal(record.get("exit_cost_bps", "0")))
    return plain(plan) == record["expected"]


def audit_report(records):
    # Repeated/overlapping explicit paths must not double-count fills.
    rows = list({json.dumps(r, sort_keys=True): r for r in records}.values())
    if len({r["dry_run"] for r in rows}) > 1:
        raise ValueError("Report cannot combine dry-run and live sessions")
    decisions = [r for r in rows if r["kind"] == "decision"]
    fills = list({(r["session"], r["fill_id"]): r for r in rows if r["kind"] == "fill"}.values())
    fill_ids = {(r["session"], r["fill_id"]) for r in fills}
    markouts = {(r["session"], r["fill_id"], r["horizon"]): r for r in rows
                if r["kind"] == "markout" and (r["session"], r["fill_id"]) in fill_ids}
    accounts = [r for r in rows if r["kind"] == "account"]
    users = {r["risk"]["anchor"]["user"] for r in accounts if r.get("risk", {}).get("anchor")}
    if len(users) > 1:
        raise ValueError("Report cannot combine different Gate accounts")
    latest = max(accounts, key=lambda r: r["at"]) if accounts else None
    by_horizon = {}
    for horizon in QualityRecorder.horizons:
        observations = [r for r in markouts.values() if r["horizon"] == horizon and r["observed"]]
        by_horizon[str(horizon)] = dict(observed=len(observations), fills=len(fills),
                                      missing_or_pending=max(0, len(fills) - len(observations)),
                                      mean_bps=str(sum((Decimal(r["bps"]) for r in observations), Decimal(0)) / len(observations))
                                      if observations else None)
    by_pair = {}
    for pair in sorted({r["pair"] for r in fills}):
        pair_fills = [r for r in fills if r["pair"] == pair]
        order_ids = {(r["session"], r["order_id"]) for r in pair_fills}
        samples = []
        for fill in pair_fills:
            obs = markouts.get((fill["session"], fill["fill_id"], 5))
            if obs and obs["observed"] and obs.get("fee_adjusted_estimate_usdt") is not None:
                samples.append((Decimal(obs["fee_adjusted_estimate_usdt"]), Decimal(fill["amount"]) * Decimal(fill["price"])))
        notional = sum((row[1] for row in samples), Decimal(0))
        by_pair[pair] = dict(fills=len(pair_fills), distinct_filled_orders=len(order_ids),
                            maker_fills=sum(not r["market"] for r in pair_fills),
                            taker_fills=sum(r["market"] for r in pair_fills),
                            five_second_observations=len(samples),
                            five_second_fee_adjusted_bps=str(sum((r[0] for r in samples), Decimal(0)) / notional * 10000)
                            if notional else None,
                            cancel_requests=sum(r["kind"] == "cancel_requested" and r.get("pair") == pair for r in rows),
                            quote_retained=sum(r["kind"] == "quote_retained" and r.get("pair") == pair for r in rows))
    return dict(fills=len(fills), close_fills=sum(r["close"] for r in fills), by_pair=by_pair,
                replay_checked=len(decisions), replay_mismatches=sum(not replay_decision(r) for r in decisions),
                markouts=by_horizon, latest_account=latest,
                note="Observed markouts are not realized strategy profit; no synthetic fills. Missing rotated logs limit coverage.")
