import contextlib
import copy
import io
import json
import tempfile
import types
import unittest
from dataclasses import replace
from decimal import Decimal as D
from pathlib import Path
from unittest.mock import patch

from hummingbot.strategy.gate_avellaneda.adaptive import AdaptiveSettings, quote_plan
from hummingbot.strategy.gate_avellaneda.core import Market, Settings
from hummingbot.strategy.gate_avellaneda.microstructure import MicroSignal
from hummingbot.strategy.gate_avellaneda.verification import (
    QualityRecorder, TelemetrySettings, audit_report, fee_in_usdt, plain, replay_decision,
)
from hummingbot.strategy.gate_avellaneda.verify import main, preflight


def market(at=100, mid="10"):
    price = D(mid)
    return Market("A-USDT", price - D("0.01"), price + D("0.01"), D("0.002"), D(200), D(100),
                  D(10000000), D(1000), D(1000), D("0.0001"), D(0), D("0.0002"),
                  D("0.001"), D("0.001"), D("0.001"), at)


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "quality.jsonl"
        self.recorder = QualityRecorder(TelemetrySettings(path=str(self.path)), False)

    def rows(self):
        return [json.loads(row) for row in self.path.read_text().splitlines()]

    def fill(self, buy=True, price="9.99", fee=None, timestamp=100, now=100):
        self.recorder.fill(now, "A-USDT", "order", "trade", D(1), D(price), buy, False, False,
                           timestamp, market(), fee)

    def marks(self):
        return [row for row in self.rows() if row["kind"] == "markout"]

    def test_buy_markout_edge_move_fee_estimate_decomposition(self):
        fee = types.SimpleNamespace(percent=D("0.0002"), percent_token=None, flat_fees=[])
        self.fill(fee=fee)
        self.recorder.sample(101, {"A-USDT": market(101, "9.98")}, 2)
        result = self.marks()[0]
        self.assertEqual(D(result["gross_usdt"]), D("-0.01"))
        self.assertEqual(D(result["entry_edge_usdt"]), D("0.01"))
        self.assertEqual(D(result["post_fill_move_usdt"]), D("-0.02"))
        self.assertEqual(D(result["fee_adjusted_estimate_usdt"]), D("-0.011998"))
        self.assertTrue(result["observed"])

    def test_sell_markout_sign(self):
        self.fill(buy=False, price="10.01")
        self.recorder.sample(101, {"A-USDT": market(101, "9.98")}, 2)
        self.assertEqual(D(self.marks()[0]["gross_usdt"]), D("0.03"))

    def test_horizons_each_emitted_once_and_pending_released(self):
        self.fill()
        for at in (101, 101, 105, 105, 130, 130):
            self.recorder.sample(at, {"A-USDT": market(at)}, 2)
        self.assertEqual([r["horizon"] for r in self.marks()], [1, 5, 30])
        self.assertEqual(self.recorder.pending, [])

    def test_missing_stale_or_future_horizon_censored_no_fabricated_pnl(self):
        for book in ({}, {"A-USDT": market(100)}, {"A-USDT": market(105)}):
            self.fill()
            self.recorder.sample(104, book, 2)
        self.assertEqual(len(self.marks()), 3)
        self.assertTrue(all(not row["observed"] and "gross_usdt" not in row for row in self.marks()))

    def test_late_delivery_does_not_reconstruct_missed_markouts_or_anchor(self):
        self.fill(timestamp=90, now=100)
        self.recorder.sample(100, {"A-USDT": market(100)}, 2)
        self.assertIsNone(next(r for r in self.rows() if r["kind"] == "fill")["anchor_mid"])
        self.assertEqual([r["observed"] for r in self.marks()], [False, False])

    def test_observation_must_follow_target(self):
        self.fill()
        self.recorder.sample(101, {"A-USDT": market(100.9)}, 2)
        self.assertEqual(self.marks(), [])
        self.recorder.sample(102, {"A-USDT": market(101.5)}, 2)
        self.assertTrue(self.marks()[0]["observed"])

    def test_rotation_keeps_bounded_backups_and_mode_paths_separate(self):
        recorder = QualityRecorder(TelemetrySettings(path=str(self.path), max_bytes=10000, backups=2), True)
        for at in range(50):
            self.assertTrue(recorder.record("payload", at, value="x" * 2000))
        self.assertNotEqual(recorder.path, self.path)
        self.assertEqual(len(list(self.path.parent.glob("*.jsonl*"))), 3)
        self.assertLessEqual(recorder.path.stat().st_size, 10000)

    def test_io_error_is_visible_and_recovers(self):
        with patch("hummingbot.strategy.gate_avellaneda.verification.os.chmod", side_effect=OSError("disk")):
            self.assertFalse(self.recorder.record("event", 100))
        self.assertIn("disk", self.recorder.error)
        self.assertTrue(self.recorder.record("event", 101))
        self.assertEqual(self.recorder.error, "")

    def test_unknown_and_invalid_fee_not_guessed_as_zero(self):
        for fee in (None, types.SimpleNamespace(percent=D("0.0002")),
                    types.SimpleNamespace(percent=D("NaN"), flat_fees=[]),
                    types.SimpleNamespace(percent=D("0.001"), percent_token="BTC", flat_fees=[]),
                    types.SimpleNamespace(percent=D(0), flat_fees=[types.SimpleNamespace(token="POINT", amount=D(1))])):
            self.assertIsNone(fee_in_usdt(fee, D(1), D(10)))

    def test_flat_usdt_fee_conversion(self):
        fee = types.SimpleNamespace(percent=D(0), flat_fees=[types.SimpleNamespace(token="USDT", amount=D("0.1"))])
        self.assertEqual(fee_in_usdt(fee, D(1), D(10)), D("0.1"))

    def decision(self, signal=None, allow_open=True):
        risk, adaptive = Settings(), AdaptiveSettings()
        plan = quote_plan(market(), D(1), 100, risk, adaptive, 100, signal, allow_open=allow_open, gamma=D("1.3"))
        self.recorder.decision(100, market(), D(1), 100, risk, adaptive, signal, allow_open, None, D("1.3"), plan)
        return self.rows()[-1]

    def test_decision_replay_with_and_without_micro_protection(self):
        for signal in (None, MicroSignal(True, "", 100, reference=D("10.005"), imbalance=D("0.8"),
                                        flow=D("0.7"), flow_sufficient=True)):
            for allow_open in (True, False):
                self.assertTrue(replay_decision(self.decision(signal, allow_open)))

    def test_replay_detects_changed_output(self):
        record = self.decision()
        record["expected"]["reference"] = "999"
        self.assertFalse(replay_decision(record))

    def test_report_deduplicates_and_uses_latest_cumulative_account(self):
        self.fill()
        self.recorder.sample(101, {"A-USDT": market(101)}, 2)
        for at in (100, 102):
            self.recorder.record("account", at, risk=dict(pnl="-1", anchor=dict(user="42")), cash_totals=dict(fee="-1"))
        self.decision()
        report = audit_report(self.rows() * 2)
        self.assertEqual(report["fills"], 1)
        self.assertEqual(report["latest_account"]["cash_totals"]["fee"], "-1")
        self.assertEqual(report["markouts"]["1"]["observed"], 1)
        self.assertEqual(report["markouts"]["30"]["missing_or_pending"], 1)
        self.assertEqual(report["replay_mismatches"], 0)

    def test_orphan_markout_is_not_counted_as_complete_fill_coverage(self):
        self.fill()
        self.recorder.sample(101, {"A-USDT": market(101)}, 2)
        report = audit_report([r for r in self.rows() if r["kind"] != "fill"])
        self.assertEqual(report["fills"], 0)
        self.assertEqual(report["markouts"]["1"]["observed"], 0)

    def test_report_refuses_mixed_live_and_dry_run(self):
        self.fill()
        rows = self.rows()
        other = copy.deepcopy(rows[0])
        other["dry_run"] = True
        with self.assertRaisesRegex(ValueError, "dry-run"):
            audit_report(rows + [other])

    def test_cli_exit_code_detects_replay_failure_and_malformed_log(self):
        record = self.decision()
        record["expected"]["reference"] = "999"
        self.path.write_text(json.dumps(record) + "\n")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--report", str(self.path)]), 1)
            self.path.write_text("broken")
            self.assertEqual(main(["--report", str(self.path)]), 1)

    def test_preflight_explicit_missing_runtime_or_network_is_a_failure(self):
        with patch("hummingbot.strategy.gate_avellaneda.verify.importlib.import_module", side_effect=ImportError("not compiled")):
            result = preflight(runtime=True)
        self.assertFalse(result["ok"])
        self.assertFalse(result["live_ready"])
        with patch("hummingbot.strategy.gate_avellaneda.verify.urlopen", side_effect=TimeoutError("blocked")):
            self.assertFalse(preflight(gate_public=True)["ok"])
        self.assertEqual(preflight()["checks"], {})

    def test_preflight_checks_actual_imports_and_public_candidate_schema(self):
        imports = types.SimpleNamespace(OrderType=types.SimpleNamespace(LIMIT_MAKER=True),
                                        PositionAction=types.SimpleNamespace(CLOSE=True),
                                        GateAvellanedaPortfolioConfig=lambda: None)
        from test.hummingbot.strategy.gate_avellaneda.support import FakeConnector
        connector = FakeConnector(["A-USDT", "B-USDT"])
        responses = [contextlib.closing(io.BytesIO(json.dumps(value).encode()))
                     for value in (connector.contracts, connector.tickers)]
        with patch("hummingbot.strategy.gate_avellaneda.verify.urlopen", side_effect=responses), \
                patch("hummingbot.strategy.gate_avellaneda.verify.importlib.import_module", return_value=imports):
            result = preflight(runtime=True, gate_public=True)
        self.assertTrue(result["ok"])
        self.assertFalse(result["live_ready"])
        self.assertEqual(len(result["checks"]["gate_public"]["candidates"]), 2)

    def test_disabled_recorder_emits_no_files_or_synthetic_fills(self):
        disabled = QualityRecorder(TelemetrySettings(enabled=False, path=str(self.path)), False)
        disabled.fill(100, "A-USDT", "order", "trade", D(1), D(10), True, False, False, 100)
        disabled.sample(101, {"A-USDT": market(101)}, 2)
        self.assertEqual(disabled.pending, [])
        self.assertFalse(self.path.exists())
