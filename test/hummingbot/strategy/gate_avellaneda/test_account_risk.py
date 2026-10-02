import copy
import json
import os
import tempfile
import unittest
from datetime import datetime
from decimal import Decimal as D
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from hummingbot.strategy.gate_avellaneda.account_risk import AccountRisk, AccountRiskSettings, RiskStateStore, mode_path


def account(pnl="0", fee="0", fund="0", dnw="100", refr="0", upnl="0"):
    history = dict(pnl=pnl, fee=fee, fund=fund, dnw=dnw, refr=refr)
    wallet = sum(map(D, history.values()))
    return dict(user=42, currency="USDT", margin_mode=0, enable_credit=False, point="0", bonus="0",
                total=str(wallet), available=str(wallet), unrealised_pnl=upnl, history=history)


def records(**kinds):
    return {kind: dict(kind=kind, pair="OLD-USDT", time=101, change=D(value)) for kind, value in kinds.items()}


class AccountRiskTests(unittest.TestCase):
    def setUp(self):
        self.risk = AccountRisk(AccountRiskSettings())
        self.risk.observe(account(), {}, D(0), 100, D(20))

    def observe(self, cash=None, rows=None, now=101):
        cash = cash or account()
        self.risk.observe(cash, rows or {}, D(cash["unrealised_pnl"]), now, D(20))

    def test_net_settlements_include_fees_and_funding_for_exited_pairs(self):
        self.observe(account(pnl="-8", fee="-1", fund="-1"), records(pnl="-8", fee="-1", fund="-1"))
        self.assertEqual(self.risk.pnl, D(-10))
        self.assertTrue(self.risk.latched)
        self.assertTrue(self.risk.reconciliation_error == "")

    def test_exact_boundary_and_no_reset_by_deposit_recovery_or_day(self):
        self.observe(account(pnl="-9.99"), records(pnl="-9.99"))
        self.assertFalse(self.risk.latched)
        self.observe(account(pnl="-10"), records(pnl="-10"), 102)
        self.observe(account(pnl="5", dnw="200"), records(pnl="5", dnw="100"), 100000)
        self.assertTrue(self.risk.latched)
        self.assertFalse(self.risk.allow_open)
        self.assertEqual(self.risk.reason, "session loss limit")

    def test_deposits_cannot_hide_loss(self):
        self.observe(account(pnl="-11", dnw="200"), records(pnl="-11", dnw="100"))
        self.assertTrue(self.risk.latched)
        self.assertEqual(self.risk.pnl, D(-11))

    def test_withdrawal_and_rebate_are_not_trading_pnl(self):
        self.observe(account(dnw="80", refr="1"), records(dnw="-20", refr="1"))
        self.assertEqual(self.risk.pnl, D(0))
        self.assertTrue(self.risk.allow_open)

    def test_withdrawal_to_reserve_still_latches(self):
        self.observe(account(dnw="20"), records(dnw="-80"))
        self.assertEqual(self.risk.pnl, D(0))
        self.assertEqual(self.risk.reason, "account reserve floor")

    def test_drawdown_from_profit_is_independent_of_absolute_loss(self):
        self.observe(account(pnl="8"), records(pnl="8"))
        self.observe(account(pnl="-2"), records(pnl="-2"), 102)
        self.assertEqual(self.risk.drawdown, D(10))
        self.assertEqual(self.risk.reason, "profit drawdown limit")

    def test_daily_loss_from_prior_profit(self):
        self.observe(account(pnl="20"), records(pnl="20"))
        self.observe(account(pnl="20"), records(pnl="20"), 86400)
        self.observe(account(pnl="10"), records(pnl="10"), 86401)
        self.assertEqual(self.risk.daily_pnl, D(-10))
        self.assertEqual(self.risk.reason, "daily loss limit")

    def test_taipei_day_boundary_keeps_unobserved_gap(self):
        before = datetime(2026, 10, 2, 23, 59, 59, tzinfo=ZoneInfo("Asia/Taipei")).timestamp()
        self.observe(account(pnl="5"), records(pnl="5"), before)
        self.observe(account(pnl="-4"), records(pnl="-4"), before + 2)
        self.assertEqual(self.risk.day, "2026-10-03")
        self.assertEqual(self.risk.daily_pnl, D(-9))

    def test_history_catches_missing_late_loss_without_double_counting(self):
        self.observe(account(fee="-10"), {})
        self.assertEqual(self.risk.pnl, D(-10))
        self.assertTrue(self.risk.latched)
        self.assertIn("mismatch", self.risk.reconciliation_error)
        self.observe(account(fee="-10"), records(fee="-10"), 102)
        self.assertEqual(self.risk.pnl, D(-10))
        self.assertEqual(self.risk.reconciliation_error, "")

    def test_wallet_mismatch_pauses_then_recovers(self):
        cash = account()
        cash["total"] = "99"
        self.observe(cash)
        self.assertFalse(self.risk.allow_open)
        self.assertFalse(self.risk.latched)
        self.observe(now=102)
        self.assertTrue(self.risk.allow_open)

    def test_unrealized_mismatch_uses_more_negative_value(self):
        self.risk.observe(account(upnl="-1"), {}, D(-10), 101, D(20))
        self.assertEqual(self.risk.pnl, D(-10))
        self.assertTrue(self.risk.latched)
        self.assertIn("unrealized", self.risk.reconciliation_error)

    def test_unsupported_flow_pauses_opening(self):
        self.observe(rows=records(point_fee="-1"))
        self.assertFalse(self.risk.allow_open)
        self.assertIn("unsupported", self.risk.reconciliation_error)

    def test_initial_book_records_are_excluded(self):
        risk = AccountRisk(AccountRiskSettings())
        risk.observe(account(pnl="-40"), records(pnl="-40"), D(0), 100, D(20))
        risk.observe(account(pnl="-40"), records(pnl="-40"), D(0), 101, D(20))
        self.assertEqual(risk.pnl, D(0))
        self.assertTrue(risk.allow_open)

    def test_repeated_poll_is_idempotent(self):
        for now in range(101, 110):
            self.observe(account(fee="-1"), records(fee="-1"), now)
        self.assertEqual(self.risk.pnl, D(-1))

    def test_restore_preserves_peak_latch_day_and_identity(self):
        self.observe(account(pnl="8"), records(pnl="8"))
        self.observe(account(pnl="-2"), records(pnl="-2"), 102)
        other = AccountRisk(AccountRiskSettings())
        other.restore(self.risk.dump())
        self.assertEqual(other.peak, D(8))
        self.assertTrue(other.latched)
        self.assertFalse(other.allow_open)
        other.observe(account(pnl="-2"), records(pnl="-2"), D(0), 103, D(20))
        self.assertTrue(other.latched)

    def test_wrong_account_is_rejected(self):
        cash = account()
        cash["user"] = 43
        with self.assertRaisesRegex(ValueError, "another Gate"):
            self.observe(cash)

    def test_invalid_account_modes_cash_history_and_nonfinite_are_rejected(self):
        for changes in (dict(margin_mode=3), dict(currency="BTC"), dict(enable_credit=True), dict(user=None),
                        dict(history={}), dict(point="1"), dict(bonus="1"), dict(total="NaN")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.observe(dict(account(), **changes))

    def test_clock_rollback_and_corrupt_state_refuse_restore(self):
        self.observe(now=102)
        with self.assertRaisesRegex(ValueError, "backwards"):
            self.observe(now=101)
        for field, bad in (("pnl", "NaN"), ("latched", "false"), ("last_at", -1), ("drawdown", "1")):
            payload = copy.deepcopy(self.risk.dump())
            payload[field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                AccountRisk(AccountRiskSettings()).restore(payload)


class RiskStateStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "state.json"
        self.store = RiskStateStore(self.path)
        self.addCleanup(self.store.close)

    def test_atomic_save_checksum_and_private_permissions(self):
        self.assertIsNone(self.store.load())
        payload = dict(pnl="-10", latched=True)
        self.store.save(payload)
        self.assertEqual(self.store.load(), payload)
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)
        self.assertEqual(mode_path(self.path, True).name, "state.dry_run.json")

    def test_second_writer_cannot_acquire_lock(self):
        with self.assertRaisesRegex(ValueError, "another process"):
            RiskStateStore(self.path)
        self.store.close()
        other = RiskStateStore(self.path)
        other.close()

    def test_damage_or_unknown_version_is_not_overwritten(self):
        self.store.save(dict(value=1))
        original = json.loads(self.path.read_text())
        for field, value in (("sha256", "bad"), ("version", 2), ("payload", dict(value=2))):
            self.path.write_text(json.dumps(dict(original, **{field: value})))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.store.load()
        self.path.write_text("partial")
        with self.assertRaises(ValueError):
            self.store.load()

    def test_failed_replace_preserves_prior_state_and_removes_temporary_file(self):
        self.store.save(dict(value=1))
        with patch("hummingbot.strategy.gate_avellaneda.account_risk.os.replace", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                self.store.save(dict(value=2))
        self.assertEqual(self.store.load(), dict(value=1))
        self.assertEqual(sorted(p.name for p in self.path.parent.iterdir()), ["state.json", "state.json.lock"])
