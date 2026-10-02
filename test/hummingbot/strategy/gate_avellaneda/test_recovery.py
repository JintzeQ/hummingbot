import ast
import copy
import re
import unittest
from dataclasses import asdict
from decimal import Decimal as D

from hummingbot.strategy.gate_avellaneda.adaptive import AdaptiveSettings, InventoryTracker
from hummingbot.strategy.gate_avellaneda.core import Intent, Portfolio, Settings, Slot
from hummingbot.strategy.gate_avellaneda.recovery import RecoveryLedger, RecoverySettings, recovery_pairs
from hummingbot.strategy.gate_avellaneda.verification import plain


class RecoveryLedgerTests(unittest.TestCase):
    def setUp(self):
        self.settings = RecoverySettings()
        self.ledger = RecoveryLedger(self.settings)
        self.portfolio = Portfolio(Settings())
        self.portfolio.slots["A-USDT"] = Slot("A-USDT", selected_at=90)
        self.inventory = InventoryTracker(AdaptiveSettings())
        self.inventory.fill("A-USDT", "first", True, D(0), D(1), False, 95)
        self.intent = Intent("A-USDT", True, D(1), D(10), False, False)
        self.cid = self.ledger.new_id()
        self.ledger.prepare(self.cid, self.intent, 100)

    def raw_order(self, **changes):
        return dict(dict(id=123, text=self.cid, contract="A_USDT", user=42, size="1000", left="1000",
                         price="10", tif="poc", is_reduce_only=False, status="open"), **changes)

    def raw_trade(self, **changes):
        return dict(dict(id="fill1", order_id="123", text=self.cid, contract="A_USDT", size="100",
                         price="10", create_time=101, fee="0", point_fee="0"), **changes)

    def state(self):
        return self.ledger.dump({"A-USDT": D(1)}, self.portfolio, self.inventory)

    def test_ids_are_unique_gate_length_and_owner_bound(self):
        ids = {self.ledger.new_id() for _ in range(1000)}
        self.assertEqual(len(ids), 1000)
        self.assertTrue(all(len(cid) == 28 and re.fullmatch(r"t-ga[0-9a-f]{24}", cid) for cid in ids))
        with self.assertRaises(ValueError):
            self.ledger.prepare("t-other", self.intent, 100)
        with self.assertRaises(ValueError):
            self.ledger.prepare(self.cid, self.intent, 100)

    def test_fill_deduplication_and_quantity_bounds(self):
        self.assertTrue(self.ledger.fill(self.cid, "1", D("0.2"), D(10), 101))
        self.assertFalse(self.ledger.fill(self.cid, "1", D("0.2"), D(10), 101.1))
        self.assertEqual(self.ledger.filled(self.cid), D("0.2"))
        with self.assertRaises(ValueError):
            self.ledger.fill(self.cid, "1", D("0.3"), D(10), 101)
        with self.assertRaises(ValueError):
            self.ledger.fill(self.cid, "2", D(1), D(10), 101)
        with self.assertRaises(ValueError):
            self.ledger.fill(self.cid, "", D("0.1"), D(10), 101)

    def test_validated_order_user_side_type_reduce_only_and_identity(self):
        self.assertEqual(self.ledger.validate_order(self.cid, self.raw_order(left="800"), D("0.001"), "42"), D("0.2"))
        for change in (dict(user=43), dict(text="other"), dict(contract="B_USDT"), dict(id=124), dict(size="-1000"),
                       dict(size="999"), dict(left="1001"), dict(status="other"), dict(price="9"), dict(tif="gtc"),
                       dict(is_reduce_only=True)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.ledger.validate_order(self.cid, self.raw_order(**change), D("0.001"), "42")

    def test_market_close_validates_zero_price_ioc_and_correct_side(self):
        self.cid = self.ledger.new_id()
        self.ledger.prepare(self.cid, Intent("A-USDT", False, D(1), D(10), True, True), 100)
        raw = self.raw_order(size="-1000", price="0", tif="ioc", is_reduce_only=True)
        self.ledger.validate_order(self.cid, raw, D("0.001"), "42")
        with self.assertRaises(ValueError):
            self.ledger.validate_order(self.cid, dict(raw, tif="poc"), D("0.001"), "42")

    def test_trade_identity_sign_time_points_and_duplicates(self):
        self.ledger.validate_order(self.cid, self.raw_order(), D("0.001"), "42")
        rows = self.ledger.trade_rows(self.cid, [self.raw_trade()], D("0.001"), 102)
        self.assertEqual(rows["fill1"]["amount"], D("0.1"))
        for change in (dict(id=None), dict(order_id="other"), dict(text="other"), dict(contract="B_USDT"),
                       dict(size="-100"), dict(size="0"), dict(price="0"), dict(create_time=98), dict(create_time=104),
                       dict(point_fee="1"), dict(size="1001")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.ledger.trade_rows(self.cid, [self.raw_trade(**change)], D("0.001"), 102)
        with self.assertRaises(ValueError):
            self.ledger.trade_rows(self.cid, [self.raw_trade(), self.raw_trade()], D("0.001"), 102)

    def test_rest_cannot_erase_or_change_previously_seen_fill(self):
        self.ledger.validate_order(self.cid, self.raw_order(), D("0.001"), "42")
        self.ledger.fill(self.cid, "fill1", D("0.1"), D(10), 101)
        with self.assertRaises(RuntimeError):
            self.ledger.trade_rows(self.cid, [], D("0.001"), 102)
        with self.assertRaises(ValueError):
            self.ledger.trade_rows(self.cid, [self.raw_trade(price="11")], D("0.001"), 102)

    def test_restore_preserves_slots_age_streak_and_order_fills(self):
        self.ledger.fill(self.cid, "fill1", D("0.1"), D(10), 101)
        self.inventory.cooldowns["A-USDT"] = True, 110
        self.portfolio.slots["A-USDT"].state = "retiring"
        self.portfolio.slots["A-USDT"].force_market = True
        other = RecoveryLedger(self.settings)
        portfolio, inventory = Portfolio(Settings()), InventoryTracker(AdaptiveSettings())
        expected = other.restore(self.state(), portfolio, inventory)
        self.assertEqual(expected, {"A-USDT": D(1)})
        self.assertTrue(other.recovering)
        self.assertEqual(portfolio.slots["A-USDT"].selected_at, 90)
        self.assertTrue(portfolio.slots["A-USDT"].force_market)
        self.assertEqual(inventory.opened_at["A-USDT"], 95)
        self.assertEqual(inventory.streaks["A-USDT"], (True, 1, "first"))
        self.assertEqual(inventory.cooldowns["A-USDT"], (True, 110))
        self.assertEqual(other.filled(self.cid), D("0.1"))

    def test_corrupt_schema_inventory_or_ownership_refuses_restore(self):
        changes = [lambda s: s.update(schema=2), lambda s: s.update(owner="bad"),
                   lambda s: s["expected_positions"].update({"B-USDT": "1"}),
                   lambda s: s["inventory"].update(opened_at={}),
                   lambda s: s["slots"]["A-USDT"].update(force_market="yes"),
                   lambda s: s["orders"][self.cid].update(terminal="yes"),
                   lambda s: s["inventory"].update(streaks={"A-USDT": [True, 0, "first"]}),
                   lambda s: s["inventory"].update(cooldowns={"A-USDT": ["buy", 100]}),
                   lambda s: s["inventory"].update(counted_orders={"A-USDT": [None]})]
        for mutate in changes:
            state = copy.deepcopy(self.state())
            mutate(state)
            with self.subTest(state=state), self.assertRaises(ValueError):
                RecoveryLedger(self.settings).restore(state, Portfolio(Settings()), InventoryTracker(AdaptiveSettings()))

    def test_confirmation_requires_matching_snapshots_and_quiet_time(self):
        self.assertFalse(self.ledger.confirm("a", 100))
        self.assertFalse(self.ledger.confirm("a", 101))
        self.assertFalse(self.ledger.confirm("b", 105))
        self.assertTrue(self.ledger.confirm("b", 110))
        self.ledger.wait("API unavailable")
        self.assertFalse(self.ledger.confirm("b", 111))

    def test_recovery_pairs_include_out_of_current_pool_orders_and_positions(self):
        self.assertEqual(recovery_pairs(dict(recovery=self.state())), {"A-USDT"})
        self.assertEqual(recovery_pairs({}), set())
