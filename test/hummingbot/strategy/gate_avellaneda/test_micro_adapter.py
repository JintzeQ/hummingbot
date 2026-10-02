import types
import unittest
from dataclasses import replace
from decimal import Decimal as D

from hummingbot.strategy.gate_avellaneda.microstructure import MicroSettings
from test.hummingbot.strategy.gate_avellaneda.support import FakeConnector, OrderType, PositionAction, contract, load_adapter

adapter = load_adapter()


class MicroAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.pairs = ["A-USDT", "B-USDT", "C-USDT"]
        adapter.GateAvellanedaPortfolio._initial_contracts = {p: contract(p) for p in self.pairs}
        self.connector = FakeConnector(self.pairs)
        adapter.GateAvellanedaPortfolio._initial_tickers = {
            t["contract"].replace("_", "-"): t for t in self.connector.tickers
        }
        self.config = adapter.GateAvellanedaPortfolioConfig(deadman=dict(enabled=False), execution=dict(retain_quotes=False), quality_control=dict(enabled=False), account_risk=dict(enabled=False), telemetry=dict(enabled=False), recovery=dict(enabled=False),
            adaptive=dict(enabled=False),
            micro=MicroSettings(warmup_seconds=1, min_samples=3, confirm_seconds=1, recover_seconds=2, retire_seconds=5),
        )
        self.bot = adapter.GateAvellanedaPortfolio({"gate_io_perpetual": self.connector}, self.config)
        self.sequence = {p: 100 for p in self.pairs}

    def frames(self, timestamp, a_bids=None, a_asks=None, full=False):
        self.bot.current_timestamp = timestamp
        for pair in self.pairs:
            self.sequence[pair] += 1
            raw = dict(U=self.sequence[pair], u=self.sequence[pair], t=timestamp * 1000, l="100",
                       b=a_bids if pair == "A-USDT" and a_bids is not None else [],
                       a=a_asks if pair == "A-USDT" and a_asks is not None else [])
            if pair == "A-USDT" and full:
                raw["full"] = True
            self.bot.signal_feed.observe_depth(pair, raw, timestamp)

    async def ready(self, live=True):
        self.config.dry_run = not live
        await self.bot._refresh()
        for t in (100, 100.5, 101):
            self.frames(t)
            self.bot._sample()
        self.bot.next_account = self.bot.next_monitor = 999
        self.bot.portfolio.evaluate(self.bot._snapshots(), 101, True, {}, set())

    async def position(self, pnl="0"):
        self.connector.positions = [dict(contract="A_USDT", size="1000", mode="single", mark_price="10", unrealised_pnl=pnl)]
        self.bot.expected_positions = {"A-USDT": D("1")}
        await self.bot._refresh()

    def collapse(self):
        self.frames(102, a_bids=[dict(p="9.99", s="20000")])
        self.bot._snapshots()
        self.frames(103)
        self.bot._snapshots()

    async def test_rest_only_cannot_select_or_quote_when_micro_enabled(self):
        await self.bot._refresh()
        self.bot._sample()
        self.bot.next_account = 999
        self.bot.on_tick()
        self.assertEqual(self.bot.portfolio.slots, {})
        self.assertEqual(self.connector.sent, [])

    async def test_dry_run_uses_ws_candidates_and_exposes_signal_status(self):
        await self.ready(live=False)
        self.bot.on_tick()
        self.assertEqual(set(self.bot.portfolio.slots), {"A-USDT", "B-USDT"})
        self.assertEqual(self.connector.sent, [])
        self.assertIn("micro=protect", self.bot.format_status())
        self.assertEqual(len(self.bot.last_quotes), 2)

    async def test_collapse_pauses_opening_but_preserves_reduce_only_quotes(self):
        await self.ready()
        await self.position()
        self.collapse()
        self.bot.on_tick()
        orders = [o for o in self.connector.sent if o[2] == "A-USDT"]
        self.assertEqual(len(orders), 1)
        self.assertFalse(orders[0][1])
        self.assertEqual(orders[0][5]["position_action"], PositionAction.CLOSE)
        self.assertEqual(orders[0][4], OrderType.LIMIT_MAKER)
        self.assertTrue(self.bot.signals["A-USDT"].paused)

    async def test_flat_paused_pair_keeps_slot_and_other_pair_can_quote(self):
        await self.ready()
        self.collapse()
        self.bot.on_tick()
        self.assertFalse(any(o[2] == "A-USDT" for o in self.connector.sent))
        self.assertTrue(any(o[2] == "B-USDT" for o in self.connector.sent))
        self.assertIn("A-USDT", self.bot.portfolio.slots)
        self.assertNotIn("C-USDT", self.bot.portfolio.slots)

    async def test_observe_records_anomaly_without_changing_orders_or_retiring(self):
        self.config.micro = replace(self.config.micro, mode="observe")
        await self.ready()
        self.collapse()
        self.bot.on_tick()
        orders = [o for o in self.connector.sent if o[2] == "A-USDT"]
        self.assertEqual(len(orders), 2)
        self.assertTrue(all(o[5]["position_action"] == PositionAction.OPEN for o in orders))
        for t in range(104, 110):
            self.frames(t)
            self.bot.on_tick()
        self.assertEqual(self.bot.portfolio.slots["A-USDT"].state, "active")
        self.assertTrue(self.bot.signals["A-USDT"].retire)

    async def test_soft_conflict_cancels_old_quotes_and_requotes_smaller_opening(self):
        await self.ready()
        self.bot.on_tick()
        first_buy = next(o for o in self.connector.sent if o[2] == "A-USDT" and o[1])
        self.bot.signal_feed.observe_trade("A-USDT", 1, 102, "9.99", "10", False, 102)
        self.frames(102, a_bids=[dict(p="9.99", s="500000")])
        self.bot.on_tick()
        self.frames(103)
        self.bot.on_tick()
        ids = {oid for oid, order in self.bot.orders.items() if order.intent.pair == "A-USDT"}
        self.assertTrue(ids.issubset(set(self.connector.cancels)))
        for oid in ids:
            self.bot.did_cancel_order(types.SimpleNamespace(order_id=oid))
        self.connector.open_orders = [o for o in self.connector.open_orders if o["text"] not in ids]
        await self.bot._refresh()
        self.bot.on_tick()
        new_buy = [o for o in self.connector.sent if o[2] == "A-USDT" and o[1]][-1]
        self.assertLess(new_buy[3], first_buy[3])
        self.assertEqual(self.bot.signals["A-USDT"].buy_scale, D("0.5"))

    async def test_reference_move_reprices_early_but_respects_minimum_interval(self):
        await self.ready()
        self.bot.on_tick()
        self.frames(102, a_bids=[dict(p="10.01", s="100000")],
                    a_asks=[dict(p="10.03", s="100000")], full=True)
        self.bot.on_tick()
        self.assertEqual(self.connector.cancels, [])
        self.frames(103)
        self.bot.on_tick()
        ids = {oid for oid, order in self.bot.orders.items() if order.intent.pair == "A-USDT"}
        self.assertTrue(ids.issubset(set(self.connector.cancels)))
        self.assertEqual(self.bot.next_quote["A-USDT"], 0)

    async def test_sequence_gap_cancels_pair_without_churning_and_requests_snapshot(self):
        await self.ready()
        self.bot.on_tick()
        self.bot.current_timestamp = 102
        self.bot.signal_feed.observe_depth("A-USDT", dict(U=110, u=110, t=102000, b=[], a=[], l="100"), 102)
        self.frames(102)
        self.bot.on_tick()
        self.assertIn("A-USDT", self.bot.portfolio.slots)
        self.assertNotIn("C-USDT", self.bot.portfolio.slots)
        self.assertFalse(self.bot.signals["A-USDT"].ready)
        self.connector.book["id"] = 109
        await self.bot._refresh()
        params = [r for path, r in self.connector.requests if path == "book"][-1]["params"]
        self.assertEqual(params["limit"], 100)
        self.assertEqual(params["with_id"], "true")

    async def test_healthy_ws_does_not_get_overwritten_by_periodic_rest(self):
        await self.ready()
        before = len([r for r in self.connector.requests if r[0] == "book"])
        self.bot.next_books = 0
        await self.bot._refresh()
        self.assertEqual(before, len([r for r in self.connector.requests if r[0] == "book"]))
        self.assertTrue(self.bot.signal_feed.signal("A-USDT", 101).ready)

    async def test_loss_stop_overrides_missing_ws_and_all_micro_guards(self):
        await self.ready()
        await self.position(pnl="-5")
        self.bot.signal_feed.reset()
        self.bot.on_tick()
        orders = [o for o in self.connector.sent if o[2] == "A-USDT"]
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0][4], OrderType.MARKET)
        self.assertEqual(orders[0][3], D("1"))
        self.assertEqual(orders[0][5]["position_action"], PositionAction.CLOSE)
        self.assertIn("A-USDT", self.bot.portfolio.excluded)

    async def test_persistent_anomaly_uses_general_exit_not_loss_exclusion(self):
        await self.ready()
        await self.position()
        self.collapse()
        for t in range(104, 109):
            self.frames(t)
            self.bot._snapshots()
        self.bot.on_tick()
        self.assertEqual(self.bot.portfolio.slots["A-USDT"].state, "retiring")
        self.assertFalse(self.bot.portfolio.slots["A-USDT"].force_market)
        self.assertNotIn("A-USDT", self.bot.portfolio.excluded)
        self.assertEqual([o for o in self.connector.sent if o[2] == "A-USDT"][0][4], OrderType.LIMIT_MAKER)

    async def test_exclusive_feed_ownership_and_stop_detaches(self):
        with self.assertRaises(ValueError):
            adapter.GateAvellanedaPortfolio({"gate_io_perpetual": self.connector}, self.config)
        await self.bot.on_stop()
        self.assertFalse(hasattr(self.connector, "_gate_market_signal_feed"))

    def test_nested_config_validates_mode_and_thresholds(self):
        config = adapter.GateAvellanedaPortfolioConfig(deadman=dict(enabled=False), execution=dict(retain_quotes=False), quality_control=dict(enabled=False), micro=dict(mode="observe", micro_weight="0.1"))
        self.assertEqual(config.micro.micro_weight, D("0.1"))
        for values in [dict(mode="bad"), dict(micro_weight="NaN"), dict(max_age=0), dict(depth_floor_ratio="1")]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                adapter.GateAvellanedaPortfolioConfig(deadman=dict(enabled=False), execution=dict(retain_quotes=False), quality_control=dict(enabled=False), micro=values)
