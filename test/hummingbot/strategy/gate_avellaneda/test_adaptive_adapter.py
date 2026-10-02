import types
import unittest
from dataclasses import replace
from decimal import Decimal as D

from hummingbot.strategy.gate_avellaneda.adaptive import AdaptiveSettings
from hummingbot.strategy.gate_avellaneda.core import Intent
from hummingbot.strategy.gate_avellaneda.gamma import GammaController, GammaSettings
from hummingbot.strategy.gate_avellaneda.microstructure import MicroSettings
from test.hummingbot.strategy.gate_avellaneda.support import FakeConnector, OrderType, PositionAction, contract, load_adapter

adapter = load_adapter()


class AdaptiveAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.pairs = ["A-USDT", "B-USDT", "C-USDT"]
        adapter.GateAvellanedaPortfolio._initial_contracts = {p: contract(p) for p in self.pairs}
        self.connector = FakeConnector(self.pairs)
        adapter.GateAvellanedaPortfolio._initial_tickers = {
            t["contract"].replace("_", "-"): t for t in self.connector.tickers
        }
        self.config = adapter.GateAvellanedaPortfolioConfig(kline_volatility=dict(enabled=False), deadman=dict(enabled=False), execution=dict(retain_quotes=False), quality_control=dict(enabled=False), account_risk=dict(enabled=False), telemetry=dict(enabled=False), recovery=dict(enabled=False),
            micro=MicroSettings(warmup_seconds=1, min_samples=3, confirm_seconds=1, recover_seconds=2, retire_seconds=5),
            adaptive=AdaptiveSettings(require_correlation=False), dry_run=False,
        )
        self.bot = adapter.GateAvellanedaPortfolio({"gate_io_perpetual": self.connector}, self.config)
        self.sequences = {p: 100 for p in self.pairs}

    def frame(self, now, a_bids=None, a_asks=None, full=False):
        self.bot.current_timestamp = now
        for p in self.pairs:
            self.sequences[p] += 1
            raw = dict(U=self.sequences[p], u=self.sequences[p], t=now * 1000, l="100",
                       b=a_bids if p == "A-USDT" and a_bids is not None else [],
                       a=a_asks if p == "A-USDT" and a_asks is not None else [])
            if p == "A-USDT" and full:
                raw["full"] = True
            self.bot.signal_feed.observe_depth(p, raw, now)

    async def ready(self):
        await self.bot._refresh()
        for t in (100, 100.5, 101):
            self.frame(t)
            self.bot._sample()
        self.bot.next_account = self.bot.next_monitor = 999
        self.bot.portfolio.evaluate(self.bot._snapshots(), 101, True, {}, set(),
                                    entry_check=self.bot._entry_check, candidate_filter=self.bot._candidate_filter)

    async def positions(self, values, pnl="0"):
        self.connector.positions = [dict(contract=p.replace("-", "_"), size=str(int(amount * 1000)),
                                         mode="single", mark_price="10", unrealised_pnl=pnl)
                                    for p, amount in values.items() if amount]
        self.bot.expected_positions = values.copy()
        await self.bot._refresh()

    def clear_orders(self):
        for order_id in list(self.bot.orders):
            self.bot.did_cancel_order(types.SimpleNamespace(order_id=order_id))
        self.connector.open_orders = []

    async def test_shallow_market_still_quotes_close_before_retirement(self):
        await self.ready()
        await self.positions({"A-USDT": D("1")})
        self.frame(102, a_bids=[dict(p="9.99", s="5000")], a_asks=[dict(p="10.01", s="5000")])
        self.bot.on_tick()
        orders = [o for o in self.connector.sent if o[2] == "A-USDT"]
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0][5]["position_action"], PositionAction.CLOSE)
        self.assertEqual(self.bot.portfolio.slots["A-USDT"].state, "active")
        self.assertIn("depth", self.bot.plans["A-USDT"].entry_reason)

    async def test_tight_market_quotes_outside_best_prices_with_cost_floor(self):
        await self.ready()
        self.frame(102, a_bids=[dict(p="9.999", s="100000")], a_asks=[dict(p="10.001", s="100000")], full=True)
        self.bot.on_tick()
        orders = [o for o in self.connector.sent if o[2] == "A-USDT"]
        self.assertEqual(len(orders), 2)
        self.assertLess(orders[0][5]["price"], D("9.999"))
        self.assertGreater(orders[1][5]["price"], D("10.001"))
        self.assertNotIn("book spread below fee", self.bot.format_status())

    async def test_near_position_cap_prevents_opening_and_improves_passive_close(self):
        await self.ready()
        await self.positions({"A-USDT": D("1.6")})
        self.bot.on_tick()
        orders = [o for o in self.connector.sent if o[2] == "A-USDT"]
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0][5]["position_action"], PositionAction.CLOSE)
        self.assertGreater(orders[0][5]["price"], D("9.99"))
        self.assertLess(orders[0][5]["price"], D("10.01"))
        self.assertEqual(self.bot.plans["A-USDT"].stage, "reduce_only")

    async def test_smaller_dynamic_cap_cancels_pending_openings_before_requote(self):
        await self.ready()
        self.bot.on_tick()
        self.assertEqual(len(self.connector.sent), 4)
        self.config.adaptive = replace(self.config.adaptive, volatility_reference=D("0.00005"))
        self.frame(102)
        self.bot.on_tick()
        self.assertEqual(set(self.connector.cancels), set(self.bot.orders))
        self.assertEqual(len(self.connector.sent), 4)
        self.clear_orders()
        await self.bot._refresh()
        self.bot.on_tick()
        new = self.connector.sent[4:]
        self.assertTrue(new)
        self.assertTrue(all(o[3] * o[5]["price"] <= D("1.25") for o in new))

    async def test_unique_same_side_opening_fills_block_that_side(self):
        await self.ready()
        for i in range(3):
            self.bot._submit(Intent("A-USDT", True, D("0.1"), D("9.99")))
            oid = self.connector.sent[-1][0]
            self.bot.did_fill_order(types.SimpleNamespace(order_id=oid, amount=D("0.1")))
            self.bot.did_complete_buy_order(types.SimpleNamespace(order_id=oid))
        self.connector.open_orders = []
        await self.positions({"A-USDT": D("0.3")})
        self.bot.on_tick()
        new = [o for o in self.connector.sent[3:] if o[2] == "A-USDT"]
        self.assertTrue(new)
        self.assertTrue(all(o[5]["position_action"] == PositionAction.CLOSE for o in new))
        self.assertIs(self.bot.inventory.blocked_side("A-USDT", 101), True)

    async def test_partial_fills_of_one_order_do_not_trigger_streak_cooldown(self):
        await self.ready()
        self.bot._submit(Intent("A-USDT", True, D("0.3"), D("9.99")))
        oid = self.connector.sent[-1][0]
        for _ in range(3):
            self.bot.did_fill_order(types.SimpleNamespace(order_id=oid, amount=D("0.1")))
        self.bot.did_complete_buy_order(types.SimpleNamespace(order_id=oid))
        self.connector.open_orders = []
        await self.positions({"A-USDT": D("0.3")})
        self.bot.on_tick()
        self.assertTrue(any(o[2] == "A-USDT" and o[1] and o[5]["position_action"] == PositionAction.OPEN
                            for o in self.connector.sent[1:]))

    async def test_holding_time_retires_pair_without_overriding_loss_policy(self):
        self.config.adaptive = replace(self.config.adaptive, age_ramp_seconds=2, max_hold_seconds=5)
        await self.ready()
        await self.positions({"A-USDT": D("1")})
        for t in range(102, 107):
            self.frame(t)
        self.bot.on_tick()
        slot = self.bot.portfolio.slots["A-USDT"]
        self.assertEqual(slot.state, "retiring")
        self.assertIn("holding time", slot.reason)
        self.assertFalse(slot.force_market)
        self.assertNotIn("A-USDT", self.bot.portfolio.excluded)

    async def test_loss_stop_overrides_circuit_and_stale_micro_data(self):
        await self.ready()
        await self.positions({"A-USDT": D("1")}, pnl="-5")
        self.bot.loss_circuit.until = 200
        self.bot.signal_feed.reset()
        self.bot.on_tick()
        orders = [o for o in self.connector.sent if o[2] == "A-USDT"]
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0][4], OrderType.MARKET)
        self.assertEqual(orders[0][3], D("1"))
        self.assertFalse(self.bot.entry_allowed)

    async def test_two_real_pair_stops_block_replacement_and_keep_both_market_exits(self):
        await self.ready()
        await self.positions({"A-USDT": D("1"), "B-USDT": D("-1")}, pnl="-5")
        self.bot.on_tick()
        self.assertEqual(len(self.connector.sent), 2)
        self.assertTrue(all(o[4] == OrderType.MARKET and o[5]["position_action"] == PositionAction.CLOSE
                            for o in self.connector.sent))
        self.assertFalse(self.bot.entry_allowed)
        self.assertNotIn("C-USDT", self.bot.portfolio.slots)
        self.assertEqual(len(self.bot.loss_circuit.events), 2)

    async def test_circuit_cancels_openings_while_existing_close_remains_working(self):
        await self.ready()
        await self.positions({"A-USDT": D("1")})
        self.bot.on_tick()
        close_ids = {oid for oid, order in self.bot.orders.items() if order.intent.close}
        open_ids = set(self.bot.orders) - close_ids
        self.bot.loss_circuit.record("X", 101)
        self.bot.loss_circuit.record("Y", 101)
        self.bot.on_tick()
        self.assertTrue(open_ids.issubset(set(self.connector.cancels)))
        self.assertFalse(close_ids & set(self.connector.cancels))

    async def test_unknown_correlation_leaves_second_slot_vacant_and_status_explains(self):
        self.bot.return_history.settings = replace(self.config.adaptive, require_correlation=True)
        await self.ready()
        self.assertEqual(len(self.bot.portfolio.slots), 1)
        self.assertIn("correlation unavailable", self.bot.format_status())
        self.bot.on_tick()
        self.assertEqual(len(self.connector.sent), 2)

    async def test_selection_skips_correlated_candidate_and_fills_with_independent_pair(self):
        await self.ready()
        self.bot.return_history.settings = replace(self.config.adaptive, require_correlation=True,
                                                  correlation_min_samples=3, correlation_history_samples=10)
        self.bot.return_history.prices.clear()
        patterns = {"A-USDT": ("-.001", ".001", ".002", "-.002"),
                    "B-USDT": ("-.001", ".001", ".002", "-.002"),
                    "C-USDT": (".001", ".001", "-.001", "-.001")}
        for pair, pattern in patterns.items():
            price = D("10")
            for i in range(9):
                self.bot.return_history.observe(pair, 60 + i * 5, price)
                price *= 1 + D(pattern[i % 4])
        self.bot.portfolio.slots.clear()
        self.bot.portfolio.evaluate(self.bot._snapshots(), 101, True, {}, set(),
                                    entry_check=self.bot._entry_check, candidate_filter=self.bot._candidate_filter)
        self.assertEqual(set(self.bot.portfolio.slots), {"A-USDT", "C-USDT"})
        self.assertIn("B-USDT", self.bot.selection_rejections)

    async def test_stale_ws_cancels_orders_without_simulated_rest_replacement(self):
        await self.ready()
        self.bot.on_tick()
        self.bot.current_timestamp = 104
        self.bot.on_tick()
        self.assertEqual(set(self.connector.cancels), set(self.bot.orders))
        self.assertEqual(len(self.connector.sent), 4)
        self.assertEqual(len(self.bot.portfolio.slots), 2)

    async def test_terminal_exchange_order_visibility_still_counts_in_exposure(self):
        await self.ready()
        self.bot._submit(Intent("A-USDT", True, D("0.5"), D("9.99")))
        oid = self.connector.sent[-1][0]
        self.bot.did_cancel_order(types.SimpleNamespace(order_id=oid))
        await self.bot._refresh()
        signed, pending, caps = self.bot._exposure_inputs(self.bot._snapshots())
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].amount, D("0.5"))
        self.assertGreaterEqual(pending[0].price, D("10"))
        self.assertEqual(caps["A-USDT"], D("20"))

    async def test_conflict_requotes_at_neutral_reference_after_cancel_and_rest_confirmation(self):
        await self.ready()
        self.bot.on_tick()
        self.bot.signal_feed.observe_trade("A-USDT", 1, 102, "9.99", "10", False, 102)
        self.frame(102, a_bids=[dict(p="9.99", s="500000")])
        self.bot.on_tick()
        self.assertEqual(self.bot.plans["A-USDT"].confidence, 0)
        self.assertEqual(self.bot.plans["A-USDT"].reference, D("10"))
        self.assertTrue(self.connector.cancels)
        self.clear_orders()
        await self.bot._refresh()
        self.bot.on_tick()
        self.assertTrue(any(o[2] == "A-USDT" for o in self.connector.sent[4:]))

    async def test_recovered_circuit_forces_selection_check_without_waiting_monitor_period(self):
        await self.ready()
        self.bot.portfolio.slots.clear()
        self.bot.loss_circuit.until = 101
        self.bot.loss_circuit.settings = replace(self.config.adaptive, recovery_seconds=2)
        self.bot.entry_allowed = False
        for t in (101, 102):
            self.frame(t)
            self.bot.on_tick()
            self.assertEqual(self.bot.portfolio.slots, {})
        self.frame(103)
        self.bot.on_tick()
        self.assertTrue(self.bot.entry_allowed)
        self.assertEqual(len(self.bot.portfolio.slots), 2)

    def test_gamma_config_checks_baseline_and_keeps_legacy_mode_available(self):
        with self.assertRaises(ValueError):
            adapter.GateAvellanedaPortfolioConfig(kline_volatility=dict(enabled=False), deadman=dict(enabled=False), execution=dict(retain_quotes=False), quality_control=dict(enabled=False), risk={"gamma": "4"})
        fixed = adapter.GateAvellanedaPortfolioConfig(kline_volatility=dict(enabled=False), deadman=dict(enabled=False), execution=dict(retain_quotes=False), quality_control=dict(enabled=False), risk={"gamma": "4"}, gamma_control={"enabled": False})
        self.assertEqual(fixed.risk.gamma, 4)
        legacy = adapter.GateAvellanedaPortfolioConfig(kline_volatility=dict(enabled=False), deadman=dict(enabled=False), execution=dict(retain_quotes=False), quality_control=dict(enabled=False), risk={"gamma": "4"}, adaptive={"enabled": False})
        self.assertEqual(legacy.risk.gamma, 4)

    async def test_pair_net_loss_adapts_gamma_without_affecting_other_pair(self):
        await self.ready()
        await self.positions({"A-USDT": D("1")}, pnl="-4")
        self.bot.on_tick()
        for now in (102, 103, 104):
            self.frame(now)
        self.bot.on_tick()
        self.assertGreater(self.bot.gamma.value("A-USDT"), 1)
        self.assertEqual(self.bot.gamma.states["A-USDT"].target, D("1.8"))
        self.assertEqual(self.bot.gamma.value("B-USDT"), 1)
        self.assertEqual(self.bot.plans["A-USDT"].gamma, self.bot.gamma.value("A-USDT"))
        self.assertIn("gamma_target=1.8", self.bot.format_status())

    async def test_material_gamma_change_waits_for_cancel_and_rest_before_requote(self):
        await self.ready()
        await self.positions({"A-USDT": D("1")})
        self.bot.on_tick()
        before = len(self.connector.sent)
        old_a = {oid for oid, order in self.bot.orders.items() if order.intent.pair == "A-USDT"}
        for now in (102, 103, 104):
            self.frame(now)
        state = self.bot.gamma.states["A-USDT"]
        state.value, state.updated_at = D("3"), 104
        self.bot.on_tick()
        self.assertTrue(old_a.issubset(set(self.connector.cancels)))
        self.assertEqual(len(self.connector.sent), before)
        self.clear_orders()
        await self.bot._refresh()
        self.frame(105)
        self.bot.on_tick()
        self.assertEqual(self.bot.last_gamma_values["A-USDT"], D("3"))
        self.assertTrue(any(order[2] == "A-USDT" for order in self.connector.sent[before:]))

    async def test_tiny_gamma_change_does_not_churn_working_quotes(self):
        await self.ready()
        self.bot.on_tick()
        for now in (102, 103, 104):
            self.frame(now)
        state = self.bot.gamma.states["A-USDT"]
        state.value, state.updated_at = D("1.001"), 104
        self.bot.on_tick()
        self.assertEqual(self.connector.cancels, [])
        self.assertEqual(len(self.connector.sent), 4)
        self.assertEqual(self.bot.last_gamma_values["A-USDT"], 1)

    async def test_gamma_reprice_respects_minimum_submission_interval(self):
        await self.ready()
        await self.positions({"A-USDT": D("1")})
        self.bot.on_tick()
        state = self.bot.gamma.states["A-USDT"]
        state.value, state.updated_at = D("3"), 102
        self.frame(102)
        self.bot.on_tick()
        self.assertEqual(self.connector.cancels, [])
        self.frame(103)
        self.bot.on_tick()
        self.assertTrue(self.connector.cancels)
        self.assertEqual(len(self.connector.sent), 4)

    async def test_gamma_does_not_update_on_stale_book_or_unreconciled_account(self):
        await self.ready()
        await self.positions({"A-USDT": D("1")}, pnl="-4")
        self.bot.on_tick()
        state = self.bot.gamma.states["A-USDT"]
        self.frame(104)
        self.bot.account_dirty = True
        self.bot.on_tick()
        self.assertEqual(state.updated_at, 101)
        self.bot.account_dirty = False
        self.bot.signal_feed.reset()
        self.bot.on_tick()
        self.assertEqual(state.updated_at, 101)
        self.assertEqual(state.value, 1)

    async def test_observe_mode_does_not_apply_micro_stress_to_gamma(self):
        self.config.micro = replace(self.config.micro, mode="observe")
        await self.ready()
        self.bot.on_tick()
        self.bot.signal_feed.observe_trade("A-USDT", 1, 102, "9.99", "10", False, 102)
        for now in (102, 103, 104):
            self.frame(now, a_bids=[dict(p="9.99", s="500000")])
        self.bot.on_tick()
        self.assertTrue(self.bot.signals["A-USDT"].book_flow_conflict)
        self.assertEqual(self.bot.gamma.states["A-USDT"].components["micro"], 0)
        self.assertEqual(self.bot.gamma.value("A-USDT"), 1)

    async def test_fixed_gamma_is_connected_to_adaptive_quotes(self):
        self.config.risk = replace(self.config.risk, gamma=D("2"))
        self.config.gamma_control = GammaSettings(enabled=False)
        self.bot.gamma = GammaController(self.config.risk, self.config.adaptive, self.config.gamma_control)
        await self.ready()
        self.bot.on_tick()
        self.assertEqual(self.bot.plans["A-USDT"].gamma, 2)
        self.assertEqual(self.bot.gamma.states, {})
        self.assertIn("gamma_mode=fixed", self.bot.format_status())

    async def test_high_gamma_cannot_delay_loss_market_exit_with_stale_ws(self):
        await self.ready()
        self.bot.on_tick()
        self.bot.gamma.states["A-USDT"].value = D("3")
        self.clear_orders()
        await self.positions({"A-USDT": D("1")}, pnl="-5")
        self.bot.signal_feed.reset()
        self.bot.on_tick()
        close = self.connector.sent[-1]
        self.assertEqual(close[2], "A-USDT")
        self.assertEqual(close[3], D("1"))
        self.assertEqual(close[4], OrderType.MARKET)
        self.assertEqual(close[5]["position_action"], PositionAction.CLOSE)

    async def test_observation_quotes_also_refresh_on_material_gamma_change(self):
        self.config.dry_run = True
        await self.ready()
        await self.positions({"A-USDT": D("1")})
        self.bot.on_tick()
        old_prices = [intent.price for intent in self.bot.last_quotes["A-USDT"]]
        for now in (102, 103, 104):
            self.frame(now)
        state = self.bot.gamma.states["A-USDT"]
        state.value, state.updated_at = D("3"), 104
        self.bot.on_tick()
        self.assertNotEqual(old_prices, [intent.price for intent in self.bot.last_quotes["A-USDT"]])
        self.assertEqual(self.bot.last_gamma_values["A-USDT"], 3)
        self.assertEqual(self.connector.sent, [])
