import unittest
from dataclasses import replace
from decimal import Decimal as D

from hummingbot.strategy.gate_avellaneda.adaptive import AdaptiveSettings, entry_rejection, quote_plan
from hummingbot.strategy.gate_avellaneda.core import Settings
from hummingbot.strategy.gate_avellaneda.gamma import GammaController, GammaSettings
from hummingbot.strategy.gate_avellaneda.microstructure import MicroSignal
from test.hummingbot.strategy.gate_avellaneda.test_core import market


class GammaControllerTests(unittest.TestCase):
    def setUp(self):
        self.risk, self.adaptive, self.settings = Settings(), AdaptiveSettings(), GammaSettings()
        self.control = GammaController(self.risk, self.adaptive, self.settings)

    def update(self, now, pnl="0", signal=None, **changes):
        return self.control.update(market(observed_at=now, **changes), D(pnl), now, signal)

    def test_settings_reject_invalid_bounds_weights_and_timing(self):
        for changes in [dict(minimum=D("0")), dict(maximum=D("0.5")), dict(loss_weight=D("-1")),
                        dict(micro_weight=D("NaN")), dict(max_step_ratio=D("0")), dict(max_step_ratio=D("1")),
                        dict(reprice_bps=D("0")), dict(ema_seconds=0), dict(update_seconds=float("inf")),
                        dict(min_reprice_seconds=float("nan"))]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                GammaSettings(**changes)
        with self.assertRaises(ValueError):
            GammaController(replace(self.risk, gamma=D("4")), self.adaptive, self.settings)

    def test_healthy_market_and_unknown_pair_use_base_gamma(self):
        self.assertEqual(self.control.value("unseen"), 1)
        self.assertEqual(self.update(100), 1)
        self.assertEqual(self.update(103), 1)
        state = self.control.states["AAA-USDT"]
        self.assertEqual(state.target, 1)
        self.assertTrue(all(value == 0 for value in state.components.values()))

    def test_disabled_controller_preserves_fixed_baseline_without_state(self):
        self.control = GammaController(replace(self.risk, gamma=D("2")), self.adaptive,
                                       replace(self.settings, enabled=False))
        self.assertEqual(self.update(100, pnl="-4", volatility=D("0.04")), 2)
        self.assertEqual(self.control.states, {})

    def test_volatility_and_depth_raise_target_independently(self):
        for changes, component in [(dict(volatility=D("0.02")), "volatility"),
                                   (dict(bid_depth=D("50")), "depth")]:
            with self.subTest(component=component):
                self.control.states.clear()
                self.update(100, **changes)
                state = self.control.states["AAA-USDT"]
                self.assertGreater(state.target, 1)
                self.assertGreater(state.components[component], 0)

    def test_loss_uses_net_loss_limit_and_profit_cannot_lower_baseline(self):
        self.update(100, pnl="-4")
        self.assertEqual(self.control.states["AAA-USDT"].target, D("1.8"))
        self.assertEqual(self.control.states["AAA-USDT"].components["loss"], D("0.8"))
        self.update(103, pnl="2")
        self.assertEqual(self.control.states["AAA-USDT"].target, 1)

    def test_conflict_depth_loss_and_pause_raise_micro_stress(self):
        signals = [MicroSignal(True, "conflict", flow=D("-0.8"), flow_sufficient=True, book_flow_conflict=True),
                   MicroSignal(True, "depth", bid_loss=D("0.7")), MicroSignal(True, "paused", paused=True)]
        for signal in signals:
            with self.subTest(reason=signal.reason):
                self.control.states.clear()
                self.update(100, signal=signal)
                self.assertGreater(self.control.states["AAA-USDT"].components["micro"], 0)
                self.assertGreater(self.control.states["AAA-USDT"].target, 1)

    def test_directional_flow_without_conflict_does_not_duplicate_premium(self):
        self.update(100, signal=MicroSignal(True, "aligned", flow=D("0.9"), flow_sufficient=True))
        self.assertEqual(self.control.states["AAA-USDT"].components["micro"], 0)

    def test_combined_stress_is_clipped_to_maximum(self):
        signal = MicroSignal(True, "paused", paused=True)
        self.update(100, pnl="-20", signal=signal, volatility=D("1"), bid_depth=D("0"))
        state = self.control.states["AAA-USDT"]
        self.assertEqual(state.target, 3)
        self.assertTrue(all(0 <= value <= 1 for value in state.components.values()))
        self.assertEqual(state.value, 1)

    def test_repeated_or_early_ticks_cannot_accumulate_updates(self):
        self.update(100, pnl="-4")
        for now in (100, 101, 102, 99):
            self.assertEqual(self.update(now, pnl="-4"), 1)
        self.assertEqual(self.update(103, pnl="-4"), D("1.1"))

    def test_gap_performs_only_one_bounded_update(self):
        self.update(100, pnl="-4")
        self.assertEqual(self.update(1000, pnl="-4"), D("1.1"))
        self.assertEqual(self.control.states["AAA-USDT"].updated_at, 1000)

    def test_invalid_stale_and_warming_inputs_freeze_gamma(self):
        self.update(100, pnl="-4")
        state = self.control.states["AAA-USDT"]
        for m, pnl, signal in [(market(observed_at=80), D("-4"), None),
                               (market(observed_at=103, ready=False), D("-4"), None),
                               (market(observed_at=103, volatility=D("NaN")), D("-4"), None),
                               (market(observed_at=103), D("NaN"), None),
                               (market(observed_at=103), D("-4"), MicroSignal(False, "stale")),
                               (market(observed_at=103), D("-4"), MicroSignal(True, "invalid", flow=D("NaN")))]:
            self.assertEqual(self.control.update(m, pnl, 103, signal), 1)
            self.assertEqual(state.updated_at, 100)

    def test_stress_recovers_smoothly_and_each_pair_has_independent_state(self):
        last = self.update(100, pnl="-4")
        for now in range(103, 145, 3):
            value = self.update(now, pnl="-4")
            self.assertLessEqual(abs(value - last), last * D("0.1"))
            last = value
        self.assertGreater(last, D("1.7"))
        value = self.update(145)
        self.assertLess(value, last)
        self.assertGreater(value, 1)
        self.assertLessEqual(last - value, last * D("0.1"))
        self.control.update(market(pair="OTHER-USDT", observed_at=145), D("0"), 145)
        self.assertEqual(self.control.value("OTHER-USDT"), 1)
        self.assertEqual(self.control.value("AAA-USDT"), value)

    def test_configured_lower_baseline_stays_within_bounds(self):
        self.control = GammaController(replace(self.risk, gamma=D("0.5")), self.adaptive, self.settings)
        self.update(100, pnl="-4")
        self.assertEqual(self.update(103, pnl="-4"), D("0.55"))
        self.assertEqual(self.control.states["AAA-USDT"].target, D("0.9"))


class GammaQuoteTests(unittest.TestCase):
    def setUp(self):
        self.risk, self.adaptive = Settings(), AdaptiveSettings()
        self.market = market(bid=D("9.9999"), ask=D("10.0001"), volatility=D("0.01"), tick=D("0.00001"))

    def plan(self, gamma, position="0", **kwargs):
        return quote_plan(self.market, D(position), 0, self.risk, self.adaptive, 100, gamma=D(gamma), **kwargs)

    def test_higher_gamma_widens_flat_opening_volatility_component(self):
        base, cautious = self.plan("1"), self.plan("2")
        self.assertLess(cautious.intents[0].price, base.intents[0].price)
        self.assertGreater(cautious.intents[1].price, base.intents[1].price)
        self.assertEqual(cautious.position_cap, base.position_cap)
        self.assertEqual(cautious.order_quote, base.order_quote)

    def test_default_quote_and_screening_use_configured_baseline(self):
        risk = replace(self.risk, gamma=D("2"))
        plan = quote_plan(self.market, D("0"), 0, risk, self.adaptive, 100)
        self.assertEqual(plan.intents, self.plan("2").intents)
        self.assertEqual(plan.gamma, 2)
        extreme = replace(self.risk, gamma=D("50"))
        self.assertIsNotNone(entry_rejection(self.market, extreme, self.adaptive, 100))
        self.assertIsNone(entry_rejection(self.market, extreme, self.adaptive, 100, gamma=D("1")))

    def test_higher_gamma_improves_reducing_side_and_discourages_accumulation(self):
        for position in ("0.5", "-0.5"):
            with self.subTest(position=position):
                base, cautious = self.plan("1", position), self.plan("2", position)
                self.assertEqual(cautious.pressure, 2 * base.pressure)
                base_close = next(i for i in base.intents if i.close)
                cautious_close = next(i for i in cautious.intents if i.close)
                base_open = next(i for i in base.intents if not i.close)
                cautious_open = next(i for i in cautious.intents if not i.close)
                if D(position) > 0:
                    self.assertLess(cautious_close.price, base_close.price)
                    self.assertLess(cautious_open.price, base_open.price)
                else:
                    self.assertGreater(cautious_close.price, base_close.price)
                    self.assertGreater(cautious_open.price, base_open.price)
                if cautious_close.buy:
                    self.assertLessEqual(cautious_close.price, self.market.ask - self.market.tick)
                else:
                    self.assertGreaterEqual(cautious_close.price, self.market.bid + self.market.tick)
                self.assertLessEqual(cautious_close.amount, abs(D(position)))

    def test_invalid_gamma_blocks_openings_but_keeps_safe_closes(self):
        for gamma in ("0", "-1", "NaN", "Infinity"):
            plan = self.plan(gamma, "0.5")
            self.assertTrue(plan.intents and all(i.close for i in plan.intents))
            self.assertEqual(plan.entry_reason, "invalid effective Gamma")
            self.assertIsNotNone(entry_rejection(self.market, self.risk, self.adaptive, 100, gamma=D(gamma)))

    def test_gamma_does_not_override_micro_or_global_opening_pause(self):
        for kwargs in [dict(allow_open=False), dict(signal=MicroSignal(True, "paused", paused=True))]:
            self.assertTrue(all(i.close for i in self.plan("3", "0.5", **kwargs).intents))
            self.assertEqual(self.plan("3", **kwargs).intents, [])

    def test_low_gamma_still_obeys_fee_floor_and_order_budget(self):
        plan = self.plan("0.5")
        floor = self.market.maker_fee + self.risk.min_net_spread / 2
        for intent in plan.intents:
            self.assertGreaterEqual(abs(intent.price - plan.reference) / self.market.mid, floor)
            self.assertLessEqual(intent.notional, D("5"))

    def test_effective_gamma_preserves_return_unit_normalization(self):
        a = self.plan("2")
        scaled = replace(self.market, bid=self.market.bid * 100, ask=self.market.ask * 100,
                         volatility=self.market.volatility * 100, tick=self.market.tick * 100,
                         kappa=self.market.kappa / 100, step=self.market.step / 100, minimum=self.market.minimum / 100)
        b = quote_plan(scaled, D("0"), 0, self.risk, self.adaptive, 100, gamma=D("2"))
        self.assertEqual([i.price * 100 for i in a.intents], [i.price for i in b.intents])

    def test_too_wide_gamma_quotes_are_rejected_by_entry_screening(self):
        self.assertIsNotNone(entry_rejection(self.market, self.risk, self.adaptive, 100, gamma=D("50")))
        self.assertTrue(all(i.close for i in self.plan("50", "0.5").intents))
