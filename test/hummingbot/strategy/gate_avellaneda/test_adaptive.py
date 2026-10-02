import itertools
import unittest
from dataclasses import replace
from decimal import Decimal as D

from hummingbot.strategy.gate_avellaneda.adaptive import (
    AdaptiveSettings, InventoryTracker, LossCircuit, ReturnHistory, entry_rejection,
    exposure_bounds, exposure_scale, limit_exposure, quote_plan, signal_adjustment,
)
from hummingbot.strategy.gate_avellaneda.core import Intent, Settings
from hummingbot.strategy.gate_avellaneda.microstructure import MicroSignal
from test.hummingbot.strategy.gate_avellaneda.test_core import market


class AdaptiveQuoteTests(unittest.TestCase):
    def setUp(self):
        self.risk, self.settings = Settings(), AdaptiveSettings()

    def plan(self, m=None, position="0", age=0, **kwargs):
        return quote_plan(m or market(), D(position), age, self.risk, self.settings, 100, **kwargs)

    def test_defaults_validate_and_invalid_settings_are_rejected(self):
        self.assertTrue(self.settings.enabled)
        invalid = [dict(soft_ratio=D("0.8")), dict(min_size_scale=D("0")), dict(inventory_strength=D("NaN")),
                   dict(micro_risk_bps=D("-1")), dict(max_hold_seconds=120), dict(stop_count=1),
                   dict(max_directional_quote=D("50")), dict(correlation_min_samples=2),
                   dict(correlation_history_samples=10), dict(max_abs_correlation=D("1")),
                   dict(age_ramp_seconds=float("inf")), dict(spread_multiplier=D("0"))]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                AdaptiveSettings(**values)

    def test_flat_quotes_are_passive_cost_covered_and_within_budget(self):
        plan = self.plan()
        self.assertEqual(len(plan.intents), 2)
        self.assertEqual(plan.position_cap, D("20"))
        self.assertLessEqual(plan.intents[0].price, market().bid)
        self.assertGreaterEqual(plan.intents[1].price, market().ask)
        self.assertTrue(all(q.notional <= D("5") and not q.close for q in plan.intents))

    def test_tight_book_can_quote_outside_top_without_relaxing_cost_floor(self):
        m = market(bid=D("9.999"), ask=D("10.001"))
        self.assertIsNotNone(m.rejection(self.risk, 100))
        self.assertIsNone(entry_rejection(m, self.risk, self.settings, 100))
        plan = self.plan(m)
        self.assertLess(plan.intents[0].price, m.bid)
        self.assertGreater(plan.intents[1].price, m.ask)

    def test_depth_volatility_trend_and_indicator_failures_allow_safe_closes(self):
        for changes in [dict(bid_depth=D("1")), dict(volatility=D("1")), dict(trend=D("0.1")),
                        dict(alpha=D("0"), kappa=D("0"), ready=False), dict(maker_fee=D("NaN"))]:
            for position in ("1", "-1"):
                with self.subTest(changes=changes, position=position):
                    plan = self.plan(market(**changes), position)
                    self.assertEqual(len(plan.intents), 1)
                    self.assertTrue(plan.intents[0].close)
                    self.assertLessEqual(plan.intents[0].amount, abs(D(position)))

    def test_stale_crossed_and_invalid_execution_data_prevent_all_quotes(self):
        for changes in [dict(observed_at=80), dict(bid=D("11")), dict(tick=D("0")), dict(bid=D("NaN"))]:
            with self.subTest(changes=changes):
                self.assertEqual(self.plan(market(**changes), "1").intents, [])

    def test_old_inventory_close_can_improve_inside_spread_without_crossing(self):
        plan = self.plan(position="1", age=130)
        close = next(q for q in plan.intents if q.close)
        self.assertGreater(close.price, market().bid)
        self.assertLess(close.price, market().ask)

    def test_inventory_pressure_is_nonlinear_and_age_increases_it(self):
        small = self.plan(position="0.5")
        larger = self.plan(position="1")
        aged = self.plan(position="1", age=120)
        self.assertGreater(larger.pressure, 2 * small.pressure)
        self.assertGreater(aged.pressure, larger.pressure)
        self.assertEqual(larger.stage, "caution")

    def test_near_cap_or_aged_inventory_is_reduce_only_for_both_sides(self):
        for position, age in [("1.6", 0), ("-1.6", 0), ("0.5", 240), ("-0.5", 240)]:
            plan = self.plan(position=position, age=age)
            self.assertEqual(plan.stage, "reduce_only")
            self.assertTrue(plan.intents and all(q.close for q in plan.intents))

    def test_dynamic_caps_and_order_sizes_only_shrink(self):
        for changes in [dict(volatility=D("0.04")), dict(bid_depth=D("50"), ask_depth=D("50"))]:
            plan = self.plan(market(**changes))
            self.assertLess(plan.position_cap, D("20"))
            self.assertLess(plan.order_quote, D("5"))
            self.assertGreaterEqual(plan.position_cap, D("5"))
        self.assertEqual(exposure_scale(market(), self.risk, self.settings), 1)

    def test_shrinking_cap_does_not_shrink_close_amount(self):
        normal = self.plan(position="1.5")
        reduced = self.plan(market(bid_depth=D("50"), ask_depth=D("50")), "1.5")
        self.assertTrue(all(q.close for q in reduced.intents))
        self.assertGreaterEqual(reduced.intents[0].amount, normal.intents[0].amount)

    def test_micro_conflict_zeroes_reference_shift_and_widens_adverse_side(self):
        m = market(bid=D("9.999"), ask=D("10.001"))
        signal = MicroSignal(True, "conflict", reference=D("10.0005"), imbalance=D("0.8"),
                             flow=D("-0.8"), flow_quote=D("100"), flow_sufficient=True, book_flow_conflict=True)
        reference, confidence, buy, sell = signal_adjustment(m, signal, self.settings)
        self.assertEqual(reference, m.mid)
        self.assertEqual(confidence, 0)
        self.assertGreater(buy, sell)
        original = self.plan(m)
        guarded = self.plan(m, signal=signal)
        self.assertLess(guarded.intents[0].price, original.intents[0].price)

    def test_consistent_flow_preserves_bounded_confidence_and_missing_flow_reduces_it(self):
        signal = MicroSignal(True, "normal", reference=D("10.002"), imbalance=D("0.8"), flow=D("0.8"),
                             flow_quote=D("100"), flow_sufficient=True)
        reference, confidence, _, _ = signal_adjustment(market(), signal, self.settings)
        self.assertEqual(confidence, D("0.8"))
        self.assertEqual(reference, D("10.0016"))
        quiet = replace(signal, flow_sufficient=False)
        self.assertEqual(signal_adjustment(market(), quiet, self.settings)[1], D("0.5"))
        self.assertEqual(signal_adjustment(market(), None, self.settings)[0], market().mid)

    def test_micro_pause_and_global_pause_preserve_closes(self):
        signal = MicroSignal(True, "paused", paused=True, reference=D("10.002"))
        for kwargs in [dict(signal=signal), dict(allow_open=False)]:
            plan = self.plan(position="1", **kwargs)
            self.assertTrue(plan.intents and all(q.close for q in plan.intents))
            self.assertEqual(self.plan(**kwargs).intents, [])

    def test_blocked_opening_side_and_minimum_rounding(self):
        plan = self.plan(blocked_side=True)
        self.assertTrue(all(not q.buy for q in plan.intents))
        impossible = market(step=D("0.5"), minimum=D("0.5"), bid_depth=D("50"), ask_depth=D("50"))
        self.assertIsNotNone(entry_rejection(impossible, self.risk, self.settings, 100))

    def test_return_normalization_is_invariant_under_price_unit_change(self):
        a = self.plan()
        m = market(bid=D("999"), ask=D("1001"), volatility=D("0.2"), kappa=D("2"),
                   tick=D("0.1"), step=D("0.00001"), minimum=D("0.00001"))
        b = self.plan(m)
        self.assertEqual([q.price * 100 for q in a.intents], [q.price for q in b.intents])
        self.assertEqual(a.position_cap, b.position_cap)
        gamma = quote_plan(market(), D("0"), 0, replace(self.risk, gamma=D("20")), self.settings, 100)
        self.assertEqual(a.intents, gamma.intents)

    def test_scheduled_funding_cost_widens_only_paying_opening(self):
        for funding in (D("0.001"), D("-0.001")):
            with self.subTest(funding=funding):
                m = market(bid=D("9.999"), ask=D("10.001"), funding=funding, funding_next_at=110)
                normal = self.plan(replace(m, funding_next_at=10000))
                crossing = self.plan(m)
                if funding > 0:
                    self.assertLess(crossing.intents[0].price, normal.intents[0].price)
                    self.assertEqual(crossing.intents[1].price, normal.intents[1].price)
                else:
                    self.assertGreater(crossing.intents[1].price, normal.intents[1].price)
                    self.assertEqual(crossing.intents[0].price, normal.intents[0].price)

    def test_unreliable_proposed_opening_does_not_block_close(self):
        settings = replace(self.settings, min_arrival_attenuation=D("0.9"))
        plan = quote_plan(market(), D("1"), 0, self.risk, settings, 100)
        self.assertTrue(plan.intents and all(q.close for q in plan.intents))
        self.assertIn("too far", plan.entry_reason)
        self.assertIsNotNone(entry_rejection(market(), self.risk, settings, 100))

    def test_invalid_funding_schedule_prevents_opening_but_allows_close(self):
        plan = self.plan(market(funding_interval=0), "1")
        self.assertEqual(plan.entry_reason, "invalid funding schedule")
        self.assertTrue(plan.intents and all(q.close for q in plan.intents))


class InventoryTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = InventoryTracker(AdaptiveSettings())

    def test_partial_fills_do_not_count_multiple_orders_or_reset_age(self):
        for i in range(3):
            self.tracker.fill("A", "order", True, D(i), D(i + 1), False, 100 + i)
        self.assertEqual(self.tracker.age("A", D("3"), 110), 10)
        self.assertIsNone(self.tracker.blocked_side("A", 103))
        self.assertEqual(self.tracker.streaks["A"][1], 1)

    def test_three_opening_orders_trigger_side_cooldown(self):
        for i in range(3):
            self.tracker.fill("A", str(i), False, -D(i), -D(i + 1), False, 100 + i)
        self.assertIs(self.tracker.blocked_side("A", 103), False)
        self.assertIsNone(self.tracker.blocked_side("A", 117))

    def test_interleaved_partial_fills_count_each_opening_order_once(self):
        for i, order in enumerate(("first", "second", "first")):
            self.tracker.fill("A", order, True, D(i), D(i + 1), False, 100 + i)
        self.assertEqual(self.tracker.streaks["A"][1], 2)
        self.assertIsNone(self.tracker.blocked_side("A", 103))

    def test_reductions_preserve_age_and_flattening_clears_state(self):
        self.tracker.fill("A", "open", True, D("0"), D("1"), False, 100)
        self.tracker.fill("A", "close", False, D("1"), D("0.5"), True, 110)
        self.tracker.reconcile({"A": D("0.5")}, 120)
        self.assertEqual(self.tracker.age("A", D("0.5"), 120), 20)
        self.tracker.fill("A", "close2", False, D("0.5"), D("0"), True, 130)
        self.assertEqual(self.tracker.opened_at, {})
        self.assertEqual(self.tracker.streaks, {})

    def test_reversal_restarts_age_and_reconciliation_initializes_unobserved_position(self):
        self.tracker.fill("A", "open", True, D("0"), D("1"), False, 100)
        self.tracker.fill("A", "reverse", False, D("1"), D("-1"), False, 110)
        self.assertEqual(self.tracker.age("A", D("-1"), 120), 10)
        self.tracker.reconcile({"B": D("1")}, 120)
        self.assertEqual(self.tracker.opened_at, {"B": 120})


class ExposureTests(unittest.TestCase):
    def setUp(self):
        self.settings = AdaptiveSettings()
        self.markets = {p: market(p) for p in ("A", "B")}
        self.caps = {p: D("20") for p in self.markets}

    def test_pending_opposite_orders_do_not_cancel_each_other_in_risk_bounds(self):
        pending = [Intent("A", True, D("0.5"), D("10")), Intent("A", False, D("0.5"), D("10"))]
        longs, shorts = exposure_bounds({}, pending)
        self.assertEqual(longs["A"], D("5.05"))
        self.assertEqual(shorts["A"], D("5.05"))

    def test_directional_and_pair_caps_shrink_new_order_to_quantum(self):
        signed = {"A": D("15"), "B": D("9")}
        intents = [Intent("A", True, D("0.5"), D("10")), Intent("B", True, D("0.5"), D("10"))]
        selected = limit_exposure(intents, signed, [], self.markets, self.caps, self.settings)
        self.assertLess(selected[0].amount, D("0.1"))
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].amount % D("0.001"), 0)

    def test_closes_are_allowed_even_when_existing_exposure_exceeds_caps(self):
        close = Intent("A", False, D("3"), D("10"), close=True, market=True)
        result = limit_exposure([close, Intent("B", True, D("0.5"), D("10"))],
                                {"A": D("30")}, [], self.markets, self.caps, self.settings)
        self.assertEqual(result, [close])

    def test_terminal_pending_and_partial_remaining_size_count(self):
        pending = [Intent("A", True, D("0.25"), D("10"))]
        selected = limit_exposure([Intent("A", True, D("0.5"), D("10"))], {"A": D("15")}, pending,
                                  self.markets, self.caps, self.settings)
        self.assertLess(selected[0].amount, D("0.25"))

    def test_every_fill_subset_respects_gross_directional_and_pair_limits(self):
        settings = replace(self.settings, max_gross_quote=D("12"), max_directional_quote=D("9"))
        pending = [Intent("A", True, D("0.2"), D("10")), Intent("A", False, D("0.2"), D("10"))]
        proposed = [Intent(p, buy, D("0.5"), D("10")) for p in ("A", "B") for buy in (True, False)]
        selected = limit_exposure(proposed, {}, pending, self.markets, self.caps, settings)
        orders = pending + selected
        for mask in itertools.product((0, 1), repeat=len(orders)):
            positions = {"A": D("0"), "B": D("0")}
            for fill, order in zip(mask, orders):
                positions[order.pair] += fill * (1 if order.buy else -1) * order.notional
            self.assertLessEqual(sum(abs(v) for v in positions.values()), settings.max_gross_quote)
            self.assertLessEqual(sum(max(D("0"), v) for v in positions.values()), settings.max_directional_quote)
            self.assertLessEqual(sum(max(D("0"), -v) for v in positions.values()), settings.max_directional_quote)
            self.assertTrue(all(abs(v) <= self.caps[p] for p, v in positions.items()))


class CorrelationTests(unittest.TestCase):
    def setUp(self):
        self.history = ReturnHistory(AdaptiveSettings(correlation_min_samples=3, correlation_history_samples=10))

    def samples(self, first, second):
        a = b = D("10")
        for i in range(9):
            self.history.observe("A", i * 5, a)
            self.history.observe("B", i * 5, b)
            a *= 1 + D(first[i % len(first)])
            b *= 1 + D(second[i % len(second)])

    def test_identical_and_opposite_returns_are_filtered(self):
        for second in [["-.001", ".001"], [".001", "-.001"]]:
            self.history.prices.clear()
            self.samples(["-.001", ".001"], second)
            self.assertGreater(abs(self.history.correlation("A", "B")), .99)
            self.assertFalse(self.history.eligible("B", ["A"]))

    def test_low_correlation_can_fill_second_slot(self):
        self.samples(["-.001", ".001", ".002", "-.002"], [".001", ".001", "-.001", "-.001"])
        self.assertAlmostEqual(self.history.correlation("A", "B"), 0, places=8)
        self.assertTrue(self.history.eligible("B", ["A"]))

    def test_unknown_constant_and_gapped_history_wait_for_samples(self):
        self.assertTrue(self.history.eligible("A", []))
        self.assertFalse(self.history.eligible("B", ["A"]))
        self.samples(["0"], ["0"])
        self.assertIsNone(self.history.correlation("A", "B"))
        self.history.observe("A", 100, D("11"))
        self.assertEqual(len(self.history.prices["A"]), 1)
        self.history.observe("A", 100, D("12"))
        self.assertEqual(len(self.history.prices["A"]), 1)

    def test_old_history_cannot_qualify_replacement_without_new_samples(self):
        self.samples(["-.001", ".001", ".002", "-.002"], [".001", ".001", "-.001", "-.001"])
        self.assertTrue(self.history.eligible("B", ["A"], 40))
        self.assertTrue(self.history.eligible("B", ["A"], 45))
        self.assertFalse(self.history.eligible("B", ["A"], 50))

    def test_invalid_price_and_timestamp_do_not_change_history(self):
        for timestamp, price in [(float("nan"), D("10")), (-1, D("10")), (0, D("NaN")), (0, D("0"))]:
            self.history.observe("A", timestamp, price)
        self.assertEqual(self.history.prices, {})


class LossCircuitTests(unittest.TestCase):
    def setUp(self):
        self.circuit = LossCircuit(AdaptiveSettings(stop_window_seconds=100, stop_pause_seconds=10, recovery_seconds=3))

    def test_two_stops_pause_and_duplicates_cannot_extend_circuit(self):
        self.circuit.record("A", 10)
        self.assertTrue(self.circuit.allow_open(10, True, 2))
        self.circuit.record("B", 20)
        self.circuit.record("B", 25)
        self.assertEqual(self.circuit.until, 30)
        self.assertFalse(self.circuit.allow_open(29, True, 2))

    def test_cooldown_requires_continuous_healthy_recovery(self):
        self.circuit.record("A", 10)
        self.circuit.record("B", 20)
        for t in (30, 31, 32):
            self.assertFalse(self.circuit.allow_open(t, True, 2))
        self.assertTrue(self.circuit.allow_open(33, True, 2))
        self.assertEqual(self.circuit.until, 0)

    def test_unhealthy_or_missing_ticks_restart_recovery(self):
        self.circuit.record("A", 10)
        self.circuit.record("B", 20)
        self.assertFalse(self.circuit.allow_open(30, True, 2))
        self.assertFalse(self.circuit.allow_open(31, False, 2))
        self.assertFalse(self.circuit.allow_open(32, True, 2))
        self.assertFalse(self.circuit.allow_open(35, True, 2))
        for t in (36, 37, 38):
            self.assertFalse(self.circuit.allow_open(t, True, 2))
        self.assertTrue(self.circuit.allow_open(39, True, 2))

    def test_old_stop_expires_and_new_stop_extends_active_pause(self):
        self.circuit.record("A", 0)
        self.circuit.record("B", 101)
        self.assertEqual(self.circuit.until, 0)
        self.circuit.record("C", 102)
        self.assertEqual(self.circuit.until, 112)
        self.circuit.record("D", 105)
        self.assertEqual(self.circuit.until, 115)
