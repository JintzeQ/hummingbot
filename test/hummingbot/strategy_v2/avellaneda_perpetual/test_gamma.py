import math
import unittest
from dataclasses import replace
from decimal import Decimal as D

from pydantic import ValidationError

from hummingbot.strategy_v2.utils.avellaneda_perpetual import AvellanedaEngine, AvellanedaSettings, Quote, Snapshot
from hummingbot.strategy_v2.utils.avellaneda_perpetual_gamma import GammaCalibration, GammaController


def btc(t=0, **changes):
    mid = D(str(80000 * (1 + 0.0005 * math.sin(t / 17) + 0.0002 * math.sin(t / 5))))
    return replace(Snapshot(t, mid - D("0.05"), mid + D("0.05"), D("0"), D("100"), D("100"),
                            D("0.1"), D("0.0001"), D("0.0001"), D("1")), **changes)


def calibration_engine(**changes):
    values = dict(gamma_mode="adaptive", warmup_samples=3)
    values.update(changes)
    e = AvellanedaEngine(AvellanedaSettings(**values))
    for t in range(901):
        e.calibration_observations.observe(btc(t), e.config)
    return e


def profile():
    e = calibration_engine()
    return e.calibrate(btc(900))


class CalibrationTests(unittest.TestCase):
    def test_40u_defaults_and_quantized_first_lot(self):
        e = calibration_engine()
        p = e.calibrate(btc(900))
        self.assertEqual(D("40"), e.config.max_position_quote)
        self.assertEqual(D("50"), e.config.capital_budget_quote)
        self.assertEqual(D("0.0001"), p.reference_amount)
        self.assertNotEqual(0.25, p.reference_inventory)
        self.assertAlmostEqual(float(p.reference_amount * p.reference_mid / D("40")), p.reference_inventory)
        raw_ticks = float(p.reference_mid / p.price_tick) * (
            1 - math.exp(-p.reference_inventory * p.gamma_base * p.variance_per_second * p.horizon_seconds))
        self.assertAlmostEqual(2, raw_ticks, places=8)
        self.assertGreaterEqual(max(abs(p.long_bid_change_ticks), abs(p.long_ask_change_ticks)), 1)
        self.assertGreaterEqual(max(p.short_bid_change_ticks, p.short_ask_change_ticks), 1)
        self.assertEqual(p, GammaCalibration.model_validate_json(p.model_dump_json()))

    def test_insufficient_data_zero_volatility_and_absolute_bound_fail(self):
        e = calibration_engine()
        e.calibration_observations.reset("test")
        for t in range(450):
            e.calibration_observations.observe(btc(t), e.config)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            e.calibrate(btc(449))
        e.calibration_observations.reset("test")
        for t in range(901):
            e.calibration_observations.observe(btc(t, bid=D("79999.95"), ask=D("80000.05")), e.config)
        with self.assertRaisesRegex(ValueError, "near-zero"):
            e.calibrate(btc(900))
        e = calibration_engine()
        with self.assertRaisesRegex(ValueError, "outside"):
            e.calibrate(btc(900, price_tick=D("100")))

    def test_reject_invisible_post_only_shift_and_invalid_minimum_lot(self):
        e = calibration_engine()
        s = btc(900)
        with self.assertRaisesRegex(ValueError, "visibly shift"):
            e.calibrate(replace(s, bid=s.mid * D("0.999"), ask=s.mid * D("1.001")))
        with self.assertRaisesRegex(ValueError, "both first orders"):
            e.calibrate(replace(s, min_amount=D("0.001")))
        with self.assertRaisesRegex(ValueError, "both first orders"):
            e.calibrate(replace(s, available=D("5")))

    def test_gap_and_rule_change_require_a_new_continuous_interval(self):
        e = calibration_engine()
        e.calibration_observations.observe(btc(920), e.config)
        self.assertEqual(0, e.calibration_observations.elapsed)
        e.calibration_observations.observe(btc(921, price_tick=D("0.2")), e.config)
        self.assertEqual(1, e.calibration_observations.count)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            e.calibrate(btc(921))

    def test_invalid_profiles_cannot_bypass_equation_or_tick_check(self):
        p = profile().model_dump()
        for values in ({"gamma_base": float("nan")}, {"gamma_base": p["gamma_base"] * 2},
                       {"reference_inventory": 0.5}, {"long_bid_change_ticks": 0, "long_ask_change_ticks": 0},
                       {"reference_amount": D("0.00011")}, {"valid_samples": 5},
                       {"variance_per_second": 0}, {"version": "other"}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                GammaCalibration.model_validate(dict(p, **values))

    def test_metadata_mismatch_and_expired_profile_prevent_live_quotes(self):
        p = profile()
        c = AvellanedaSettings(gamma_mode="adaptive", gamma_calibration=p, dry_run=False)
        g = GammaController(c)
        s = btc(900)
        self.assertIsNone(g.compatibility_error(s))
        for new_s in (replace(s, price_tick=D("0.2")), replace(s, amount_step=D("0.001")),
                      replace(s, min_notional=D("5")), replace(s, timestamp=899),
                      replace(s, timestamp=900 + 86401)):
            self.assertIsNotNone(g.compatibility_error(new_s))
        for changes in ({"max_position_quote": D("20")}, {"trading_pair": "ETH-USDT"},
                        {"horizon_seconds": 60}, {"kappa": 5000}, {"order_amount_quote": D("9")},
                        {"min_spread": D("0.001")}, {"capital_budget_quote": D("60")}):
            c = AvellanedaSettings(gamma_mode="adaptive", gamma_calibration=p, dry_run=False, **changes)
            self.assertIsNotNone(GammaController(c).compatibility_error(s))
        with self.assertRaisesRegex(ValidationError, "requires"):
            AvellanedaSettings(gamma_mode="adaptive", dry_run=False)

    def test_preview_autocalibrates_without_creating_orders(self):
        e = AvellanedaEngine(AvellanedaSettings(gamma_mode="adaptive", warmup_samples=3))
        for t in range(926):
            self.assertEqual([], e.step(btc(t)))
        self.assertIsNotNone(e.gamma.profile)
        self.assertGreater(e.gamma.current, 1)
        self.assertEqual({}, e.orders)
        self.assertEqual(2, len(e.preview))
        self.assertTrue(e.metrics["spread_floor_active"])

    def test_calibration_only_samples_once_per_second(self):
        e = calibration_engine()
        before = e.calibration_observations.count
        for t in (900.1, 900.2, 900.9):
            e.calibration_observations.observe(btc(t), e.config)
        self.assertEqual(before, e.calibration_observations.count)

    def test_calibration_timestamp_jitter_does_not_drop_alternate_seconds(self):
        e = calibration_engine()
        e.calibration_observations.reset("test")
        for t in range(901):
            e.calibration_observations.observe(btc(t + (0.01 if t % 2 else 0.02)), e.config)
        self.assertEqual(901, e.calibration_observations.count)


class AdaptiveTests(unittest.TestCase):
    def controller(self, **changes):
        return GammaController(AvellanedaSettings(gamma_mode="adaptive", gamma_calibration=profile(), **changes))

    def test_inventory_and_volatility_targets(self):
        g = self.controller()
        g.update(900, 0.5, g.profile.variance_per_second)
        self.assertAlmostEqual(g.base * 1.25, g.target)
        self.assertAlmostEqual(g.base * 1.05, g.current)
        g.update(905, -0.5, g.profile.variance_per_second * 4)
        self.assertAlmostEqual(g.base * 1.25 * 1.5, g.target)
        self.assertLessEqual(g.current, g.base * 1.05 * 1.1)

    def test_frequency_and_frozen_updates(self):
        g = self.controller()
        g.update(900, 1, g.profile.variance_per_second * 100)
        previous = g.current
        g.update(904.9, 0, 0)
        self.assertEqual(previous, g.current)
        g.update(910, 0, 0, eligible=False, reason="stale")
        self.assertEqual(previous, g.current)
        self.assertIn("stale", g.reason)
        g.update(911, 0, 0)
        self.assertGreaterEqual(g.current, previous * 0.9)

    def test_smoothing_and_step_limit_are_both_applied(self):
        g = self.controller(gamma_smoothing=0.2)
        g.update(900, 1, g.profile.variance_per_second)
        self.assertAlmostEqual(g.base * 1.1, g.current)
        self.assertAlmostEqual(g.base * 2, g.target)
        slow = self.controller(gamma_smoothing=0.01)
        slow.update(900, 1, slow.profile.variance_per_second)
        self.assertAlmostEqual(slow.base * 1.01, slow.current)

    def test_bounds_stay_finite_under_large_inventory_and_volatility(self):
        g = self.controller()
        for t in range(900, 1100, 5):
            before = g.current
            g.update(t, 10, g.profile.variance_per_second * 1e12)
            self.assertLessEqual(g.current, before * 1.1 + 1e-10)
            self.assertLessEqual(g.current, min(g.base * 3, 10000))
            self.assertTrue(math.isfinite(g.current))
        self.assertEqual(g.bounds[1], g.target)
        self.assertTrue(g.bounds_active)

    def test_center_and_spread_use_same_current_gamma_and_zero_inventory_has_zero_skew(self):
        p = profile()
        e = AvellanedaEngine(AvellanedaSettings(gamma_mode="adaptive", gamma_calibration=p))
        e.variance = p.variance_per_second * 100
        e.gamma.update(900, 0.8, e.variance)
        s = btc(900, position=D("0.0003"))
        e.quotes(s)
        q = float(s.position * s.mid / D("40"))
        self.assertAlmostEqual(float(e.reservation_price / s.mid),
                               math.exp(-q * e.gamma.current * e.variance * 30))
        self.assertAlmostEqual(float(e.spread), e.gamma.current * e.variance * 30
                               + 2 * math.log1p(e.gamma.current / 10000) / e.gamma.current)
        e.quotes(replace(s, position=D("0")))
        self.assertEqual(s.mid, e.reservation_price)
        e.variance = 0
        e.quotes(s)
        self.assertEqual(s.mid, e.reservation_price)
        self.assertGreaterEqual(e.spread, D("0.0004"))

    def test_engine_freezes_on_stale_reconciliation_warmup_and_halt(self):
        p = profile()
        for kind in ("stale", "reconciliation", "warmup", "halt"):
            e = AvellanedaEngine(AvellanedaSettings(gamma_mode="adaptive", gamma_calibration=p))
            e.started = True
            e.initial_equity = D("100")
            for t in range(880, 901):
                e.observe(btc(t))
            s = btc(901)
            if kind == "stale":
                s = replace(s, book_age=11)
            elif kind == "reconciliation":
                e.expected_position, e.reconcile_since = D("0.0001"), 901
            elif kind == "warmup":
                e.samples.clear()
            else:
                e.halt("test")
            before = e.gamma.current
            e.step(s)
            self.assertEqual(before, e.gamma.current)
            self.assertIsNone(e.gamma.last_update)

    def test_40u_position_and_margin_limits_do_not_depend_on_gamma(self):
        p = profile()
        e = AvellanedaEngine(AvellanedaSettings(gamma_mode="adaptive", gamma_calibration=p, dry_run=False))
        s = btc(900)
        capacity = D("40") / s.mid
        quotes = e.quotes(replace(s, position=capacity))
        self.assertEqual(["sell"], [q.side for q in quotes])
        self.assertTrue(quotes[0].close)
        e.step(replace(s, position=capacity + s.amount_step))
        self.assertIn("position notional", e.halt_reason)
        self.assertEqual(50, e.config.capital_budget_quote)

    def test_warmup_after_gap_cancels_existing_live_quotes(self):
        e = AvellanedaEngine(AvellanedaSettings(dry_run=False, warmup_samples=3))
        for t in range(3):
            commands = e.step(btc(t))
        for i, command in enumerate(commands):
            e.register(str(i), command.quote, 2)
        commands = e.step(btc(20))
        self.assertEqual(["cancel", "cancel"], [c.kind for c in commands])
        self.assertIn("warming", e.status)

    def test_expiring_profile_halts_and_cancels_own_live_orders(self):
        p = profile()
        e = AvellanedaEngine(AvellanedaSettings(
            dry_run=False, gamma_mode="adaptive", gamma_calibration=p, calibration_max_age_seconds=900, warmup_samples=3))
        for t in range(1798, 1801):
            commands = e.step(btc(t))
        for i, command in enumerate(commands):
            e.register(str(i), command.quote, 1800)
        commands = e.step(btc(1801))
        self.assertIn("expired", e.halt_reason)
        self.assertEqual(["cancel", "cancel"], [c.kind for c in commands])


class MarkoutTests(unittest.TestCase):
    def test_nonfinite_fill_timestamp_requires_position_reconciliation(self):
        e = AvellanedaEngine(AvellanedaSettings())
        e.register("a", Quote("buy", D("0.0001"), D("80000")), 0)
        e.filled("a", D("0.0001"), float("nan"), "fill")
        self.assertIn("invalid", e.halt_reason)
        self.assertIsNone(e.expected_position)
        self.assertEqual(set(), e.seen_fills)

    def test_actual_fee_and_rebate_estimate_remain_separate_from_equity(self):
        e = AvellanedaEngine(AvellanedaSettings())
        e.register("a", Quote("buy", D("0.0001"), D("80000")), 0)
        e.record_fill_metrics("a", "fill", 0, D("0.0001"), D("80000"), "0.0016")
        for t, mid in ((5, "80010"), (30, "79990"), (60, "80020")):
            e.observe_markouts(btc(t, bid=D(mid) - D("0.05"), ask=D(mid) + D("0.05")))
        row = e.completed_markouts[0]
        self.assertEqual("0.0016", row["actual_fee_quote"])
        self.assertEqual(D("0.00096"), D(row["estimated_uncredited_rebate"]))
        self.assertEqual(D("0.001"), D(row["markouts"]["5"]["value_quote"]))
        self.assertEqual(D("-0.001"), D(row["markouts"]["30"]["value_quote"]))
        self.assertEqual(D("0.002"), D(row["markouts"]["60"]["value_quote"]))
        self.assertEqual([], e.markouts)
        self.assertIsNone(e.initial_equity)

    def test_missing_fee_and_late_observation_are_explicit(self):
        e = AvellanedaEngine(AvellanedaSettings())
        e.register("a", Quote("sell", D("0.0001"), D("80000"), market=True), 0)
        e.record_fill_metrics("a", "fill", 0, D("0.0001"), D("80000"))
        e.observe_markouts(btc(80, bid=D("79989.95"), ask=D("79990.05")))
        row = e.completed_markouts[0]
        self.assertIsNone(row["actual_fee_quote"])
        self.assertIsNone(row["credited_rebate_quote"])
        self.assertFalse(row["funding_attribution_complete"])
        self.assertTrue(row["markouts"]["60"]["late"])
        self.assertEqual(80, row["markouts"]["60"]["observed_delay_seconds"])
        self.assertEqual(D("0.001"), D(row["markouts"]["60"]["value_quote"]))
