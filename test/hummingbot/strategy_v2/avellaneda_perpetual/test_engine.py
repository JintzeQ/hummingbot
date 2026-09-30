import unittest
from dataclasses import replace
from decimal import Decimal as D

from pydantic import ValidationError

from hummingbot.strategy_v2.utils.avellaneda_perpetual import AvellanedaEngine, AvellanedaSettings, Quote, Snapshot


def snapshot(timestamp=0, **overrides):
    s = Snapshot(timestamp, D("99.99"), D("100.01"), D("0"), D("100"), D("100"),
                 D("0.01"), D("0.001"), D("0.001"), D("1"))
    return replace(s, **overrides)


def engine(**overrides):
    # Retain the original 20U scenarios as regression coverage after defaults
    # move to 40U. New-default risk boundaries are tested separately.
    values = dict(dry_run=False, warmup_samples=3, max_position_quote=D("20"), capital_budget_quote=D("30"))
    values.update(overrides)
    return AvellanedaEngine(AvellanedaSettings(**values))


def warmed(**overrides):
    e = engine(**overrides)
    for t in range(3):
        e.step(snapshot(t))
    return e


class ModelTests(unittest.TestCase):
    def test_fee_rates_and_cost_floor(self):
        c = AvellanedaSettings()
        self.assertEqual(D("0.00008"), c.maker_fee * (1 - c.rebate_rate))
        self.assertEqual(D("0.00026"), c.fee_floor)
        self.assertEqual(D("0.00028"), (c.maker_fee + c.taker_fee) * (1 - c.rebate_rate))

    def test_invalid_configurations(self):
        for kwargs in ({"risk_factor": 0}, {"risk_factor": float("nan")},
                       {"horizon_seconds": float("inf")}, {"maker_fee": D("NaN")},
                       {"rebate_rate": D("1.01")}, {"connector": "binance_perpetual"},
                       {"trading_pair": "BTC-USD"}, {"warmup_samples": 201},
                       {"capital_budget_quote": D("1")}, {"max_spread": D("0.0001")},
                       {"order_amount_quote": D("41")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                AvellanedaSettings(**kwargs)

    def test_long_and_short_shift_reservation_price(self):
        e = engine()
        e.variance = 0.0001
        e.quotes(snapshot(position=D("0.1")))
        self.assertLess(e.reservation_price, D("100"))
        e.quotes(snapshot(position=D("-0.1")))
        self.assertGreater(e.reservation_price, D("100"))

    def test_higher_volatility_widens_spread(self):
        e = engine()
        e.quotes(snapshot())
        low = e.spread
        e.variance = 0.00001
        e.quotes(snapshot())
        self.assertGreater(e.spread, low)

    def test_quote_rounding_and_post_only(self):
        e = engine()
        for q in e.quotes(snapshot()):
            self.assertEqual(0, q.price % D("0.01"))
            self.assertEqual(0, q.amount % D("0.001"))
            self.assertLessEqual(q.amount * q.price, e.config.order_amount_quote)
            if q.side == "buy":
                self.assertLessEqual(q.price, D("99.99"))
            else:
                self.assertGreaterEqual(q.price, D("100.01"))

    def test_fee_floor_with_no_rebate(self):
        e = engine(rebate_rate=D("0"), min_spread=D("0.0001"))
        e.quotes(snapshot())
        self.assertGreaterEqual(e.spread, D("0.0005"))

    def test_reduce_only_side_is_clipped_to_current_position(self):
        e = engine()
        for position, close_side in ((D("0.019"), "sell"), (D("-0.019"), "buy")):
            quotes = e.quotes(snapshot(position=position))
            close = next(q for q in quotes if q.side == close_side)
            self.assertTrue(close.close)
            self.assertLessEqual(close.amount, abs(position))

    def test_at_inventory_cap_only_reduces(self):
        quotes = engine().quotes(snapshot(position=D("0.2")))
        self.assertEqual(["sell"], [q.side for q in quotes])
        self.assertTrue(quotes[0].close)

    def test_open_order_capacity_respects_inventory_limit(self):
        e = engine()
        s = snapshot(position=D("0.185"))
        buy = next(q for q in e.quotes(s) if q.side == "buy")
        self.assertLessEqual((s.position + buy.amount) * s.mid, e.config.max_position_quote)

    def test_budget_reserves_gross_fees_not_rebated_fees(self):
        e = engine()
        s = snapshot(available=D("15.01"))
        quotes = e.quotes(s)
        cost = sum(q.amount * q.price * (D(1) + e.config.maker_fee) for q in quotes)
        self.assertLessEqual(cost, s.available - D("5") - D("20") * e.config.taker_fee)

    def test_minimum_contract_does_not_upsize_budget(self):
        self.assertEqual([], engine().quotes(snapshot(amount_step=D("1"), min_amount=D("1"))))

    def test_oversized_model_spread_pauses(self):
        e = engine()
        e.variance = 0.1
        self.assertEqual([], e.quotes(snapshot()))

    def test_irregular_sample_times_scale_variance_per_second(self):
        e = engine()
        e.observe(snapshot(0))
        e.observe(snapshot(2, bid=D("109.99"), ask=D("110.01")))
        import math
        self.assertAlmostEqual(math.log(1.1) ** 2 / 2, e.variance)

    def test_long_gap_requires_warmup_again(self):
        e = warmed()
        self.assertEqual([], e.step(snapshot(20)))
        self.assertEqual(1, len(e.samples))

    def test_runtime_settings_are_detached_from_user_config(self):
        config = AvellanedaSettings()
        e = AvellanedaEngine(config)
        config.dry_run = False
        self.assertTrue(e.config.dry_run)


class LifecycleTests(unittest.TestCase):
    def test_preview_never_emits_commands(self):
        e = AvellanedaEngine(AvellanedaSettings(warmup_samples=3))
        for t in range(10):
            self.assertEqual([], e.step(snapshot(t)))
        self.assertEqual(2, len(e.preview))
        e.halt("test")
        self.assertEqual([], e.step(snapshot(11, position=D("0.1"))))

    def test_refresh_waits_for_all_cancel_acknowledgements(self):
        e = warmed()
        commands = e.step(snapshot(3))
        for i, command in enumerate(commands):
            e.register(str(i), command.quote, 3)
        self.assertEqual(["cancel", "cancel"], [x.kind for x in e.step(snapshot(8))])
        self.assertEqual([], e.step(snapshot(9)))
        e.terminal("0")
        self.assertEqual([], e.step(snapshot(10)))
        e.terminal("1")
        self.assertEqual(["create", "create"], [x.kind for x in e.step(snapshot(11))])

    def test_partial_fill_cancels_and_waits_for_position_sync(self):
        e = warmed()
        e.register("buy", Quote("buy", D("0.1"), D("99.98")), 2)
        e.register("sell", Quote("sell", D("0.1"), D("100.02")), 2)
        e.filled("buy", D("0.03"), 3, "fill-1")
        self.assertEqual(["cancel", "cancel"], [x.kind for x in e.step(snapshot(3))])
        e.terminal("buy")
        e.terminal("sell")
        self.assertEqual([], e.step(snapshot(4)))
        commands = e.step(snapshot(5, position=D("0.03")))
        self.assertEqual(2, len(commands))
        self.assertTrue(next(x.quote for x in commands if x.quote.side == "sell").close)

    def test_synced_fill_still_cancels_opposite_order(self):
        e = warmed()
        e.register("buy", Quote("buy", D("0.1"), D("99.98")), 2)
        e.filled("buy", D("0.03"), 3, "f")
        self.assertEqual(["cancel"], [c.kind for c in e.step(snapshot(3, position=D("0.03")))])

    def test_duplicate_fill_and_late_fill_after_cancel(self):
        e = warmed()
        e.register("b", Quote("buy", D("0.1"), D("100")), 2)
        e.terminal("b")
        e.filled("b", D("0.02"), 3, "same")
        e.filled("b", D("0.02"), 3, "same")
        self.assertEqual(D("0.02"), e.expected_position)

    def test_reconciliation_timeout_requires_rest_ack_before_close(self):
        e = warmed()
        e.register("b", Quote("buy", D("0.1"), D("100")), 2)
        e.filled("b", D("0.1"), 3, "f")
        e.terminal("b")
        self.assertEqual([], e.step(snapshot(20, position=D("0.08"))))
        self.assertIn("reconciliation timeout", e.halt_reason)
        e.accept_rest_position(D("0.08"))
        close = e.step(snapshot(21, position=D("0.08")))[0].quote
        self.assertTrue(close.close and close.market)
        self.assertEqual(D("0.08"), close.amount)

    def test_stale_data_cancels_and_recovery_can_quote(self):
        e = warmed()
        e.register("b", Quote("buy", D("0.1"), D("100")), 2)
        self.assertEqual(["cancel"], [c.kind for c in e.step(snapshot(3, book_age=11))])
        e.terminal("b")
        self.assertEqual(2, len(e.step(snapshot(4))))
        self.assertIsNone(e.halt_reason)

    def test_invalid_snapshot_cancels_without_orders(self):
        e = warmed()
        e.register("b", Quote("buy", D("0.1"), D("100")), 2)
        self.assertEqual(["cancel"], [c.kind for c in e.step(snapshot(3, bid=D("NaN")))])

    def test_loss_halts_then_reduces_without_duplicate_close(self):
        e = warmed()
        e.confirmed_position = D("0.1")
        commands = e.step(snapshot(3, position=D("0.1"), equity=D("99")))
        self.assertIn("loss limit", e.halt_reason)
        self.assertEqual([], commands)
        e.accept_rest_position(D("0.1"))
        commands = e.step(snapshot(3, position=D("0.1"), equity=D("99")))
        close = commands[0].quote
        self.assertTrue(close.close and close.market)
        self.assertEqual("sell", close.side)
        e.register("close", close, 3)
        self.assertEqual([], e.step(snapshot(4, position=D("0.1"), equity=D("99"))))

    def test_cancel_timeout_halts_but_never_forgets_live_order(self):
        e = warmed()
        e.register("b", Quote("buy", D("0.1"), D("100")), 2)
        e.step(snapshot(7))
        commands = e.step(snapshot(17))
        self.assertEqual(["cancel"], [c.kind for c in commands])
        self.assertIn("cancel acknowledgement", e.halt_reason)
        self.assertIn("b", e.orders)

    def test_close_retries_are_bounded_and_delay_after_failure(self):
        e = warmed(max_close_attempts=2)
        e.halt("test")
        e.accept_rest_position(D("-0.1"))
        s = snapshot(3, position=D("-0.1"))
        close = e.step(s)[0].quote
        self.assertEqual("buy", close.side)
        e.register("c1", close, 3)
        e.terminal("c1", failed=True, timestamp=3)
        self.assertEqual([], e.step(replace(s, timestamp=4)))
        e.accept_rest_position(D("-0.1"))
        e.register("c2", e.step(replace(s, timestamp=8))[0].quote, 8)
        e.terminal("c2", failed=True, timestamp=8)
        e.accept_rest_position(D("-0.1"))
        self.assertEqual([], e.step(replace(s, timestamp=14)))
        self.assertIn("exhausted", e.status)

    def test_market_terminal_always_requires_new_rest_position(self):
        e = warmed()
        e.halt("test")
        e.accept_rest_position(D("0.1"))
        s = snapshot(3, position=D("0.1"))
        e.register("close", e.step(s)[0].quote, 3)
        e.terminal("close")
        self.assertTrue(e.rest_ack_required)
        self.assertEqual([], e.step(replace(s, timestamp=4)))
        e.accept_rest_position(D("0"))
        self.assertEqual([], e.step(snapshot(5)))
        self.assertIn("flat", e.status)

    def test_unknown_events_do_not_alter_inventory_or_failures(self):
        e = warmed()
        e.filled("foreign", D("1"), 3, "f")
        e.terminal("foreign", failed=True)
        self.assertIsNone(e.expected_position)
        self.assertEqual(0, e.failures)

    def test_position_cap_halts(self):
        e = warmed()
        e.confirmed_position = D("0.21")
        e.step(snapshot(3, position=D("0.21")))
        self.assertIn("position notional", e.halt_reason)

    def test_new_halt_cannot_close_before_authenticated_ack(self):
        e = warmed()
        self.assertEqual([], e.step(snapshot(3, position=D("0.1"))))
        self.assertIn("without a recorded fill", e.halt_reason)
        self.assertTrue(e.rest_ack_required)
        e.accept_rest_position(D("0.1"))
        self.assertEqual(1, len(e.step(snapshot(4, position=D("0.1")))))

    def test_failure_limit_and_leave_position_on_stop(self):
        e = warmed(max_order_failures=1)
        e.register("b", Quote("buy", D("0.1"), D("100")), 2)
        e.terminal("b", failed=True)
        self.assertIn("order failures", e.halt_reason)
        e.halt("stop", flatten=False)
        e.accept_rest_position(D("0.1"))
        self.assertEqual([], e.step(snapshot(3, position=D("0.1"))))


if __name__ == "__main__":
    unittest.main()
