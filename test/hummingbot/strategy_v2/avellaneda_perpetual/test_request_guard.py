import tempfile
import unittest
from decimal import Decimal as D
from pathlib import Path
from test.hummingbot.strategy_v2.avellaneda_perpetual.test_engine import snapshot, warmed

from hummingbot.connector.derivative.gate_io_perpetual.gate_io_perpetual_request_guard import (
    GateRequestDeferred,
    GateRequestGuard,
)
from hummingbot.strategy_v2.utils.avellaneda_perpetual_simulation import SimulationSettings, simulate

ORDERS = "https://api.gateio.ws/api/v4/futures/usdt/orders"


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.now = 1700000000.0
        self.guard = GateRequestGuard(clock=lambda: self.now)

    def attempt(self, method="POST", data=None):
        self.guard.before_request(method, ORDERS, data)

    def test_quote_limit_reserves_two_safety_slots_and_counts_failed_attempts(self):
        for _ in range(4):
            self.attempt()
        with self.assertRaises(GateRequestDeferred):
            self.attempt()
        self.attempt("DELETE")
        self.attempt(data={"reduce_only": True, "price": "0"})
        with self.assertRaises(GateRequestDeferred):
            self.attempt("DELETE")
        self.assertEqual(6, self.guard.stats()["requests_24h"])
        self.now += 1
        self.attempt()
        self.assertEqual(1, self.guard.stats()["requests_1s"])

    def test_reduce_only_maker_is_still_a_normal_quote(self):
        for _ in range(4):
            self.attempt()
        with self.assertRaises(GateRequestDeferred):
            self.attempt(data={"reduce_only": True, "price": "100"})

    def test_rolling_daily_budget_pauses_quotes_but_allows_safety(self):
        self.guard = GateRequestGuard(daily=100, safety_daily=10, clock=lambda: self.now)
        for _ in range(90):
            self.attempt()
            self.now += 2
        with self.assertRaises(GateRequestDeferred):
            self.attempt()
        self.attempt("DELETE")
        self.attempt(data={"reduce_only": True, "price": "0"})
        self.now += 86400
        self.attempt()
        self.assertEqual(1, self.guard.stats()["requests_24h"])

    def test_reset_header_milliseconds_and_server_date_clock_skew(self):
        from email.utils import formatdate
        server = self.now - 30
        with self.assertRaises(GateRequestDeferred):
            self.guard.after_response("POST", ORDERS, 429, {
                "X-Gate-RateLimit-Reset-Timestamp": str((server + 12) * 1000),
                "Date": formatdate(server, usegmt=True), "Retry-After": "5"})
        self.assertAlmostEqual(12.25, self.guard.stats()["cooldown_seconds"])
        self.assertFalse(self.guard.allowed("POST", ORDERS, True, True))
        self.assertTrue(self.guard.allowed("DELETE", ORDERS + "/123", True, True))
        self.now += 12.3
        self.assertEqual("", self.guard.quote_reason())

    def test_cancellation_bucket_shared_across_order_ids(self):
        with self.assertRaises(GateRequestDeferred):
            self.guard.after_response("DELETE", ORDERS + "/1", 429, {"Retry-After": "10"})
        self.assertFalse(self.guard.allowed("DELETE", ORDERS + "/2", True, True))
        self.assertTrue(self.guard.allowed("POST", ORDERS, True, True))
        self.assertFalse(self.guard.allowed("POST", ORDERS, True, False))

    def test_fallback_backoff_and_success_cannot_shorten_cooldown(self):
        for expected in (5.25, 10.25, 20.25, 40.25, 60.25, 60.25):
            with self.assertRaises(GateRequestDeferred):
                self.guard.after_response("POST", ORDERS, 429, {})
            self.assertAlmostEqual(expected, self.guard.stats()["cooldown_seconds"])
        self.guard.after_response("POST", ORDERS, 200, {})
        self.assertAlmostEqual(60.25, self.guard.stats()["cooldown_seconds"])

    def test_zero_remaining_success_sets_cooldown_without_failure(self):
        self.guard.after_response("POST", ORDERS, 201, {
            "x-gate-ratelimit-requests-remain": "0", "x-gate-ratelimit-reset-timestamp": str(self.now + 2)})
        self.assertFalse(self.guard.allowed("POST", ORDERS, True))
        self.assertEqual(0, self.guard.rate_limited)

    def test_nontrading_reads_do_not_consume_daily_budget(self):
        self.guard.before_request("GET", ORDERS)
        self.guard.before_request("GET", ORDERS.replace("orders", "accounts"))
        self.assertEqual(0, self.guard.stats()["requests_24h"])

    def test_queued_pre_cooldown_price_is_rejected_even_after_reset(self):
        self.guard.pending_quotes["old"] = (self.guard.generation, self.now)
        with self.assertRaises(GateRequestDeferred):
            self.guard.after_response("POST", ORDERS, 429, {"Retry-After": "1"})
        self.now += 2
        with self.assertRaises(GateRequestDeferred):
            self.attempt(data={"text": "old", "price": "100"})
        self.guard.pending_quotes["fresh"] = (self.guard.generation, self.now)
        self.attempt(data={"text": "fresh", "price": "101"})
        self.assertEqual(1, self.guard.total_attempts)

    def test_stale_queued_quote_is_rejected_and_invalid_reset_does_not_hang(self):
        self.guard.pending_quotes["old"] = (0, self.now)
        self.now += 11
        with self.assertRaises(GateRequestDeferred):
            self.attempt(data={"text": "old"})
        with self.assertRaises(GateRequestDeferred):
            self.guard.after_response("POST", ORDERS, 429, {"X-Gate-RateLimit-Reset-Timestamp": "inf"})
        self.assertEqual(5.25, self.guard.stats()["cooldown_seconds"])

    def test_disk_failure_disables_quotes_but_preserves_safety_requests(self):
        from unittest.mock import Mock
        self.guard._stream = Mock()
        self.guard._stream.write.side_effect = OSError("disk full")
        with self.assertRaises(RuntimeError):
            self.attempt()
        self.assertIn("journal", self.guard.quote_reason())
        self.attempt("DELETE")
        self.attempt(data={"reduce_only": True, "price": "0"})
        self.assertEqual(2, self.guard.total_attempts)

    def test_restart_preserves_attempts_cooldown_and_account_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "requests.jsonl"
            first = GateRequestGuard(journal=path, clock=lambda: self.now)
            try:
                first.before_request("POST", ORDERS)
                with self.assertRaises(GateRequestDeferred):
                    first.after_response("POST", ORDERS, 429, {"Retry-After": "20"})
                with self.assertRaises(OSError):
                    GateRequestGuard(journal=path, clock=lambda: self.now)
            finally:
                first.close()
            second = GateRequestGuard(journal=path, clock=lambda: self.now)
            try:
                self.assertEqual(1, second.stats()["requests_24h"])
                self.assertGreater(second.stats()["cooldown_seconds"], 20)
            finally:
                second.close()

    def test_corrupt_journal_and_backwards_clock_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "requests.jsonl"
            path.write_text("not json\n")
            with self.assertRaises(ValueError):
                GateRequestGuard(journal=path, clock=lambda: self.now)
        self.attempt()
        self.now -= 1
        with self.assertRaises(RuntimeError):
            self.attempt()


class ProtectedEngineTests(unittest.TestCase):
    def test_identical_quotes_keep_queue_until_maximum_age(self):
        engine = warmed()
        for n, quote in enumerate(engine.preview):
            engine.register(str(n), quote, 2)
        self.assertEqual([], engine.step(snapshot(7)))
        self.assertEqual(2, engine.retained_quotes)
        self.assertEqual([], engine.step(snapshot(12)))
        commands = engine.step(snapshot(32))
        self.assertEqual(["cancel", "cancel"], [c.kind for c in commands])

    def test_partial_fill_is_not_retained_as_an_unchanged_quote(self):
        engine = warmed()
        quote = engine.preview[0]
        engine.register("buy", quote, 2)
        engine.filled("buy", D("0.01"), 3, "fill")
        commands = engine.step(snapshot(3, position=D("0.01")))
        self.assertEqual(["cancel"], [c.kind for c in commands])

    def test_cancel_429_waits_then_retries_without_ack_timeout_halt(self):
        engine = warmed()
        now = [3.0]
        guard = engine.request_guard = GateRequestGuard(clock=lambda: now[0])
        engine.register("buy", engine.preview[0], 2)
        engine._cancel_all(3)
        guard.deferred_cancels.add("buy")
        with self.assertRaises(GateRequestDeferred):
            guard.after_response("DELETE", "futures/usdt/orders/1", 429, {"Retry-After": "20"})
        now[0] = 14
        self.assertEqual([], engine.step(snapshot(14)))
        self.assertIsNone(engine.halt_reason)
        self.assertIsNone(engine.orders["buy"].cancel_at)
        now[0] = 24
        commands = engine.step(snapshot(24))
        self.assertEqual(["cancel"], [c.kind for c in commands])
        self.assertIsNone(engine.halt_reason)

    def test_daily_exhaustion_halts_and_still_permits_emergency_close(self):
        engine = warmed()
        guard = engine.request_guard = GateRequestGuard(daily=100, safety_daily=10, clock=lambda: 1000)
        guard.attempts.extend(range(800, 890))
        engine.confirmed_position = D("0.1")
        engine.step(snapshot(1000, position=D("0.1")))
        self.assertIn("24h", engine.halt_reason)
        engine.accept_rest_position(D("0.1"))
        commands = engine.step(snapshot(1001, position=D("0.1")))
        self.assertEqual(1, len(commands))
        self.assertTrue(commands[0].quote.market)
        self.assertTrue(commands[0].quote.close)


class ProtectedSimulationTests(unittest.TestCase):
    def test_injected_429_is_counted_and_quotes_resume_after_cooldown(self):
        result = simulate(SimulationSettings(duration_seconds=300, simulated_429_after_seconds=30))
        for mode in result["results"]:
            self.assertEqual(1, mode["summary"]["request_guard"]["http_429"])
            paused = [s for s in mode["samples"] if s["request_guard"]["cooldown_seconds"] > 0]
            self.assertTrue(paused)
            self.assertTrue(any("cooldown" in s["status"] for s in paused))
            self.assertTrue(mode["summary"]["shutdown_complete"])
            self.assertGreater(mode["summary"]["request_guard"]["session_requests"], 10)

    def test_low_daily_budget_cancels_and_flattens(self):
        result = simulate(SimulationSettings(duration_seconds=600, requests_per_24h=100, safety_requests_per_24h=10))
        for mode in result["results"]:
            summary = mode["summary"]
            self.assertIn("24h", summary["halt_reason"])
            self.assertTrue(summary["shutdown_complete"])
            self.assertLessEqual(summary["max_position_quote"], 40)
