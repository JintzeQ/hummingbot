import json
import unittest
from dataclasses import replace
from decimal import Decimal as D
from io import BytesIO
from types import SimpleNamespace

from pydantic import ValidationError

from hummingbot.strategy_v2.utils.avellaneda_perpetual import AvellanedaEngine, Command, Quote, Snapshot
from hummingbot.strategy_v2.utils.avellaneda_perpetual_simulation import (
    PaperBroker,
    SimulationSettings,
    calibrate_tape,
    capture_tape,
    simulate,
    synthetic_tape,
)
from tools.avellaneda_perpetual_simulator import SimulatorHandler, report_html


def market(t=0, **changes):
    return replace(Snapshot(t, D("79999.9"), D("80000.1"), D("0"), D("100"), D("100"),
                            D("0.1"), D("0.0001"), D("0.0001"), D("1")), **changes)


def broker(**changes):
    s = SimulationSettings(order_latency_seconds=0, queue_ahead_lots=0, **changes)
    e = AvellanedaEngine(s.strategy(dry_run=False))
    e.step(market())
    return PaperBroker(s, e)


def post(b, q, t=0):
    b.dispatch([Command("create", quote=q)], market(t))
    b.process_events(market(t))
    return f"paper-{b.order_count}"


class PaperLedgerTests(unittest.TestCase):
    def test_round_trip_debits_gross_fees_and_credits_rebate_later_once(self):
        b = broker(rebate_delay_seconds=10)
        buy = post(b, Quote("buy", D("0.0001"), D("80000")))
        b.fill(buy, D("0.0001"), D("80000"), market(1), "test")
        self.assertEqual(D("99.99840000"), b.wallet)
        self.assertEqual(D("0.000960000"), b.rebates_pending)
        self.assertEqual(D("80000"), b.entry)
        sell = post(b, Quote("sell", D("0.0001"), D("80100"), close=True), 2)
        b.fill(sell, D("0.0001"), D("80100"), market(3), "test")
        self.assertEqual(D("0.0100"), b.realized)
        self.assertEqual(D("0.00320200"), b.fees)
        self.assertEqual(D("100.00679800"), b.wallet)
        self.assertEqual(D("0"), b.position)
        b.process_events(market(11))
        self.assertEqual(D("0.000960000"), b.rebates_credited)
        self.assertEqual(D("0.000961200"), b.rebates_pending)
        b.process_events(market(13))
        self.assertEqual(D("100.008719200"), b.wallet)
        b.process_events(market(30))
        self.assertEqual(D("100.008719200"), b.wallet)
        self.assertEqual(D("0"), b.rebates_pending)

    def test_short_linear_pnl_and_weighted_entry(self):
        b = broker()
        oid = post(b, Quote("sell", D("0.0002"), D("80100")))
        b.fill(oid, D("0.0001"), D("80100"), market(1), "test")
        b.fill(oid, D("0.0001"), D("80200"), market(2), "test")
        self.assertEqual(D("80150"), b.entry)
        self.assertEqual(D("0.0300"), b.unrealized(D("80000")))
        close = post(b, Quote("buy", D("0.0001"), D("79900"), close=True), 3)
        b.fill(close, D("0.0001"), D("79900"), market(4), "test")
        self.assertEqual(D("0.0250"), b.realized)
        self.assertEqual(D("-0.0001"), b.position)
        self.assertEqual(D("80150"), b.entry)

    def test_reduce_only_never_flips_and_rejects_wrong_direction(self):
        b = broker()
        opening = post(b, Quote("buy", D("0.0001"), D("80000")))
        b.fill(opening, D("0.0001"), D("80000"), market(1), "test")
        wrong = post(b, Quote("buy", D("0.0001"), D("80000"), close=True), 2)
        b.fill(wrong, D("0.0001"), D("80000"), market(3), "test")
        self.assertEqual(1, b.rejected)
        self.assertEqual(D("0.0001"), b.position)
        close = post(b, Quote("sell", D("0.0002"), D("80100"), close=True), 4)
        b.fill(close, D("0.0002"), D("80100"), market(5), "test")
        self.assertEqual(D("0"), b.position)
        self.assertEqual(2, len(b.fills))
        self.assertEqual(0.0001, b.fills[-1]["amount"])

    def test_pending_rebates_cannot_fund_opening_margin(self):
        b = broker()
        oid = post(b, Quote("buy", D("0.0001"), D("80000")))
        b.wallet = D("1")
        b.rebates_pending = D("100")
        b.fill(oid, D("0.0001"), D("80000"), market(1), "test")
        self.assertEqual(D("0"), b.position)
        self.assertEqual(1, b.rejected)
        self.assertEqual(D("1"), b.wallet)

    def test_funding_uses_signed_position_and_does_not_generate_rebates(self):
        b = broker(funding_rate_8h=D("0.001"))
        b.position, b.entry = D("0.0002"), D("80000")
        b.settle_funding(market(1))
        self.assertEqual(D("-0.01600000"), b.funding)
        b.position = D("-0.0002")
        b.settle_funding(market(2))
        self.assertEqual(D("0"), b.funding)
        self.assertEqual(D("100"), b.wallet)
        self.assertEqual(D("0"), b.rebates_pending)


class PaperLifecycleTests(unittest.TestCase):
    def test_touch_is_not_fill_and_queue_consumes_assumed_volume(self):
        from hummingbot.strategy_v2.utils.avellaneda_perpetual_simulation import MarketTick
        b = broker()
        oid = post(b, Quote("buy", D("0.0001"), D("80000")))
        b.orders[oid].queue_ahead = D("0.0002")
        b.match(MarketTick(market(1, bid=D("79999.8"), ask=D("80000")), D("0.001")))
        self.assertEqual(0, len(b.fills))
        through = market(2, bid=D("79998"), ask=D("79999"))
        b.match(MarketTick(through, D("0.0002")))
        self.assertEqual(D("0"), b.orders[oid].queue_ahead)
        self.assertEqual(0, len(b.fills))
        b.match(MarketTick(replace(through, timestamp=3), D("0.0001")))
        self.assertEqual(1, len(b.fills))

    def test_cancel_request_does_not_remove_order_until_ack_and_can_race_fill(self):
        from hummingbot.strategy_v2.utils.avellaneda_perpetual_simulation import MarketTick
        b = broker(cancel_latency_seconds=2)
        oid = post(b, Quote("buy", D("0.0001"), D("80000")))
        b.dispatch(b.engine._cancel_all(1), market(1))
        b.process_events(market(2))
        self.assertIn(oid, b.orders)
        b.match(MarketTick(market(2, bid=D("79998"), ask=D("79999")), D("0.0001")))
        self.assertTrue(b.fills[0]["during_cancel"])
        b.process_events(market(3))
        self.assertEqual(1, len(b.fills))
        self.assertNotIn(oid, b.engine.orders)

    def test_post_only_rejected_at_arrival_has_no_taker_fill(self):
        s = SimulationSettings(order_latency_seconds=2)
        b = PaperBroker(s, AvellanedaEngine(s.strategy(dry_run=False)))
        b.dispatch([Command("create", quote=Quote("buy", D("0.0001"), D("80000")))], market())
        b.process_events(market(1))
        self.assertFalse(b.orders["paper-1"].active)
        b.process_events(market(2, bid=D("79998"), ask=D("79999")))
        self.assertEqual(1, b.rejected)
        self.assertEqual([], b.fills)
        self.assertEqual(D("0"), b.fees)

    def test_fill_position_visibility_is_delayed_separately(self):
        b = broker(position_latency_seconds=3)
        oid = post(b, Quote("buy", D("0.0001"), D("80000")))
        b.fill(oid, D("0.0001"), D("80000"), market(1), "test")
        self.assertEqual(D("0.0001"), b.position)
        self.assertEqual(D("0"), b.snapshot(market(2)).position)
        self.assertEqual(D("0.0001"), b.engine.expected_position)
        b.process_events(market(4))
        b.engine.step(b.snapshot(market(4)))
        self.assertIsNone(b.engine.expected_position)
        self.assertEqual(D("0.0001"), b.engine.confirmed_position)

    def test_market_close_includes_spread_slippage_and_taker_fee(self):
        b = broker(market_slippage_bps=D("2"))
        opening = post(b, Quote("buy", D("0.0001"), D("80000")))
        b.fill(opening, D("0.0001"), D("80000"), market(1), "test")
        post(b, Quote("sell", D("0.0001"), D("80000"), close=True, market=True), 2)
        f = b.fills[-1]
        self.assertEqual("taker", f["liquidity"])
        self.assertLess(f["price"], float(market().bid))
        self.assertAlmostEqual(f["notional"] * .0005, f["fee"])
        self.assertEqual(D("0"), b.position)
        self.assertTrue(b.engine.rest_ack_required)


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = simulate()

    def test_reproducible_same_tape_and_calibration_before_any_fill(self):
        self.assertEqual(self.result, simulate())
        r = self.result
        self.assertEqual(3, len(r["results"]))
        common = [(s["timestamp"], s["mid"]) for s in r["results"][0]["samples"]]
        # Downsampling can preserve different inventory transitions per mode.
        for mode in r["results"]:
            mapping = {s["timestamp"]: s["mid"] for s in mode["samples"]}
            self.assertTrue(all(mapping[t] == p for t, p in common if t in mapping))
            self.assertTrue(all(f["timestamp"] > r["calibration"]["created_at"] for f in mode["fills"]))
        other = synthetic_tape(SimulationSettings(seed=43))
        self.assertNotEqual(other[0].snapshot.mid, synthetic_tape(SimulationSettings())[0].snapshot.mid)

    def test_cash_and_economic_pnl_reconcile_and_shutdown_leaves_no_orders(self):
        for mode in self.result["results"]:
            s = mode["summary"]
            self.assertAlmostEqual(s["realized"] - s["fees"] + s["rebates_credited"] + s["funding"] + s["unrealized"], s["cash_pnl"], places=10)
            self.assertAlmostEqual(s["cash_pnl"] + s["rebates_pending"], s["economic_pnl"], places=10)
            self.assertAlmostEqual(s["fees"] * .6, s["rebates_credited"] + s["rebates_pending"], places=10)
            self.assertTrue(s["shutdown_complete"])
            self.assertEqual(0, s["final_position"])
            self.assertEqual(0, s["unresolved_orders"])
            for point in mode["samples"]:
                self.assertAlmostEqual(point["wallet"] + point["unrealized"], point["equity"], places=10)

    def test_cash_rebate_delay_changes_credit_not_total_earned(self):
        immediate = simulate(SimulationSettings(rebate_delay_seconds=0))
        for delayed, fast in zip(self.result["results"], immediate["results"]):
            self.assertEqual(delayed["summary"]["fill_count"], fast["summary"]["fill_count"])
            self.assertEqual(0, fast["summary"]["rebates_pending"])
            self.assertAlmostEqual(delayed["summary"]["economic_pnl"], fast["summary"]["economic_pnl"], places=10)

    def test_capture_gap_has_no_imputed_fills_and_recovers_after_warmup(self):
        r = simulate(SimulationSettings(source="capture"))
        self.assertEqual(1, len(r["metadata"]["gaps"]))
        gap = r["metadata"]["gaps"][0]
        self.assertGreater(gap["seconds"], 300)
        for mode in r["results"]:
            self.assertEqual("simulation end", mode["summary"]["halt_reason"])
            self.assertTrue(mode["summary"]["shutdown_complete"])
            self.assertFalse(any(gap["after"] < f["timestamp"] <= gap["before"] for f in mode["fills"]))
            self.assertTrue(any(s["timestamp"] > gap["before"] and s["status"] == "quoting" for s in mode["samples"]))

    def test_loss_limit_runs_real_halt_and_reduce_only_close(self):
        r = simulate(SimulationSettings(scenario="uptrend", duration_seconds=1800, max_session_loss_quote=D("0.03")))
        for mode in r["results"]:
            self.assertEqual("session equity loss limit", mode["summary"]["halt_reason"])
            self.assertTrue(mode["summary"]["shutdown_complete"])
            self.assertTrue(any(f["liquidity"] == "taker" and f["reduce_only"] for f in mode["fills"]))

    def test_profile_rules_and_fee_assumptions_are_recalibrated(self):
        settings = SimulationSettings(max_position_quote=D("30"), rebate_rate=D("0"))
        tape = capture_tape(settings)
        _, p = calibrate_tape(tape, settings)
        self.assertEqual(D("30"), p.max_position_quote)
        self.assertEqual("0", p.quote_settings["rebate_rate"])
        self.assertNotEqual(p.gamma_base, self.result["calibration"]["gamma_base"])

    def test_validation_rejects_unsafe_or_nonfinite_configuration(self):
        for fields in ({"initial_capital": 30}, {"max_position_quote": 60}, {"order_amount_quote": 41},
                       {"volatility_bps": float("nan")}, {"rebate_delay_seconds": float("inf")},
                       {"source": "live"}, {"api_key": "never-accepted"}, {"duration_seconds": 100000}):
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                SimulationSettings(**fields)

    def test_portable_report_embeds_data_without_remote_runtime(self):
        page = report_html(self.result)
        self.assertIn('id="embedded-result"', page)
        self.assertNotIn("<!-- SIMULATION_DATA -->", page)
        self.assertNotIn('src="https://', page)
        adversarial = dict(self.result, extra="</script><script>alert(1)</script>")
        self.assertNotIn("</script><script>alert", report_html(adversarial))

    def test_api_rejects_cross_origin_or_oversized_request_before_simulation(self):
        for headers in ({"Host": "evil.example:8765", "Content-Length": "2"},
                        {"Host": "127.0.0.1:8765", "Origin": "https://evil.example", "Content-Length": "2"},
                        {"Host": "127.0.0.1:8765", "Content-Length": "999999"}):
            handler = object.__new__(SimulatorHandler)
            handler.path = "/api/simulate"
            handler.headers = headers
            handler.server = SimpleNamespace(server_port=8765)
            handler.rfile = BytesIO(b"{}")
            replies = []
            handler.reply = lambda status, body: replies.append((status, json.loads(body)))
            handler.do_POST()
            self.assertIn(replies[0][0], (400, 403))


if __name__ == "__main__":
    unittest.main()
