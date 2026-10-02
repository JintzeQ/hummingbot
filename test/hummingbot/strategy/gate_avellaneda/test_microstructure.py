import unittest
from collections import deque
from dataclasses import replace
from decimal import Decimal as D

from hummingbot.strategy.gate_avellaneda.microstructure import GateMarketSignalFeed, MicroSettings

PAIR = "A-USDT"


class MicrostructureTests(unittest.TestCase):
    def setUp(self):
        self.settings = MicroSettings(warmup_seconds=1, min_samples=3, confirm_seconds=1, recover_seconds=2, retire_seconds=5)
        self.feed = GateMarketSignalFeed({PAIR: D("0.001")}, self.settings)
        self.sequence = 100

    def snapshot(self, bid=100000, ask=100000, sequence=100):
        return dict(id=sequence, bids=[dict(p="9.99", s=str(bid))], asks=[dict(p="10.01", s=str(ask))])

    def update(self, now, bid=None, ask=None, **extra):
        self.sequence += 1
        raw = dict(U=self.sequence, u=self.sequence, t=now * 1000, l="100",
                   b=[] if bid is None else [dict(p="9.99", s=str(bid))],
                   a=[] if ask is None else [dict(p="10.01", s=str(ask))])
        raw.update(extra)
        return self.feed.observe_depth(PAIR, raw, now)

    def warm(self, bid=100000, ask=100000):
        self.feed.seed_snapshot(PAIR, self.snapshot(bid, ask), 100)
        for t in (100, 100.5, 101):
            self.update(t)
        return self.feed.signal(PAIR, 101)

    def trade(self, now, buy=False, quantity="10", price="9.99", trade_id=1, **kwargs):
        return self.feed.observe_trade(PAIR, trade_id, now, price, quantity, buy, now, **kwargs)

    def test_balanced_book_reference_equals_mid(self):
        signal = self.warm()
        self.assertTrue(signal.ready)
        self.assertEqual(signal.reference, D("10"))
        self.assertEqual(signal.imbalance, 0)

    def test_bid_heavy_weighted_mid_and_quarter_blend(self):
        signal = self.warm(500000, 100000)
        expected = D("10") + D("2") / 3 * D("0.01")
        self.assertEqual(signal.weighted_mid, expected)
        self.assertEqual(signal.reference, D("10") + (expected - 10) / 4)
        self.assertLess(signal.reference, signal.ask)

    def test_ask_heavy_reference_moves_down(self):
        signal = self.warm(100000, 500000)
        self.assertLess(signal.reference, D("10"))
        self.assertGreater(signal.reference, signal.bid)

    def test_zero_weight_disables_reference_shift(self):
        self.feed.settings = replace(self.settings, micro_weight=D("0"))
        self.assertEqual(self.warm(500000, 100000).reference, D("10"))

    def test_ema_smooths_imbalance_instead_of_lagging_market_price(self):
        self.warm()
        self.update(102, bid=500000)
        state = self.feed.states[PAIR]
        self.assertGreater(state.smoothed_imbalance, 0)
        self.assertLess(state.smoothed_imbalance, D("2") / 3)
        self.update(103, full=True, b=[dict(p="10.99", s="500000")], a=[dict(p="11.01", s="100000")])
        self.assertGreater(self.feed.signal(PAIR, 103).reference, D("11"))

    def test_contract_multiplier_converts_quantity_and_depth(self):
        signal = self.warm()
        self.assertEqual(signal.bid_depth, D("999"))
        self.assertEqual(self.feed.states[PAIR].bids[D("9.99")], D("100"))

    def test_rest_snapshot_alone_never_becomes_ready(self):
        self.feed.seed_snapshot(PAIR, self.snapshot(), 100)
        self.assertFalse(self.feed.signal(PAIR, 100).ready)
        self.update(100)
        self.assertFalse(self.feed.signal(PAIR, 100).ready)

    def test_bootstrap_buffers_and_replays_overlapping_sequence_bridge(self):
        self.assertFalse(self.update(100, U=99, u=101))
        self.feed.seed_snapshot(PAIR, self.snapshot(), 100)
        self.assertTrue(self.feed.states[PAIR].synced)
        self.assertEqual(self.feed.states[PAIR].update_id, 101)

    def test_gap_invalidates_and_new_snapshot_can_bridge(self):
        self.warm()
        self.assertFalse(self.update(102, U=110, u=110))
        self.assertTrue(self.feed.needs_snapshot(PAIR))
        self.feed.seed_snapshot(PAIR, self.snapshot(sequence=109), 102)
        self.assertEqual(self.feed.states[PAIR].update_id, 110)
        self.assertFalse(self.feed.signal(PAIR, 102).ready)

    def test_duplicate_cannot_extend_freshness_or_warmup(self):
        self.warm()
        count = self.feed.states[PAIR].samples
        self.assertTrue(self.update(104, U=103, u=103))
        self.assertEqual(self.feed.states[PAIR].samples, count)
        self.assertFalse(self.feed.signal(PAIR, 104).ready)

    def test_full_snapshot_replaces_old_levels(self):
        self.warm()
        self.update(102, full=True, b=[dict(p="10.01", s="100000")], a=[dict(p="10.03", s="100000")])
        self.assertNotIn(D("9.99"), self.feed.states[PAIR].bids)
        self.assertEqual(self.feed.signal(PAIR, 102).bid, D("10.01"))

    def test_full_snapshot_without_rest_bootstraps_but_still_warms(self):
        self.assertTrue(self.update(100, full=True, b=self.snapshot()["bids"], a=self.snapshot()["asks"]))
        self.assertFalse(self.feed.signal(PAIR, 100).ready)
        self.update(100.5)
        self.update(101)
        self.assertTrue(self.feed.signal(PAIR, 101).ready)

    def test_periodic_rest_does_not_replace_healthy_ws(self):
        self.warm()
        self.assertFalse(self.feed.seed_snapshot(PAIR, self.snapshot(bid=1, sequence=999), 101))
        self.assertEqual(self.feed.states[PAIR].update_id, 103)

    def test_zero_quantity_deletes_level_and_empty_side_invalidates(self):
        self.warm()
        self.assertFalse(self.update(102, bid=0))
        self.assertFalse(self.feed.signal(PAIR, 102).ready)

    def test_crossed_negative_duplicate_levels_fail_closed(self):
        for rows in [[dict(p="10.02", s="1")], [dict(p="9.99", s="-1")],
                     [dict(p="9.99", s="1"), dict(p="9.99", s="2")]]:
            with self.subTest(rows=rows):
                self.feed.reset()
                self.sequence = 100
                self.warm()
                self.assertFalse(self.update(102, b=rows))

    def test_stale_book_and_quiet_interval_require_resynchronization(self):
        self.warm()
        self.assertFalse(self.feed.signal(PAIR, 104).ready)
        self.assertFalse(self.update(104))
        self.assertTrue(self.feed.needs_snapshot(PAIR))

    def test_delayed_future_wrong_level_and_backwards_time_are_rejected(self):
        for kwargs in [dict(t=99000), dict(t=104000), dict(l="20"), dict(t=100500)]:
            with self.subTest(kwargs=kwargs):
                self.feed.reset()
                self.sequence = 100
                self.warm()
                self.assertFalse(self.update(102, **kwargs))

    def test_trade_deduplication_and_internal_print_exclusion(self):
        self.warm()
        self.trade(101)
        self.trade(101)
        self.trade(101, trade_id=2, internal=True)
        self.assertEqual(len(self.feed.states[PAIR].trades), 1)

    def test_flow_is_notional_weighted_and_expires(self):
        self.warm()
        self.trade(101, quantity="10", buy=False, price="10")
        self.trade(101, quantity="30", buy=True, price="10", trade_id=2)
        self.assertEqual(self.feed.signal(PAIR, 101).flow, D("0.5"))
        for t in range(102, 108):
            self.update(t)
        self.assertEqual(self.feed.signal(PAIR, 107).flow_quote, 0)
        self.assertEqual(self.feed.states[PAIR].trade_ids, set())

    def test_invalid_trade_identity_amount_direction_and_delay_fail_closed(self):
        for values in [(None, 101, "10", "1", True, 101), (1, 101, "10", "0", True, 101),
                       (1, 101, "NaN", "1", True, 101), (1, 101, "10", "1", "buy", 101),
                       (1, 99, "10", "1", True, 102), (1, 105, "10", "1", True, 102)]:
            with self.subTest(values=values):
                self.feed.reset()
                self.assertFalse(self.feed.observe_trade(PAIR, *values))

    def test_matching_sell_print_explains_bid_reduction_after_grace(self):
        self.warm()
        self.update(102, bid=50000)
        self.assertEqual(self.feed.signal(PAIR, 102).bid_loss, 0)
        self.trade(102, quantity="50")
        self.update(102.5)
        self.assertEqual(self.feed.signal(PAIR, 102.5).bid_loss, 0)

    def test_wrong_side_or_wrong_price_does_not_explain_reduction(self):
        self.warm()
        self.update(102, bid=50000)
        self.trade(102, quantity="50", buy=True)
        self.trade(102, quantity="50", price="9.98", trade_id=2)
        self.update(103)
        self.assertEqual(self.feed.signal(PAIR, 103).bid_loss, D("0.5"))

    def test_soft_bid_conflict_requires_confirmation_and_scales_buy(self):
        self.warm()
        self.trade(102)
        self.update(102, bid=500000)
        self.assertEqual(self.feed.signal(PAIR, 102).buy_scale, 1)
        self.update(103)
        signal = self.feed.signal(PAIR, 103)
        self.assertEqual(signal.buy_scale, D("0.5"))
        self.assertEqual(signal.sell_scale, 1)
        self.assertFalse(signal.paused)

    def test_confidence_flags_follow_configured_thresholds(self):
        self.feed.settings = replace(self.settings, imbalance_threshold=D("0.8"), min_flow_quote=D("150"))
        self.warm(500000, 100000)
        self.trade(102)
        self.update(102)
        signal = self.feed.signal(PAIR, 102)
        self.assertFalse(signal.flow_sufficient)
        self.assertFalse(signal.book_flow_conflict)
        self.feed.settings = replace(self.feed.settings, min_flow_quote=D("25"))
        signal = self.feed.signal(PAIR, 102)
        self.assertTrue(signal.flow_sufficient)
        self.assertFalse(signal.book_flow_conflict)
        self.feed.settings = replace(self.feed.settings, imbalance_threshold=D("0.6"))
        self.assertTrue(self.feed.signal(PAIR, 102).book_flow_conflict)

    def test_ask_conflict_and_buy_pressure_scale_sell(self):
        for bid, ask in [(100000, 500000), (500000, 100000)]:
            self.feed.reset()
            self.sequence = 100
            self.warm(bid, ask)
            self.trade(102, buy=True, price="10.01")
            self.update(102)
            self.feed.signal(PAIR, 102)
            self.update(103)
            self.assertEqual(self.feed.signal(PAIR, 103).sell_scale, D("0.5"))

    def test_sell_pressure_scales_buy_but_small_flow_is_ignored(self):
        self.warm(100000, 500000)
        self.trade(102, quantity="1")
        self.update(102)
        self.feed.signal(PAIR, 102)
        self.update(103)
        self.assertEqual(self.feed.signal(PAIR, 103).buy_scale, 1)
        self.trade(103, quantity="10", trade_id=2)
        self.update(104)
        self.feed.signal(PAIR, 104)
        self.update(105)
        self.assertEqual(self.feed.signal(PAIR, 105).buy_scale, D("0.5"))

    def test_liquidity_collapse_pauses_and_persistent_pause_retires(self):
        self.warm()
        self.update(102, bid=20000)
        self.assertFalse(self.feed.signal(PAIR, 102).paused)
        for t in range(103, 109):
            self.update(t)
            signal = self.feed.signal(PAIR, t)
        self.assertTrue(signal.paused)
        self.assertTrue(signal.retire)

    def test_unmatched_loss_with_opposing_flow_is_severe(self):
        for side in ("bid", "ask"):
            self.feed.reset()
            self.sequence = 100
            self.warm(100000, 10000) if side == "bid" else self.warm(10000, 100000)
            self.trade(102, buy=side == "ask", quantity="3", price="9.99" if side == "bid" else "10.01")
            self.update(102, bid=25000 if side == "bid" else 5000, ask=5000 if side == "bid" else 25000)
            self.feed.signal(PAIR, 102)
            for t in (103, 104):
                self.update(t)
                signal = self.feed.signal(PAIR, t)
            self.assertIn("unmatched", signal.reason)
            self.assertTrue(signal.paused)

    def test_recovery_hysteresis_prevents_churn_and_retirement_during_recovery(self):
        self.warm()
        self.update(102, bid=20000)
        self.feed.signal(PAIR, 102)
        self.update(103)
        self.feed.signal(PAIR, 103)
        for t in range(104, 108):
            self.update(t)
            self.feed.signal(PAIR, t)
        self.update(108, bid=100000)
        self.assertFalse(self.feed.signal(PAIR, 108).retire)
        self.assertTrue(self.feed.signal(PAIR, 108).paused)
        self.update(109)
        self.assertTrue(self.feed.signal(PAIR, 109).paused)
        self.update(110)
        self.assertFalse(self.feed.signal(PAIR, 110).paused)

    def test_repeated_reads_cannot_confirm_or_recover_without_new_books(self):
        self.warm()
        self.update(102, bid=20000)
        self.assertFalse(self.feed.signal(PAIR, 102).paused)
        self.assertFalse(self.feed.signal(PAIR, 103.5).paused)
        self.update(103.5)
        self.assertTrue(self.feed.signal(PAIR, 103.5).paused)

    def test_reconnect_unknown_pair_and_invalid_snapshot_fail_closed(self):
        self.warm()
        self.trade(101)
        self.feed.reset()
        self.assertTrue(self.feed.needs_snapshot(PAIR))
        self.assertFalse(self.feed.signal(PAIR, 101).ready)
        self.assertEqual(len(self.feed.states[PAIR].trades), 0)
        self.assertFalse(self.feed.observe_depth("OTHER", {}, 101))
        self.assertFalse(self.feed.seed_snapshot("OTHER", {}, 101))
        self.assertFalse(self.feed.observe_trade("OTHER", 1, 101, "10", "1", True, 101))
        self.assertFalse(self.feed.seed_snapshot(PAIR, dict(id=-1), 101))

    def test_settings_multipliers_and_event_buffer_bounds(self):
        for values in [dict(mode="bad"), dict(min_samples=1), dict(max_age=0), dict(micro_weight=D("NaN")),
                       dict(caution_scale=D("2")), dict(flow_threshold=D("1")), dict(min_flow_quote=D("0")),
                       dict(baseline_seconds=60), dict(reprice_bps=D("0"))]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                MicroSettings(**values)
        with self.assertRaises(ValueError):
            GateMarketSignalFeed({PAIR: D("0")}, self.settings)
        self.warm()
        self.feed.states[PAIR].trades = deque([(102, "x", True, D("10"), D("1"))], maxlen=1)
        self.assertFalse(self.trade(102, trade_id=2))
        self.assertFalse(self.feed.signal(PAIR, 102).ready)
