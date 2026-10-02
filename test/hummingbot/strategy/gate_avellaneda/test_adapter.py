import asyncio
import io
import json
import types
import unittest
from decimal import Decimal as D
from unittest.mock import AsyncMock, patch

from hummingbot.strategy.gate_avellaneda.core import Intent
from test.hummingbot.strategy.gate_avellaneda.support import (
    FakeConnector, NetworkStatus, OrderType, PositionAction,
    TradeType, constants, contract, load_adapter, method_from_source,
)

adapter = load_adapter()


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        pairs = ["A-USDT", "B-USDT", "C-USDT"]
        adapter.GateAvellanedaPortfolio._initial_contracts = {p: contract(p) for p in pairs}
        adapter.GateAvellanedaPortfolio._initial_tickers = {
            p: dict(contract=p.replace("-", "_"), last="10", volume_24h_quote="10000000", funding_rate_indicative="0.0001")
            for p in pairs
        }
        self.connector = FakeConnector(pairs)
        self.config = adapter.GateAvellanedaPortfolioConfig(micro=dict(enabled=False))
        self.bot = adapter.GateAvellanedaPortfolio({"gate_io_perpetual": self.connector}, self.config)

    async def ready(self, live=False):
        self.config.dry_run = not live
        await self.bot._refresh()
        self.bot._sample()

    async def finish_task(self):
        if self.bot.refresh_task:
            await self.bot.refresh_task

    def position(self, pair, size):
        return dict(contract=pair.replace("-", "_"), size=str(size), mode="single", mark_price="10",
                    unrealised_pnl="0")

    async def test_observer_selects_two_and_never_sends(self):
        await self.ready()
        self.bot.on_tick()
        await self.finish_task()
        self.assertEqual(len(self.bot.portfolio.slots), 2)
        self.assertEqual(self.connector.sent, [])
        self.assertEqual(len(self.bot.last_quotes), 2)
        self.assertIn("DRY RUN", self.bot.format_status())

    async def test_live_batch_sends_four_with_shared_budget(self):
        await self.ready(live=True)
        self.bot.on_tick()
        self.bot.on_tick()
        await self.finish_task()
        self.assertEqual(len(self.connector.sent), 4)
        self.assertTrue(all(o[4] == OrderType.LIMIT_MAKER for o in self.connector.sent))
        self.assertTrue(all(o[5]["position_action"] == PositionAction.OPEN for o in self.connector.sent))

    async def test_second_submission_batch_requires_new_account_revision(self):
        await self.ready(live=True)
        self.bot.on_tick()
        await self.finish_task()
        self.bot.orders.clear()
        self.bot.next_quote.clear()
        self.bot.used_revision = self.bot.revision
        self.bot.next_account = 999
        self.bot.on_tick()
        self.assertEqual(len(self.connector.sent), 4)

    async def test_fill_blocks_quotes_until_reconciled_and_close_is_explicit(self):
        await self.ready(live=True)
        self.bot.on_tick()
        await self.finish_task()
        order_id = next(iter(self.bot.orders))
        pair = self.bot.orders[order_id].intent.pair
        self.bot.did_fill_order(types.SimpleNamespace(order_id=order_id, amount=D("0.1")))
        self.assertTrue(self.bot.account_dirty)
        self.assertLess(self.bot.orders[order_id].remaining, self.bot.orders[order_id].intent.amount)
        self.bot.on_tick()
        self.assertEqual(len(self.connector.sent), 4)
        for oid in list(self.bot.orders):
            self.bot.did_cancel_order(types.SimpleNamespace(order_id=oid))
        self.connector.open_orders = []
        self.connector.positions = [self.position(pair, 100)]
        await self.bot._refresh()
        self.bot.next_quote.clear()
        self.bot.on_tick()
        await self.finish_task()
        closes = [o for o in self.connector.sent[4:] if o[5]["position_action"] == PositionAction.CLOSE]
        self.assertEqual(len(closes), 1)
        self.assertLessEqual(closes[0][3], D("0.1"))

    async def test_retirement_cancels_opens_and_waits_for_confirmation(self):
        await self.ready(live=True)
        self.bot.on_tick()
        await self.finish_task()
        pair = next(iter(self.bot.portfolio.slots))
        self.bot.portfolio.retire(pair, "depth", 100)
        self.bot.positions[pair] = D("0.1")
        self.bot.on_tick()
        self.assertTrue(self.connector.cancels)
        self.assertEqual(len(self.connector.sent), 4)
        self.assertIn(pair, self.bot.portfolio.slots)

    async def test_retirement_maker_then_market_fallback(self):
        await self.ready(live=True)
        self.bot.portfolio.slots.clear()
        self.bot.portfolio.evaluate(self.bot._snapshots(), 100, True, {}, set())
        self.bot.portfolio.retire("A-USDT", "depth", 100)
        self.bot.positions = {"A-USDT": D("0.1")}
        self.bot.expected_positions = {"A-USDT": D("0.1")}
        self.bot.position_quote = {"A-USDT": D("1")}
        self.bot.exchange_open_pairs = set()
        self.bot.next_monitor = 999
        self.bot.on_tick()
        await self.finish_task()
        first = self.connector.sent[0]
        self.assertEqual(first[4], OrderType.LIMIT_MAKER)
        self.assertEqual(first[5]["position_action"], PositionAction.CLOSE)
        for oid in list(self.bot.orders):
            self.bot.did_cancel_order(types.SimpleNamespace(order_id=oid))
        self.connector.open_orders = []
        self.connector.positions = [self.position("A-USDT", 100)]
        self.bot.current_timestamp = 170
        await self.bot._refresh()
        self.bot.on_tick()
        await self.finish_task()
        self.assertTrue(any(o[4] == OrderType.MARKET for o in self.connector.sent))

    async def test_no_market_fallback_if_disabled_or_stale(self):
        await self.ready()
        m = self.bot._snapshots()["A-USDT"]
        self.bot.positions = {"A-USDT": D("-0.1")}
        self.bot.current_timestamp = 170
        self.assertIsNone(self.bot._exit_intent("A-USDT", m, 100))
        self.bot.current_timestamp = 100
        self.config.allow_market_exit = False
        intent = self.bot._exit_intent("A-USDT", m, 0)
        self.assertTrue(intent.buy)
        self.assertFalse(intent.market)
        self.assertIsNone(self.bot._exit_intent("B-USDT", m, 0))

    async def test_network_or_expired_account_cancels_owned_orders(self):
        await self.ready(live=True)
        self.bot._submit(Intent("A-USDT", True, D("0.1"), D("9.99")))
        self.connector.network_status = NetworkStatus.NOT_CONNECTED
        self.bot.on_tick()
        await self.finish_task()
        self.assertEqual(self.connector.cancels, ["t-1"])
        self.connector.network_status = NetworkStatus.CONNECTED
        self.bot.current_timestamp = 120
        self.bot.next_account = 999
        self.bot.on_tick()
        self.assertEqual(len(self.connector.cancels), 2)

    async def test_preexisting_position_or_order_prevents_live_start(self):
        for field, value in [("positions", [self.position("A-USDT", 1)]),
                             ("open_orders", [dict(text="external", contract="A_USDT")])]:
            with self.subTest(field=field):
                setattr(self.connector, field, value)
                self.config.dry_run = False
                await self.bot._refresh()
                self.assertIn("Start requires", self.bot.halt_reason)
                self.assertEqual(self.connector.sent, [])
                setattr(self.connector, field, [])

    async def test_unowned_orders_and_unmanaged_positions_halt(self):
        await self.ready(live=True)
        self.connector.open_orders = [dict(text="external", contract="A_USDT")]
        await self.bot._refresh()
        self.assertIn("Unowned", self.bot.halt_reason)
        self.connector.open_orders = []
        self.connector.positions = [self.position("OTHER-USDT", 1)]
        await self.bot._refresh()
        self.assertIn("unmanaged", self.bot.halt_reason)

    async def test_unknown_position_value_and_invalid_balance_halt(self):
        await self.ready()
        p = self.position("A-USDT", 1)
        p["mark_price"] = "0"
        self.connector.positions = [p]
        await self.bot._refresh()
        self.assertIn("Cannot value", self.bot.halt_reason)
        self.connector.positions = []
        self.connector.account["available"] = "NaN"
        await self.bot._refresh()
        self.assertIn("Invalid account", self.bot.halt_reason)

    async def test_rest_failure_pauses_without_replacing_pairs(self):
        await self.ready()
        self.bot.on_tick()
        await self.finish_task()
        before = set(self.bot.portfolio.slots)
        self.connector.error = OSError("offline")
        await self.bot._refresh()
        self.assertTrue(self.bot.account_dirty)
        self.assertEqual(set(self.bot.portfolio.slots), before)

    async def test_fill_during_snapshot_discards_snapshot(self):
        await self.ready(live=True)
        original = self.connector._api_get

        async def changing(path_url, **kwargs):
            result = await original(path_url, **kwargs)
            if path_url == "positions":
                self.bot.account_epoch += 1
            return result

        self.connector._api_get = changing
        revision = self.bot.revision
        await self.bot._refresh()
        self.assertEqual(self.bot.revision, revision)

    async def test_delayed_zero_position_does_not_release_retiring_slot(self):
        await self.ready(live=True)
        self.bot.on_tick()
        await self.finish_task()
        oid = next(iter(self.bot.orders))
        pair = self.bot.orders[oid].intent.pair
        self.bot.did_fill_order(types.SimpleNamespace(order_id=oid, amount=D("0.1")))
        self.bot.portfolio.retire(pair, "illiquid", 100)
        for order_id in list(self.bot.orders):
            self.bot.did_cancel_order(types.SimpleNamespace(order_id=order_id))
        self.connector.open_orders = []
        # Gate's REST position response is temporarily still zero after a fill.
        self.connector.positions = []
        revision = self.bot.revision
        await self.bot._refresh()
        self.assertTrue(self.bot.account_dirty)
        self.assertEqual(self.bot.revision, revision)
        self.bot.on_tick()
        self.assertIn(pair, self.bot.portfolio.slots)
        self.connector.positions = [self.position(pair, 100)]
        await self.bot._refresh()
        self.assertFalse(self.bot.account_dirty)
        self.assertEqual(self.bot.positions[pair], D("0.1"))

    async def test_confirmed_close_fill_allows_replacement_only_after_rest_matches(self):
        await self.ready(live=True)
        self.bot.portfolio.evaluate(self.bot._snapshots(), 100, True, {}, set())
        self.bot.portfolio.retire("A-USDT", "illiquid", 100)
        self.bot.positions = {"A-USDT": D("0.1")}
        self.bot.expected_positions = {"A-USDT": D("0.1")}
        self.bot._submit(Intent("A-USDT", False, D("0.1"), D("10.01"), close=True))
        self.bot.did_fill_order(types.SimpleNamespace(order_id="t-1", amount=D("0.1")))
        self.bot.did_complete_sell_order(types.SimpleNamespace(order_id="t-1"))
        self.connector.open_orders = []
        self.connector.positions = [self.position("A-USDT", 100)]
        await self.bot._refresh()
        self.assertTrue(self.bot.account_dirty)
        self.connector.positions = []
        await self.bot._refresh()
        self.bot.next_monitor = 0
        self.bot.on_tick()
        await self.finish_task()
        self.assertEqual(set(self.bot.portfolio.slots), {"B-USDT", "C-USDT"})

    async def test_setup_mode_or_leverage_failure_halts(self):
        self.config.dry_run = False
        self.connector.setup_success = False
        await self.bot._refresh()
        self.assertIn("One-way", self.bot.halt_reason)
        self.connector._trading_pair_position_mode_set = AsyncMock(return_value=(True, ""))
        await self.bot._refresh()
        self.assertIn("1x leverage", self.bot.halt_reason)

    async def test_order_pagination_and_limit(self):
        self.connector.open_orders = [dict(text=str(i), contract="A_USDT") for i in range(101)]
        self.assertEqual(len(await self.bot._open_orders()), 101)
        self.connector.open_orders *= 10
        with self.assertRaises(ValueError):
            await self.bot._open_orders()

    async def test_failed_book_is_not_marked_fresh(self):
        await self.ready()
        self.bot.current_timestamp = 120
        original = self.connector._api_get

        async def failing(path_url, **kwargs):
            if path_url == "book":
                raise OSError("book timeout")
            return await original(path_url, **kwargs)

        self.connector._api_get = failing
        await self.bot._refresh()
        self.assertEqual(self.bot.books["A-USDT"][1], 100)

    async def test_contract_delisting_invalid_book_and_warmup(self):
        await self.ready()
        self.connector.contracts = self.connector.contracts[1:]
        self.bot.next_public = 0
        await self.bot._refresh()
        self.assertTrue(self.bot.contracts["A-USDT"]["in_delisting"])
        self.bot.books["A-USDT"] = (dict(bids=[], asks=[]), 100)
        self.assertNotIn("A-USDT", self.bot._snapshots())
        self.bot.indicators["B-USDT"][0].is_sampling_buffer_full = False
        self.assertFalse(self.bot._snapshots()["B-USDT"].ready)

    async def test_budget_checker_rejects_or_shrinks_never_increases(self):
        await self.ready(live=True)
        intent = Intent("A-USDT", True, D("0.1"), D("9.99"), close=True)

        def zero(candidate, **kwargs):
            candidate.amount = D("0")
            return candidate

        self.connector.budget_checker.adjust_candidate = zero
        self.bot._submit(intent)
        self.assertEqual(self.connector.sent, [])
        self.connector.budget_checker.adjust_candidate = lambda c, **k: types.SimpleNamespace(amount=D("1"))
        with self.assertRaises(ValueError):
            self.bot._submit(intent)

    async def test_cancel_retry_and_terminal_events(self):
        await self.ready(live=True)
        self.bot._submit(Intent("A-USDT", True, D("0.1"), D("9.99")))
        self.bot._cancel_pair("B-USDT")
        self.bot._cancel_pair("A-USDT")
        self.bot._cancel_pair("A-USDT")
        self.assertEqual(self.connector.cancels, ["t-1"])
        self.bot.current_timestamp = 106
        self.bot._cancel_pair("A-USDT")
        self.assertEqual(len(self.connector.cancels), 2)
        self.bot.did_fail_order(types.SimpleNamespace(order_id="t-1"))
        self.assertEqual(self.bot.orders, {})
        self.bot.did_fill_order(types.SimpleNamespace(order_id="unknown", amount=D("1")))
        self.bot.did_cancel_order(types.SimpleNamespace(order_id="unknown"))

    async def test_reserve_floor_retires_both(self):
        await self.ready()
        self.bot.on_tick()
        await self.finish_task()
        self.bot.equity = D("20")
        self.bot.on_tick()
        self.assertTrue(all(s.state == "retiring" for s in self.bot.portfolio.slots.values()))

    async def test_stop_cancels_task_and_orders_and_prevents_new_ticks(self):
        await self.ready(live=True)
        self.bot._submit(Intent("A-USDT", True, D("0.1"), D("9.99")))
        self.bot.refresh_task = asyncio.create_task(asyncio.sleep(10))
        await self.bot.on_stop()
        self.assertTrue(self.bot.stopping)
        self.assertTrue(self.bot.refresh_task.cancelled())
        self.bot.on_tick()
        self.assertEqual(len(self.connector.sent), 1)

    async def test_status_before_selection_and_when_halted(self):
        self.assertIn("No qualifying", self.bot.format_status())
        self.bot.halt_reason = "test"
        self.assertIn("HALTED", self.bot.format_status())

    def book_record(self, record_id, change, kind="pnl", pair="A-USDT", timestamp=100):
        return dict(id=str(record_id), change=str(change), type=kind,
                    contract=pair.replace("-", "_"), time=timestamp)

    async def selected(self):
        await self.ready(live=True)
        self.bot.portfolio.evaluate(self.bot._snapshots(), 100, True, {}, set())
        self.bot.next_account = self.bot.next_monitor = 999

    async def test_pair_net_pnl_sums_settlements_fees_funding_and_float(self):
        await self.selected()
        self.connector.account_book = [self.book_record(1, "-0.5"),
                                       self.book_record(2, "-0.5", "fee"),
                                       self.book_record(3, "-0.5", "fund")]
        position = dict(self.position("A-USDT", 1000), unrealised_pnl="-3.5", realised_pnl="-1.5")
        self.connector.positions = [position]
        self.bot.expected_positions = {"A-USDT": D("1")}
        await self.bot._refresh()
        # realised_pnl repeats the settlements and must not be added again.
        self.assertEqual(self.bot._pair_pnl("A-USDT"), D("-5"))
        self.assertEqual(self.bot._pair_pnl("B-USDT"), D("0"))
        self.assertIn("net_PnL=-5.0", self.bot.format_status())

    async def test_pair_net_pnl_offsets_profit_and_ignores_transfers_other_pairs(self):
        await self.selected()
        self.connector.account_book = [self.book_record(1, "-3"), self.book_record(2, "2"),
                                       self.book_record(3, "-0.4", "fee"),
                                       self.book_record(4, "-0.6", "fund"),
                                       self.book_record(5, "100", "dnw"),
                                       self.book_record(6, "-100", pair="C-USDT")]
        await self.bot._refresh()
        self.assertEqual(self.bot._pair_pnl("A-USDT"), D("-2"))
        self.bot.on_tick()
        self.assertNotIn("A-USDT", self.bot.portfolio.excluded)

    async def test_old_records_initial_baseline_and_previous_selection_do_not_count(self):
        self.connector.account_book = [self.book_record(1, "-100")]
        await self.ready(live=True)
        self.bot.portfolio.evaluate(self.bot._snapshots(), 101, True, {}, set())
        self.bot.current_timestamp = 110
        self.connector.account_book.extend([self.book_record(2, "-100", timestamp=100.5),
                                           self.book_record(3, "-2", timestamp=101),
                                           self.book_record(4, "-1", "fee", timestamp=110)])
        await self.bot._refresh()
        self.assertEqual(self.bot._pair_pnl("A-USDT"), D("-3"))

    async def test_late_settlement_is_captured_and_repeated_refresh_never_double_counts(self):
        await self.selected()
        self.bot.current_timestamp = 120
        await self.bot._refresh()
        # The record appears late, with a timestamp from an earlier poll.
        self.connector.account_book = [self.book_record(1, "-5", timestamp=101)]
        await self.bot._refresh()
        self.assertEqual(self.bot._pair_pnl("A-USDT"), D("-5"))
        await self.bot._refresh()
        self.assertEqual(self.bot._pair_pnl("A-USDT"), D("-5"))

    async def test_loss_stop_immediately_uses_market_for_long_and_short(self):
        for size in (1000, -1000):
            with self.subTest(size=size):
                self.setUp()
                await self.selected()
                self.config.allow_market_exit = False
                self.connector.positions = [dict(self.position("A-USDT", size), unrealised_pnl="-5")]
                self.bot.expected_positions = {"A-USDT": D(size) / 1000}
                await self.bot._refresh()
                self.bot.on_tick()
                stops = [o for o in self.connector.sent if o[2] == "A-USDT"]
                self.assertEqual(len(stops), 1)
                self.assertEqual(stops[0][4], OrderType.MARKET)
                self.assertEqual(stops[0][3], D("1"))
                self.assertEqual(stops[0][1], size < 0)
                self.assertEqual(stops[0][5]["position_action"], PositionAction.CLOSE)
                self.assertTrue(self.bot.portfolio.slots["A-USDT"].force_market)
                self.assertNotIn("C-USDT", self.bot.portfolio.slots)

    async def test_loss_stop_cancels_both_open_and_maker_close_before_market(self):
        await self.selected()
        self.bot._submit(Intent("A-USDT", True, D("0.1"), D("9.99")))
        self.bot._submit(Intent("A-USDT", False, D("0.5"), D("10.01"), close=True))
        self.connector.positions = [dict(self.position("A-USDT", 1000), unrealised_pnl="-5")]
        self.bot.expected_positions = {"A-USDT": D("1")}
        await self.bot._refresh()
        self.bot.on_tick()
        self.assertEqual(set(self.connector.cancels), {"t-1", "t-2"})
        self.assertFalse(any(o[4] == OrderType.MARKET for o in self.connector.sent))
        for oid in list(self.bot.orders):
            self.bot.did_cancel_order(types.SimpleNamespace(order_id=oid))
        # Local cancel callbacks alone are insufficient: exchange still lists them.
        await self.bot._refresh()
        self.bot.on_tick()
        self.assertFalse(self.bot.halt_reason)
        self.assertFalse(any(o[4] == OrderType.MARKET for o in self.connector.sent))
        self.connector.open_orders = []
        await self.bot._refresh()
        self.bot.on_tick()
        stops = [o for o in self.connector.sent if o[2] == "A-USDT" and o[4] == OrderType.MARKET]
        self.assertEqual(len(stops), 1)

    async def test_stop_partial_close_and_delayed_rest_block_replacement_until_flat(self):
        await self.selected()
        self.connector.positions = [dict(self.position("A-USDT", 1000), unrealised_pnl="-5")]
        self.bot.expected_positions = {"A-USDT": D("1")}
        await self.bot._refresh()
        self.bot.on_tick()
        oid = next(k for k, o in self.bot.orders.items() if o.intent.pair == "A-USDT")
        self.bot.did_fill_order(types.SimpleNamespace(order_id=oid, amount=D("0.4")))
        self.bot.did_complete_sell_order(types.SimpleNamespace(order_id=oid))
        self.connector.open_orders = [o for o in self.connector.open_orders if o["text"] != oid]
        self.connector.positions = [dict(self.position("A-USDT", 600), unrealised_pnl="-1")]
        await self.bot._refresh()
        self.bot.on_tick()
        remaining = next(k for k, o in self.bot.orders.items() if o.intent.pair == "A-USDT")
        self.assertEqual(self.bot.orders[remaining].intent.amount, D("0.6"))
        self.assertTrue(self.bot.orders[remaining].intent.market)
        self.assertNotIn("C-USDT", self.bot.portfolio.slots)
        self.bot.did_fill_order(types.SimpleNamespace(order_id=remaining, amount=D("0.6")))
        self.bot.did_complete_sell_order(types.SimpleNamespace(order_id=remaining))
        self.connector.open_orders = [o for o in self.connector.open_orders if o["text"] != remaining]
        # The final close is confirmed by fills, but the REST position is delayed.
        await self.bot._refresh()
        self.assertTrue(self.bot.account_dirty)
        self.bot.on_tick()
        self.assertNotIn("C-USDT", self.bot.portfolio.slots)
        self.connector.positions = []
        await self.bot._refresh()
        self.bot.on_tick()
        self.assertEqual(set(self.bot.portfolio.slots), {"B-USDT", "C-USDT"})
        self.assertIn("A-USDT", self.bot.portfolio.excluded)

    async def test_flat_realised_loss_triggers_replacement_without_a_close_order(self):
        await self.selected()
        self.connector.account_book = [self.book_record(1, "-5")]
        await self.bot._refresh()
        self.bot.on_tick()
        self.assertEqual(set(self.bot.portfolio.slots), {"B-USDT", "C-USDT"})
        self.assertFalse(any(o[2] == "A-USDT" for o in self.connector.sent))

    async def test_stop_still_executes_when_market_book_missing_and_indicators_unready(self):
        await self.selected()
        self.connector.positions = [dict(self.position("A-USDT", 1000), unrealised_pnl="-5")]
        self.bot.expected_positions = {"A-USDT": D("1")}
        await self.bot._refresh()
        del self.bot.books["A-USDT"]
        self.bot.on_tick()
        stops = [o for o in self.connector.sent if o[2] == "A-USDT"]
        self.assertEqual(stops[0][4], OrderType.MARKET)

    async def test_stale_account_cannot_trigger_or_submit_a_loss_stop(self):
        await self.selected()
        self.bot.positions = {"A-USDT": D("1")}
        self.bot.position_marks = {"A-USDT": D("10")}
        self.bot.unrealised_pnl = {"A-USDT": D("-5")}
        self.bot.account_dirty = True
        self.bot.on_tick()
        self.assertEqual(self.connector.sent, [])
        self.assertIsNone(self.bot._exit_intent("A-USDT", None, 100, force_market=True))

    async def test_account_book_paginates_fixed_window_and_deduplicates_by_id(self):
        await self.selected()
        self.connector.account_book = [self.book_record(i, "-0.001") for i in range(1001)]
        rows = await self.bot._account_book()
        self.assertEqual(len(rows), 1001)
        requests = [kwargs["params"] for path, kwargs in self.connector.requests if path == "futures/usdt/account_book"]
        self.assertEqual(requests[-2]["offset"], 0)
        self.assertEqual(requests[-1]["offset"], 1000)
        self.assertEqual(requests[-2]["from"], requests[-1]["from"])
        self.assertEqual(requests[-2]["to"], requests[-1]["to"])
        self.connector.account_book = [self.book_record(1, "-1"), self.book_record(1, "-1")]
        with self.assertRaisesRegex(RuntimeError, "pagination"):
            await self.bot._account_book()

    async def test_ledger_pagination_limit_halts_instead_of_using_incomplete_losses(self):
        await self.selected()
        self.connector.account_book = [self.book_record(i, "-0.001") for i in range(10000)]
        await self.bot._refresh()
        self.assertIn("Account-book pagination limit", self.bot.halt_reason)
        self.assertTrue(self.bot.account_dirty)

    async def test_invalid_ledger_data_and_non_usdt_fees_are_rejected(self):
        await self.selected()
        for changes in (dict(change="NaN"), dict(time=float("nan")), dict(id=""), dict(id=None),
                        dict(contract=""), dict(type="point_fee"), dict(type="bonus_offset")):
            with self.subTest(changes=changes):
                self.connector.account_book = [dict(self.book_record(1, "-1"), **changes)]
                # Test malformed time directly; fake range filtering would hide NaN.
                with patch.object(self.connector, "_api_get", AsyncMock(return_value=self.connector.account_book)):
                    with self.assertRaises(ValueError):
                        await self.bot._account_book()

    async def test_ledger_failure_pauses_new_orders_and_never_fakes_zero_loss(self):
        await self.selected()
        original = self.connector._api_get

        async def failure(path_url, **kwargs):
            if path_url == "futures/usdt/account_book":
                raise OSError("ledger unavailable")
            return await original(path_url, **kwargs)

        self.connector._api_get = failure
        await self.bot._refresh()
        self.bot.on_tick()
        self.assertTrue(self.bot.account_dirty)
        self.assertEqual(self.connector.sent, [])

    async def test_invalid_unrealised_pnl_and_dry_run_stop_never_submit(self):
        await self.selected()
        self.connector.positions = [dict(self.position("A-USDT", 1000), unrealised_pnl="NaN")]
        self.bot.expected_positions = {"A-USDT": D("1")}
        await self.bot._refresh()
        self.assertIn("Invalid per-pair", self.bot.halt_reason)
        self.setUp()
        await self.ready()
        self.bot.portfolio.evaluate(self.bot._snapshots(), 100, True, {}, set())
        self.connector.positions = [dict(self.position("A-USDT", 1000), unrealised_pnl="-5")]
        await self.bot._refresh()
        self.bot.on_tick()
        self.assertIn("A-USDT", self.bot.portfolio.excluded)
        self.assertEqual(self.connector.sent, [])

    async def test_late_open_fill_after_cancel_is_reconciled_before_stop_close(self):
        await self.selected()
        self.bot._submit(Intent("A-USDT", True, D("0.1"), D("9.99")))
        self.bot.did_cancel_order(types.SimpleNamespace(order_id="t-1"))
        self.bot.did_fill_order(types.SimpleNamespace(order_id="t-1", amount=D("0.1")))
        self.assertEqual(self.bot.expected_positions["A-USDT"], D("0.1"))
        self.assertTrue(self.bot.account_dirty)
        self.connector.open_orders = []
        self.connector.positions = [dict(self.position("A-USDT", 100), unrealised_pnl="-5")]
        await self.bot._refresh()
        self.bot.on_tick()
        closes = [o for o in self.connector.sent if o[4] == OrderType.MARKET]
        self.assertEqual(closes[0][3], D("0.1"))

    async def test_pending_market_stop_waits_for_fill_without_duplicate_close_or_cancel(self):
        await self.selected()
        self.connector.positions = [dict(self.position("A-USDT", 1000), unrealised_pnl="-5")]
        self.bot.expected_positions = {"A-USDT": D("1")}
        await self.bot._refresh()
        self.bot.on_tick()
        stop_id = next(oid for oid, o in self.bot.orders.items() if o.intent.pair == "A-USDT")
        self.bot.on_tick()
        self.bot.on_tick()
        self.assertNotIn(stop_id, self.connector.cancels)
        self.assertEqual(len([o for o in self.connector.sent if o[2] == "A-USDT"]), 1)


class BootstrapTests(unittest.TestCase):
    def test_public_startup_builds_candidate_subscription_pool(self):
        config = adapter.GateAvellanedaPortfolioConfig(candidate_limit=2)
        contracts = [contract(p) for p in ("A-USDT", "B-USDT", "C-USDT")]
        tickers = [dict(contract=p.replace("-", "_"), last="10", volume_24h_quote="10000000")
                   for p in ("A-USDT", "B-USDT", "C-USDT")]
        responses = [io.BytesIO(json.dumps(contracts).encode()), io.BytesIO(json.dumps(tickers).encode())]
        with patch.object(adapter, "urlopen", side_effect=responses):
            adapter.GateAvellanedaPortfolio.init_markets(config)
        self.assertEqual(len(adapter.GateAvellanedaPortfolio.markets["gate_io_perpetual"]), 2)

    def test_empty_universe_fails_with_message(self):
        with patch.object(adapter, "urlopen", side_effect=[io.BytesIO(b"[]"), io.BytesIO(b"[]")]):
            with self.assertRaisesRegex(ValueError, "No affordable"):
                adapter.GateAvellanedaPortfolio.init_markets(adapter.GateAvellanedaPortfolioConfig())

    def test_pydantic_validates_nested_budget(self):
        with self.assertRaises(ValueError):
            adapter.GateAvellanedaPortfolioConfig(risk=dict(capital="50"))

    def test_refresh_period_must_fit_staleness_threshold(self):
        with self.assertRaisesRegex(ValueError, "max_age"):
            adapter.GateAvellanedaPortfolioConfig(book_refresh_seconds=30)


class ConnectorRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_gate_trading_rule_uses_minimum_contract_count(self):
        namespace = dict(Decimal=D, TradingRule=lambda *a, **k: types.SimpleNamespace(args=a, **k))
        method = method_from_source(
            "hummingbot/connector/derivative/gate_io_perpetual/gate_io_perpetual_derivative.py",
            "GateIoPerpetualDerivative", "_format_trading_rules", namespace,
        )
        exchange = types.SimpleNamespace(trading_pair_associated_to_exchange_symbol=AsyncMock(return_value="A-USDT"),
                                         logger=lambda: types.SimpleNamespace(exception=lambda *a: None))
        namespace["web_utils"] = types.SimpleNamespace(is_exchange_information_valid=lambda r: True)
        rule = dict(name="A_USDT", quanto_multiplier="0.01", order_size_min="3", order_price_round="0.001")
        rules = await method(exchange, [rule])
        self.assertEqual(rules[0].min_order_size, D("0.03"))

    async def test_real_gate_order_method_sends_reduce_only_for_close(self):
        namespace = dict(Decimal=D, PositionAction=PositionAction, OrderType=OrderType, CONSTANTS=constants)
        method = method_from_source(
            "hummingbot/connector/derivative/gate_io_perpetual/gate_io_perpetual_derivative.py",
            "GateIoPerpetualDerivative", "_place_order", namespace,
        )
        exchange = types.SimpleNamespace(
            exchange_symbol_associated_to_pair=AsyncMock(return_value="A_USDT"),
            _format_amount_to_size=lambda pair, amount: amount / D("0.001"),
            _api_post=AsyncMock(return_value=dict(id=123, finish_as="open")), current_timestamp=100,
        )
        for action, kind in [(PositionAction.OPEN, OrderType.LIMIT_MAKER),
                             (PositionAction.CLOSE, OrderType.LIMIT_MAKER),
                             (PositionAction.CLOSE, OrderType.MARKET)]:
            await method(exchange, "t-test", "A-USDT", D("0.1"), TradeType.SELL, kind,
                         D("10"), position_action=action)
            payload = exchange._api_post.call_args.kwargs["data"]
            self.assertEqual(payload.get("reduce_only", False), action == PositionAction.CLOSE)
            self.assertEqual(payload["size"], -100)
            self.assertEqual(payload["tif"], "ioc" if kind == OrderType.MARKET else "poc")
        exchange._api_post.return_value = dict(id=124, finish_as="cancelled")
        with self.assertRaises(IOError):
            await method(exchange, "t-test", "A-USDT", D("0.1"), TradeType.BUY, OrderType.LIMIT, D("10"))

    async def test_real_gate_trade_parser_accepts_string_sizes(self):
        namespace = dict(Decimal=D, TradeType=TradeType,
                         OrderBookMessage=lambda **kwargs: types.SimpleNamespace(**kwargs),
                         OrderBookMessageType=types.SimpleNamespace(TRADE="trade"))
        method = method_from_source(
            "hummingbot/connector/derivative/gate_io_perpetual/gate_io_perpetual_api_order_book_data_source.py",
            "GateIoPerpetualAPIOrderBookDataSource", "_parse_trade_message", namespace,
        )
        connector = types.SimpleNamespace(trading_pair_associated_to_exchange_symbol=AsyncMock(return_value="A-USDT"),
                                          _format_size_to_amount=lambda pair, size: size * D("0.001"))
        source = types.SimpleNamespace(_connector=connector)
        queue = asyncio.Queue()
        await method(source, dict(result=[dict(create_time_ms=100000, contract="A_USDT", size="-100", id=1, price="10")]), queue)
        message = queue.get_nowait()
        self.assertEqual(message.content["amount"], D("0.1"))
        self.assertEqual(message.content["trade_type"], float(TradeType.SELL.value))


if __name__ == "__main__":
    unittest.main()

