import json
import tempfile
import types
import unittest
from decimal import Decimal as D
from pathlib import Path
from unittest.mock import patch

from test.hummingbot.strategy.gate_avellaneda.support import FakeConnector, OrderType, PositionAction, contract, load_adapter
from test.hummingbot.strategy.gate_avellaneda.test_account_risk import account

adapter = load_adapter()


class AccountAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.pairs = ["A-USDT", "B-USDT", "C-USDT"]
        adapter.GateAvellanedaPortfolio._initial_contracts = {p: contract(p) for p in self.pairs}
        self.connector = FakeConnector(self.pairs)
        self.connector.account = account()
        adapter.GateAvellanedaPortfolio._initial_tickers = {
            t["contract"].replace("_", "-"): t for t in self.connector.tickers}
        self.config = adapter.GateAvellanedaPortfolioConfig(deadman=dict(enabled=False), execution=dict(retain_quotes=False), quality_control=dict(enabled=False),
            dry_run=False, micro=dict(enabled=False), adaptive=dict(require_correlation=False), recovery=dict(enabled=False),
            account_risk=dict(state_path=self.directory.name + "/risk.json"),
            telemetry=dict(path=self.directory.name + "/quality.jsonl"))
        self.bot = self.new_bot()

    def new_bot(self):
        bot = adapter.GateAvellanedaPortfolio({"gate_io_perpetual": self.connector}, self.config)
        self.addAsyncCleanup(bot.on_stop)
        return bot

    async def ready(self):
        await self.bot._refresh()
        self.assertFalse(self.bot.halt_reason)
        self.bot._sample()
        self.bot.portfolio.evaluate(self.bot._snapshots(), 100, True, {}, set(), entry_check=self.bot._entry_check)
        self.bot.next_account = self.bot.next_monitor = 999
        self.assertEqual(len(self.bot.portfolio.slots), 2)

    async def positions(self, values, pnls=None, cash=None, now=101):
        pnls = pnls or {}
        self.connector.positions = [dict(contract=p.replace("-", "_"), size=str(int(amount * 1000)),
                                         mode="single", mark_price="10", unrealised_pnl=str(pnls.get(p, 0)))
                                    for p, amount in values.items() if amount]
        self.bot.expected_positions = values.copy()
        self.connector.account = cash or account(upnl=str(sum(map(D, map(str, pnls.values())), D(0))))
        self.bot.current_timestamp = now
        await self.bot._refresh()
        self.bot.next_account = self.bot.next_monitor = 999
        self.assertFalse(self.bot.halt_reason)

    def book(self, rid, kind, change, pair="A-USDT", at=101):
        return dict(id=rid, type=kind, change=str(change), contract=pair.replace("-", "_"), time=at)

    async def test_joint_stop_exits_both_with_stale_book_and_market_exit_disabled(self):
        await self.ready()
        self.bot.config.allow_market_exit = False
        self.connector.account_book = [self.book(1, "fee", "-2", "EXITED-USDT")]
        await self.positions({"A-USDT": D(1), "B-USDT": D(-1)}, {"A-USDT": -4, "B-USDT": -4},
                             account(fee="-2", upnl="-8"), now=120)
        self.bot.books.clear()
        self.bot.on_tick()
        self.assertTrue(self.bot.account_risk.latched)
        self.assertEqual(self.bot.account_risk.pnl, D(-10))
        sent = self.connector.sent
        self.assertEqual({o[2] for o in sent}, {"A-USDT", "B-USDT"})
        self.assertTrue(all(o[4] == OrderType.MARKET and o[5]["position_action"] == PositionAction.CLOSE for o in sent))
        self.assertEqual({o[2]: o[1] for o in sent}, {"A-USDT": False, "B-USDT": True})
        self.assertFalse(self.bot.entry_allowed)
        self.bot.on_tick()
        self.assertEqual(len(self.connector.sent), 2)
        self.assertEqual(self.connector.cancels, [])

    async def test_global_stop_waits_for_cancels_and_reconciles_late_fill(self):
        await self.ready()
        self.bot.on_tick()
        opening = next(oid for oid, order in self.bot.orders.items() if order.intent.pair == "A-USDT" and order.intent.buy)
        self.connector.account_book = [self.book(1, "pnl", "-10", "EXITED-USDT")]
        await self.positions({}, cash=account(pnl="-10"), now=101)
        self.bot.on_tick()
        self.assertTrue(self.connector.cancels)
        self.assertTrue(all(not row[5]["position_action"] == PositionAction.CLOSE for row in self.connector.sent))
        for oid in list(self.bot.orders):
            self.bot.did_cancel_order(types.SimpleNamespace(order_id=oid))
        # Exchange still exposes the cancelled order: no close or refill yet.
        self.bot.did_fill_order(types.SimpleNamespace(order_id=opening, exchange_trade_id="late", amount=D("0.2"),
                                                     price=D(10), timestamp=101))
        await self.positions({"A-USDT": D("0.2")}, cash=account(pnl="-10"), now=102)
        count = len(self.connector.sent)
        self.bot.on_tick()
        self.assertEqual(len(self.connector.sent), count)
        self.connector.open_orders = []
        await self.positions({"A-USDT": D("0.2")}, cash=account(pnl="-10"), now=103)
        self.bot.on_tick()
        self.assertEqual(self.connector.sent[-1][3], D("0.2"))
        self.assertEqual(self.connector.sent[-1][4], OrderType.MARKET)

    async def test_partial_global_close_waits_until_terminal_then_closes_remainder(self):
        await self.ready()
        await self.positions({"A-USDT": D(1)}, {"A-USDT": -10})
        self.bot.on_tick()
        oid = self.connector.sent[-1][0]
        self.bot.did_fill_order(types.SimpleNamespace(order_id=oid, exchange_trade_id="partial", amount=D("0.4"),
                                                     price=D(10), timestamp=101))
        await self.positions({"A-USDT": D("0.6")}, {"A-USDT": -10}, now=102)
        self.bot.on_tick()
        self.assertEqual(len(self.connector.sent), 1)
        self.assertNotIn(("A-USDT", oid), self.connector.cancels)
        self.bot.did_fail_order(types.SimpleNamespace(order_id=oid))
        self.connector.open_orders = []
        await self.positions({"A-USDT": D("0.6")}, {"A-USDT": -10}, now=103)
        self.bot.on_tick()
        self.assertEqual(self.connector.sent[-1][3], D("0.6"))
        self.assertEqual(len(self.connector.sent), 2)

    async def test_no_refill_after_confirmed_flat_global_stop(self):
        await self.ready()
        self.connector.account_book = [self.book(1, "pnl", "-10", "OLD-USDT")]
        await self.positions({}, cash=account(pnl="-10"))
        self.bot.next_monitor = 0
        self.bot.on_tick()
        self.assertEqual(self.bot.portfolio.slots, {})
        self.assertEqual(self.connector.sent, [])

    async def test_disk_failure_blocks_open_but_forced_close_still_runs(self):
        await self.ready()
        self.bot._saved_payload = None
        with patch.object(self.bot.risk_store, "save", side_effect=OSError("disk full")):
            self.bot.on_tick()
            self.assertEqual(self.connector.sent, [])
            self.assertIn("disk full", self.bot.persistence_error)
            await self.positions({"A-USDT": D(1)}, {"A-USDT": -10})
            self.bot.on_tick()
            self.assertEqual(self.connector.sent[-1][4], OrderType.MARKET)
        self.bot._checkpoint()
        self.assertEqual(self.bot.persistence_error, "")

    async def test_logging_failure_blocks_new_opening_orders(self):
        await self.ready()
        with patch("hummingbot.strategy.gate_avellaneda.verification.os.chmod", side_effect=OSError("log disk")):
            self.bot.on_tick()
            self.assertEqual(self.connector.sent, [])
            self.assertIn("log disk", self.bot.recorder.error)

    async def test_cash_mismatch_preserves_closes_while_pausing_opens(self):
        await self.ready()
        cash = account()
        cash["total"] = "99"
        await self.positions({"A-USDT": D(1)}, cash=cash)
        self.bot.on_tick()
        self.assertTrue(self.connector.sent)
        self.assertTrue(all(o[5]["position_action"] == PositionAction.CLOSE for o in self.connector.sent))
        self.assertFalse(self.bot.account_risk.allow_open)

    async def test_per_pair_stop_and_loss_circuit_exclusions_survive_restart(self):
        await self.ready()
        await self.positions({"A-USDT": D(1)}, {"A-USDT": -5})
        self.bot.on_tick()
        self.assertIn("A-USDT", self.bot.portfolio.excluded)
        # Stop only after an independently confirmed flat account for restart.
        self.connector.positions = self.connector.open_orders = []
        self.connector.account = account(pnl="-5")
        self.connector.account_book = [self.book(1, "pnl", "-5")]
        await self.bot.on_stop()
        restarted = self.new_bot()
        restarted.current_timestamp = 102
        self.assertIn("A-USDT", restarted.portfolio.excluded)
        self.assertIn("A-USDT", restarted.loss_circuit.seen)
        await restarted._refresh()
        self.assertFalse(restarted.halt_reason)
        self.assertEqual(restarted.account_risk.pnl, D(-5))
        self.assertEqual(restarted.run_started_at, 100)

    async def test_global_latch_and_profit_peak_survive_restart(self):
        await self.ready()
        self.connector.account_book = [self.book(1, "pnl", "8")]
        await self.positions({}, cash=account(pnl="8"))
        self.connector.account_book = [self.book(1, "pnl", "-2")]
        await self.positions({}, cash=account(pnl="-2"), now=102)
        self.bot.on_tick()
        await self.bot.on_stop()
        restarted = self.new_bot()
        restarted.current_timestamp = 103
        self.assertTrue(restarted.account_risk.latched)
        self.assertEqual(restarted.account_risk.peak, D(8))
        with patch.object(restarted, "_setup_live", side_effect=AssertionError("must not reconfigure stopped account")):
            await restarted._refresh()
        self.assertFalse(restarted.halt_reason)
        restarted.next_account = 999
        restarted.on_tick()
        self.assertEqual(restarted.portfolio.slots, {})

    async def test_restart_does_not_adopt_unfinished_position(self):
        await self.ready()
        await self.positions({"A-USDT": D(1)})
        await self.bot.on_stop()
        restarted = self.new_bot()
        restarted.current_timestamp = 102
        await restarted._refresh()
        self.assertIn("flat", restarted.halt_reason)
        self.assertEqual(self.connector.sent, [])

    async def test_account_mismatch_is_rejected_before_mode_changes(self):
        await self.ready()
        await self.bot.on_stop()
        self.connector.account["user"] = 43
        restarted = self.new_bot()
        restarted.current_timestamp = 102
        with patch.object(restarted, "_setup_live", side_effect=AssertionError("wrong account")):
            await restarted._refresh()
        self.assertIn("another Gate", restarted.halt_reason)

    async def test_duplicate_trade_callback_changes_position_only_once(self):
        await self.ready()
        self.bot.on_tick()
        oid = next(iter(self.bot.orders))
        fill = types.SimpleNamespace(order_id=oid, exchange_trade_id="unique", amount=D("0.1"), price=D(10), timestamp=100)
        self.bot.did_fill_order(fill)
        first = self.bot.expected_positions.copy()
        self.bot.did_fill_order(fill)
        self.assertEqual(self.bot.expected_positions, first)
        rows = [json.loads(line) for line in Path(self.config.telemetry.path).read_text().splitlines()]
        self.assertEqual(sum(r["kind"] == "fill" for r in rows), 1)

    async def test_global_guard_also_applies_without_adaptive_or_gamma(self):
        self.bot.config.adaptive = type(self.bot.config.adaptive)(enabled=False)
        self.bot.config.gamma_control = type(self.bot.config.gamma_control)(enabled=False)
        await self.ready()
        self.connector.account_book = [self.book(1, "pnl", "-10", "OLD-USDT")]
        await self.positions({}, cash=account(pnl="-10"))
        self.bot.on_tick()
        self.assertTrue(self.bot.account_risk.latched)
        self.assertEqual(self.connector.sent, [])

    async def test_all_account_book_records_validated_outside_candidate_pool(self):
        await self.ready()
        self.connector.account_book = [self.book(1, "fee", "-1", "OLD-USDT"), self.book(2, "dnw", "50", "")]
        self.bot.current_timestamp = 102
        rows = await self.bot._account_book()
        self.assertEqual(len(rows), 0)
        self.assertEqual(rows.cash_totals["fee"], D("-1"))
        self.assertEqual(rows.cash_totals["dnw"], D("50"))
        self.connector.account_book.append(self.book(2, "dnw", "50", ""))
        with self.assertRaises(RuntimeError):
            await self.bot._account_book()

    async def test_corrupt_state_refuses_constructor(self):
        await self.ready()
        await self.bot.on_stop()
        Path(self.config.account_risk.state_path).write_text("broken")
        with self.assertRaisesRegex(ValueError, "restored"):
            self.new_bot()

    async def test_public_api_outage_does_not_block_authenticated_market_exit(self):
        await self.ready()
        self.bot.next_public = 0
        original = self.connector._api_get

        async def api(path_url, **kwargs):
            if path_url in ("futures/usdt/contracts", "tickers"):
                raise TimeoutError("public outage")
            return await original(path_url, **kwargs)

        with patch.object(self.connector, "_api_get", side_effect=api):
            await self.positions({"A-USDT": D(1)}, {"A-USDT": -10})
        self.assertTrue(self.bot._account_fresh())
        self.bot.on_tick()
        self.assertEqual(self.connector.sent[-1][4], OrderType.MARKET)

    async def test_stop_circuit_and_cooldowns_restore_without_healthy_shortcut(self):
        await self.ready()
        self.bot.loss_circuit.record("A-USDT", 100)
        self.bot.loss_circuit.record("B-USDT", 101)
        self.bot.portfolio.cooldowns["C-USDT"] = 701
        self.bot._checkpoint()
        await self.bot.on_stop()
        restarted = self.new_bot()
        self.assertEqual(restarted.loss_circuit.until, 701)
        self.assertEqual(restarted.portfolio.cooldowns["C-USDT"], 701)
        self.assertFalse(restarted.loss_circuit.allow_open(701, True, 2))
        self.assertIsNotNone(restarted.loss_circuit.healthy_since)

    def test_live_persistence_cannot_be_disabled_while_account_guard_enabled(self):
        with self.assertRaisesRegex(ValueError, "persistent"):
            adapter.GateAvellanedaPortfolioConfig(deadman=dict(enabled=False), execution=dict(retain_quotes=False), quality_control=dict(enabled=False), dry_run=False, account_risk=dict(persist=False))

    async def test_latched_market_close_survives_cash_endpoint_outage(self):
        await self.ready()
        await self.positions({'A-USDT':D('.05')}, {'A-USDT':-5})
        self.bot.portfolio.stop_loss('A-USDT', D(-5), 101)
        original = self.connector._api_get

        async def failed_book(path_url, **kwargs):
            if path_url == 'futures/usdt/account_book':
                raise OSError('cash endpoint timeout')
            return await original(path_url, **kwargs)

        self.connector._api_get = failed_book
        self.bot.current_timestamp = 102
        await self.bot._refresh()
        self.assertTrue(self.bot.cash_error)
        self.assertTrue(self.bot._account_fresh())
        self.assertFalse(self.bot._opening_ready())
        self.bot.on_tick()
        self.assertEqual(len(self.connector.sent), 1)
        self.assertEqual(self.connector.sent[0][3], D('.05'))
        self.assertEqual(self.connector.sent[0][4], OrderType.MARKET)
        self.assertEqual(self.connector.sent[0][5]['position_action'], PositionAction.CLOSE)

    async def test_production_cash_pagination_resumes_after_ten_thousand(self):
        await self.ready()
        self.connector.account_book = [self.book(i, 'fee', '-0.000001') for i in range(10001)]
        self.connector.account = account(fee='-0.010001')
        self.bot.current_timestamp = 102
        await self.bot._refresh()
        self.assertTrue(self.bot.cash_error)
        self.assertFalse(self.bot.halt_reason)
        self.assertTrue(self.bot._account_fresh())
        self.assertEqual(self.bot.cash_journal.page_range(102)[2], 10000)
        await self.bot._refresh()
        self.assertFalse(self.bot.cash_error)
        self.assertFalse(self.bot.account_risk.reconciliation_error)
        self.assertEqual(self.bot.account_risk.cash_totals['fee'], D('-0.010001'))
        self.assertTrue(self.bot._opening_ready())

    async def test_released_pair_late_fee_exclusion_and_loss_circuit_persist(self):
        await self.ready()
        self.bot._checkpoint()
        self.connector.account_book = [self.book(1, 'pnl', '-4.999')]
        self.connector.account = account(pnl='-4.999')
        self.bot.current_timestamp = 102
        await self.bot._refresh()
        self.bot.portfolio.retire('A-USDT', 'test completed lifecycle', 102)
        self.bot.next_monitor = 0
        self.bot.on_tick()
        self.assertNotIn('A-USDT', self.bot.portfolio.slots)
        self.assertNotIn('A-USDT', self.bot.portfolio.excluded)
        self.assertEqual(self.bot.cash_journal.closed_cycles()[0][:2], ('A-USDT', 100))
        self.connector.account_book.append(self.book(2, 'fee', '-0.002', at=103))
        self.connector.account = account(pnl='-4.999', fee='-0.002')
        self.bot.current_timestamp = 104
        await self.bot._refresh()
        self.assertIn('A-USDT', self.bot.portfolio.excluded)
        self.assertIn('A-USDT', self.bot.loss_circuit.seen)
        self.assertEqual(self.bot.account_risk.pnl, D('-5.001'))
        self.bot._checkpoint()
        await self.bot.on_stop()
        self.bot = self.new_bot()
        self.assertIn('A-USDT', self.bot.portfolio.excluded)
        self.assertIn('A-USDT', self.bot.loss_circuit.seen)

    async def test_cash_disk_error_does_not_invalidate_confirmed_position_exit(self):
        await self.ready()
        await self.positions({'A-USDT':D('.05')}, {'A-USDT':-5})
        self.bot.portfolio.stop_loss('A-USDT', D(-5), 101)
        with patch.object(self.bot.cash_journal, 'ingest', side_effect=OSError('disk full')):
            self.bot.current_timestamp = 102
            await self.bot._refresh()
            self.bot.on_tick()
        self.assertTrue(self.bot.cash_error)
        self.assertTrue(self.bot._account_fresh())
        self.assertTrue(all(row[5]['position_action'] == PositionAction.CLOSE for row in self.connector.sent))
        self.assertEqual(len(self.connector.sent), 1)

    async def test_periodic_rescan_finds_old_small_late_fee_below_tolerance(self):
        await self.ready()
        self.bot._checkpoint()
        self.bot.current_timestamp = 1000
        await self.bot._refresh()
        self.connector.account_book = [self.book(1, 'fee', '-0.002', at=101)]
        self.connector.account = account(fee='-0.002')
        self.bot.current_timestamp = 1003
        await self.bot._refresh()
        self.assertEqual(self.bot.cash_journal.snapshot().cash_totals['fee'], 0)
        self.assertFalse(self.bot.account_risk.reconciliation_error)
        self.bot.current_timestamp = 3701
        await self.bot._refresh()
        self.assertEqual(self.bot.cash_journal.snapshot().cash_totals['fee'], D('-0.002'))

    async def test_closed_cycle_database_failure_keeps_safe_exit_available(self):
        await self.ready()
        await self.positions({'A-USDT':D('.05')}, {'A-USDT':-5})
        self.bot.portfolio.stop_loss('A-USDT',D(-5),101)
        with patch.object(self.bot.cash_journal, 'closed_cycles', side_effect=OSError('read failure')):
            self.bot.current_timestamp = 102
            await self.bot._refresh()
            self.assertTrue(self.bot._account_fresh())
            self.assertTrue(self.bot.cash_error)
            self.bot.on_tick()
        self.assertEqual(len(self.connector.sent),1)
        self.assertEqual(self.connector.sent[0][5]['position_action'],PositionAction.CLOSE)

    async def test_rescan_metadata_failure_still_allows_latched_close(self):
        await self.ready()
        await self.positions({'A-USDT':D('.05')}, {'A-USDT':-5})
        self.bot.portfolio.stop_loss('A-USDT',D(-5),101)
        self.connector.account = account(fee='-1',upnl='-5')
        with patch.object(self.bot.cash_journal, 'request_rescan', side_effect=OSError('metadata failure')):
            self.bot.current_timestamp = 102
            await self.bot._refresh()
            self.assertTrue(self.bot._account_fresh())
            self.assertFalse(self.bot._opening_ready())
            self.bot.on_tick()
        self.assertEqual(len(self.connector.sent),1)
        self.assertEqual(self.connector.sent[0][5]['position_action'],PositionAction.CLOSE)
