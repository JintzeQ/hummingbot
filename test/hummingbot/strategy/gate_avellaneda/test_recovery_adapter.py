import ast
import asyncio
import copy
import json
import re
import tempfile
import types
import unittest
from decimal import Decimal as D
from pathlib import Path
from unittest.mock import patch

from hummingbot.strategy.gate_avellaneda.core import Intent
from hummingbot.strategy.gate_avellaneda.recovery import RecoveryLedger, RecoverySettings
from hummingbot.strategy.gate_avellaneda.gamma import GammaState
from test.hummingbot.strategy.gate_avellaneda.support import (
    ROOT, FakeConnector, OrderType, PositionAction, TradeType, contract, load_adapter, method_from_source,
)
from test.hummingbot.strategy.gate_avellaneda.test_account_risk import account

adapter = load_adapter()
CONNECTOR_PATH = "hummingbot/connector/derivative/gate_io_perpetual/gate_io_perpetual_derivative.py"


class RecoveryConnector(FakeConnector):
    reserve_portfolio_order_id = method_from_source(
        CONNECTOR_PATH, "GateIoPerpetualDerivative", "reserve_portfolio_order_id", {"re": re})

    def __init__(self, pairs):
        super().__init__(pairs)
        self.account = account()
        self.details = {}
        self.trades = {}
        self.position_sizes = {}
        self.pnl = {}
        self.rest_cancels = []
        self.stopped = []
        self.before_send = None
        self.stale_open_visibility = False
        self.order_error = None

    def send(self, buy, pair, amount, order_type, kwargs):
        cid = getattr(self, "_gate_portfolio_client_order_id", None)
        if cid is None:
            return super().send(buy, pair, amount, order_type, kwargs)
        del self._gate_portfolio_client_order_id
        if self.before_send is not None:
            self.before_send(cid)
        exchange_id = str(len(self.details) + 1000)
        size = amount * 1000 * (1 if buy else -1)
        raw = dict(id=exchange_id, user=42, text=cid, contract=pair.replace("-", "_"), size=str(size),
                   left=str(abs(size)), price="0" if order_type == OrderType.MARKET else str(kwargs["price"]),
                   tif="ioc" if order_type == OrderType.MARKET else "poc", status="open",
                   is_reduce_only=kwargs["position_action"] == PositionAction.CLOSE)
        self.details[cid] = raw
        self.trades[exchange_id] = []
        self.open_orders.append(copy.deepcopy(raw))
        self.sent.append((cid, buy, pair, amount, order_type, kwargs))
        return cid

    def fill(self, client_id, size, trade_id, timestamp=140, price="10", callback_bot=None):
        raw = self.details[client_id]
        exchange_id = raw["id"]
        raw["left"] = str(D(raw["left"]) - abs(D(size)))
        trade = dict(id=trade_id, order_id=exchange_id, text=client_id, contract=raw["contract"],
                     size=str(size), price=price, create_time=timestamp, fee="0", point_fee="0")
        self.trades[exchange_id].append(trade)
        pair = raw["contract"].replace("_", "-")
        self.position_sizes[pair] = self.position_sizes.get(pair, D(0)) + D(size)
        if D(raw["left"]) == 0:
            raw["status"] = "finished"
            self.open_orders = [o for o in self.open_orders if o["text"] != client_id]
        self.update_positions()
        if callback_bot is not None:
            callback_bot.current_timestamp = timestamp
            callback_bot.did_fill_order(types.SimpleNamespace(order_id=client_id, exchange_trade_id=trade_id,
                                                              amount=abs(D(size)) / 1000, price=D(price), timestamp=timestamp))
        return trade

    def update_positions(self):
        self.positions = [dict(contract=p.replace("-", "_"), size=str(size), mode="single", leverage="1",
                               mark_price="10", unrealised_pnl=str(self.pnl.get(p, 0)))
                          for p, size in self.position_sizes.items() if size]
        self.account["unrealised_pnl"] = str(sum((D(str(self.pnl.get(p, 0))) for p, size in self.position_sizes.items() if size), D(0)))

    async def _api_get(self, path_url, **kwargs):
        if path_url.startswith("futures/usdt/orders/"):
            self.requests.append((path_url, kwargs))
            if self.order_error:
                raise self.order_error
            identity = path_url.rsplit("/", 1)[1]
            raw = self.details.get(identity) or next((r for r in self.details.values() if r["id"] == identity), None)
            if raw is None:
                raise IOError('{"label":"ORDER_NOT_FOUND"}')
            return copy.deepcopy(raw)
        if path_url == "futures/usdt/my_trades":
            self.requests.append((path_url, kwargs))
            query = kwargs["params"]
            rows = self.trades.get(query["order"], [])
            return copy.deepcopy(rows[query["offset"]:query["offset"] + query["limit"]])
        return await super()._api_get(path_url, **kwargs)

    async def _api_delete(self, path_url, **kwargs):
        exchange_id = path_url.rsplit("/", 1)[1]
        raw = next(r for r in self.details.values() if r["id"] == exchange_id)
        self.rest_cancels.append(raw["text"])
        raw["status"] = "finished"
        if not self.stale_open_visibility:
            self.open_orders = [o for o in self.open_orders if o["text"] != raw["text"]]
        return copy.deepcopy(raw)

    def stop_tracking_order(self, client_id):
        self.stopped.append(client_id)


class RecoveryAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.pairs = ["A-USDT", "B-USDT", "C-USDT"]
        adapter.GateAvellanedaPortfolio._initial_contracts = {p: contract(p) for p in self.pairs}
        self.connector = RecoveryConnector(self.pairs)
        adapter.GateAvellanedaPortfolio._initial_tickers = {t["contract"].replace("_", "-"): t for t in self.connector.tickers}
        self.config = adapter.GateAvellanedaPortfolioConfig(
            dry_run=False, micro=dict(enabled=False), adaptive=dict(require_correlation=False),
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
        self.bot.next_account = self.bot.next_monitor = 9999

    def send(self, buy=True, amount="0.5", close=False, market=False, pair="A-USDT"):
        before = len(self.connector.sent)
        self.bot._submit(Intent(pair, buy, D(amount), D(10), close, market))
        self.assertEqual(len(self.connector.sent), before + 1)
        return self.connector.sent[-1][0]

    def crash(self):
        # A killed process cannot run on_stop or flush later in-memory state.
        self.bot.stopping = True
        self.bot.risk_store.close()
        self.bot = self.new_bot()
        self.bot.current_timestamp = 200

    async def refresh(self, at):
        self.bot.current_timestamp = at
        await self.bot._refresh()
        self.bot.next_account = self.bot.next_monitor = 9999

    async def recover(self, at=200):
        for now in (at, at + 1, at + 6, at + 12):
            await self.refresh(now)
        self.assertFalse(self.bot.halt_reason, self.bot.halt_reason)
        self.assertEqual(self.bot.recovery.phase, "ready", self.bot.recovery.reason)

    async def test_write_ahead_metadata_exists_before_exchange_submission(self):
        await self.ready()

        def assertion(cid):
            payload = self.bot.risk_store.load()
            self.assertIn(cid, payload["recovery"]["orders"])
            self.assertEqual(payload["recovery"]["orders"][cid]["intent"]["amount"], "0.5")
            self.assertEqual(payload["recovery"]["slots"]["A-USDT"]["selected_at"], 100)

        self.connector.before_send = assertion
        cid = self.send()
        self.assertEqual(len(cid), 28)

    async def test_replays_offline_partial_fill_cancels_old_order_then_resumes(self):
        await self.ready()
        cid = self.send()
        self.connector.fill(cid, 200, "offline", timestamp=140)
        self.crash()
        await self.refresh(200)
        self.assertIn(cid, self.connector.rest_cancels)
        self.assertEqual(self.bot.expected_positions["A-USDT"], D("0.2"))
        self.assertTrue(self.bot.recovery.recovering)
        self.bot.on_tick()
        self.assertEqual(len(self.connector.sent), 1)
        await self.recover(201)
        self.assertEqual(self.bot.positions["A-USDT"], D("0.2"))
        self.assertEqual(self.bot.inventory.opened_at["A-USDT"], 140)
        self.assertEqual(self.bot.portfolio.slots["A-USDT"].selected_at, 100)
        self.assertIn(cid, self.connector.stopped)
        self.assertNotIn(cid, self.bot.recovery.orders)
        self.bot.on_tick()
        self.assertTrue(any(row[5]["position_action"] == PositionAction.OPEN for row in self.connector.sent[1:]))

    async def test_already_persisted_fill_not_counted_again_after_restart(self):
        await self.ready()
        cid = self.send(amount="1")
        self.connector.fill(cid, 400, "known", timestamp=110, callback_bot=self.bot)
        self.crash()
        await self.recover()
        self.assertEqual(self.bot.expected_positions["A-USDT"], D("0.4"))
        rows = [json.loads(line) for line in Path(self.config.telemetry.path).read_text().splitlines()]
        self.assertEqual(sum(r["kind"] == "fill" and r["trade_id"] == "known" for r in rows), 1)

    async def test_completed_order_without_any_callback_is_fully_recovered(self):
        await self.ready()
        cid = self.send()
        self.connector.fill(cid, 500, "done", timestamp=120)
        self.crash()
        await self.recover()
        self.assertEqual(self.bot.expected_positions["A-USDT"], D("0.5"))
        self.assertEqual(self.connector.rest_cancels, [])

    async def test_late_fill_after_cancel_request_included_before_resume(self):
        await self.ready()
        cid = self.send()
        self.crash()
        await self.refresh(200)
        self.connector.fill(cid, 100, "cancel-race", timestamp=200.5)
        await self.recover(201)
        self.assertEqual(self.bot.expected_positions["A-USDT"], D("0.1"))

    async def test_stale_open_visibility_cannot_release_or_become_unowned(self):
        await self.ready()
        cid = self.send()
        self.connector.stale_open_visibility = True
        self.crash()
        for at in (200, 201, 208):
            await self.refresh(at)
        self.assertFalse(self.bot.halt_reason)
        self.assertTrue(self.bot.recovery.recovering)
        self.assertIn(cid, self.bot.recovery.orders)
        self.connector.open_orders = []
        await self.recover(209)

    async def test_offline_pair_loss_continues_market_exit_before_opening_recovery_ready(self):
        await self.ready()
        cid = self.send(amount="1")
        self.connector.fill(cid, 1000, "opening", timestamp=110, callback_bot=self.bot)
        self.connector.pnl["A-USDT"] = -5
        self.connector.update_positions()
        self.crash()
        await self.refresh(200)
        self.assertTrue(self.bot._recovery_verified)
        self.assertTrue(self.bot.recovery.recovering)
        self.bot.config.allow_market_exit = False
        self.bot.books.clear()
        self.bot.on_tick()
        exit_order = self.connector.sent[-1]
        self.assertEqual(exit_order[4], OrderType.MARKET)
        self.assertEqual(exit_order[5]["position_action"], PositionAction.CLOSE)
        self.assertEqual(exit_order[3], D(1))
        self.assertIn("A-USDT", self.bot.portfolio.excluded)

    async def test_restarting_again_during_partial_loss_exit_preserves_latch_and_remaining(self):
        await self.ready()
        cid = self.send(amount="1")
        self.connector.fill(cid, 1000, "opening", timestamp=110, callback_bot=self.bot)
        self.bot.did_complete_buy_order(types.SimpleNamespace(order_id=cid))
        self.connector.pnl["A-USDT"] = -10
        self.connector.update_positions()
        await self.refresh(120)
        self.bot.on_tick()
        close_id = self.connector.sent[-1][0]
        self.assertNotEqual(close_id, cid)
        self.assertEqual(self.connector.sent[-1][4], OrderType.MARKET)
        self.connector.fill(close_id, -400, "close-partial", timestamp=121, callback_bot=self.bot)
        self.connector.details[close_id]["status"] = "finished"
        self.connector.open_orders = []
        self.crash()
        await self.refresh(200)
        self.bot.on_tick()
        self.assertTrue(self.bot.account_risk.latched)
        self.assertEqual(self.connector.sent[-1][3], D("0.6"))
        self.assertEqual(self.connector.sent[-1][4], OrderType.MARKET)
        self.assertFalse(self.bot.entry_allowed)

    async def test_existing_market_reduce_order_not_cancelled_or_duplicated_on_restart(self):
        await self.ready()
        cid = self.send(amount="1")
        self.connector.fill(cid, 1000, "opening", timestamp=110, callback_bot=self.bot)
        close_id = self.send(buy=False, amount="1", close=True, market=True)
        self.crash()
        await self.refresh(200)
        self.bot.on_tick()
        self.assertNotIn(close_id, self.connector.rest_cancels)
        self.assertEqual(len(self.connector.sent), 2)
        self.assertTrue(self.bot.recovery.recovering)

    async def test_unowned_order_blocks_recovery_without_cancelling_it(self):
        await self.ready()
        self.send()
        self.crash()
        self.connector.open_orders.append(dict(text="manual", contract="A_USDT"))
        await self.refresh(200)
        self.assertIn("unowned", self.bot.halt_reason)
        self.assertEqual(self.connector.rest_cancels, [])

    async def test_unexplained_position_change_blocks_recovery(self):
        await self.ready()
        cid = self.send()
        self.connector.fill(cid, 500, "opening", timestamp=110, callback_bot=self.bot)
        self.crash()
        self.connector.position_sizes["A-USDT"] += 100
        self.connector.update_positions()
        await self.refresh(200)
        self.assertFalse(self.bot._recovery_verified)
        self.assertIn("positions differ", self.bot.recovery.reason)
        self.bot.on_tick()
        self.assertEqual(len(self.connector.sent), 1)

    async def test_missing_order_waits_grace_and_multiple_stable_snapshots(self):
        await self.ready()
        cid = self.bot.recovery.new_id()
        self.bot.recovery.prepare(cid, Intent("A-USDT", True, D("0.5"), D(10), False, False), 100)
        self.bot._checkpoint()  # Crash before any request was scheduled.
        self.crash()
        await self.refresh(200)
        self.assertTrue(self.bot.recovery.recovering)
        self.assertIn("grace", self.bot.recovery.reason)
        await self.refresh(220)
        self.assertTrue(self.bot.recovery.recovering)
        await self.refresh(226)
        self.assertEqual(self.bot.recovery.phase, "ready")
        self.assertEqual(self.connector.sent, [])

    async def test_filled_order_not_found_is_never_assumed_zero(self):
        await self.ready()
        cid = self.send()
        self.connector.fill(cid, 500, "opening", timestamp=110, callback_bot=self.bot)
        self.connector.details.clear()
        self.crash()
        await self.refresh(300)
        self.assertIn("previously filled", self.bot.halt_reason)

    async def test_api_error_not_interpreted_as_order_not_found(self):
        await self.ready()
        self.send()
        self.crash()
        self.connector.order_error = IOError('{"label":"INTERNAL_SERVER_ERROR"}')
        await self.refresh(300)
        self.assertTrue(self.bot.recovery.recovering)
        self.assertEqual(self.connector.rest_cancels, [])

    async def test_modified_order_identity_size_or_reduce_only_blocks_before_cancel(self):
        for changed in (dict(user=43), dict(size="501"), dict(is_reduce_only=True), dict(price="11")):
            with self.subTest(changed=changed):
                await self.ready()
                cid = self.send()
                self.connector.details[cid].update(changed)
                self.crash()
                await self.refresh(200)
                self.assertTrue(self.bot.halt_reason)
                self.assertNotIn(cid, self.connector.rest_cancels)
                # Reset the fixture with a new durable state for the next case.
                await self.bot.on_stop()
                self.config.account_risk.state_path += ".next"
                self.connector = RecoveryConnector(self.pairs)
                self.bot = self.new_bot()

    async def test_terminal_status_fill_sum_mismatch_never_resumes(self):
        await self.ready()
        cid = self.send()
        self.connector.fill(cid, 500, "opening", timestamp=110)
        self.connector.trades[self.connector.details[cid]["id"]] = []
        self.crash()
        await self.refresh(200)
        self.assertIn("totals disagree", self.bot.recovery.reason)
        self.assertTrue(self.bot.recovery.recovering)

    async def test_recovered_position_wrong_leverage_or_hedge_mode_not_changed(self):
        await self.ready()
        cid = self.send()
        self.connector.fill(cid, 500, "opening", timestamp=110, callback_bot=self.bot)
        self.crash()
        self.connector.positions[0]["leverage"] = "3"
        with patch.object(self.connector, "_set_trading_pair_leverage", side_effect=AssertionError("must not mutate exposure")):
            await self.refresh(200)
        self.assertIn("not confirmed 1x", self.bot.halt_reason)

    async def test_holding_age_and_retirement_timer_include_offline_time(self):
        await self.ready()
        cid = self.send()
        self.connector.fill(cid, 500, "opening", timestamp=110, callback_bot=self.bot)
        self.bot.portfolio.retire("A-USDT", "holding timeout", 120)
        self.bot._checkpoint()
        self.crash()
        await self.recover(800)
        self.assertEqual(self.bot.portfolio.slots["A-USDT"].retiring_since, 120)
        self.assertGreater(self.bot.inventory.age("A-USDT", D("0.5"), 810), 600)
        self.bot.on_tick()
        exits = [row for row in self.connector.sent if row[2] == "A-USDT" and row[5]["position_action"] == PositionAction.CLOSE]
        self.assertEqual(exits[-1][4], OrderType.MARKET)

    async def test_missing_fill_id_waits_for_rest_replay_then_keeps_exact_position(self):
        await self.ready()
        cid = self.send()
        self.connector.fill(cid, 500, "opening", timestamp=110)
        self.bot.current_timestamp = 110
        self.bot.did_fill_order(types.SimpleNamespace(order_id=cid, amount=D("0.5"), price=D(10), timestamp=110))
        self.assertTrue(self.bot._recovery_block_open)
        await self.refresh(111)
        await self.refresh(112)
        self.assertEqual(self.bot.expected_positions["A-USDT"], D("0.5"))

    async def test_confirmed_terminal_pruned_only_after_fill_and_position_agreement(self):
        await self.ready()
        cid = self.send()
        self.connector.fill(cid, 500, "opening", timestamp=110, callback_bot=self.bot)
        self.bot.did_complete_buy_order(types.SimpleNamespace(order_id=cid))
        await self.refresh(111)
        self.assertNotIn(cid, self.bot.recovery.orders)
        self.assertEqual(self.bot.expected_positions["A-USDT"], D("0.5"))

    async def test_persistence_failure_sends_no_open_and_preserves_reduce_only_fallback(self):
        await self.ready()
        with patch.object(self.bot.risk_store, "save", side_effect=OSError("disk full")):
            self.bot._submit(Intent("A-USDT", True, D("0.5"), D(10), False, False))
            self.assertEqual(self.connector.sent, [])
            self.bot._submit(Intent("A-USDT", False, D("0.5"), D(10), True, True))
            self.assertEqual(self.connector.sent[-1][5]["position_action"], PositionAction.CLOSE)

    async def test_dry_run_restart_restores_state_without_sticking_in_recovery(self):
        await self.bot.on_stop()
        self.config.dry_run = True
        self.bot = self.new_bot()
        await self.ready()
        self.bot._checkpoint()
        self.crash()
        await self.refresh(200)
        self.assertEqual(self.bot.recovery.phase, "ready")
        self.assertFalse(self.bot.halt_reason)
        self.bot.on_tick()
        self.assertEqual(self.connector.sent, [])

    async def test_restored_two_slots_wait_for_fresh_correlation_before_opening(self):
        await self.ready()
        self.bot._checkpoint()
        self.config.adaptive = type(self.config.adaptive)(require_correlation=True)
        self.crash()
        await self.recover()
        self.bot.on_tick()
        self.assertEqual(self.connector.sent, [])
        self.assertEqual(self.bot._recovery_correlation_pairs, {"A-USDT", "B-USDT"})
        self.bot.next_quote = {}
        with patch.object(self.bot.return_history, "eligible", return_value=True):
            self.bot.on_tick()
        self.assertTrue(self.connector.sent)

    async def test_gamma_and_same_side_cooldown_survive_restart(self):
        await self.ready()
        cid = self.send()
        self.connector.fill(cid, 500, "opening", timestamp=110, callback_bot=self.bot)
        self.bot.gamma.states["A-USDT"] = GammaState(D("2.3"), D("2.5"), 110, dict(loss=D("0.5")))
        self.bot.inventory.cooldowns["A-USDT"] = True, 300
        self.bot._checkpoint()
        self.crash()
        self.assertEqual(self.bot.gamma.value("A-USDT"), D("2.3"))
        self.assertEqual(self.bot.inventory.blocked_side("A-USDT", 200), True)
        await self.recover()
        self.assertEqual(self.bot.inventory.opened_at["A-USDT"], 110)

    async def test_cash_mismatch_blocks_recovery_even_when_positions_match(self):
        await self.ready()
        self.bot._checkpoint()
        self.crash()
        self.connector.account["total"] = "99"
        for at in (200, 206, 212):
            await self.refresh(at)
        self.assertTrue(self.bot.recovery.recovering)
        self.assertIn("mismatch", self.bot.recovery.reason)
        self.assertFalse(self.bot._recovery_verified)

    async def test_new_slots_remain_owned_until_terminal_settlement_is_proven(self):
        await self.ready()
        cid = self.send()
        self.bot.did_cancel_order(types.SimpleNamespace(order_id=cid))
        self.bot.portfolio.retire("A-USDT", "replace", 100)
        self.bot.account_dirty = False
        self.bot.next_monitor = 0
        self.bot.on_tick()
        self.assertIn("A-USDT", self.bot.portfolio.slots)

    async def test_out_of_pool_owned_pair_is_subscribed_even_if_volume_drops(self):
        await self.ready()
        cid = self.send()
        self.connector.fill(cid, 500, "opening", timestamp=110, callback_bot=self.bot)
        tickers = copy.deepcopy(self.connector.tickers)
        tickers[0]["volume_24h_quote"] = "1"
        import io
        responses = [io.BytesIO(json.dumps(value).encode()) for value in (self.connector.contracts, tickers)]
        with patch.object(adapter, "urlopen", side_effect=responses):
            adapter.GateAvellanedaPortfolio.init_markets(self.config)
        self.assertIn("A-USDT", adapter.GateAvellanedaPortfolio.markets["gate_io_perpetual"])

    async def test_changed_allowlist_retires_owned_pair_instead_of_adding_new_exposure(self):
        await self.ready()
        self.bot._checkpoint()
        self.config.candidate_pairs = ["B-USDT", "C-USDT"]
        self.crash()
        self.assertEqual(self.bot.portfolio.slots["A-USDT"].state, "retiring")

    async def test_fills_during_recovery_do_not_double_count_rest_replay(self):
        await self.ready()
        cid = self.send()
        self.crash()
        self.connector.fill(cid, 200, "offline", timestamp=140, callback_bot=self.bot)
        self.assertNotIn("A-USDT", self.bot.expected_positions)
        await self.recover()
        self.assertEqual(self.bot.expected_positions["A-USDT"], D("0.2"))

    async def test_recovery_fill_pages_use_one_order_and_bound_offsets(self):
        await self.ready()
        self.connector.trades["paged"] = [dict(id=str(i)) for i in range(1001)]
        rows = await self.bot._order_fills("paged", "A-USDT")
        self.assertEqual(len(rows), 1001)
        params = [r[1]["params"] for r in self.connector.requests if r[0] == "futures/usdt/my_trades"]
        self.assertEqual([p["offset"] for p in params], [0, 1000])
        self.assertTrue(all(p["order"] == "paged" and p["contract"] == "A_USDT" for p in params))

    async def test_framework_submission_exception_keeps_ownership_and_cleans_reservation(self):
        await self.ready()
        with patch.object(self.bot, "buy", side_effect=RuntimeError("framework failure")):
            self.bot._submit(Intent("A-USDT", True, D("0.5"), D(10), False, False))
        self.assertEqual(self.connector.sent, [])
        self.assertFalse(hasattr(self.connector, "_gate_portfolio_client_order_id"))
        self.assertEqual(len(self.bot.recovery.orders), 1)
        self.assertTrue(next(iter(self.bot.recovery.orders.values()))["terminal"])
        self.crash()
        await self.recover(220)

    async def test_fill_pagination_cap_blocks_incomplete_recovery(self):
        await self.ready()
        self.connector.trades["full"] = [dict(id=str(i)) for i in range(10000)]
        with self.assertRaisesRegex(ValueError, "pagination"):
            await self.bot._order_fills("full", "A-USDT")

    async def test_wrong_account_refuses_recovery_before_any_exchange_cancellation(self):
        await self.ready()
        self.send()
        self.crash()
        self.connector.account["user"] = 43
        await self.refresh(200)
        self.assertIn("another Gate", self.bot.halt_reason)
        self.assertEqual(self.connector.rest_cancels, [])

    async def test_old_schema_with_open_position_is_not_silently_adopted(self):
        await self.ready()
        payload = self.bot.risk_store.load()
        payload.pop("recovery")
        self.bot.risk_store.save(payload)
        self.connector.position_sizes["A-USDT"] = D(500)
        self.connector.update_positions()
        self.crash()
        await self.refresh(200)
        self.assertIn("flat", self.bot.halt_reason)

    async def test_stop_cancels_pending_native_submission_tasks(self):
        await self.ready()
        cid = self.send()
        future = asyncio.create_task(asyncio.Event().wait())
        self.connector._gate_portfolio_order_tasks = {cid: future}
        await self.bot.on_stop()
        self.assertTrue(future.cancelled())

    async def test_scope_missing_owned_market_refuses_before_trading(self):
        await self.ready()
        self.send()
        self.bot._checkpoint()
        self.bot.stopping = True
        self.bot.risk_store.close()
        adapter.GateAvellanedaPortfolio._initial_contracts = {p: contract(p) for p in ("B-USDT", "C-USDT")}
        with self.assertRaisesRegex(ValueError, "absent"):
            self.new_bot()

    async def test_recovery_never_changes_leverage_before_position_ownership_matches(self):
        await self.ready()
        cid = self.send()
        self.connector.fill(cid, 500, "opening", timestamp=110, callback_bot=self.bot)
        self.crash()
        self.connector.position_sizes["A-USDT"] += 100
        self.connector.update_positions()
        with patch.object(self.connector, "_set_trading_pair_leverage", side_effect=AssertionError("must wait for ownership")):
            await self.refresh(200)
        self.assertFalse(self.bot.halt_reason)
        self.assertIn("positions differ", self.bot.recovery.reason)

    def test_recovery_requires_persistent_risk(self):
        for account_risk in (dict(enabled=False), dict(persist=False)):
            with self.assertRaisesRegex(ValueError, "persistent"):
                adapter.GateAvellanedaPortfolioConfig(account_risk=account_risk)


class GateReservedOrderProductionTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_connector_methods_preserve_framework_path_and_cancelable_task(self):
        tree = ast.parse((ROOT / CONNECTOR_PATH).read_text())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GateIoPerpetualDerivative")
        wanted = {"reserve_portfolio_order_id", "buy", "sell", "_order_with_reserved_id"}
        klass.body = [node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        klass.bases = [ast.Name(id="Base", ctx=ast.Load())]
        captured = []

        class Base:
            def buy(self, *args, **kwargs):
                return "base-buy"

            def sell(self, *args, **kwargs):
                return "base-sell"

            async def _create_order(self, **kwargs):
                captured.append(kwargs)

        namespace = dict(Base=Base, asyncio=asyncio, re=re, Decimal=D, OrderType=OrderType,
                         TradeType=TradeType, s_decimal_NaN=D("NaN"))
        code = ast.fix_missing_locations(ast.Module(body=[klass], type_ignores=[]))
        exec(compile(code, str(ROOT / CONNECTOR_PATH), "exec"), namespace)
        connector = namespace["GateIoPerpetualDerivative"]()
        self.assertEqual(connector.buy("A-USDT", D(1)), "base-buy")
        self.assertEqual(connector.sell("A-USDT", D(1)), "base-sell")
        cid = RecoveryLedger(RecoverySettings()).new_id()
        connector.reserve_portfolio_order_id(cid)
        self.assertEqual(connector.sell("A-USDT", D(1), OrderType.MARKET, D(10), position_action=PositionAction.CLOSE), cid)
        await asyncio.gather(*connector._gate_portfolio_order_tasks.values())
        self.assertEqual(captured[0]["order_id"], cid)
        self.assertEqual(captured[0]["trade_type"], TradeType.SELL)
        self.assertEqual(captured[0]["position_action"], PositionAction.CLOSE)
        self.assertFalse(hasattr(connector, "_gate_portfolio_client_order_id"))
        self.assertEqual(connector._gate_portfolio_order_tasks, {})
