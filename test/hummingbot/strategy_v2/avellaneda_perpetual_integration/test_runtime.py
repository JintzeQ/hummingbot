"""Compiled Clock/strategy/Gate integration with only transport mocked.

Requires a built Hummingbot installation. No real network or credentials.
"""

import asyncio
import math
import unittest
from decimal import Decimal as D
from types import SimpleNamespace
from unittest.mock import AsyncMock, PropertyMock, patch

from bidict import bidict

from hummingbot.connector.derivative.gate_io_perpetual.gate_io_perpetual_derivative import (
    GateIoPerpetualDerivative as Gate,
)
from hummingbot.connector.derivative.gate_io_perpetual.gate_io_perpetual_request_guard import (
    GateRequestDeferred,
    GateRequestGuard,
)
from hummingbot.connector.derivative.position import Position
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.clock import Clock
from hummingbot.core.clock_mode import ClockMode
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionSide, TradeType
from hummingbot.core.data_type.in_flight_order import OrderState, OrderUpdate, TradeUpdate
from hummingbot.core.data_type.order_book import OrderBook
from hummingbot.core.data_type.order_book_row import OrderBookRow
from hummingbot.core.data_type.trade_fee import AddedToCostTradeFee, TokenAmount
from hummingbot.core.web_assistant.connections.data_types import RESTMethod
from hummingbot.strategy_v2.utils.avellaneda_perpetual import Command, Quote
from scripts.avellaneda_perpetual import AvellanedaPerpetual, AvellanedaPerpetualConfig


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_rest_boundary_counts_429_and_defers_until_reset(self):
        s, ex, book = self.setup_strategy(dry_run=False)
        now = [1700000000.0]
        ex.request_guard.clock = lambda: now[0]
        assistant = await ex._web_assistants_factory.get_rest_assistant()
        response = SimpleNamespace(method=RESTMethod.POST,
                                   url="https://api.gateio.ws/api/v4/futures/usdt/orders",
                                   status=429, headers={"X-Gate-RateLimit-Reset-Timestamp": str(now[0] + 20)},
                                   json=AsyncMock(return_value={"id": "123"}), text=AsyncMock(return_value="limited"))
        try:
            with patch.object(ex._web_assistants_factory, "get_rest_assistant", AsyncMock(return_value=assistant)), \
                    patch.object(assistant._connection, "call", AsyncMock(return_value=response)) as transport:
                args = ("t-HBOT-guard", "BTC-USDT", D("0.0001"), TradeType.BUY,
                        OrderType.LIMIT_MAKER, D("79900"))
                with self.assertRaises(GateRequestDeferred):
                    await ex._place_order(*args, position_action=PositionAction.OPEN)
                self.assertIn("t-HBOT-guard", ex.request_guard.deferred_orders)
                self.assertEqual(1, ex.request_guard.stats()["requests_24h"])
                self.assertEqual(1, transport.await_count)
                with self.assertRaises(GateRequestDeferred):
                    await ex._place_order(*args, position_action=PositionAction.OPEN)
                self.assertEqual(1, transport.await_count)
                self.assertEqual(1, ex.request_guard.stats()["requests_24h"])
                now[0] += 21
                response.status, response.headers = 201, {}
                result = await ex._place_order(*args, position_action=PositionAction.OPEN)
                self.assertEqual("123", result[0])
                self.assertEqual(2, transport.await_count)
                self.assertEqual(2, ex.request_guard.stats()["requests_24h"])
        finally:
            await ex._web_assistants_factory.close()

    async def test_429_failure_event_does_not_escalate_or_consume_emergency_attempt(self):
        s, ex, book = self.setup_strategy(dry_run=False)
        assistant = await ex._web_assistants_factory.get_rest_assistant()
        response = SimpleNamespace(method=RESTMethod.POST,
                                   url="https://api.gateio.ws/api/v4/futures/usdt/orders",
                                   status=429, headers={"Retry-After": "10"},
                                   json=AsyncMock(), text=AsyncMock(return_value="limited"))
        try:
            with patch.object(Gate, "ready", new_callable=PropertyMock, return_value=True), \
                    patch.object(ex._web_assistants_factory, "get_rest_assistant", AsyncMock(return_value=assistant)), \
                    patch.object(assistant._connection, "call", AsyncMock(return_value=response)) as transport:
                clock = Clock(ClockMode.BACKTEST, 1, 0, 100)
                clock.add_iterator(s)
                clock.backtest_til(1)
                ex._set_current_timestamp(1)
                s.engine.halt("test loss")
                quote = Quote("sell", D("0.0001"), D("80000"), close=True, market=True)
                s._execute([Command("create", quote=quote)])
                for _ in range(100):
                    await asyncio.sleep(0)
                    if transport.await_count and not s.engine.orders:
                        break
                self.assertEqual(1, transport.await_count)
                self.assertEqual({}, s.engine.orders)
                self.assertEqual(0, s.engine.failures)
                self.assertEqual(0, s.engine.close_attempts)
                self.assertTrue(s.engine.rest_ack_required)
                self.assertEqual(1, ex.request_guard.rate_limited)
        finally:
            await ex._web_assistants_factory.close()

    async def test_actual_cancel_429_preserves_working_order_until_retry_ack(self):
        s, ex, book = self.setup_strategy(dry_run=False)
        now = [1700000000.0]
        ex.request_guard.clock = lambda: now[0]
        assistant = await ex._web_assistants_factory.get_rest_assistant()
        response = SimpleNamespace(method=RESTMethod.DELETE,
                                   url="https://api.gateio.ws/api/v4/futures/usdt/orders/123",
                                   status=429, headers={"Retry-After": "20"},
                                   json=AsyncMock(return_value={"finish_as": "cancelled"}), text=AsyncMock())
        tracked = SimpleNamespace(get_exchange_order_id=AsyncMock(return_value="123"))
        s.engine.register("order", Quote("buy", D("0.0001"), D("79000")), 1)
        s.engine._cancel_all(2)
        try:
            with patch.object(ex._web_assistants_factory, "get_rest_assistant", AsyncMock(return_value=assistant)), \
                    patch.object(assistant._connection, "call", AsyncMock(return_value=response)) as transport:
                with self.assertRaises(GateRequestDeferred):
                    await ex._place_cancel("order", tracked)
                s.engine.sync_deferred_cancels()
                self.assertIsNone(s.engine.orders["order"].cancel_at)
                self.assertEqual([], s.engine._cancel_all(15))
                self.assertIsNone(s.engine.halt_reason)
                self.assertEqual(1, transport.await_count)
                now[0] += 21
                response.status, response.headers = 200, {}
                self.assertTrue(await ex._place_cancel("order", tracked))
                self.assertEqual(2, ex.request_guard.total_attempts)
                self.assertEqual(2, transport.await_count)
        finally:
            await ex._web_assistants_factory.close()

    def setup_strategy(self, **settings):
        ex = Gate("dummy-key", "dummy-secret", "dummy-id", trading_pairs=["BTC-USDT"], trading_required=False)
        ex.request_guard = GateRequestGuard()  # Test transport only; no persistent account journal.
        ex._account_balances["USDT"] = D("100")
        ex._account_available_balances["USDT"] = D("100")
        ex._trading_rules["BTC-USDT"] = TradingRule(
            "BTC-USDT", min_price_increment=D("0.1"), min_base_amount_increment=D("0.0001"),
            min_order_size=D("0.0001"), min_notional_size=D("1"),
            buy_order_collateral_token="USDT", sell_order_collateral_token="USDT")
        ex._set_trading_pair_symbol_map(bidict({"BTC_USDT": "BTC-USDT"}))
        ex._perpetual_trading.set_leverage("BTC-USDT", 1)
        ex._set_current_timestamp(0)
        book = OrderBook()
        ex.order_book_tracker._order_books["BTC-USDT"] = book
        self.set_book(book, 0)
        config = AvellanedaPerpetualConfig(**settings)
        AvellanedaPerpetual.init_markets(config)
        return AvellanedaPerpetual({"gate_io_perpetual": ex}, config), ex, book

    @staticmethod
    def set_book(book, t):
        mid = 80000 * (1 + 0.0005 * math.sin(t / 17) + 0.0002 * math.sin(t / 5))
        book.apply_snapshot([OrderBookRow(mid - 0.05, 1, t + 1)],
                            [OrderBookRow(mid + 0.05, 1, t + 1)], t + 1)

    async def test_compiled_clock_runs_adaptive_preview_and_exports_once_without_api_mutations(self):
        s, ex, book = self.setup_strategy(gamma_mode="adaptive")
        clock = Clock(ClockMode.BACKTEST, 1, 0, 1200)
        clock.add_iterator(s)
        with patch.object(Gate, "ready", new_callable=PropertyMock, return_value=True), \
                patch.object(ex, "_api_get", AsyncMock()) as get, \
                patch.object(ex, "_api_post", AsyncMock()) as post, \
                patch.object(s, "_export_calibration") as export:
            for t in range(1, 1001):
                self.set_book(book, t)
                clock.backtest_til(t)
            self.assertEqual(1000, s.current_timestamp)
            self.assertIsNotNone(s.engine.gamma.profile)
            self.assertEqual(2, len(s.engine.preview))
            self.assertEqual({}, ex.in_flight_orders)
            self.assertEqual({}, s.engine.orders)
            self.assertGreater(s.engine.gamma.current, 1)
            self.assertIn("cap: 40", s.format_status())
            export.assert_called_once()
            get.assert_not_called()
            post.assert_not_called()
            await s.on_stop()
            post.assert_not_called()

    async def test_real_strategy_order_dispatch_and_compiled_fill_cancel_event_bridge(self):
        s, ex, book = self.setup_strategy(dry_run=False, warmup_samples=3)
        position = Position("BTC-USDT", PositionSide.LONG, D("0"), D("80000"), D("0.0001"), 1)
        ex._perpetual_trading._account_positions["BTC-USDT"] = position
        replies = iter(("101", "102"))

        async def submit(**kwargs):
            return {"id": next(replies), "finish_as": ""}

        with patch.object(Gate, "ready", new_callable=PropertyMock, return_value=True), \
                patch.object(ex, "_api_post", AsyncMock(side_effect=submit)) as post:
            # Limit-order tracking needs a real Clock timestamp, as at CLI start.
            clock = Clock(ClockMode.BACKTEST, 1, 0, 100)
            clock.add_iterator(s)
            clock.backtest_til(1)
            ex._set_current_timestamp(1)
            s.engine.step(s._snapshot())
            s._execute([Command("create", quote=q) for q in s.engine.quotes(s._snapshot())])
            for _ in range(50):
                await asyncio.sleep(0)
                if post.await_count == 2:
                    break
            self.assertEqual(2, post.await_count)
            payloads = [call.kwargs["data"] for call in post.call_args_list]
            self.assertEqual(["poc", "poc"], [p["tif"] for p in payloads])
            self.assertNotIn("reduce_only", payloads[0])
            self.assertIs(True, payloads[1]["reduce_only"])
            self.assertEqual([1, -1], [p["size"] for p in payloads])
            close_id = next(oid for oid, order in ex.in_flight_orders.items() if order.trade_type == TradeType.SELL)
            tracked = ex.in_flight_orders[close_id]
            update = TradeUpdate("fill-1", close_id, tracked.exchange_order_id, "BTC-USDT", 1,
                                 tracked.price, D("0.0001"), tracked.price * D("0.0001"),
                                 AddedToCostTradeFee(flat_fees=[TokenAmount("USDT", D("0.0016"))]))
            ex._order_tracker.process_trade_update(update)
            ex._order_tracker.process_trade_update(update)
            self.assertEqual(D("0"), s.engine.expected_position)
            self.assertEqual(1, len(s.engine.markouts))
            self.assertEqual("0.0016", s.engine.markouts[0]["actual_fee_quote"])
            for oid, order in list(ex.in_flight_orders.items()):
                ex._order_tracker.process_order_update(OrderUpdate(
                    "BTC-USDT", 2, OrderState.FILLED if oid == close_id else OrderState.CANCELED,
                    client_order_id=oid, exchange_order_id=order.exchange_order_id))
            for _ in range(50):
                await asyncio.sleep(0)
                if not s.engine.orders:
                    break
            self.assertEqual({}, s.engine.orders)
            self.assertEqual({}, ex.in_flight_orders)

    async def test_real_emergency_market_close_reaches_gate_as_reduce_only_ioc(self):
        s, ex, book = self.setup_strategy(dry_run=False)
        with patch.object(ex, "_api_post", AsyncMock(return_value={"id": "101", "finish_as": ""})) as post:
            s.engine.halt("test loss guard")
            s._execute([Command("create", quote=Quote("sell", D("0.0001"), D("80000"), close=True, market=True))])
            for _ in range(50):
                await asyncio.sleep(0)
                if post.await_count:
                    break
            self.assertEqual(1, post.await_count)
            data = post.call_args.kwargs["data"]
            self.assertIs(True, data["reduce_only"])
            self.assertEqual("ioc", data["tif"])
            self.assertEqual("0", data["price"])
            self.assertEqual(-1, data["size"])
