import asyncio
import json
import types
import unittest
from decimal import Decimal as D
from unittest.mock import AsyncMock

from hummingbot.strategy.gate_avellaneda.microstructure import GateMarketSignalFeed, MicroSettings
from test.hummingbot.strategy.gate_avellaneda.support import TradeType, method_from_source


class MarketFeedHookTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.feed = GateMarketSignalFeed({"A-USDT": D("0.001")}, MicroSettings(warmup_seconds=1, min_samples=2))
        self.connector = types.SimpleNamespace(
            _gate_market_signal_feed=self.feed,
            trading_pair_associated_to_exchange_symbol=AsyncMock(return_value="A-USDT"),
            exchange_symbol_associated_to_pair=AsyncMock(return_value="A_USDT"),
            _format_size_to_amount=lambda pair, size: size * D("0.001"),
        )
        self.snapshot = dict(id=100, bids=[dict(p="9.99", s="100000")], asks=[dict(p="10.01", s="100000")])
        self.rest = types.SimpleNamespace(execute_request=AsyncMock(return_value=self.snapshot))
        self.ws = types.SimpleNamespace(connect=AsyncMock(), send=AsyncMock())
        self.source = types.SimpleNamespace(
            _connector=self.connector, _time=lambda: 100, _trading_pairs=["A-USDT"],
            _request_order_book_snapshot=AsyncMock(return_value=self.snapshot),
            _api_factory=types.SimpleNamespace(get_rest_assistant=AsyncMock(return_value=self.rest),
                                               get_ws_assistant=AsyncMock(return_value=self.ws)),
            logger=lambda: types.SimpleNamespace(info=lambda *args: None, error=lambda *args: None),
        )

    def method(self, name):
        def message(message_type, content, timestamp):
            return types.SimpleNamespace(message_type=message_type, content=content, timestamp=timestamp)
        namespace = dict(
            asyncio=asyncio, json=json, Decimal=D, TradeType=TradeType, OrderBookMessage=message,
            OrderBookMessageType=types.SimpleNamespace(TRADE="trade", SNAPSHOT="snapshot", DIFF="diff"),
            WSJSONRequest=lambda **kwargs: types.SimpleNamespace(**kwargs), RESTMethod=types.SimpleNamespace(GET="get"),
            CONSTANTS=types.SimpleNamespace(ORDER_BOOK_PATH_URL="book", TRADES_ENDPOINT_NAME="trades",
                                            ORDERS_UPDATE_ENDPOINT_NAME="depth", WS_URL="ws", PING_TIMEOUT=30),
            web_utils=types.SimpleNamespace(public_rest_url=lambda endpoint: endpoint),
        )
        return method_from_source(
            "hummingbot/connector/derivative/gate_io_perpetual/gate_io_perpetual_api_order_book_data_source.py",
            "GateIoPerpetualAPIOrderBookDataSource", name, namespace,
        )

    async def test_snapshot_bootstraps_feed_and_converts_contract_sizes(self):
        result = await self.method("_order_book_snapshot")(self.source, "A-USDT")
        self.assertEqual(result.message_type, "snapshot")
        self.assertEqual(result.content["bids"][0][1], D("100"))
        self.assertEqual(self.feed.states["A-USDT"].update_id, 100)
        self.assertFalse(self.feed.signal("A-USDT", 100).ready)

    async def test_rest_snapshot_limit_matches_ws_and_includes_sequence_id(self):
        await self.method("_request_order_book_snapshot")(self.source, "A-USDT")
        self.assertEqual(self.rest.execute_request.call_args.kwargs["params"],
                         dict(contract="A_USDT", with_id="true", limit=100))

    async def test_trade_hook_uses_aggressor_sign_base_amount_and_ignores_internal(self):
        queue = asyncio.Queue()
        rows = [dict(create_time_ms=100000, contract="A_USDT", size=size, id=i, price="10", is_internal=internal)
                for i, size, internal in [(1, "-100", False), (2, "200", False), (3, "900", True)]]
        await self.method("_parse_trade_message")(self.source, dict(result=rows), queue)
        trades = list(self.feed.states["A-USDT"].trades)
        self.assertEqual([(t[2], t[4]) for t in trades], [(False, D("0.1")), (True, D("0.2"))])
        self.assertEqual(queue.qsize(), 3)
        self.assertEqual(queue.get_nowait().content["trade_type"], float(TradeType.SELL.value))

    async def test_full_depth_is_snapshot_and_following_diff_bridges_feed(self):
        queue = asyncio.Queue()
        full = dict(U=100, u=100, t=100000, l="100", s="A_USDT", full=True,
                    b=self.snapshot["bids"], a=self.snapshot["asks"])
        method = self.method("_parse_order_book_diff_message")
        await method(self.source, dict(result=full), queue)
        self.assertEqual(queue.get_nowait().message_type, "snapshot")
        self.source._time = lambda: 101
        delta = dict(U=101, u=101, t=101000, l="100", s="A_USDT", b=[dict(p="9.99", s="200000")], a=[])
        await method(self.source, dict(result=delta), queue)
        result = queue.get_nowait()
        self.assertEqual(result.message_type, "diff")
        self.assertEqual(result.content["first_update_id"], 101)
        self.assertEqual(result.content["bids"][0][1], D("200"))
        self.assertTrue(self.feed.signal("A-USDT", 101).ready)

    async def test_subscription_requests_explicit_matching_depth_level(self):
        await self.method("_subscribe_channels")(self.source, self.ws)
        requests = [call.args[0].payload for call in self.ws.send.call_args_list]
        self.assertEqual(requests[0]["payload"], ["A_USDT"])
        self.assertEqual(requests[1]["payload"], ["A_USDT", "100ms", "100"])

    async def test_websocket_reconnect_discards_previous_book_and_flow(self):
        self.feed.seed_snapshot("A-USDT", self.snapshot, 100)
        self.feed.observe_trade("A-USDT", 1, 100, "10", "1", True, 100)
        result = await self.method("_connected_websocket_assistant")(self.source)
        self.assertIs(result, self.ws)
        self.ws.connect.assert_awaited_once_with(ws_url="ws", ping_timeout=30)
        self.assertTrue(self.feed.needs_snapshot("A-USDT"))
        self.assertEqual(len(self.feed.states["A-USDT"].trades), 0)
