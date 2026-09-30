"""Run Gate's actual order request method without its compiled connector base.

The full connector suite additionally tests this change in an installed runtime.
Only dependency-heavy class construction is replaced here; the request code is
read directly from the connector source, so this is not a duplicate builder.
"""

import ast
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
from unittest.mock import AsyncMock

from hummingbot.connector.derivative.gate_io_perpetual.gate_io_perpetual_request_guard import GateRequestDeferred
from hummingbot.core.data_type.common import OrderType, PositionAction, TradeType

ROOT = Path(__file__).resolve().parents[4]


def load_place_order():
    source = ROOT / "hummingbot/connector/derivative/gate_io_perpetual/gate_io_perpetual_derivative.py"
    parsed = ast.parse(source.read_text())
    cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef)
               and node.name == "GateIoPerpetualDerivative")
    method = next(node for node in cls.body if isinstance(node, ast.AsyncFunctionDef)
                  and node.name == "_place_order")
    namespace = {"Decimal": Decimal, "Tuple": Tuple, "OrderType": OrderType,
                 "TradeType": TradeType, "PositionAction": PositionAction, "GateRequestDeferred": GateRequestDeferred,
                 "CONSTANTS": SimpleNamespace(ORDER_CREATE_PATH_URL="futures/usdt/orders")}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_place_order"]


class GateRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_open_close_maker_market_buy_sell_requests(self):
        place_order = load_place_order()
        for action in (PositionAction.OPEN, PositionAction.CLOSE):
            for order_type in (OrderType.LIMIT_MAKER, OrderType.MARKET):
                for side in (TradeType.BUY, TradeType.SELL):
                    with self.subTest(action=action, order_type=order_type, side=side):
                        connector = SimpleNamespace(
                            exchange_symbol_associated_to_pair=AsyncMock(return_value="BTC_USDT"),
                            _format_amount_to_size=lambda pair, amount: Decimal("1"),
                            _api_post=AsyncMock(return_value={"id": "123", "finish_as": "filled"}),
                            current_timestamp=123.0)
                        result = await place_order(connector, "t-HBOT-test", "BTC-USDT", Decimal("0.0001"),
                                                   side, order_type, Decimal("10000"), position_action=action)
                        self.assertEqual(("123", 123.0), result)
                        data = connector._api_post.call_args.kwargs["data"]
                        self.assertEqual(action == PositionAction.CLOSE, data.get("reduce_only", False))
                        if action == PositionAction.OPEN:
                            self.assertNotIn("reduce_only", data)
                        self.assertEqual(1.0 if side == TradeType.BUY else -1.0, data["size"])
                        self.assertEqual("poc" if order_type == OrderType.LIMIT_MAKER else "ioc", data["tif"])

    async def test_rejected_order_still_raises(self):
        place_order = load_place_order()
        connector = SimpleNamespace(exchange_symbol_associated_to_pair=AsyncMock(return_value="BTC_USDT"),
                                    _format_amount_to_size=lambda pair, amount: Decimal("1"),
                                    _api_post=AsyncMock(return_value={"finish_as": "failed"}))
        with self.assertRaises(IOError):
            await place_order(connector, "test", "BTC-USDT", Decimal("0.0001"), TradeType.SELL,
                              OrderType.LIMIT_MAKER, Decimal("10000"), position_action=PositionAction.CLOSE)
