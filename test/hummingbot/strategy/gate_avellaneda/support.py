"""Load the Python adapter with exchange doubles, without Cython or credentials.

Patches are scoped to module loading and restored immediately, so these tests do
not replace installed Hummingbot modules for the rest of the repository suite.
"""

import ast
import importlib.util
import logging
import sys
import types
from decimal import Decimal as D
from enum import Enum
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[4]


class OrderType(Enum):
    LIMIT = 1
    LIMIT_MAKER = 2
    MARKET = 3

    def is_limit_type(self):
        return self != OrderType.MARKET


class TradeType(Enum):
    BUY = 1
    SELL = 2


class PositionAction(Enum):
    OPEN = 1
    CLOSE = 2


class PositionMode(Enum):
    ONEWAY = 1
    HEDGE = 2


class NetworkStatus(Enum):
    CONNECTED = 1
    NOT_CONNECTED = 2


class Indicator:
    def __init__(self, *args, **kwargs):
        self.current_value = D("0.002")
        self.is_sampling_buffer_full = True

    def add_sample(self, price):
        pass


class Intensity(Indicator):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.current_value = (D("100"), D("200"))

    def calculate(self, timestamp):
        pass


class Candidate:
    def __init__(self, **kwargs):
        vars(self).update(kwargs)


class ScriptBase:
    def __init__(self, connectors, config):
        self.connectors, self.config = connectors, config
        self.current_timestamp = 100

    def logger(self):
        logger = logging.getLogger("gate_portfolio_tests")
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        return logger

    def buy(self, connector_name, pair, amount, order_type, **kwargs):
        return self.connectors[connector_name].send(True, pair, amount, order_type, kwargs)

    def sell(self, connector_name, pair, amount, order_type, **kwargs):
        return self.connectors[connector_name].send(False, pair, amount, order_type, kwargs)

    def cancel(self, connector_name, pair, order_id):
        self.connectors[connector_name].cancels.append(order_id)


def module(name, **attributes):
    result = types.ModuleType(name)
    vars(result).update(attributes)
    return result


constants = module("hummingbot.connector.derivative.gate_io_perpetual.gate_io_perpetual_constants",
                   POSITION_INFORMATION_URL="positions", USER_BALANCES_PATH_URL="account",
                   USER_ORDERS_PATH_URL="orders", NETWORK_CHECK_PATH_URL="network",
                   TICKER_PATH_URL="tickers", ORDER_BOOK_PATH_URL="book", ORDER_CREATE_PATH_URL="create")


def load_adapter():
    common = dict(OrderType=OrderType, TradeType=TradeType, PositionAction=PositionAction, PositionMode=PositionMode)
    modules = {
        "hummingbot.client.config.config_data_types": module("config", BaseClientModel=BaseModel),
        constants.__name__: constants,
        "hummingbot.core.data_type.common": module("common", **common),
        "hummingbot.core.data_type.order_candidate": module("candidate", PerpetualOrderCandidate=Candidate),
        "hummingbot.core.network_iterator": module("network", NetworkStatus=NetworkStatus),
        "hummingbot.strategy.__utils__.trailing_indicators.instant_volatility": module("vol", InstantVolatilityIndicator=Indicator),
        "hummingbot.strategy.__utils__.trailing_indicators.trading_intensity": module("intensity", TradingIntensityIndicator=Intensity),
        "hummingbot.strategy.order_book_asset_price_delegate": module("delegate", OrderBookAssetPriceDelegate=lambda info: info),
        "hummingbot.strategy.market_trading_pair_tuple": module("tuple", MarketTradingPairTuple=lambda *args: args),
        "hummingbot.strategy.script_strategy_base": module("script", ScriptStrategyBase=ScriptBase),
    }
    spec = importlib.util.spec_from_file_location("gate_portfolio_adapter_under_test", ROOT / "scripts/gate_avellaneda_portfolio.py")
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(adapter)
    return adapter


def contract(pair):
    return dict(name=pair.replace("-", "_"), quanto_multiplier="0.001", order_size_min="1",
                order_price_round="0.001", in_delisting=False, mark_price="10")


class FakeConnector:
    def __init__(self, pairs):
        self.network_status = NetworkStatus.CONNECTED
        self.contracts = [contract(p) for p in pairs]
        self.tickers = [dict(contract=p.replace("-", "_"), last="10", volume_24h_quote="10000000",
                             funding_rate_indicative="0.0001") for p in pairs]
        self.positions = []
        self.open_orders = []
        self.account = dict(available="100", total="100", unrealised_pnl="0")
        self.book = dict(bids=[dict(p="9.99", s="100000")], asks=[dict(p="10.01", s="100000")])
        self.sent, self.cancels, self.requests = [], [], []
        self.error = None
        self.setup_success = True
        self.budget_checker = types.SimpleNamespace(adjust_candidate=lambda c, **kwargs: c)
        self._perpetual_trading = types.SimpleNamespace(set_position_mode=lambda *args: None,
                                                        set_leverage=lambda *args: None)

    async def _api_get(self, path_url, **kwargs):
        self.requests.append((path_url, kwargs))
        if self.error:
            raise self.error
        if path_url == "positions":
            return self.positions
        if path_url == "orders":
            offset = kwargs.get("params", {}).get("offset", 0)
            return self.open_orders[offset:offset + 100]
        return {"account": self.account, "book": self.book, "tickers": self.tickers,
                "futures/usdt/contracts": self.contracts}[path_url]

    async def _trading_pair_position_mode_set(self, *args):
        return self.setup_success, "mode rejected"

    async def _set_trading_pair_leverage(self, *args):
        return self.setup_success, "leverage rejected"

    def get_order_book(self, pair):
        return object()

    def get_price(self, pair, is_buy):
        return D("10.01") if is_buy else D("9.99")

    def get_fee(self, *args):
        return types.SimpleNamespace(percent=D("0.0002"))

    def send(self, buy, pair, amount, order_type, kwargs):
        order_id = f"t-{len(self.sent) + 1}"
        self.sent.append((order_id, buy, pair, amount, order_type, kwargs))
        self.open_orders.append(dict(text=order_id, contract=pair.replace("-", "_")))
        return order_id


def method_from_source(path, class_name, method_name, namespace):
    """Execute the production method body against exchange doubles."""
    tree = ast.parse((ROOT / path).read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in klass.body if getattr(n, "name", "") == method_name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    extracted = ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[]))
    exec(compile(extracted, str(ROOT / path), "exec"), namespace)
    return namespace[method_name]
