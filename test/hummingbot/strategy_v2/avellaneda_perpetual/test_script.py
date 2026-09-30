"""Execute the real script with connector/base-class boundary doubles.

These tests do not replace compiled Hummingbot or authenticated Gate tests.
"""

import asyncio
import importlib.util
import sys
import unittest
from decimal import Decimal as D
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

from hummingbot.strategy_v2.utils.avellaneda_perpetual import Quote

ROOT = Path(__file__).resolve().parents[4]


class OrderType(Enum):
    MARKET = 1
    LIMIT_MAKER = 3


class PositionAction(Enum):
    OPEN = "OPEN"
    CLOSE = "CLOSE"


class PositionMode(Enum):
    ONEWAY = "ONEWAY"


class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"


class ScriptBoundary:
    def __init__(self, connectors, config):
        self.connectors = connectors
        self.config = config
        self.current_timestamp = 100.0
        self.calls = []

    @classmethod
    def logger(cls):
        import logging
        return logging.getLogger("avellaneda-test")

    def buy(self, *args, **kwargs):
        self.calls.append(("buy", args, kwargs))
        return f"order-{len(self.calls)}"

    def sell(self, *args, **kwargs):
        self.calls.append(("sell", args, kwargs))
        return f"order-{len(self.calls)}"

    def cancel(self, *args):
        self.calls.append(("cancel", args, {}))


def module(name, **attrs):
    value = ModuleType(name)
    value.__dict__.update(attrs)
    return value


def load_script():
    # Use the repository's actual config model and serializer. Only its external
    # connector-name validator is replaced; this config has a Literal connector.
    config_spec = importlib.util.spec_from_file_location(
        "test_client_config", ROOT / "hummingbot/client/config/config_data_types.py")
    client_config = importlib.util.module_from_spec(config_spec)
    config_replacements = {
        "test_client_config": client_config,
        "hummingbot.client.config.config_validators": module("validators", validate_connector=lambda value: None),
    }
    with patch.dict(sys.modules, config_replacements):
        config_spec.loader.exec_module(client_config)
    replacements = {
        "hummingbot.client.config.config_data_types": client_config,
        "hummingbot.strategy.script_strategy_base": module("script", ScriptStrategyBase=ScriptBoundary),
        "hummingbot.core.data_type.common": module(
            "common", OrderType=OrderType, PositionAction=PositionAction,
            PositionMode=PositionMode, PositionSide=PositionSide),
        "hummingbot.connector.derivative.gate_io_perpetual.gate_io_perpetual_constants": module(
            "gate", POSITION_INFORMATION_URL="positions", USER_ORDERS_PATH_URL="orders",
            USER_BALANCES_PATH_URL="account"),
    }
    spec = importlib.util.spec_from_file_location("test_avellaneda_script", ROOT / "scripts/avellaneda_perpetual.py")
    loaded = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, replacements):
        spec.loader.exec_module(loaded)
    return loaded


SCRIPT = load_script()


def connector():
    async def api_get(path_url, **kwargs):
        return {"total": "100", "enable_credit": False} if path_url == "account" else []

    return SimpleNamespace(
        ready=True, account_positions={}, in_flight_orders={},
        _api_get=AsyncMock(side_effect=api_get),
        _trading_pair_position_mode_set=AsyncMock(return_value=(True, "")),
        _set_trading_pair_leverage=AsyncMock(return_value=(True, "")),
        _perpetual_trading=SimpleNamespace(set_position_mode=lambda x: None, set_leverage=lambda p, x: None),
        _update_positions=AsyncMock(), _update_balances=AsyncMock(),
        get_order_book=lambda p: SimpleNamespace(snapshot_uid=100, last_diff_uid=100),
        get_price=lambda p, ask: D("100.01") if ask else D("99.99"),
        get_available_balance=lambda a: D("100"), get_balance=lambda a: D("100"),
        trading_rules={"BTC-USDT": SimpleNamespace(min_price_increment=D("0.01"),
                                                   min_base_amount_increment=D("0.001"),
                                                   min_order_size=D("0.001"), min_notional_size=D("1"))},
    )


def strategy(**overrides):
    c = SCRIPT.AvellanedaPerpetualConfig(warmup_samples=3, **overrides)
    return SCRIPT.AvellanedaPerpetual({"gate_io_perpetual": connector()}, c)


class ScriptTests(unittest.IsolatedAsyncioTestCase):
    async def test_preview_has_no_order_or_account_mutation(self):
        s = strategy()
        for t in range(100, 106):
            s.current_timestamp = t
            s.on_tick()
        self.assertEqual([], s.calls)
        self.assertEqual(2, len(s.engine.preview))
        s.exchange._trading_pair_position_mode_set.assert_not_awaited()
        s.exchange._set_trading_pair_leverage.assert_not_awaited()
        await s.on_stop()
        self.assertEqual([], s.calls)

    async def test_live_waits_for_acknowledgement(self):
        s = strategy(dry_run=False)
        gate = asyncio.Event()

        async def wait_for_mode(*args):
            await gate.wait()
            return True, ""

        s.exchange._trading_pair_position_mode_set.side_effect = wait_for_mode
        s.on_tick()
        await asyncio.sleep(0)
        self.assertFalse(s._configured)
        self.assertEqual([], s.calls)
        gate.set()
        await s._setup_task
        self.assertTrue(s._configured)
        s.exchange._set_trading_pair_leverage.assert_awaited_once_with("BTC-USDT", 1)

    async def test_startup_position_is_rejected_before_mode_change(self):
        s = strategy(dry_run=False)
        s.exchange._api_get.side_effect = [{}, [{"size": "1"}]]
        with self.assertLogs("avellaneda-test", level="ERROR"):
            await s._configure_live()
        self.assertFalse(s._configured)
        self.assertFalse(s.engine.flatten)
        s.exchange._trading_pair_position_mode_set.assert_not_awaited()

    async def test_startup_existing_order_is_rejected(self):
        s = strategy(dry_run=False)
        s.exchange._api_get.side_effect = [{}, [], [{"id": "foreign-order"}]]
        with self.assertLogs("avellaneda-test", level="ERROR"):
            await s._configure_live()
        self.assertTrue(s._setup_failed)
        self.assertEqual([], s.calls)

    async def test_mode_and_leverage_failures_never_enable_trading(self):
        for method in ("_trading_pair_position_mode_set", "_set_trading_pair_leverage"):
            s = strategy(dry_run=False)
            getattr(s.exchange, method).return_value = (False, "rejected")
            with self.assertLogs("avellaneda-test", level="ERROR"):
                await s._configure_live()
            self.assertFalse(s._configured)
            self.assertEqual([], s.calls)

    async def test_order_route_post_only_and_close(self):
        s = strategy(dry_run=False)
        await s._configure_live()
        for t in range(100, 104):
            s.current_timestamp = t
            s.on_tick()
        self.assertEqual(2, len(s.calls))
        for _, _, kwargs in s.calls:
            self.assertEqual(PositionAction.OPEN, kwargs["position_action"])
        self.assertEqual(OrderType.LIMIT_MAKER, s.calls[0][1][3])
        from hummingbot.strategy_v2.utils.avellaneda_perpetual import Command
        s._execute([Command("create", quote=Quote("sell", D("0.03"), D("100"), close=True))])
        self.assertEqual(PositionAction.CLOSE, s.calls[-1][2]["position_action"])

    async def test_signed_position_and_unrealized_equity(self):
        s = strategy()
        s.exchange.account_positions = {"p": SimpleNamespace(
            trading_pair="BTC-USDT", position_side=PositionSide.SHORT, amount=D("0.1"), entry_price=D("101"))}
        snapshot = s._snapshot()
        self.assertEqual(D("-0.1"), snapshot.position)
        self.assertEqual(D("100.1"), snapshot.equity)

    async def test_stale_connector_cancels_registered_quotes(self):
        s = strategy(dry_run=False)
        await s._configure_live()
        s.engine.register("our-order", Quote("buy", D("0.1"), D("99")), 100)
        s.exchange.ready = False
        s.on_tick()
        self.assertEqual("cancel", s.calls[0][0])

    async def test_halt_rest_ack_does_not_starve_market_close(self):
        s = strategy(dry_run=False)
        await s._configure_live()
        s.exchange.account_positions = {"p": SimpleNamespace(
            trading_pair="BTC-USDT", position_side=PositionSide.LONG, amount=D("0.1"), entry_price=D("100"))}
        s.engine.halt("test")
        s.on_tick()
        self.assertEqual([], s.calls)
        await s._rest_ack_task
        with self.assertLogs("avellaneda-test", level="ERROR"):
            s.on_tick()
        self.assertEqual("sell", s.calls[0][0])
        self.assertEqual(OrderType.MARKET, s.calls[0][1][3])
        self.assertEqual(PositionAction.CLOSE, s.calls[0][2]["position_action"])
        s.on_tick()
        self.assertEqual(1, len(s.calls))

    async def test_failed_account_refresh_cannot_acknowledge_close(self):
        s = strategy(dry_run=False)
        s.exchange._update_positions.side_effect = RuntimeError("offline")
        self.assertFalse(await s._refresh_account(acknowledge=True))
        self.assertFalse(s._halt_position_acknowledged)
        self.assertIn("account refresh failed", s.engine.halt_reason)

    async def test_flat_stop_no_order(self):
        s = strategy(dry_run=False)
        await s._configure_live()
        await s.on_stop()
        self.assertEqual([], s.calls)

    async def test_synchronous_close_dispatch_failure_counts_attempt(self):
        s = strategy(dry_run=False)
        await s._configure_live()
        s.engine.halt("test")
        s.engine.accept_rest_position(D("0.1"))
        from hummingbot.strategy_v2.utils.avellaneda_perpetual import Command
        with patch.object(s, "sell", side_effect=RuntimeError("dispatch failure")):
            with self.assertLogs("avellaneda-test", level="ERROR"):
                s._execute([Command("create", quote=Quote("sell", D("0.1"), D("100"),
                                                          close=True, market=True))])
        self.assertEqual(1, s.engine.close_attempts)
        self.assertGreater(s.engine.next_close_at, s.current_timestamp)

    async def test_stop_flattens_and_waits_for_terminal_event(self):
        s = strategy(dry_run=False, stop_timeout=1)
        await s._configure_live()
        s.exchange.account_positions = {"p": SimpleNamespace(
            trading_pair="BTC-USDT", position_side=PositionSide.LONG, amount=D("0.1"), entry_price=D("100"))}

        async def fill_close():
            for _ in range(50):
                await asyncio.sleep(0.01)
                if s.engine.orders:
                    oid = next(iter(s.engine.orders))
                    s.did_fill_order(SimpleNamespace(
                        order_id=oid, amount=D("0.1"), timestamp=100, exchange_trade_id="closing-fill"))
                    s.exchange.account_positions = {}
                    s.did_complete_sell_order(SimpleNamespace(order_id=oid))
                    return
            self.fail("No close was submitted")

        fill_task = asyncio.create_task(fill_close())
        await s.on_stop()
        await fill_task
        self.assertEqual(1, len(s.calls))
        self.assertEqual(PositionAction.CLOSE, s.calls[0][2]["position_action"])
        self.assertEqual({}, s.engine.orders)
        self.assertEqual(D("0"), s._position())

    async def test_sample_config_validates(self):
        import yaml
        with (ROOT / "examples/avellaneda_perpetual_gate.yml").open() as f:
            config = SCRIPT.AvellanedaPerpetualConfig(**yaml.safe_load(f))
        self.assertTrue(config.dry_run)
        SCRIPT.AvellanedaPerpetual.init_markets(config)
        self.assertEqual({"gate_io_perpetual": {"BTC-USDT"}}, SCRIPT.AvellanedaPerpetual.markets)
        self.assertIn("dry_run", SCRIPT.AvellanedaPerpetualConfig.model_json_schema()["properties"])

    async def test_unified_credit_wallet_is_rejected(self):
        for account in ({"enable_credit": True}, {"margin_mode": 3}, {"enable_dual_plus": True}):
            s = strategy(dry_run=False)
            s.exchange._api_get.side_effect = [account]
            with self.assertLogs("avellaneda-test", level="ERROR"):
                await s._configure_live()
            self.assertTrue(s._setup_failed)
            self.assertEqual([], s.calls)
