"""Configurable Gate USDT perpetual Avellaneda market maker (preview by default)."""

import asyncio
import time
from decimal import Decimal

from hummingbot.client.config.config_data_types import BaseClientModel
from hummingbot.connector.derivative.gate_io_perpetual import gate_io_perpetual_constants as GATE
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, PositionSide
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.strategy_v2.utils.avellaneda_perpetual import AvellanedaEngine, AvellanedaSettings, Snapshot


class AvellanedaPerpetualConfig(AvellanedaSettings, BaseClientModel):
    pass


class AvellanedaPerpetual(ScriptStrategyBase):
    @classmethod
    def init_markets(cls, config: AvellanedaPerpetualConfig):
        cls.markets = {config.connector: {config.trading_pair}}

    def __init__(self, connectors, config: AvellanedaPerpetualConfig):
        if config is None:
            raise ValueError("Start avellaneda_perpetual.py with --conf <config.yml>")
        super().__init__(connectors, config)
        self.engine = AvellanedaEngine(config)
        self.settings = self.engine.config
        self._setup_task = None
        self._account_task = None
        self._rest_ack_task = None
        self._configured = config.dry_run
        self._setup_failed = False
        self._last_account_at = 0.0
        self._book_uid = None
        self._book_updated_at = 0.0
        self._stopping = False
        self._last_logged_halt = None
        self._halt_position_acknowledged = False

    @property
    def exchange(self):
        return self.connectors[self.settings.connector]

    def _position(self):
        total = Decimal("0")
        for position in self.exchange.account_positions.values():
            if position.trading_pair == self.settings.trading_pair:
                if position.position_side == PositionSide.LONG:
                    total += abs(position.amount)
                elif position.position_side == PositionSide.SHORT:
                    total -= abs(position.amount)
                else:
                    total += position.amount
        return total

    async def _configure_live(self):
        """Wait for exchange acknowledgements, not the connector's default mode.

        Gate's public set_position_mode is fire-and-forget and may skip an API call
        when its local default is ONEWAY. This Gate-specific adapter explicitly
        awaits its existing REST methods before enabling order creation.
        """
        try:
            await asyncio.wait_for(self._configure_live_inner(), timeout=self.settings.stop_timeout)
            if not self._stopping:
                self._configured = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._setup_failed = True
            self.engine.halt(f"live setup failed: {exc}", flatten=False)
            self.logger().error(str(self.engine.halt_reason))

    async def _configure_live_inner(self):
        ex, c = self.exchange, self.settings
        account = await ex._api_get(
            path_url=GATE.USER_BALANCES_PATH_URL, is_auth_required=True, limit_id=GATE.USER_BALANCES_PATH_URL)
        if (not isinstance(account, dict) or str(account.get("enable_credit", False)).lower() == "true"
                or int(account.get("margin_mode", 0)) != 0
                or str(account.get("enable_dual_plus", False)).lower() == "true"):
            raise ValueError("Use a standard dedicated USDT futures wallet; unified credit accounts are unsupported")
        positions = await ex._api_get(path_url=GATE.POSITION_INFORMATION_URL, is_auth_required=True,
                                      limit_id=GATE.POSITION_INFORMATION_URL)
        if any(Decimal(str(p["size"])) != 0 for p in positions):
            raise ValueError("Live start requires a dedicated, flat USDT futures account")
        orders = await ex._api_get(path_url=GATE.USER_ORDERS_PATH_URL, params={"status": "open"},
                                   is_auth_required=True, limit_id=GATE.USER_ORDERS_PATH_URL)
        if orders or ex.in_flight_orders:
            raise ValueError("Live start requires no existing futures orders")
        success, message = await ex._trading_pair_position_mode_set(PositionMode.ONEWAY, c.trading_pair)
        if not success:
            raise ValueError(f"ONEWAY mode rejected: {message}")
        ex._perpetual_trading.set_position_mode(PositionMode.ONEWAY)
        success, message = await ex._set_trading_pair_leverage(c.trading_pair, c.leverage)
        if not success:
            raise ValueError(f"Leverage rejected: {message}")
        ex._perpetual_trading.set_leverage(c.trading_pair, c.leverage)
        await ex._update_positions()
        await ex._update_balances()
        if self._position() != 0:
            raise ValueError("Account acquired a position during setup")
        self._last_account_at = self.current_timestamp

    async def _refresh_account(self, acknowledge=False):
        try:
            await asyncio.wait_for(self.exchange._update_positions(), self.settings.stale_seconds)
            await asyncio.wait_for(self.exchange._update_balances(), self.settings.stale_seconds)
            self._last_account_at = self.current_timestamp
            if acknowledge:
                self.engine.accept_rest_position(self._position())
                self._halt_position_acknowledged = True
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.engine.halt(f"authenticated account refresh failed: {exc}")
            return False

    def _snapshot(self):
        ex, c, now = self.exchange, self.settings, self.current_timestamp
        book = ex.get_order_book(c.trading_pair)
        uid = (book.snapshot_uid, book.last_diff_uid)
        if uid != self._book_uid:
            self._book_uid = uid
            self._book_updated_at = now
        bid, ask = ex.get_price(c.trading_pair, False), ex.get_price(c.trading_pair, True)
        mid = (bid + ask) / 2
        # Mid-valued linear USDT equity includes unrealized inventory PnL.
        # Rebates enter only when actually credited to the futures wallet.
        pnl = Decimal("0")
        for p in ex.account_positions.values():
            if p.trading_pair == c.trading_pair:
                amount = abs(p.amount) if p.position_side == PositionSide.LONG else -abs(p.amount)
                if p.position_side == PositionSide.BOTH:
                    amount = p.amount
                pnl += (mid - p.entry_price) * amount
        rule = ex.trading_rules[c.trading_pair]
        account_fresh = c.dry_run or now - self._last_account_at <= c.stale_seconds
        return Snapshot(now, bid, ask, self._position(), ex.get_available_balance("USDT"),
                        ex.get_balance("USDT") + pnl, rule.min_price_increment,
                        rule.min_base_amount_increment, rule.min_order_size, rule.min_notional_size,
                        ready=ex.ready and account_fresh, book_age=now - self._book_updated_at)

    def _execute(self, commands):
        if self.settings.dry_run:
            return
        for command in commands:
            try:
                if command.kind == "cancel":
                    self.cancel(self.settings.connector, self.settings.trading_pair, command.order_id)
                else:
                    q = command.quote
                    if self._stopping and not q.close:
                        continue
                    if self.engine.halt_reason and not q.market:
                        continue
                    place = self.buy if q.side == "buy" else self.sell
                    order_id = place(self.settings.connector, self.settings.trading_pair, q.amount,
                                     OrderType.MARKET if q.market else OrderType.LIMIT_MAKER,
                                     price=q.price,
                                     position_action=PositionAction.CLOSE if q.close else PositionAction.OPEN)
                    self.engine.register(order_id, q, self.current_timestamp)
            except Exception as exc:
                if command.kind == "create" and command.quote.market:
                    self.engine.close_attempts += 1
                    self.engine.next_close_at = self.current_timestamp + self.settings.refresh_seconds
                self.engine.halt(f"order dispatch failed: {exc}")
                self.logger().error(str(self.engine.halt_reason))

    def on_tick(self):
        if self._stopping or self._setup_failed:
            return
        if not self.exchange.ready:
            self._execute(self.engine._cancel_all(self.current_timestamp))
            self.engine.status = "connector not ready; quotes paused"
            return
        if not self._configured:
            if self._setup_task is None:
                self._setup_task = asyncio.create_task(self._configure_live())
            self.engine.status = "waiting for live account/mode/leverage acknowledgements"
            return
        if not self.settings.dry_run:
            if (self.engine.halt_reason and not self.engine.orders
                    and (self.engine.rest_ack_required or not self._halt_position_acknowledged or
                         self.current_timestamp - self._last_account_at >= self.settings.stale_seconds / 2)):
                if self._rest_ack_task is None or self._rest_ack_task.done():
                    self._rest_ack_task = asyncio.create_task(self._refresh_account(acknowledge=True))
                # Do not close from a cached position while a REST acknowledgement is pending.
                if not self._rest_ack_task.done():
                    return
            elif (self.current_timestamp - self._last_account_at >= min(5.0, self.settings.stale_seconds / 2)
                  and (self._account_task is None or self._account_task.done())):
                self._account_task = asyncio.create_task(self._refresh_account())
        try:
            self._execute(self.engine.step(self._snapshot()))
        except Exception as exc:
            self.engine.halt(f"snapshot/quote error: {exc}")
            self._execute(self.engine._cancel_all(self.current_timestamp))
        if self.engine.halt_reason != self._last_logged_halt:
            self._last_logged_halt = self.engine.halt_reason
            self.logger().error(f"Strategy halted: {self.engine.halt_reason}")

    def did_fill_order(self, event):
        self._halt_position_acknowledged = False
        self.engine.filled(event.order_id, event.amount, event.timestamp, event.exchange_trade_id)

    def did_cancel_order(self, event):
        self.engine.terminal(event.order_id)

    def did_expire_order(self, event):
        self.engine.terminal(event.order_id)

    def did_fail_order(self, event):
        self.engine.terminal(event.order_id, failed=True, timestamp=event.timestamp)

    def did_complete_buy_order(self, event):
        self.engine.terminal(event.order_id)

    def did_complete_sell_order(self, event):
        self.engine.terminal(event.order_id)

    async def on_stop(self):
        self._stopping = True
        self.engine.halt("stop requested", flatten=self.settings.flatten_on_stop)
        for task in (self._setup_task, self._account_task, self._rest_ack_task):
            if task and not task.done():
                task.cancel()
        if self.settings.dry_run or not self._configured:
            return
        deadline = time.monotonic() + self.settings.stop_timeout
        try:
            while time.monotonic() < deadline:
                if not self.engine.orders:
                    refreshed = await asyncio.wait_for(self._refresh_account(acknowledge=True),
                                                       max(0.1, deadline - time.monotonic()))
                    if not refreshed:
                        raise RuntimeError("Could not verify exchange position during shutdown")
                self._execute(self.engine.step(self._snapshot()))
                if not self.engine.orders and (self._position() == 0 or not self.settings.flatten_on_stop):
                    return
                await asyncio.sleep(0.25)
        except Exception as exc:
            self.logger().error(f"Shutdown refresh/close failed: {exc}")
        self.logger().error("Shutdown could not confirm all orders terminal and position flat. Inspect Gate futures.")

    def format_status(self):
        e, c = self.engine, self.settings
        rows = [f"Avellaneda perpetual | {c.connector} {c.trading_pair}",
                f"Mode: {'PREVIEW (no orders)' if c.dry_run else 'LIVE'} | {e.status}",
                f"Position: {self._position()} | cap: {c.max_position_quote} USDT | leverage: {c.leverage}",
                f"Variance/s: {e.variance:.8g} | reservation: {e.reservation_price} | full spread: {e.spread}",
                f"Net maker: {c.maker_fee * (1 - c.rebate_rate):.6%} | cost floor: {c.fee_floor:.6%}",
                f"Own unresolved orders: {len(e.orders)} | emergency attempts: {e.close_attempts}"]
        if e.halt_reason:
            rows.append(f"HALTED: {e.halt_reason}")
        rows.extend(f"Preview {q.side}: {q.amount} @ {q.price} ({'reduce-only' if q.close else 'open'})"
                    for q in e.preview)
        return "\n".join(rows)
