"""Quote mathematics and order lifecycle for the Gate perpetual Avellaneda script.

Prices and quantities use Decimal. The AS model uses dimensionless log returns,
normalized signed inventory and a fixed arrival-decay parameter (kappa).
This module deliberately has no connector or compiled Hummingbot dependencies.
"""

import math
from collections import deque
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

D = Decimal
ZERO = D("0")


def setting(default, **constraints):
    return Field(default=default, json_schema_extra={"prompt": None}, **constraints)


class AvellanedaSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    script_file_name: str = setting("avellaneda_perpetual.py")
    connector: Literal["gate_io_perpetual"] = setting("gate_io_perpetual")
    trading_pair: str = setting("BTC-USDT", pattern=r"^[A-Z0-9]+-USDT$")
    dry_run: bool = setting(True)
    leverage: int = setting(1, ge=1, le=3)
    capital_budget_quote: Decimal = setting(D("30"), gt=0)
    order_amount_quote: Decimal = setting(D("10"), gt=0)
    max_position_quote: Decimal = setting(D("20"), gt=0)
    available_balance_reserve: Decimal = setting(D("5"), ge=0)
    risk_factor: float = setting(1.0, gt=0, le=10000)
    kappa: float = setting(10000.0, gt=0, le=10000000)
    horizon_seconds: float = setting(30.0, gt=0, le=3600)
    min_spread: Decimal = setting(D("0.0004"), gt=0, lt=D("0.1"))
    adverse_selection_buffer: Decimal = setting(D("0.0001"), ge=0, lt=D("0.1"))
    max_spread: Decimal = setting(D("0.02"), gt=0, lt=D("0.5"))
    maker_fee: Decimal = setting(D("0.0002"), ge=0, lt=D("0.1"))
    taker_fee: Decimal = setting(D("0.0005"), ge=0, lt=D("0.1"))
    rebate_rate: Decimal = setting(D("0.6"), ge=0, le=1)
    volatility_samples: int = setting(200, ge=10, le=10000)
    warmup_samples: int = setting(20, ge=3)
    refresh_seconds: float = setting(5.0, gt=0)
    stale_seconds: float = setting(10.0, gt=0)
    reconciliation_timeout: float = setting(15.0, gt=0)
    cancel_timeout: float = setting(10.0, gt=0)
    max_session_loss_quote: Decimal = setting(D("1"), gt=0)
    max_order_failures: int = setting(3, ge=1, le=20)
    max_close_attempts: int = setting(3, ge=1, le=10)
    flatten_on_stop: bool = setting(True)
    stop_timeout: float = setting(30.0, gt=0, le=60)

    @model_validator(mode="after")
    def check_relationships(self):
        for name in self.__class__.model_fields:
            value = getattr(self, name)
            if isinstance(value, (float, Decimal)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.warmup_samples > self.volatility_samples:
            raise ValueError("warmup_samples must not exceed volatility_samples")
        if self.order_amount_quote > self.max_position_quote:
            raise ValueError("order_amount_quote must not exceed max_position_quote")
        if self.max_position_quote / self.leverage > self.capital_budget_quote:
            raise ValueError("capital budget must cover maximum position margin")
        if self.max_spread < max(self.min_spread, self.fee_floor):
            raise ValueError("max_spread must cover min_spread and net maker costs")
        return self

    @property
    def fee_floor(self) -> Decimal:
        return 2 * self.maker_fee * (1 - self.rebate_rate) + self.adverse_selection_buffer


@dataclass(frozen=True)
class Snapshot:
    timestamp: float
    bid: Decimal
    ask: Decimal
    position: Decimal  # signed base quantity, never collateral balance
    available: Decimal
    equity: Decimal  # wallet balance + unrealized PnL; before uncredited rebates
    price_tick: Decimal
    amount_step: Decimal
    min_amount: Decimal
    min_notional: Decimal
    ready: bool = True
    book_age: float = 0.0

    @property
    def mid(self):
        return (self.bid + self.ask) / 2

    @property
    def valid(self):
        values = (self.bid, self.ask, self.position, self.available, self.equity,
                  self.price_tick, self.amount_step, self.min_amount, self.min_notional)
        return (all(v.is_finite() for v in values) and 0 < self.bid < self.ask
                and math.isfinite(self.timestamp) and math.isfinite(self.book_age)
                and self.price_tick > 0 and self.amount_step > 0
                and self.min_amount >= 0 and self.min_notional >= 0)


@dataclass(frozen=True)
class Quote:
    side: Literal["buy", "sell"]
    amount: Decimal
    price: Decimal
    close: bool = False
    market: bool = False


@dataclass(frozen=True)
class Command:
    kind: Literal["cancel", "create"]
    order_id: str = ""
    quote: Optional[Quote] = None


@dataclass
class WorkingOrder:
    quote: Quote
    remaining: Decimal
    created_at: float
    cancel_at: Optional[float] = None


def round_step(value: Decimal, step: Decimal, up=False) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_CEILING if up else ROUND_FLOOR) * step


class AvellanedaEngine:
    def __init__(self, config: AvellanedaSettings):
        # Keep runtime risk limits fixed even if a client edits its config object.
        self.config = config.model_copy(deep=True)
        self.samples = deque(maxlen=config.volatility_samples)
        self.orders: dict[str, WorkingOrder] = {}
        self.known_orders: dict[str, Quote] = {}
        self.seen_fills = set()
        self.expected_position: Optional[Decimal] = None
        self.confirmed_position = ZERO
        self.reconcile_since: Optional[float] = None
        self.initial_equity: Optional[Decimal] = None
        self.halt_reason: Optional[str] = None
        self.status = "warming up"
        self.preview: list[Quote] = []
        self.variance = 0.0
        self.reservation_price = ZERO
        self.spread = ZERO
        self.failures = 0
        self.close_attempts = 0
        self.next_close_at = 0.0
        self.started = False
        self.flatten = True
        self.inventory_changed = False
        self.rest_ack_required = False

    def observe(self, s: Snapshot):
        if self.samples and s.timestamp - self.samples[-1][0] > self.config.stale_seconds:
            self.samples.clear()
            self.variance = 0.0
        if not self.samples or s.timestamp > self.samples[-1][0]:
            self.samples.append((s.timestamp, float(s.mid)))
        if len(self.samples) >= 2:
            elapsed = self.samples[-1][0] - self.samples[0][0]
            if elapsed > 0:
                self.variance = sum(math.log(b[1] / a[1]) ** 2
                                    for a, b in zip(self.samples, list(self.samples)[1:])) / elapsed

    def quotes(self, s: Snapshot) -> list[Quote]:
        c = self.config
        inventory = float(s.position * s.mid / c.max_position_quote)
        risk = c.risk_factor * self.variance * c.horizon_seconds
        self.spread = max(c.min_spread, c.fee_floor,
                          D(str(risk + 2 * math.log1p(c.risk_factor / c.kappa) / c.risk_factor)))
        if self.spread > c.max_spread:
            self.status = "model spread exceeds max_spread; quotes paused"
            return []
        self.reservation_price = s.mid * D(str(math.exp(max(-50.0, min(50.0, -inventory * risk)))))
        bid = round_step(min(self.reservation_price * (1 - self.spread / 2), s.bid), s.price_tick)
        ask = round_step(max(self.reservation_price * (1 + self.spread / 2), s.ask), s.price_tick, up=True)
        if bid <= 0:
            return []
        # Reserve gross fees and emergency taker fees, not anticipated rebates.
        spendable = min(s.available - c.available_balance_reserve - c.max_position_quote * c.taker_fee,
                        c.capital_budget_quote - abs(s.position) * s.mid / c.leverage)
        result = []
        for side, price, sign in (("buy", bid, 1), ("sell", ask, -1)):
            close = s.position * sign < 0
            capacity = abs(s.position) if close else max(ZERO, c.max_position_quote / s.mid - abs(s.position))
            amount = min(c.order_amount_quote / price, capacity)
            unit_cost = price * (c.maker_fee if close else D(1) / c.leverage + c.maker_fee)
            if unit_cost > 0:
                amount = min(amount, max(ZERO, spendable) / unit_cost)
            amount = round_step(amount, s.amount_step)
            if amount > 0 and amount >= s.min_amount and amount * price >= s.min_notional:
                result.append(Quote(side, amount, price, close=close))
                spendable -= amount * unit_cost
        return result

    def register(self, order_id: str, quote: Quote, timestamp: float):
        if not order_id or order_id in self.known_orders:
            raise ValueError("order ID must be unique and nonempty")
        self.orders[order_id] = WorkingOrder(quote, quote.amount, timestamp)
        self.known_orders[order_id] = quote
        if quote.market:
            self.close_attempts += 1

    def terminal(self, order_id: str, failed=False, timestamp=0.0):
        order = self.orders.pop(order_id, None)
        if order is not None and order.quote.market:
            # A market order's completion event is not a fresh position snapshot.
            self.rest_ack_required = True
        if order is not None and failed:
            self.failures += 1
            if order.quote.market:
                self.next_close_at = timestamp + self.config.refresh_seconds
            elif self.failures >= self.config.max_order_failures:
                self.halt("too many order failures")

    def filled(self, order_id: str, amount: Decimal, timestamp: float, trade_id: str):
        quote = self.known_orders.get(order_id)
        key = (order_id, trade_id)
        if quote is None or key in self.seen_fills:
            return
        if not trade_id or not amount.is_finite() or amount <= 0:
            self.halt("invalid or unidentifiable fill; position reconciliation required")
            return
        self.seen_fills.add(key)
        self.inventory_changed = True
        if self.halt_reason is not None:
            self.rest_ack_required = True
        base = self.confirmed_position if self.expected_position is None else self.expected_position
        self.expected_position = base + (amount if quote.side == "buy" else -amount)
        if self.reconcile_since is None:
            self.reconcile_since = timestamp
        order = self.orders.get(order_id)
        if order:
            order.remaining = max(ZERO, order.remaining - amount)
        self.failures = 0

    def halt(self, reason: str, flatten=True):
        if self.halt_reason is None:
            self.halt_reason = reason
            self.rest_ack_required = True
        self.flatten = self.flatten and flatten
        self.preview = []

    def _cancel_all(self, now: float) -> list[Command]:
        commands = []
        for order_id, order in self.orders.items():
            if order.quote.market:
                continue  # Never stack or cancel an unresolved emergency market close.
            if order.cancel_at is None or now - order.cancel_at >= self.config.cancel_timeout:
                if order.cancel_at is not None:
                    self.halt("cancel acknowledgement timeout")
                order.cancel_at = now
                commands.append(Command("cancel", order_id=order_id))
        return commands

    def step(self, s: Snapshot) -> list[Command]:
        c = self.config
        self.preview = []
        if not s.valid:
            self.status = "invalid market/account data; quotes paused"
            return [] if c.dry_run else self._cancel_all(s.timestamp)
        fresh = s.ready and 0 <= s.book_age <= c.stale_seconds
        if not self.started:
            self.started = True
            self.confirmed_position = s.position
            self.initial_equity = s.equity
        if self.expected_position is not None:
            if s.position == self.expected_position:
                self.confirmed_position = s.position
                self.expected_position = None
                self.reconcile_since = None
            elif s.timestamp - self.reconcile_since >= c.reconciliation_timeout:
                self.halt("fill/position reconciliation timeout")
        elif s.position != self.confirmed_position:
            self.halt("position changed without a recorded fill")
        if self.initial_equity - s.equity >= c.max_session_loss_quote:
            self.halt("session equity loss limit")
        if abs(s.position) * s.mid > c.max_position_quote:
            self.halt("position notional limit")
        if self.halt_reason is not None:
            self.status = f"HALTED: {self.halt_reason}"
            if c.dry_run:
                return []
            commands = self._cancel_all(s.timestamp)
            if any(o.quote.market and s.timestamp - o.created_at >= c.reconciliation_timeout
                   for o in self.orders.values()):
                self.status += "; unresolved market close: inspect exchange"
            if self.orders or not fresh or not self.flatten or s.timestamp < self.next_close_at:
                return commands
            # A timed-out position reconciliation requires a fresh REST refresh in
            # the adapter before reducing. Otherwise wait for exact fill acknowledgement.
            if self.expected_position is not None or self.rest_ack_required:
                return commands
            amount = round_step(abs(s.position), s.amount_step)
            if amount <= 0:
                self.status += "; flat" if s.position == 0 else "; residual dust needs manual review"
            elif self.close_attempts < c.max_close_attempts:
                commands.append(Command("create", quote=Quote(
                    "sell" if s.position > 0 else "buy", amount, s.mid, close=True, market=True)))
            else:
                self.status += "; close attempts exhausted: inspect exchange position"
            return commands
        if not fresh:
            self.status = "connector/book stale; quotes paused"
            return [] if c.dry_run else self._cancel_all(s.timestamp)
        self.observe(s)
        if self.expected_position is not None:
            self.status = "waiting for fill/position reconciliation"
            return [] if c.dry_run else self._cancel_all(s.timestamp)
        if len(self.samples) < c.warmup_samples:
            self.status = f"warming up ({len(self.samples)}/{c.warmup_samples})"
            return []
        self.status = "dry-run preview" if c.dry_run else "quoting"
        self.preview = self.quotes(s)
        if c.dry_run:
            return []
        if self.orders:
            due = any(o.cancel_at is not None or s.timestamp - o.created_at >= c.refresh_seconds
                      for o in self.orders.values())
            if self.inventory_changed or not self.preview or due:
                return self._cancel_all(s.timestamp)
            return []
        self.inventory_changed = False
        return [Command("create", quote=q) for q in self.preview]

    def accept_rest_position(self, position: Decimal):
        """Use only after all own orders are terminal and an authenticated REST refresh.

        The strategy remains halted; this acknowledgement only permits reduce-only
        liquidation after a missed/delayed fill or position event.
        """
        if self.halt_reason is not None and not self.orders:
            self.confirmed_position = position
            self.expected_position = None
            self.reconcile_since = None
            self.rest_ack_required = False
