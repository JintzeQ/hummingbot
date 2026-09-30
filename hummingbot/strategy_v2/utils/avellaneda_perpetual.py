"""Quote mathematics and order lifecycle for the Gate perpetual Avellaneda script.

Prices and quantities use Decimal. The AS model uses dimensionless log returns,
normalized signed inventory and a fixed arrival-decay parameter (kappa).
This module deliberately has no connector or compiled Hummingbot dependencies.
"""

import math
from collections import deque
from dataclasses import dataclass, replace
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hummingbot.strategy_v2.utils.avellaneda_perpetual_gamma import (
    CalibrationObservations,
    GammaCalibration,
    GammaController,
    quote_settings,
)

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
    capital_budget_quote: Decimal = setting(D("50"), gt=0)
    order_amount_quote: Decimal = setting(D("10"), gt=0)
    max_position_quote: Decimal = setting(D("40"), gt=0)
    available_balance_reserve: Decimal = setting(D("5"), ge=0)
    risk_factor: float = setting(1.0, gt=0, le=10000)
    gamma_mode: Literal["fixed", "adaptive"] = setting("fixed")
    gamma_calibration: Optional[GammaCalibration] = setting(None)
    calibration_seconds: float = setting(900.0, ge=900, le=10000)
    calibration_target_ticks: float = setting(2.0, ge=1, le=10)
    calibration_max_age_seconds: float = setting(86400.0, ge=900, le=604800)
    gamma_update_seconds: float = setting(5.0, ge=1, le=60)
    gamma_smoothing: float = setting(0.2, gt=0, le=1)
    gamma_max_step: float = setting(0.1, gt=0, le=0.1)
    gamma_min_ratio: float = setting(0.5, gt=0, le=1)
    gamma_max_ratio: float = setting(3.0, ge=1, le=3)
    gamma_inventory_weight: float = setting(1.0, ge=0, le=2)
    gamma_volatility_weight: float = setting(0.5, ge=0, le=1)
    gamma_max_volatility_multiplier: float = setting(2.0, ge=1, le=2)
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
    max_order_age_seconds: float = setting(30.0, gt=0)
    requests_per_second: int = setting(6, ge=2, le=20)
    safety_requests_per_second: int = setting(2, ge=1)
    requests_per_24h: int = setting(50000, ge=100, le=80000)
    safety_requests_per_24h: int = setting(1000, ge=10)
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
        if self.max_order_age_seconds < self.refresh_seconds:
            raise ValueError("max_order_age_seconds must cover refresh_seconds")
        if self.safety_requests_per_second >= self.requests_per_second:
            raise ValueError("Leave a positive per-second quote request budget")
        if self.safety_requests_per_24h >= self.requests_per_24h:
            raise ValueError("Leave a positive daily quote request budget")
        if self.order_amount_quote > self.max_position_quote:
            raise ValueError("order_amount_quote must not exceed max_position_quote")
        if self.max_position_quote / self.leverage > self.capital_budget_quote:
            raise ValueError("capital budget must cover maximum position margin")
        if self.max_spread < max(self.min_spread, self.fee_floor):
            raise ValueError("max_spread must cover min_spread and net maker costs")
        if self.gamma_mode == "adaptive" and not self.dry_run and self.gamma_calibration is None:
            raise ValueError("Live adaptive mode requires a preview calibration report")
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
                and math.isfinite(float(self.mid)) and float(self.mid) > 0
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


def quote_proposal(c: AvellanedaSettings, s: Snapshot, variance: float, gamma: float):
    """Pure proposal used by live quotes, calibration and sensitivity replay."""
    inventory = float(s.position * s.mid / c.max_position_quote)
    risk = gamma * variance * c.horizon_seconds
    model_spread = D(str(risk + 2 * math.log1p(gamma / c.kappa) / gamma))
    spread = max(c.min_spread, c.fee_floor, model_spread)
    reservation = s.mid * D(str(math.exp(max(-50.0, min(50.0, -inventory * risk)))))
    result = []
    if spread > c.max_spread:
        return result, reservation, spread, model_spread
    bid = round_step(min(reservation * (1 - spread / 2), s.bid), s.price_tick)
    ask = round_step(max(reservation * (1 + spread / 2), s.ask), s.price_tick, up=True)
    if bid <= 0:
        return result, reservation, spread, model_spread
    # Reserve gross fees and emergency taker fees, not anticipated rebates.
    spendable = min(s.available - c.available_balance_reserve - c.max_position_quote * c.taker_fee,
                    c.capital_budget_quote - abs(s.position) * s.mid / c.leverage)
    for side, price, sign in (("buy", bid, 1), ("sell", ask, -1)):
        close = s.position * sign < 0
        capacity = abs(s.position) if close else max(ZERO, c.max_position_quote / s.mid - abs(s.position))
        amount = min(c.order_amount_quote / price, capacity)
        unit_cost = price * (c.maker_fee if close else D(1) / c.leverage + c.maker_fee)
        amount = round_step(min(amount, max(ZERO, spendable) / unit_cost), s.amount_step)
        if amount > 0 and amount >= s.min_amount and amount * price >= s.min_notional:
            result.append(Quote(side, amount, price, close=close))
            spendable -= amount * unit_cost
    return result, reservation, spread, model_spread


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
        self.gamma = GammaController(self.config)
        self.calibration_observations = CalibrationObservations()
        self.calibration_reason = "loaded profile; checking current rules before quoting" if self.gamma.profile else "not requested"
        self.last_calibration_attempt = -math.inf
        self.metrics = {}
        self.markouts = []
        self.completed_markouts = deque(maxlen=1000)
        self.request_guard = None
        self.retained_quotes = 0
        self.last_quote_check = -math.inf

    def sync_deferred_cancels(self):
        if self.request_guard:
            for order_id in self.request_guard.deferred_cancels:
                if order_id in self.orders:
                    self.orders[order_id].cancel_at = None
            self.request_guard.deferred_cancels.clear()

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
        gamma = self.gamma.current
        result, self.reservation_price, self.spread, model_spread = quote_proposal(c, s, self.variance, gamma)
        flat, _, _, _ = quote_proposal(c, replace(s, position=ZERO), self.variance, gamma)
        flat_prices = {q.side: q.price for q in flat}
        self.metrics = dict(timestamp=s.timestamp, inventory_ratio=inventory,
                            remaining_capacity_lots=int(max(ZERO, c.max_position_quote / s.mid - abs(s.position))
                                                        / s.amount_step),
                            variance=self.variance,
                            reference_variance=self.gamma.profile.variance_per_second if self.gamma.profile else None,
                            gamma_base=self.gamma.base, gamma_current=gamma, gamma_target=self.gamma.target,
                            gamma_min=self.gamma.bounds[0], gamma_max=self.gamma.bounds[1],
                            gamma_bounds_active=self.gamma.bounds_active, gamma_reason=self.gamma.reason,
                            raw_skew_ticks=float((self.reservation_price - s.mid) / s.price_tick),
                            rounded_skew_ticks={q.side: float((q.price - flat_prices[q.side]) / s.price_tick)
                                                for q in result if q.side in flat_prices},
                            spread_floor_active=model_spread <= max(c.min_spread, c.fee_floor),
                            calibration_reason=self.calibration_reason)
        if self.spread > c.max_spread:
            self.status = "model spread exceeds max_spread; quotes paused"
        return result

    def calibrate(self, s: Snapshot) -> GammaCalibration:
        """Create a report only from a fresh preview observation interval."""
        if not self.config.dry_run or not s.valid or not s.ready or not 0 <= s.book_age <= self.config.stale_seconds:
            raise ValueError("calibration is allowed only with fresh preview data")
        c = self.config
        variance = self.calibration_observations.reference_variance(c)
        flat_s = replace(s, position=ZERO)
        flat, _, _, _ = quote_proposal(c, flat_s, variance, c.risk_factor)
        if len(flat) != 2:
            raise ValueError("both first orders must satisfy lot, minimum-notional and budget rules")
        amount = min(q.amount for q in flat)
        for _ in range(5):
            inventory = float(amount * s.mid / c.max_position_quote)
            d = c.calibration_target_ticks * float(s.price_tick / s.mid)
            if not 0 < d < 1:
                raise ValueError("invalid target tick displacement")
            gamma = -math.log1p(-d) / (inventory * variance * c.horizon_seconds)
            if not 0 < gamma <= 10000:
                raise ValueError("calibrated gamma outside (0, 10000]; remain in preview")
            flat, _, _, _ = quote_proposal(c, flat_s, variance, gamma)
            if len(flat) != 2:
                raise ValueError("calibrated spread or budget prevents two valid first orders")
            actual = min(q.amount for q in flat)
            if actual == amount:
                break
            amount = actual
        else:
            raise ValueError("first-lot calibration did not converge")
        baseline = {q.side: q.price for q in flat}
        changes = {}
        for direction, position in (("long", amount), ("short", -amount)):
            quotes, _, _, _ = quote_proposal(c, replace(s, position=position), variance, gamma)
            if len(quotes) != 2:
                raise ValueError("first inventory lot cannot produce both valid reference quotes")
            for q in quotes:
                changes[f"{direction}_{'bid' if q.side == 'buy' else 'ask'}_change_ticks"] = float(
                    (q.price - baseline[q.side]) / s.price_tick)
        return GammaCalibration(trading_pair=c.trading_pair, created_at=s.timestamp,
                                observation_seconds=self.calibration_observations.elapsed,
                                window_seconds=200, valid_samples=self.calibration_observations.count,
                                reference_mid=s.mid, price_tick=s.price_tick, amount_step=s.amount_step,
                                min_amount=s.min_amount, min_notional=s.min_notional,
                                horizon_seconds=c.horizon_seconds, max_position_quote=c.max_position_quote,
                                order_amount_quote=c.order_amount_quote, reference_amount=amount,
                                reference_inventory=inventory, variance_per_second=variance, gamma_base=gamma,
                                target_ticks=c.calibration_target_ticks, quote_settings=quote_settings(c), **changes)

    def record_fill_metrics(self, order_id, trade_id, timestamp, amount, price, fee_quote=None):
        quote = self.known_orders.get(order_id)
        if quote is None or any(r["order_id"] == order_id and r["trade_id"] == trade_id
                                for r in self.markouts):
            return
        if not all(v.is_finite() and v > 0 for v in (amount, price)):
            return
        fee = self.config.taker_fee if quote.market else self.config.maker_fee
        self.markouts.append(dict(order_id=order_id, trade_id=trade_id, timestamp=timestamp, side=quote.side,
                                  amount=str(amount), price=str(price), actual_fee_quote=fee_quote,
                                  estimated_uncredited_rebate=str(amount * price * fee * self.config.rebate_rate),
                                  markouts={}))

        self.markouts[-1].update(
            fee_source="fill_event" if fee_quote is not None else "unavailable",
            credited_rebate_quote=None, funding_attribution_complete=False)
        return self.markouts[-1]

    def observe_markouts(self, s):
        for row in self.markouts:
            age = s.timestamp - row["timestamp"]
            for horizon in (5, 30, 60):
                key = str(horizon)
                if age >= horizon and key not in row["markouts"]:
                    # Missing fresh data at the horizon is not an exact markout.
                    sign = 1 if row["side"] == "buy" else -1
                    row["markouts"][key] = dict(observed_delay_seconds=age,
                                                value_quote=str(sign * (s.mid - D(row["price"])) * D(row["amount"])),
                                                late=age - horizon > self.config.stale_seconds)
        done = [row for row in self.markouts if "60" in row["markouts"]]
        self.completed_markouts.extend(done)
        self.markouts = [row for row in self.markouts if "60" not in row["markouts"]]

    def register(self, order_id: str, quote: Quote, timestamp: float):
        if not order_id or order_id in self.known_orders:
            raise ValueError("order ID must be unique and nonempty")
        self.orders[order_id] = WorkingOrder(quote, quote.amount, timestamp)
        self.known_orders[order_id] = quote
        if self.request_guard and not quote.market:
            self.request_guard.pending_quotes[order_id] = (self.request_guard.generation, self.request_guard.clock())
        if quote.market:
            self.close_attempts += 1

    def terminal(self, order_id: str, failed=False, timestamp=0.0):
        order = self.orders.pop(order_id, None)
        if self.request_guard:
            self.request_guard.pending_quotes.pop(order_id, None)
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
        if not trade_id or not amount.is_finite() or amount <= 0 or not math.isfinite(timestamp):
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
        self.sync_deferred_cancels()
        if self.request_guard and not self.request_guard.allowed(
                "DELETE", "futures/usdt/orders", trading=True, safety=True):
            return []
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
        self.gamma.reason = "frozen: market/account data, reconciliation, warmup or halt"
        self.sync_deferred_cancels()
        if self.request_guard:
            reason = self.request_guard.quote_reason()
            if "24h" in reason or "journal" in reason:
                self.halt(reason)
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
                if self.request_guard and not self.request_guard.allowed(
                        "POST", "futures/usdt/orders", trading=True, safety=True):
                    self.status += "; emergency close waiting for request cooldown"
                    return commands
                commands.append(Command("create", quote=Quote(
                    "sell" if s.position > 0 else "buy", amount, s.mid, close=True, market=True)))
            else:
                self.status += "; close attempts exhausted: inspect exchange position"
            return commands
        if not fresh:
            self.status = "connector/book stale; quotes paused"
            return [] if c.dry_run else self._cancel_all(s.timestamp)
        if self.request_guard and self.request_guard.quote_reason():
            reason = self.request_guard.quote_reason()
            self.status = "quotes paused: " + reason
            if reason == "local quote requests/second limit":
                return []  # A short local quote pause does not require healthy quotes to be cancelled.
            return [] if c.dry_run else self._cancel_all(s.timestamp)
        self.observe(s)
        if self.expected_position is not None:
            self.status = "waiting for fill/position reconciliation"
            return [] if c.dry_run else self._cancel_all(s.timestamp)
        if len(self.samples) < c.warmup_samples:
            self.status = f"warming up ({len(self.samples)}/{c.warmup_samples})"
            return [] if c.dry_run else self._cancel_all(s.timestamp)
        self.observe_markouts(s)
        if c.gamma_mode == "adaptive":
            if self.gamma.profile is None and c.dry_run:
                self.calibration_observations.observe(s, c)
                if s.timestamp - self.last_calibration_attempt >= c.gamma_update_seconds:
                    self.last_calibration_attempt = s.timestamp
                    try:
                        self.gamma.install(self.calibrate(s))
                        self.calibration_reason = "verified; export this report before live use"
                    except ValueError as exc:
                        self.calibration_reason = str(exc)
                if self.gamma.profile is None:
                    self.status = f"preview calibration: {self.calibration_reason}"
                    self.preview = self.quotes(s)
                    return []
            error = self.gamma.compatibility_error(s)
            if error:
                self.gamma.reason = f"frozen: {error}"
                self.status = error
                if c.dry_run:
                    return []
                self.halt(error)
                return self._cancel_all(s.timestamp)
        self.gamma.update(s.timestamp, float(s.position * s.mid / c.max_position_quote), self.variance)
        self.status = "dry-run preview" if c.dry_run else "quoting"
        self.preview = self.quotes(s)
        if c.dry_run:
            return []
        if self.orders:
            pending = any(o.cancel_at is not None for o in self.orders.values())
            expired = any(s.timestamp - o.created_at >= c.max_order_age_seconds for o in self.orders.values())
            due = s.timestamp - self.last_quote_check >= c.refresh_seconds
            if self.inventory_changed or not self.preview or pending or expired:
                return self._cancel_all(s.timestamp)
            if due:
                self.last_quote_check = s.timestamp
                working = sorted((o.quote.side, o.remaining, o.quote.price, o.quote.close, o.quote.market)
                                 for o in self.orders.values())
                proposed = sorted((q.side, q.amount, q.price, q.close, q.market) for q in self.preview)
                if working != proposed:
                    return self._cancel_all(s.timestamp)
                self.retained_quotes += len(self.orders)
                self.status = "quoting; unchanged quotes retained"
            return []
        self.inventory_changed = False
        self.last_quote_check = s.timestamp
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
