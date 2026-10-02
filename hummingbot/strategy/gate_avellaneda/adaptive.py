"""Bounded quote and portfolio controls; no exchange access or order submission.

The arrival attenuation is a relative depth proxy, not a calibrated fill
probability: Hummingbot's alpha fits sampled volume, not trades per second.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, Decimal
from typing import Optional

from hummingbot.strategy.gate_avellaneda.core import Intent, Market, Settings, ZERO, floor_step
from hummingbot.strategy.gate_avellaneda.microstructure import MicroSignal
from hummingbot.strategy.gate_avellaneda.kline_volatility import calibrated_half_spread

D = Decimal
ONE = D("1")


@dataclass(frozen=True)
class AdaptiveSettings:
    enabled: bool = True
    inventory_strength: Decimal = ONE
    inventory_power: Decimal = D("2")
    soft_ratio: Decimal = D("0.5")
    reduce_only_ratio: Decimal = D("0.75")
    age_ramp_seconds: float = 120
    max_hold_seconds: float = 600
    same_side_fill_limit: int = 3
    same_side_cooldown_seconds: float = 15
    volatility_reference: Decimal = D("0.001")
    min_size_scale: Decimal = D("0.25")
    depth_target_multiple: Decimal = D("20")
    spread_multiplier: Decimal = ONE
    unconfirmed_confidence: Decimal = D("0.5")
    micro_risk_bps: Decimal = D("2")
    carry_horizon_seconds: float = 120
    min_arrival_attenuation: Decimal = D("0.01")
    max_gross_quote: Decimal = D("40")
    max_directional_quote: Decimal = D("25")
    correlation_sample_seconds: float = 5
    correlation_history_samples: int = 121
    correlation_min_samples: int = 30
    max_abs_correlation: Decimal = D("0.85")
    require_correlation: bool = True
    stop_window_seconds: float = 900
    stop_count: int = 2
    stop_pause_seconds: float = 600
    recovery_seconds: float = 30

    def __post_init__(self):
        if any(not v.is_finite() for v in vars(self).values() if isinstance(v, D)):
            raise ValueError("Adaptive settings must be finite")
        durations = (self.age_ramp_seconds, self.max_hold_seconds, self.same_side_cooldown_seconds,
                     self.carry_horizon_seconds, self.correlation_sample_seconds, self.stop_window_seconds,
                     self.stop_pause_seconds, self.recovery_seconds)
        if any(not math.isfinite(v) or v <= 0 for v in durations):
            raise ValueError("Adaptive durations must be positive and finite")
        if not ZERO < self.soft_ratio < self.reduce_only_ratio < ONE:
            raise ValueError("Inventory stages must satisfy 0 < soft < reduce-only < 1")
        if not ZERO < self.min_size_scale <= ONE or not ZERO <= self.unconfirmed_confidence <= ONE:
            raise ValueError("Invalid adaptive scale/confidence")
        if not ZERO < self.min_arrival_attenuation < ONE or not ZERO < self.max_abs_correlation < ONE:
            raise ValueError("Invalid arrival/correlation threshold")
        if min(self.inventory_strength, self.inventory_power, self.volatility_reference,
               self.depth_target_multiple, self.spread_multiplier, self.max_gross_quote,
               self.max_directional_quote) <= 0 or self.micro_risk_bps < 0:
            raise ValueError("Invalid adaptive risk coefficients")
        if (self.max_hold_seconds <= self.age_ramp_seconds or self.same_side_fill_limit < 1 or self.stop_count < 2
                or self.correlation_min_samples < 3
                or self.correlation_history_samples <= self.correlation_min_samples):
            raise ValueError("Invalid adaptive age, stop count or sampling bounds")
        if self.max_directional_quote > self.max_gross_quote:
            raise ValueError("Directional cap must fit gross cap")


@dataclass
class QuotePlan:
    gamma: Decimal = ONE
    reference: Decimal = ZERO
    confidence: Decimal = ZERO
    buy_premium: Decimal = ZERO
    sell_premium: Decimal = ZERO
    position_cap: Decimal = ZERO
    order_quote: Decimal = ZERO
    pressure: Decimal = ZERO
    stage: str = "paused"
    entry_reason: str = ""
    intents: list = field(default_factory=list)


def signal_adjustment(market: Market, signal: Optional[MicroSignal], settings: AdaptiveSettings):
    if signal is None or not signal.ready:
        return market.mid, ZERO, ZERO, ZERO
    enough = signal.flow_sufficient
    opposing = signal.book_flow_conflict
    confidence = (ZERO if signal.paused or opposing else
                  min(ONE, abs(signal.flow)) if enough and signal.imbalance * signal.flow > 0
                  else settings.unconfirmed_confidence)
    reference = market.mid + confidence * (signal.reference - market.mid)
    risk = settings.micro_risk_bps / D("10000")
    buy_risk = risk * max(signal.bid_loss, max(ZERO, -signal.flow) if enough else ZERO)
    sell_risk = risk * max(signal.ask_loss, max(ZERO, signal.flow) if enough else ZERO)
    return reference, confidence, buy_risk, sell_risk


def exposure_scale(market: Market, risk: Settings, settings: AdaptiveSettings) -> Decimal:
    volatility = max(ZERO, market.volatility / market.mid)
    vol_scale = min(ONE, settings.volatility_reference / volatility) if volatility else ONE
    depth_scale = min(ONE, max(ZERO, min(market.bid_depth, market.ask_depth))
                      / (settings.depth_target_multiple * risk.order_quote))
    return max(settings.min_size_scale, min(vol_scale, depth_scale))


def _carry(market: Market, buy: bool, now: float, settings: AdaptiveSettings) -> Decimal:
    fraction = D(str(settings.carry_horizon_seconds / market.funding_interval))
    if now < market.funding_next_at <= now + settings.carry_horizon_seconds:
        fraction = max(ONE, fraction)
    # Never assume receiving funding will cover otherwise unprofitable quotes.
    return max(ZERO, market.funding if buy else -market.funding) * fraction


def _close_quote(market, position, risk, price):
    if position == 0:
        return []
    buy = position < 0
    if buy:
        price = floor_step(min(price, market.ask - market.tick), market.tick)
    else:
        price = (max(price, market.bid + market.tick) / market.tick).to_integral_value(rounding=ROUND_CEILING) * market.tick
    if price <= 0:
        return []
    amount = floor_step(min(abs(position), risk.order_quote / price), market.step)
    return [Intent(market.pair, buy, amount, price, close=True)] if amount >= market.minimum else []


def quote_plan(market: Market, position: Decimal, age: float, risk: Settings, settings: AdaptiveSettings,
               now: float, signal: Optional[MicroSignal] = None, allow_open: bool = True,
               blocked_side: Optional[bool] = None, gamma: Optional[Decimal] = None,
               quality=None, exit_cost_bps: Decimal = ZERO, calibration=None) -> QuotePlan:
    gamma = risk.gamma if gamma is None else gamma
    plan = QuotePlan(gamma=gamma)
    unsafe = market.close_rejection(risk, now)
    if unsafe:
        plan.entry_reason = unsafe
        return plan
    reason = market.rejection(risk, now, quoted_economics=True)
    if not gamma.is_finite() or gamma <= 0:
        reason = "invalid effective Gamma"
    if (not math.isfinite(market.funding_interval) or market.funding_interval <= 0
            or not math.isfinite(market.funding_next_at)):
        reason = "invalid funding schedule"
    if reason:
        plan.entry_reason = reason
        plan.stage = "reduce_only"
        price = market.mid if age >= settings.age_ramp_seconds else (market.bid if position < 0 else market.ask)
        plan.intents = _close_quote(market, position, risk, price)
        return plan
    if (calibration and calibration.get("enabled") and calibration.get("mode") == "protect"
            and not calibration.get("ready")):
        allow_open = False
        plan.entry_reason = "K-line calibration unavailable"
    scale = exposure_scale(market, risk, settings)
    plan.position_cap = risk.max_position_quote * scale
    plan.order_quote = risk.order_quote * scale
    plan.reference, plan.confidence, plan.buy_premium, plan.sell_premium = signal_adjustment(market, signal, settings)
    q = position * market.mid / plan.position_cap
    ratio = abs(q)
    urgency = ONE + min(D("3"), D(str(max(0, age) / settings.age_ramp_seconds)))
    plan.pressure = (q * (ONE + min(D("4"), ratio ** settings.inventory_power)) * urgency
                     * settings.inventory_strength * gamma * market.volatility / market.mid)
    plan.stage = ("reduce_only" if ratio >= settings.reduce_only_ratio or age >= 2 * settings.age_ramp_seconds
                  or not allow_open or (signal is not None and signal.paused) else
                  "caution" if ratio >= settings.soft_ratio or age >= settings.age_ramp_seconds else "normal")
    # kappa is inverse price; kappa*mid and volatility/mid share return units.
    liquidity = (ONE + ONE / (market.kappa * market.mid)).ln()
    base_half = settings.spread_multiplier * (market.volatility / market.mid / 2 + liquidity)
    opening_half = settings.spread_multiplier * (gamma * market.volatility / market.mid / 2 + liquidity)
    opening_half = calibrated_half_spread(opening_half, calibration)
    fee_floor = max(ZERO, market.maker_fee) + risk.min_net_spread / 2
    center = plan.reference - plan.pressure * market.mid
    prices, close_prices = {}, {}
    for buy in (True, False):
        premium = plan.buy_premium if buy else plan.sell_premium
        floor = fee_floor + _carry(market, buy, now, settings) + premium
        evidence = (quality or {}).get(str(buy), {})
        learned = D(evidence.get("penalty_bps", "0")) / D(10000)
        extra = learned + exit_cost_bps / D(10000)
        half = max(opening_half + premium + extra, floor + extra)
        prices[buy] = center + (-ONE if buy else ONE) * half * market.mid
        # Greater risk aversion improves inventory reduction through the center;
        # it must not widen the reducing side's volatility component.
        close_prices[buy] = center + (-ONE if buy else ONE) * max(base_half + premium, floor) * market.mid
    close_price = close_prices[position < 0]
    plan.intents = _close_quote(market, position, risk, close_price)
    opens = []
    for buy in (True, False):
        evidence = (quality or {}).get(str(buy), {})
        if (buy and position < 0) or (not buy and position > 0):
            continue
        if plan.stage == "reduce_only" or blocked_side is buy or evidence.get("block", False):
            if evidence.get("block", False):
                plan.entry_reason = "observed maker execution quality is negative"
            continue
        price = (floor_step(min(prices[buy], market.bid), market.tick) if buy else
                 (max(prices[buy], market.ask) / market.tick).to_integral_value(rounding=ROUND_CEILING) * market.tick)
        if price <= 0:
            plan.entry_reason = "invalid proposed opening price"
            continue
        distance = (plan.reference - price if buy else price - plan.reference) / market.mid
        required = fee_floor + _carry(market, buy, now, settings) + (exit_cost_bps + D(evidence.get("penalty_bps", "0"))) / D(10000)
        attenuation = D(str(math.exp(-min(700, float(market.kappa * abs(price - market.mid))))))
        if distance < required:
            plan.entry_reason = "proposed opening edge below costs"
            continue
        if attenuation < settings.min_arrival_attenuation:
            plan.entry_reason = "proposed opening too far from sampled flow"
            continue
        if abs(price - market.mid) / market.mid > risk.max_quote_spread / 2:
            plan.entry_reason = "proposed opening spread too wide"
            continue
        increasing = max(ZERO, position * market.mid if buy else -position * market.mid)
        amount = min(plan.order_quote / price, max(ZERO, plan.position_cap - increasing) / max(price, market.mid))
        if increasing:
            amount *= max(ZERO, ONE - min(ONE, ratio)) ** settings.inventory_power
        if signal is not None:
            amount *= signal.buy_scale if buy else signal.sell_scale
        amount *= D(evidence.get("size_scale", "1"))
        amount = floor_step(amount, market.step)
        if amount >= market.minimum:
            opens.append(Intent(market.pair, buy, amount, price))
        else:
            plan.entry_reason = "adaptive opening below minimum"
    plan.intents.extend(opens)
    return plan


def entry_rejection(market, risk, settings, now, signal=None, gamma=None, quality=None, exit_cost_bps=ZERO, calibration=None):
    plan = quote_plan(market, ZERO, 0, risk, settings, now, signal, gamma=gamma, quality=quality, exit_cost_bps=exit_cost_bps, calibration=calibration)
    return None if len(plan.intents) == 2 else plan.entry_reason or "no executable adaptive opening"


class InventoryTracker:
    def __init__(self, settings):
        self.settings = settings
        self.opened_at, self.streaks, self.cooldowns, self.counted_orders = {}, {}, {}, {}

    def reconcile(self, positions, now):
        for pair in set(self.opened_at) | set(positions):
            if positions.get(pair, ZERO):
                self.opened_at.setdefault(pair, now)
            else:
                self.opened_at.pop(pair, None)
                self.streaks.pop(pair, None)
                self.cooldowns.pop(pair, None)
                self.counted_orders.pop(pair, None)

    def fill(self, pair, order_id, buy, before, after, close, now):
        if after == 0:
            self.opened_at.pop(pair, None)
            self.streaks.pop(pair, None)
            self.cooldowns.pop(pair, None)
            self.counted_orders.pop(pair, None)
            return
        if before == 0 or before * after < 0:
            self.opened_at[pair] = now
            self.counted_orders.pop(pair, None)
        if close:
            self.streaks.pop(pair, None)
            return
        seen = self.counted_orders.setdefault(pair, set())
        if order_id in seen:
            return  # Partial fills of one order do not count as multiple orders.
        seen.add(order_id)
        side, count, _ = self.streaks.get(pair, (buy, 0, None))
        count = count + 1 if side == buy else 1
        self.streaks[pair] = buy, count, order_id
        if count >= self.settings.same_side_fill_limit:
            self.cooldowns[pair] = buy, now + self.settings.same_side_cooldown_seconds

    def age(self, pair, position, now):
        return max(0, now - self.opened_at.get(pair, now)) if position else 0

    def blocked_side(self, pair, now):
        side, until = self.cooldowns.get(pair, (None, 0))
        return side if now < until else None


def exposure_bounds(signed_quote, pending):
    longs = {p: max(ZERO, v) for p, v in signed_quote.items()}
    shorts = {p: max(ZERO, -v) for p, v in signed_quote.items()}
    for intent in pending:
        if not intent.close:
            side = longs if intent.buy else shorts
            side[intent.pair] = side.get(intent.pair, ZERO) + intent.notional * D("1.01")
    return longs, shorts


def limit_exposure(intents, signed_quote, pending, markets, pair_caps, settings):
    longs, shorts = exposure_bounds(signed_quote, pending)
    result = []
    for intent in sorted(intents, key=lambda item: not item.close):
        if intent.close:
            result.append(intent)
            continue
        side, other = (longs, shorts) if intent.buy else (shorts, longs)
        pair = intent.pair
        pairs = set(longs) | set(shorts)
        gross = sum((max(longs.get(p, ZERO), shorts.get(p, ZERO)) for p in pairs), ZERO)
        headroom = min(pair_caps.get(pair, ZERO) - side.get(pair, ZERO),
                       settings.max_directional_quote - sum(side.values(), ZERO),
                       settings.max_gross_quote - gross + max(ZERO, other.get(pair, ZERO) - side.get(pair, ZERO)))
        # Use the larger execution price/mid so an order cannot understate exposure.
        valuation = max(intent.price, markets[pair].mid)
        amount = floor_step(min(intent.amount, max(ZERO, headroom) / (valuation * D("1.01"))), markets[pair].step)
        if amount >= markets[pair].minimum:
            adjusted = Intent(pair, intent.buy, amount, intent.price)
            result.append(adjusted)
            side[pair] = side.get(pair, ZERO) + amount * valuation * D("1.01")
    return result


class ReturnHistory:
    def __init__(self, settings):
        self.settings, self.prices = settings, {}

    def observe(self, pair, timestamp, price):
        if not math.isfinite(timestamp) or timestamp < 0 or not price.is_finite() or price <= 0:
            return
        bucket = int(timestamp / self.settings.correlation_sample_seconds)
        history = self.prices.setdefault(pair, deque(maxlen=self.settings.correlation_history_samples))
        if history and bucket <= history[-1][0]:
            return
        if history and bucket > history[-1][0] + 1:
            history.clear()
        history.append((bucket, price))

    def correlation(self, first, second):
        def returns(pair):
            history = list(self.prices.get(pair, ()))
            return {t: float(p / old - ONE) for (previous, old), (t, p) in zip(history, history[1:]) if t == previous + 1}
        a, b = returns(first), returns(second)
        times = sorted(set(a) & set(b))
        if len(times) < self.settings.correlation_min_samples:
            return None
        x, y = [a[t] for t in times], [b[t] for t in times]
        mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
        cov = sum((i - mean_x) * (j - mean_y) for i, j in zip(x, y))
        var_x, var_y = sum((i - mean_x) ** 2 for i in x), sum((j - mean_y) ** 2 for j in y)
        if min(var_x, var_y) <= 1e-24:
            return None
        return max(-1, min(1, cov / math.sqrt(var_x * var_y)))

    def eligible(self, pair, selected, now=None):
        for other in selected:
            fresh = True
            if now is not None:
                current = int(now / self.settings.correlation_sample_seconds)
                fresh = all(self.prices.get(p) and 0 <= current - self.prices[p][-1][0] <= 1
                            for p in (pair, other))
            corr = self.correlation(pair, other) if fresh else None
            if corr is None and self.settings.require_correlation:
                return False
            if corr is not None and abs(corr) >= float(self.settings.max_abs_correlation):
                return False
        return True


class LossCircuit:
    def __init__(self, settings):
        self.settings = settings
        self.events, self.seen = deque(), set()
        self.until = 0
        self.healthy_since = self.last_check = None

    def record(self, pair, now):
        if pair in self.seen:
            return
        self.seen.add(pair)
        while self.events and self.events[0][1] < now - self.settings.stop_window_seconds:
            self.events.popleft()
        self.events.append((pair, now))
        if len(self.events) >= self.settings.stop_count:
            self.until = max(self.until, now + self.settings.stop_pause_seconds)
            self.healthy_since = None

    def allow_open(self, now, healthy, max_gap):
        if not self.until:
            return True
        gap = self.last_check is not None and now - self.last_check > max_gap
        self.last_check = now
        if now < self.until or not healthy or gap:
            self.healthy_since = None
            return False
        if self.healthy_since is None:
            self.healthy_since = now
        if now - self.healthy_since < self.settings.recovery_seconds:
            return False
        self.until = 0
        self.healthy_since = None
        self.events.clear()
        return True
