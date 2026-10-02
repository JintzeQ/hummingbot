"""Per-pair bounded risk-aversion controller, not a learned profit estimator."""

import math
from dataclasses import dataclass, field
from decimal import Decimal

from hummingbot.strategy.gate_avellaneda.core import ZERO

D = Decimal
ONE = D("1")


@dataclass(frozen=True)
class GammaSettings:
    enabled: bool = True
    minimum: Decimal = D("0.5")
    maximum: Decimal = D("3")
    volatility_weight: Decimal = D("0.5")
    depth_weight: Decimal = D("0.5")
    micro_weight: Decimal = D("0.5")
    loss_weight: Decimal = ONE
    ema_seconds: float = 15
    update_seconds: float = 3
    max_step_ratio: Decimal = D("0.1")
    reprice_bps: Decimal = D("2")
    min_reprice_seconds: float = 2

    def __post_init__(self):
        if any(not value.is_finite() for value in vars(self).values() if isinstance(value, D)):
            raise ValueError("Gamma settings must be finite")
        if not ZERO < self.minimum < self.maximum:
            raise ValueError("Gamma bounds must satisfy 0 < minimum < maximum")
        if min(self.volatility_weight, self.depth_weight, self.micro_weight, self.loss_weight) < 0:
            raise ValueError("Gamma stress weights cannot be negative")
        if not ZERO < self.max_step_ratio < ONE or self.reprice_bps <= 0:
            raise ValueError("Invalid Gamma step or reprice threshold")
        if any(not math.isfinite(value) or value <= 0
               for value in (self.ema_seconds, self.update_seconds, self.min_reprice_seconds)):
            raise ValueError("Gamma timing must be positive and finite")


@dataclass
class GammaState:
    value: Decimal
    target: Decimal
    updated_at: float
    components: dict = field(default_factory=dict)


class GammaController:
    def __init__(self, risk, adaptive, settings):
        self.risk, self.adaptive, self.settings = risk, adaptive, settings
        if settings.enabled and not settings.minimum <= risk.gamma <= settings.maximum:
            raise ValueError("Base risk.gamma must fit Gamma bounds")
        self.states = {}

    def value(self, pair):
        return self.states[pair].value if self.settings.enabled and pair in self.states else self.risk.gamma

    def update(self, market, pnl, now, signal=None):
        s = self.settings
        if not s.enabled:
            return self.risk.gamma
        values = (market.volatility, market.bid_depth, market.ask_depth, pnl)
        if (market.close_rejection(self.risk, now) or not market.ready
                or any(not value.is_finite() for value in values)
                or market.volatility <= 0 or min(market.bid_depth, market.ask_depth) < 0):
            return self.value(market.pair)
        if signal is not None and (not signal.ready or any(not v.is_finite()
                                  for v in (signal.flow, signal.bid_loss, signal.ask_loss))):
            return self.value(market.pair)
        state = self.states.get(market.pair)
        if state is not None and now - state.updated_at < s.update_seconds:
            return state.value
        def clip(value):
            return max(ZERO, min(ONE, value))
        components = dict(
            volatility=clip(market.volatility / market.mid / self.adaptive.volatility_reference - ONE),
            depth=clip(ONE - min(market.bid_depth, market.ask_depth)
                       / (self.adaptive.depth_target_multiple * self.risk.order_quote)),
            micro=ZERO if signal is None else (ONE if signal.paused else clip(max(
                signal.bid_loss, signal.ask_loss, abs(signal.flow)
                if signal.flow_sufficient and signal.book_flow_conflict else ZERO))),
            loss=clip(-pnl / self.risk.pair_loss_limit),
        )
        stress = sum((value * getattr(s, name + "_weight") for name, value in components.items()), ZERO)
        target = max(s.minimum, min(s.maximum, self.risk.gamma * (ONE + stress)))
        if state is None:
            # First observation starts at the configured baseline. Other entry
            # guards and exposure caps remain active while Gamma ramps upward.
            self.states[market.pair] = GammaState(self.risk.gamma, target, now, components)
            return self.risk.gamma
        # After a gap apply one bounded update, never replay missed updates.
        elapsed = min(now - state.updated_at, s.update_seconds)
        alpha = D(str(-math.expm1(-elapsed / s.ema_seconds)))
        smoothed = state.value + alpha * (target - state.value)
        limit = state.value * s.max_step_ratio
        state.value = max(s.minimum, min(s.maximum, max(state.value - limit, min(state.value + limit, smoothed))))
        state.target, state.updated_at, state.components = target, now, components
        return state.value
