"""Bounded, rule-based gamma control and portable calibration metadata.

This controller is an engineering policy, not a fitted PnL optimizer. All
variance values use squared log returns per second, as does the quote engine.
"""

import math
from collections import deque
from decimal import Decimal
from statistics import median
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CALIBRATION_VERSION = "relative-log-inventory-v1"
ABSOLUTE_GAMMA_MAX = 10000.0


class GammaCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["relative-log-inventory-v1"] = CALIBRATION_VERSION
    trading_pair: str
    created_at: float = Field(ge=0)
    observation_seconds: float = Field(ge=900)
    window_seconds: float = Field(ge=200)
    valid_samples: int = Field(ge=201)
    reference_mid: Decimal = Field(gt=0)
    price_tick: Decimal = Field(gt=0)
    amount_step: Decimal = Field(gt=0)
    min_amount: Decimal = Field(ge=0)
    min_notional: Decimal = Field(ge=0)
    horizon_seconds: float = Field(gt=0)
    max_position_quote: Decimal = Field(gt=0)
    order_amount_quote: Decimal = Field(gt=0)
    reference_amount: Decimal = Field(gt=0)
    reference_inventory: float = Field(gt=0, le=1)
    variance_per_second: float = Field(gt=0)
    gamma_base: float = Field(gt=0, le=ABSOLUTE_GAMMA_MAX)
    target_ticks: float = Field(gt=0)
    long_bid_change_ticks: float
    long_ask_change_ticks: float
    short_bid_change_ticks: float
    short_ask_change_ticks: float
    # Bind a report to all settings affecting candidate quotes and lot capacity.
    quote_settings: dict[str, str]

    @model_validator(mode="after")
    def verify(self):
        for name in self.__class__.model_fields:
            value = getattr(self, name)
            if isinstance(value, (float, Decimal)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.reference_amount % self.amount_step != 0 or self.reference_amount < self.min_amount:
            raise ValueError("reference amount must satisfy contract quantity rules")
        q = float(self.reference_amount * self.reference_mid / self.max_position_quote)
        if not math.isclose(q, self.reference_inventory, rel_tol=1e-9):
            raise ValueError("reference inventory does not match quantized reference amount")
        d = self.target_ticks * float(self.price_tick / self.reference_mid)
        if not 0 < d < 1:
            raise ValueError("invalid reference tick displacement")
        denominator = q * self.variance_per_second * self.horizon_seconds
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError("reference risk scale must be positive and finite")
        expected = -math.log1p(-d) / denominator
        if not math.isclose(expected, self.gamma_base, rel_tol=1e-8):
            raise ValueError("gamma does not match calibration equation")
        if (max(self.long_bid_change_ticks, self.long_ask_change_ticks) > 0
                or max(abs(self.long_bid_change_ticks), abs(self.long_ask_change_ticks)) < 1
                or min(self.short_bid_change_ticks, self.short_ask_change_ticks) < 0
                or max(self.short_bid_change_ticks, self.short_ask_change_ticks) < 1):
            raise ValueError("rounded quotes must visibly shift in both inventory directions")
        return self


def quote_settings(config):
    names = ("capital_budget_quote", "order_amount_quote", "max_position_quote", "available_balance_reserve",
             "leverage", "kappa", "horizon_seconds", "min_spread", "max_spread", "maker_fee", "taker_fee",
             "rebate_rate", "adverse_selection_buffer", "volatility_samples", "calibration_target_ticks")
    # Numeric normalization lets equivalent YAML decimals (40 and 40.0) match.
    return {name: str(Decimal(str(getattr(config, name))).normalize()) for name in names}


class CalibrationObservations:
    """Require a continuous, fresh 15 minute interval and 200 second windows.

    Only one sample per second contributes. A fixed sample-count quote estimator
    still handles irregular timing; this separate calibration window uses time.
    """

    def __init__(self):
        self.window = deque()
        self.variances = deque(maxlen=10000)
        self.started_at = None
        self.last_at = None
        self.count = 0
        self.rules = None
        self.reason = "collecting at least 900 seconds of fresh observations"

    @staticmethod
    def rule_key(snapshot):
        return (snapshot.price_tick, snapshot.amount_step, snapshot.min_amount, snapshot.min_notional)

    def reset(self, reason):
        self.__init__()
        self.reason = reason

    def observe(self, s, config):
        key = self.rule_key(s)
        gap = self.last_at is not None and (s.timestamp - self.last_at > config.stale_seconds or s.timestamp < self.last_at)
        if gap or (self.rules and key != self.rules):
            self.reset("observation gap or trading-rule change; collecting again")
        if self.last_at is not None and int(s.timestamp) <= int(self.last_at):
            return
        if self.started_at is None:
            self.started_at = s.timestamp
            self.rules = key
        self.last_at = s.timestamp
        self.count += 1
        self.window.append((s.timestamp, float(s.mid)))
        # Keep the sample immediately before the 200-second boundary too.
        while len(self.window) > 2 and self.window[1][0] <= s.timestamp - 200:
            self.window.popleft()
        elapsed = self.window[-1][0] - self.window[0][0]
        if elapsed >= 200 and len(self.window) >= 21:
            points = list(self.window)
            self.variances.append(sum(math.log(b[1] / a[1]) ** 2
                                      for a, b in zip(points, points[1:])) / elapsed)

    @property
    def elapsed(self):
        return 0.0 if self.started_at is None else self.last_at - self.started_at

    def reference_variance(self, config):
        if self.elapsed < config.calibration_seconds or len(self.variances) < 20 or self.count < 201:
            raise ValueError(f"insufficient continuous observations ({self.elapsed:.0f}/{config.calibration_seconds:.0f}s)")
        value = median(self.variances)
        if not math.isfinite(value) or value <= 1e-16:
            raise ValueError("zero or near-zero reference volatility; remain in preview")
        return value


class GammaController:
    def __init__(self, config):
        self.config = config
        self.profile = config.gamma_calibration
        self.base = self.profile.gamma_base if self.profile and config.gamma_mode == "adaptive" else config.risk_factor
        self.current = self.base if config.gamma_mode == "adaptive" else config.risk_factor
        self.target = self.current
        self.last_update = None
        self.reason = "fixed gamma" if config.gamma_mode == "fixed" else "waiting for calibration"
        self.bounds_active = False

    @property
    def bounds(self):
        return (self.base * self.config.gamma_min_ratio,
                min(ABSOLUTE_GAMMA_MAX, self.base * self.config.gamma_max_ratio))

    def install(self, profile):
        self.profile = profile
        self.base = self.current = self.target = profile.gamma_base
        self.last_update = None
        self.reason = "calibration installed"

    def compatibility_error(self, s):
        p, c = self.profile, self.config
        if p is None:
            return "verified gamma calibration is required"
        if (p.trading_pair != c.trading_pair or p.horizon_seconds != c.horizon_seconds
                or p.max_position_quote != c.max_position_quote or p.order_amount_quote != c.order_amount_quote
                or p.observation_seconds < c.calibration_seconds
                or quote_settings(c) != p.quote_settings
                or CalibrationObservations.rule_key(s) != (p.price_tick, p.amount_step, p.min_amount, p.min_notional)):
            return "calibration settings or trading rules changed; recalibrate"
        if not 0 <= s.timestamp - p.created_at <= c.calibration_max_age_seconds:
            return "calibration expired or timestamp is in the future; recalibrate"
        return None

    def update(self, now, inventory, variance, eligible=True, reason="data unavailable"):
        c = self.config
        if c.gamma_mode == "fixed":
            self.reason = "fixed gamma"
            return
        if not eligible or self.profile is None:
            self.reason = f"frozen: {reason}"
            return
        if self.last_update is not None and now - self.last_update < c.gamma_update_seconds:
            self.reason = "waiting for update interval"
            return
        low, high = self.bounds
        ratio = math.sqrt(max(0, variance) / self.profile.variance_per_second)
        volatility_multiplier = min(c.gamma_max_volatility_multiplier,
                                    1 + c.gamma_volatility_weight * max(ratio - 1, 0))
        unclipped = self.base * (1 + c.gamma_inventory_weight * abs(inventory) ** 2) * volatility_multiplier
        self.target = max(low, min(high, unclipped))
        mixed = (1 - c.gamma_smoothing) * self.current + c.gamma_smoothing * self.target
        limited = max(self.current * (1 - c.gamma_max_step),
                      min(self.current * (1 + c.gamma_max_step), mixed))
        self.current = max(low, min(high, limited))
        self.bounds_active = unclipped != self.target or limited != self.current
        self.last_update = now
        self.reason = f"inventory={inventory:.4f}, volatility/reference={ratio:.4f}"
