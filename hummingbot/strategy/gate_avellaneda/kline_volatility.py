"""Closed futures-candle volatility calibration, not a fill-probability model.

The diffusion square-root-of-time assumption is a quote-distance heuristic.
It neither models queue position nor defines an optimal trading strategy.
"""
import math
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from statistics import NormalDist, stdev

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .account_risk import finite, finite_time

D = Decimal
ZERO = D(0)
INTERVALS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}


class KlineSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = True
    mode: str = Field(default="observe", pattern="^(observe|protect)$")
    interval: str = Field(default="1m", pattern="^(1m|5m|15m|1h)$")
    lookback_bars: int = Field(default=720, ge=31, le=1999)
    min_returns: int = Field(default=120, ge=30, le=1000)
    refresh_seconds: float = Field(default=60, ge=10, le=600)
    request_timeout_seconds: float = Field(default=5, gt=0, le=15)
    max_age_seconds: float = Field(default=180, ge=60, le=14400)
    quote_horizon_seconds: float = Field(default=15, ge=1, le=60)
    tail_probability: Decimal = Field(default=D("0.25"), ge=D("0.01"), le=D("0.99"))
    smoothing_seconds: float = Field(default=300, ge=60, le=3600)
    max_relative_change: Decimal = Field(default=D("0.1"), gt=0, le=D("0.25"))
    weight: Decimal = Field(default=D("0.25"), ge=0, le=D("0.5"))
    max_multiplier: Decimal = Field(default=D("1.5"), ge=1, le=2)
    max_extra_bps: Decimal = Field(default=D(5), ge=0, le=20)

    @model_validator(mode="after")
    def bounds(self):
        if self.lookback_bars < self.min_returns + 1:
            raise ValueError("K-line lookback must include enough consecutive returns")
        if self.max_age_seconds < INTERVALS[self.interval] + self.refresh_seconds:
            raise ValueError("K-line freshness must cover the bar and refresh intervals")
        if self.smoothing_seconds < self.refresh_seconds:
            raise ValueError("K-line smoothing must cover the refresh interval")
        return self


@dataclass(frozen=True)
class KlineObservation:
    pair: str
    ready: bool = False
    reason: str = "waiting for closed futures candles"
    samples: int = 0
    fetched_at: float = 0
    last_closed_at: float = 0
    raw_daily_volatility: Decimal = ZERO
    daily_volatility: Decimal = ZERO
    horizon_volatility: Decimal = ZERO
    suggested_half_spread: Decimal = ZERO


def candle_estimate(rows, now, settings):
    """Validate Gate futures object rows; never parse the spot-array schema."""
    now = finite_time(now)
    interval = INTERVALS[settings.interval]
    if not isinstance(rows, list) or len(rows) > 2000:
        raise ValueError("Invalid or oversized futures candle response")
    bars, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Expected futures candle objects")
        timestamp = finite_time(row["t"])
        if timestamp != int(timestamp) or int(timestamp) % interval or timestamp > now:
            raise ValueError("Misaligned or future candle timestamp")
        if timestamp in seen:
            raise ValueError("Duplicate futures candle timestamp")
        seen.add(timestamp)
        if timestamp + interval > now:
            continue  # The evolving current candle is never a sample.
        o, h, l, c = (finite(row[k]) for k in ("o", "h", "l", "c"))
        if min(o, h, l, c) <= 0 or h < max(o, l, c) or l > min(o, h, c):
            raise ValueError("Invalid futures candle prices")
        price = float(c)
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Candle price outside finite calculation range")
        bars.append((timestamp, price))
    bars = sorted(bars)[-settings.lookback_bars:]
    if len(bars) < settings.min_returns + 1:
        raise ValueError("Insufficient closed futures candles")
    if any(new[0] - old[0] != interval for old, new in zip(bars, bars[1:])):
        raise ValueError("Gap in closed futures candles")
    last_closed = bars[-1][0] + interval
    if not 0 <= now - last_closed <= settings.max_age_seconds:
        raise ValueError("Stale closed futures candles")
    returns = [math.log(new[1]) - math.log(old[1]) for old, new in zip(bars, bars[1:])]
    daily = stdev(returns) * math.sqrt(86400 / interval)
    if not math.isfinite(daily):
        raise ValueError("Invalid candle volatility")
    return D(str(daily)), len(returns), last_closed


class KlineVolatility:
    def __init__(self, settings):
        self.settings = settings
        self.observations = {}
        self.smoothed = {}  # public data is freshly reacquired on every restart

    def fail(self, pair, now, reason):
        old = self.observations.get(pair, KlineObservation(pair))
        value = replace(old, ready=False, reason=str(reason), fetched_at=finite_time(now))
        self.observations[pair] = value
        return value

    def update(self, pair, rows, now):
        cfg = self.settings
        try:
            raw, n, ended = candle_estimate(rows, now, cfg)
            previous = self.smoothed.get(pair)
            if previous and (ended < previous[1] or now < previous[2]):
                raise ValueError("K-line observation clock moved backwards")
            if previous is None or now - previous[2] > cfg.max_age_seconds:
                value = raw
            else:
                old_value, old_end, old_updated = previous
                if ended < old_end or now < old_updated:
                    raise ValueError("K-line observation clock moved backwards")
                value = old_value
                if ended > old_end:
                    fraction = D(str(-math.expm1(-(now - old_updated) / cfg.smoothing_seconds)))
                    desired = old_value + fraction * (raw - old_value)
                    # Zero has no relative step; seed conservatively from fresh data.
                    value = (max(old_value * (1 - cfg.max_relative_change),
                                 min(old_value * (1 + cfg.max_relative_change), desired)) if old_value > 0 else desired)
            if previous is None or ended > previous[1] or now - previous[2] > cfg.max_age_seconds:
                self.smoothed[pair] = value, ended, now
            horizon = value * D(str(math.sqrt(cfg.quote_horizon_seconds / 86400)))
            z = D(str(abs(NormalDist().inv_cdf(float(cfg.tail_probability) / 2))))
            observation = KlineObservation(pair, True, "", n, now, ended, raw, value, horizon, horizon * z)
            self.observations[pair] = observation
            return observation
        except (KeyError, TypeError, ValueError, ArithmeticError, OverflowError) as exc:
            return self.fail(pair, now, str(exc))

    def snapshot(self, pair, now):
        cfg = self.settings
        if not cfg.enabled:
            return None
        value = self.observations.get(pair, KlineObservation(pair))
        if value.ready and (not 0 <= now - value.last_closed_at <= cfg.max_age_seconds
                            or not 0 <= now - value.fetched_at <= cfg.max_age_seconds):
            value = replace(value, ready=False, reason="stale K-line calibration")
        return dict(**asdict(value), enabled=True, mode=cfg.mode,
                    interval=cfg.interval, quote_horizon_seconds=cfg.quote_horizon_seconds,
                    tail_probability=cfg.tail_probability, weight=cfg.weight,
                    max_multiplier=cfg.max_multiplier, max_extra_bps=cfg.max_extra_bps)


def calibrated_half_spread(base, calibration):
    """Return a bounded opening half-spread; never shrink or change CLOSE."""
    if not calibration or not calibration.get("enabled") or calibration.get("mode") != "protect" or not calibration.get("ready"):
        return base
    target = finite(calibration["suggested_half_spread"])
    weight = finite(calibration["weight"])
    multiplier = finite(calibration["max_multiplier"])
    extra = finite(calibration["max_extra_bps"]) / 10000
    if target < 0 or not 0 <= weight <= D("0.5") or not 1 <= multiplier <= 2 or not 0 <= extra <= D("0.002"):
        raise ValueError("Invalid K-line quote calibration bounds")
    desired = base + weight * max(ZERO, target - base)
    return min(desired, base * multiplier, base + extra)
