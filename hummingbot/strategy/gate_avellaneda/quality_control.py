"""Conservative observed execution quality; never a synthetic fill model."""
import math
from decimal import Decimal
from pydantic import BaseModel, ConfigDict, Field

from .account_risk import finite, finite_time

D = Decimal


class QualitySettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = True
    mode: str = Field(default="protect", pattern="^(observe|protect)$")
    min_orders: int = Field(default=30, ge=10, le=100)
    max_orders: int = Field(default=200, ge=100, le=1000)
    window_seconds: float = Field(default=3600, ge=300, le=86400)
    confidence_z: float = Field(default=2, ge=1, le=4)
    max_penalty_bps: Decimal = Field(default=D(20), gt=0, le=100)
    uncertain_size_scale: Decimal = Field(default=D("0.5"), gt=0, le=1)
    emergency_exit_probability: Decimal = Field(default=D("0.1"), ge=0, le=1)
    exit_slippage_bps: Decimal = Field(default=D(2), ge=0, le=100)
    soft_exit_age_seconds: float = Field(default=120, ge=30, le=600)


class QualityControl:
    def __init__(self, settings):
        self.settings = settings
        self.rows = {}

    def observe(self, fill, observation):
        if (not self.settings.enabled or observation.get("horizon") != 5 or not observation.get("observed")
                or fill["close"] or fill["market"] or fill["anchor_mid"] is None
                or fill["fee_estimate_usdt"] is None):
            return
        key = (fill["pair"], bool(fill["buy"]), fill.get("regime", "calm"))
        if key not in self.rows and len(self.rows) >= 160:
            oldest = min(self.rows, key=lambda k: max((o["at"] for o in self.rows[k].values()), default=0))
            del self.rows[oldest]
        orders = self.rows.setdefault(key, {})
        oid, fid = fill["order_id"], fill["trade_id"]
        if not fid:
            return
        order = orders.setdefault(oid, dict(at=fill["filled_at"], fills={}))
        order["at"] = max(order["at"], fill["filled_at"])
        if fid not in order["fills"] and len(order["fills"]) >= 10000:
            return
        order["fills"].setdefault(fid, dict(notional=str(fill["amount"] * fill["price"]),
                                           net=str(observation["fee_adjusted_estimate_usdt"]),
                                           move=str(observation["post_fill_move_usdt"])))
        if len(orders) > self.settings.max_orders:
            del orders[min(orders, key=lambda item: orders[item]["at"])]

    def estimate(self, pair, buy, regime, now):
        orders = self.rows.get((pair, buy, regime), {})
        samples = []
        for order in orders.values():
            if not 0 <= now - order["at"] <= self.settings.window_seconds:
                continue
            total = sum((finite(f["notional"]) for f in order["fills"].values()), D(0))
            if total > 0:
                samples.append(tuple(float(sum((finite(f[k]) for f in order["fills"].values()), D(0)) / total * 10000)
                                     for k in ("net", "move")))
        n = len(samples)
        result = dict(ready=False, orders=n, penalty_bps="0", size_scale="1", block=False, mean_net_bps="0")
        if n < self.settings.min_orders:
            return result
        means, errors = [], []
        for index in (0, 1):
            mean = sum(row[index] for row in samples) / n
            error = self.settings.confidence_z * math.sqrt(sum((row[index] - mean) ** 2 for row in samples) / (n - 1) / n)
            means.append(mean)
            errors.append(error)
        result.update(ready=True, mean_net_bps=str(means[0]), block=means[0] + errors[0] < 0,
                      size_scale=str(self.settings.uncertain_size_scale if means[0] - errors[0] < 0 else D(1)),
                      penalty_bps=str(min(self.settings.max_penalty_bps, D(str(max(0, -means[1] + errors[1]))))))
        return result

    def snapshot(self, pair, regime, now):
        if not self.settings.enabled or self.settings.mode != "protect":
            return {}
        return {str(buy): self.estimate(pair, buy, regime, now) for buy in (True, False)}

    def dump(self, now):
        self.rows = {key: {oid: row for oid, row in orders.items()
                          if 0 <= now - row["at"] <= self.settings.window_seconds}
                     for key, orders in self.rows.items()}
        self.rows = {key: orders for key, orders in self.rows.items() if orders}
        return [dict(pair=p, buy=b, regime=r, orders={oid: row for oid, row in orders.items()
                                                    if 0 <= now - row["at"] <= self.settings.window_seconds})
                for (p, b, r), orders in self.rows.items()]

    def restore(self, rows):
        if len(rows) > 160:
            raise ValueError("Quality state exceeds bounded candidate/side/regime capacity")
        for raw in rows:
            if (not isinstance(raw["pair"], str) or not raw["pair"].endswith("-USDT")
                    or type(raw["buy"]) is not bool or raw["regime"] not in ("calm", "stressed")
                    or len(raw["orders"]) > self.settings.max_orders):
                raise ValueError("Invalid execution quality state")
            for order in raw["orders"].values():
                finite_time(order["at"])
                if len(order["fills"]) > 10000:
                    raise ValueError("Quality fill capacity exceeded")
                for fill in order["fills"].values():
                    if finite(fill["notional"]) <= 0:
                        raise ValueError("Invalid quality notional")
                    finite(fill["net"])
                    finite(fill["move"])
            self.rows[(raw["pair"], raw["buy"], raw["regime"])] = raw["orders"]
