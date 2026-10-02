"""Write-ahead ownership and restart reconciliation. No exchange I/O here."""
import re
import uuid
from dataclasses import asdict
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from .account_risk import finite, finite_time
from .core import Intent, Slot
from .verification import plain

ZERO = Decimal(0)
PAIR = re.compile(r"^[A-Z0-9]+-USDT$")


class RecoverySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    confirm_snapshots: int = Field(default=2, ge=2, le=10)
    quiet_seconds: float = Field(default=5, ge=1, le=60)
    not_found_grace_seconds: float = Field(default=120, ge=120, le=600)
    max_pending_orders: int = Field(default=128, ge=8, le=1000)
    max_fills_per_order: int = Field(default=10000, ge=1000, le=10000)


def pair_value(pair):
    if not isinstance(pair, str) or not PAIR.fullmatch(pair):
        raise ValueError("Invalid recovery pair")
    return pair


def intent_from_json(data):
    if any(type(data[key]) is not bool for key in ("buy", "close", "market")):
        raise ValueError("Invalid recovery order flags")
    intent = Intent(pair_value(data["pair"]), data["buy"], finite(data["amount"]), finite(data["price"]),
                    data["close"], data["market"])
    if min(intent.amount, intent.price) <= 0:
        raise ValueError("Invalid recovery order quantity/price")
    return intent


class RecoveryLedger:
    def __init__(self, settings):
        self.settings = settings
        self.owner = uuid.uuid4().hex[:12]
        self.orders = {}
        self.restored = False
        self.phase = "ready"
        self.reason = ""
        self.confirmations = 0
        self.confirmed_since = None
        self.fingerprint = None

    @property
    def recovering(self):
        return self.phase != "ready"

    def new_id(self):
        return "t-ga" + self.owner + uuid.uuid4().hex[:12]

    def prepare(self, client_id, intent, now):
        if not re.fullmatch("t-ga" + self.owner + r"[0-9a-f]{12}", client_id) or client_id in self.orders:
            raise ValueError("Invalid or duplicate recovery order identity")
        if len(self.orders) >= self.settings.max_pending_orders:
            raise ValueError("Recovery order ledger is full; opening paused until settlement")
        self.orders[client_id] = dict(intent=plain(intent), submitted_at=finite_time(now), exchange_id=None,
                                      terminal=False, fills={})

    def filled(self, client_id):
        return sum((finite(row["amount"]) for row in self.orders[client_id]["fills"].values()), ZERO)

    def fill(self, client_id, trade_id, amount, price, timestamp):
        row = self.orders[client_id]
        if not trade_id:
            raise ValueError("Recoverable fill must have an exchange trade ID")
        value = dict(amount=str(finite(amount)), price=str(finite(price)), timestamp=finite_time(timestamp))
        if finite(amount) <= 0 or finite(price) <= 0:
            raise ValueError("Invalid recovery fill")
        if trade_id in row["fills"]:
            if any(row["fills"][trade_id][key] != value[key] for key in ("amount", "price")):
                raise ValueError("Exchange trade ID changed its fill values")
            return False
        intent = intent_from_json(row["intent"])
        if self.filled(client_id) + amount > intent.amount or len(row["fills"]) >= self.settings.max_fills_per_order:
            raise ValueError("Recovered fills exceed the order or fill ledger limit")
        row["fills"][trade_id] = value
        return True

    def validate_order(self, client_id, raw, multiplier, user):
        row = self.orders[client_id]
        intent = intent_from_json(row["intent"])
        exchange_id = str(raw.get("id", ""))
        if (not exchange_id or exchange_id == "None" or raw.get("text") != client_id
                or raw.get("contract", "").replace("_", "-") != intent.pair
                or str(raw.get("user", "")) != user):
            raise ValueError("Recovered exchange order identity/account does not match checkpoint")
        if row["exchange_id"] not in (None, exchange_id):
            raise ValueError("Recovered exchange order ID changed")
        size, left = finite(raw["size"]), abs(finite(raw["left"]))
        if abs(size) * multiplier != intent.amount or (size > 0) != intent.buy or left > abs(size):
            raise ValueError("Recovered order size/side changed")
        if bool(raw.get("is_reduce_only", False)) != intent.close:
            raise ValueError("Recovered order reduce-only flag changed")
        if raw.get("status") not in ("open", "finished"):
            raise ValueError("Invalid recovered order status")
        if (intent.market and (finite(raw["price"]) != 0 or raw.get("tif") != "ioc")
                or not intent.market and (finite(raw["price"]) != intent.price or raw.get("tif") != "poc")):
            raise ValueError("Recovered order price/type changed")
        row["exchange_id"] = exchange_id
        return (abs(size) - left) * multiplier

    def trade_rows(self, client_id, raw_rows, multiplier, now):
        row = self.orders[client_id]
        intent = intent_from_json(row["intent"])
        result = {}
        for raw in raw_rows:
            trade_id = str(raw.get("id", raw.get("trade_id", "")))
            size, price = finite(raw["size"]), finite(raw["price"])
            timestamp = finite_time(raw["create_time"])
            if (not trade_id or trade_id == "None" or trade_id in result
                    or str(raw.get("order_id")) != row["exchange_id"]
                    or raw.get("contract", "").replace("_", "-") != intent.pair
                    or raw.get("text", client_id) != client_id or size == 0 or (size > 0) != intent.buy
                    or price <= 0 or timestamp < row["submitted_at"] - 1 or timestamp > now + 1
                    or finite(raw.get("point_fee", 0)) != 0):
                raise ValueError("Invalid/misattributed/duplicate recovery trade")
            result[trade_id] = dict(amount=abs(size) * multiplier, price=price, timestamp=timestamp,
                                    fee=finite(raw["fee"]) if "fee" in raw else None)
        if sum((v["amount"] for v in result.values()), ZERO) > intent.amount:
            raise ValueError("Recovery trade sum exceeds order")
        for trade_id, known in row["fills"].items():
            if trade_id not in result:
                raise RuntimeError("Trade REST is missing a previously confirmed fill; waiting")
            # Event timestamps may be truncated by the connector; identity,
            # quantity and price must still agree exactly.
            if any(finite(known[key]) != result[trade_id][key] for key in ("amount", "price")):
                raise ValueError("REST fill conflicts with persisted execution")
        return result

    def wait(self, reason):
        self.reason = reason
        self.confirmations = 0
        self.confirmed_since = self.fingerprint = None

    def confirm(self, fingerprint, now):
        if self.fingerprint != fingerprint:
            self.fingerprint = fingerprint
            self.confirmations, self.confirmed_since = 1, now
        else:
            self.confirmations += 1
        self.phase = "confirming"
        self.reason = "waiting for stable consecutive private snapshots"
        return (self.confirmations >= self.settings.confirm_snapshots
                and now - self.confirmed_since >= self.settings.quiet_seconds)

    def dump(self, expected, portfolio, inventory):
        return dict(schema=1, owner=self.owner, expected_positions=plain(expected), orders=plain(self.orders),
                    slots={p: plain(asdict(s)) for p, s in portfolio.slots.items()},
                    inventory=dict(opened_at=inventory.opened_at.copy(), streaks=plain(inventory.streaks),
                                   cooldowns=plain(inventory.cooldowns), counted_orders={p: sorted(v) for p, v in inventory.counted_orders.items()}))

    def restore(self, payload, portfolio, inventory):
        if payload["schema"] != 1 or not re.fullmatch(r"[0-9a-f]{12}", payload["owner"]):
            raise ValueError("Invalid recovery schema/owner")
        expected = {pair_value(p): finite(a) for p, a in payload["expected_positions"].items()}
        slots = {}
        if len(payload["slots"]) > 2:
            raise ValueError("Recovery checkpoint exceeds two portfolio slots")
        for pair, raw in payload["slots"].items():
            pair_value(pair)
            if (raw["pair"] != pair or raw["state"] not in ("active", "retiring")
                    or not isinstance(raw["reason"], str) or type(raw["force_market"]) is not bool
                    or type(raw["failures"]) is not int or raw["failures"] < 0):
                raise ValueError("Invalid recovery slot")
            slots[pair] = Slot(pair, raw["state"], raw["failures"], raw["reason"],
                               finite_time(raw["retiring_since"]), finite_time(raw["selected_at"]), raw["force_market"])
        if any(amount != 0 and pair not in slots for pair, amount in expected.items()):
            raise ValueError("Persistent position has no owning portfolio slot")
        if len(payload["orders"]) > self.settings.max_pending_orders:
            raise ValueError("Recovery checkpoint exceeds pending order limit")
        self.owner = payload["owner"]
        for client_id, raw in payload["orders"].items():
            intent = intent_from_json(raw["intent"])
            self.prepare(client_id, intent, raw["submitted_at"])
            if type(raw["terminal"]) is not bool or raw["exchange_id"] is not None and not isinstance(raw["exchange_id"], str):
                raise ValueError("Invalid persistent exchange order state")
            self.orders[client_id]["exchange_id"] = raw["exchange_id"]
            self.orders[client_id]["terminal"] = raw["terminal"]
            for trade_id, fill in raw["fills"].items():
                self.fill(client_id, trade_id, finite(fill["amount"]), finite(fill["price"]), fill["timestamp"])
        state = payload["inventory"]
        inventory.opened_at = {pair_value(p): finite_time(t) for p, t in state["opened_at"].items()}
        inventory.streaks = {}
        for pair, row in state["streaks"].items():
            if len(row) != 3 or type(row[0]) is not bool or type(row[1]) is not int or row[1] < 1 or not isinstance(row[2], str):
                raise ValueError("Invalid persistent inventory streak")
            inventory.streaks[pair_value(pair)] = tuple(row)
        inventory.cooldowns = {}
        for pair, row in state["cooldowns"].items():
            if len(row) != 2 or type(row[0]) is not bool:
                raise ValueError("Invalid persistent inventory cooldown")
            inventory.cooldowns[pair_value(pair)] = row[0], finite_time(row[1])
        inventory.counted_orders = {}
        for pair, orders in state["counted_orders"].items():
            if not isinstance(orders, list) or not all(isinstance(order, str) for order in orders):
                raise ValueError("Invalid persistent inventory fill IDs")
            inventory.counted_orders[pair_value(pair)] = set(orders)
        if any(amount and pair not in inventory.opened_at for pair, amount in expected.items()) and inventory.settings.enabled:
            raise ValueError("Persistent inventory lacks original holding age")
        portfolio.slots = slots
        self.restored, self.phase = True, "reconciling"
        self.reason = "restored checkpoint; new risk frozen until exchange reconciliation"
        return expected


def recovery_pairs(payload):
    recovery = payload.get("recovery")
    if recovery is None:
        return set()
    pairs = {pair_value(pair) for pair in recovery["slots"]}
    pairs.update(pair_value(pair) for pair, amount in recovery["expected_positions"].items() if finite(amount))
    pairs.update(pair_value(row["intent"]["pair"]) for row in recovery["orders"].values())
    return pairs
