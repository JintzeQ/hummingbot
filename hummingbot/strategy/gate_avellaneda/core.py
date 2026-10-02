"""Pure decision logic. No exchange access or order submission occurs here."""

import math
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_CEILING, Decimal
from typing import Callable, Dict, List, Optional, Set

D = Decimal
ZERO = D("0")


@dataclass(frozen=True)
class Settings:
    capital: Decimal = D("100")
    reserve: Decimal = D("20")
    pair_margin: Decimal = D("40")
    max_position_quote: Decimal = D("20")
    order_quote: Decimal = D("5")
    pair_loss_limit: Decimal = D("5")
    leverage: int = 1
    gamma: Decimal = D("1")
    eta: Decimal = D("1")
    min_net_spread: Decimal = D("0.0002")
    max_book_spread: Decimal = D("0.005")
    max_quote_spread: Decimal = D("0.02")
    max_volatility: Decimal = D("0.005")
    max_trend: Decimal = D("0.03")
    max_funding: Decimal = D("0.002")
    min_volume: Decimal = D("1000000")
    min_depth: Decimal = D("50")
    max_age: float = 15
    failures_to_retire: int = 3
    cooldown: float = 600

    def __post_init__(self):
        decimals = [v for v in vars(self).values() if isinstance(v, D)]
        if any(not v.is_finite() for v in decimals):
            raise ValueError("Settings must be finite")
        if min(self.capital, self.pair_margin, self.max_position_quote, self.order_quote,
               self.pair_loss_limit, self.gamma) <= 0:
            raise ValueError("Capital, budgets, order size and gamma must be positive")
        if self.reserve < 0 or 2 * self.pair_margin + self.reserve > self.capital:
            raise ValueError("Two pair budgets plus reserve must fit total capital")
        if self.leverage != 1:
            raise ValueError("The first version supports confirmed 1x leverage only")
        if self.max_position_quote + 2 * self.order_quote > self.pair_margin:
            raise ValueError("Position and two opening orders must fit each pair budget")
        if not ZERO <= self.eta <= 1 or self.failures_to_retire < 1:
            raise ValueError("Invalid eta or retirement confirmation count")
        if not all(math.isfinite(v) and v > 0 for v in (self.max_age, self.cooldown)):
            raise ValueError("Age and cooldown must be positive")
        if min(self.min_net_spread, self.max_book_spread, self.max_quote_spread, self.max_volatility,
               self.max_trend, self.max_funding, self.min_volume, self.min_depth) <= 0:
            raise ValueError("Screening thresholds must be positive")


@dataclass(frozen=True)
class Market:
    pair: str
    bid: Decimal
    ask: Decimal
    volatility: Decimal
    kappa: Decimal
    alpha: Decimal
    volume: Decimal
    bid_depth: Decimal
    ask_depth: Decimal
    funding: Decimal
    trend: Decimal
    maker_fee: Decimal
    step: Decimal
    tick: Decimal
    minimum: Decimal
    observed_at: float
    ready: bool = True
    enabled: bool = True
    funding_interval: float = 28800
    funding_next_at: float = 0
    taker_fee: Decimal = D("0.0005")

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    def close_rejection(self, settings: Settings, now: float) -> Optional[str]:
        """Execution safety for reducing exposure, independent of entry economics."""
        if any(not x.is_finite() for x in (self.bid, self.ask, self.step, self.tick, self.minimum)):
            return "invalid executable market values"
        if (not math.isfinite(now) or not math.isfinite(self.observed_at) or now - self.observed_at > settings.max_age
                or self.observed_at > now):
            return "stale market data"
        if min(self.bid, self.step, self.tick, self.minimum) <= 0 or self.ask <= self.bid:
            return "invalid trading rules or book"
        return None

    def rejection(self, settings: Settings, now: float, quoted_economics: bool = False) -> Optional[str]:
        values = [v for v in vars(self).values() if isinstance(v, D)]
        if any(not v.is_finite() for v in values):
            return "invalid market values"
        if now - self.observed_at > settings.max_age or self.observed_at > now:
            return "stale market data"
        if not self.enabled:
            return "contract unavailable"
        if not self.ready or min(self.volatility, self.kappa, self.alpha) <= 0:
            return "indicators warming up"
        if min(self.bid, self.step, self.tick, self.minimum) <= 0 or self.ask <= self.bid:
            return "invalid trading rules or book"
        if self.minimum * self.mid > settings.order_quote:
            return "minimum order exceeds order budget"
        if floor_step(settings.order_quote / self.mid, self.step) < self.minimum:
            return "order cannot meet minimum after rounding"
        spread = (self.ask - self.bid) / self.mid
        cost = 2 * max(ZERO, self.maker_fee) + settings.min_net_spread
        if not quoted_economics and spread < cost:
            return "book spread below fee and edge threshold"
        if spread > settings.max_book_spread:
            return "book spread too wide"
        if not quoted_economics:
            model_spread = settings.gamma * self.volatility
            model_spread += 2 * (1 + settings.gamma / self.kappa).ln() / settings.gamma
            if model_spread / self.mid > settings.max_quote_spread:
                return "model spread too wide"
        if self.tick / self.mid > settings.max_book_spread / 2:
            return "price tick too coarse"
        if self.volatility / self.mid > settings.max_volatility:
            return "volatility too high"
        if abs(self.trend) > settings.max_trend:
            return "strong directional move"
        if abs(self.funding) > settings.max_funding:
            return "funding too high"
        if self.volume < settings.min_volume:
            return "volume too low"
        if min(self.bid_depth, self.ask_depth) < max(settings.min_depth, 5 * settings.order_quote):
            return "insufficient nearby depth"
        return None

    def score(self) -> float:
        """Heuristic suitability score, not an expected-profit estimate."""
        return (math.log1p(float(self.volume)) + math.log1p(float(min(self.bid_depth, self.ask_depth)))
                - 100 * float(self.volatility / self.mid) - 100 * abs(float(self.trend)))


@dataclass
class Slot:
    pair: str
    state: str = "active"
    failures: int = 0
    reason: str = ""
    retiring_since: float = 0
    selected_at: float = 0
    force_market: bool = False


class Portfolio:
    """A retiring pair occupies its slot until confirmed flat and order-free."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.slots: Dict[str, Slot] = {}
        self.cooldowns: Dict[str, float] = {}
        self.excluded: Set[str] = set()

    def evaluate(self, markets: Dict[str, Market], now: float, account_fresh: bool,
                 positions: Dict[str, Decimal], open_pairs: Set[str], allow_entries: bool = True,
                 entry_check: Optional[Callable] = None, candidate_filter: Optional[Callable] = None,
                 on_release: Optional[Callable] = None, candidate_score: Optional[Callable] = None) -> None:
        if not account_fresh:
            return
        reject = entry_check or (lambda market: market.rejection(self.settings, now))
        for pair, slot in list(self.slots.items()):
            if slot.state == "retiring":
                if positions.get(pair, ZERO) == 0 and pair not in open_pairs:
                    if on_release and not on_release(slot, now):
                        continue
                    del self.slots[pair]
                    self.cooldowns[pair] = now + self.settings.cooldown
                continue
            market = markets.get(pair)
            reason = reject(market) if market else "stale market data"
            # Data loss is a pause, not evidence that a coin's economics changed.
            if reason in ("stale market data", "invalid market values", "indicators warming up", "K-line calibration unavailable"):
                continue
            slot.failures = slot.failures + 1 if reason else 0
            slot.reason = reason or ""
            if slot.failures >= self.settings.failures_to_retire:
                self.retire(pair, reason, now)
        if not allow_entries:
            return
        candidates = sorted(
            (m for m in markets.values() if m.pair not in self.slots and m.pair not in self.excluded
             and now >= self.cooldowns.get(m.pair, 0) and reject(m) is None),
            key=lambda m: (-(candidate_score(m) if candidate_score else m.score()), m.pair),
        )
        for market in candidates:
            if len(self.slots) >= 2:
                break
            if candidate_filter and not candidate_filter(market.pair, tuple(self.slots)):
                continue
            self.slots[market.pair] = Slot(market.pair, selected_at=now)

    def retire(self, pair: str, reason: str, now: float) -> None:
        slot = self.slots[pair]
        if slot.state != "retiring":
            slot.state, slot.reason, slot.retiring_since = "retiring", reason, now

    def stop_loss(self, pair: str, pnl: Decimal, now: float) -> bool:
        """Latch the loss exit; recovery cannot re-enable this pair this run."""
        if pair not in self.slots or pair in self.excluded or not pnl.is_finite():
            return False
        if pnl > -self.settings.pair_loss_limit:
            return False
        self.retire(pair, f"pair loss limit reached: net PnL {pnl} USDT", now)
        slot = self.slots[pair]
        slot.reason = f"pair loss limit reached: net PnL {pnl} USDT"
        slot.force_market = True
        self.excluded.add(pair)
        return True


@dataclass(frozen=True)
class Intent:
    pair: str
    buy: bool
    amount: Decimal
    price: Decimal
    close: bool = False
    market: bool = False

    @property
    def notional(self) -> Decimal:
        return self.amount * self.price


def floor_step(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def avellaneda_quotes(market: Market, position: Decimal, settings: Settings,
                      reference_price=None, buy_scale: Decimal = D("1"), sell_scale: Decimal = D("1")) -> List[Intent]:
    """Hummingbot variant: absolute volatility (not variance), infinite horizon.

    Inventory is signed contract exposure / configured position cap. A reducing
    side never crosses zero: any future opposite opening requires a new snapshot.
    """
    mid = market.mid
    reference = mid if reference_price is None else reference_price
    if not reference.is_finite() or not market.bid <= reference <= market.ask:
        raise ValueError("Reference price must be within the book")
    if any(not v.is_finite() or not ZERO <= v <= D("1") for v in (buy_scale, sell_scale)):
        raise ValueError("Opening size scales must be finite and between zero and one")
    q = max(D("-1"), min(D("1"), position * mid / settings.max_position_quote))
    reservation = reference - q * settings.gamma * market.volatility
    spread = settings.gamma * market.volatility
    spread += 2 * (1 + settings.gamma / market.kappa).ln() / settings.gamma
    fee = max(ZERO, market.maker_fee)
    half_floor = mid * (2 * fee + settings.min_net_spread) / 2
    bid = min(reservation - spread / 2, mid - half_floor, market.bid)
    ask = max(reservation + spread / 2, mid + half_floor, market.ask)
    bid = floor_step(bid, market.tick)
    ask = (ask / market.tick).to_integral_value(rounding=ROUND_CEILING) * market.tick
    result = []
    for buy, price in ((True, bid), (False, ask)):
        if price <= 0:
            continue
        close = (buy and position < 0) or (not buy and position > 0)
        amount = settings.order_quote / price
        if close:
            amount = min(amount, abs(position))
        else:
            increasing = max(ZERO, position * mid if buy else -position * mid)
            amount = min(amount, max(ZERO, settings.max_position_quote - increasing) / max(mid, price))
            if (buy and q > 0) or (not buy and q < 0):
                amount *= (-settings.eta * abs(q)).exp()
            amount *= buy_scale if buy else sell_scale
        amount = floor_step(amount, market.step)
        if amount >= market.minimum:
            result.append(Intent(market.pair, buy, amount, price, close))
    return result


def allocate(intents: List[Intent], positions_quote: Dict[str, Decimal],
             outstanding: Dict[str, Decimal], available: Decimal,
             equity: Decimal, settings: Settings) -> List[Intent]:
    """Reserve the entire batch before sending either pair's orders.

    available already excludes exchange-acknowledged order margin. outstanding
    counts all local opening orders for per-pair and bot-wide limits, including
    submissions awaiting acknowledgement. Call only on a reconciled account.
    """
    pair_used = {pair: abs(value) + outstanding.get(pair, ZERO) for pair, value in positions_quote.items()}
    for pair, value in outstanding.items():
        pair_used.setdefault(pair, value)
    cap = max(ZERO, min(settings.capital, equity) - settings.reserve)
    headroom = max(ZERO, min(available - settings.reserve, cap - sum(pair_used.values(), ZERO)))
    selected = []
    # Always allow reducing orders; they do not consume a new opening allocation.
    for intent in sorted(intents, key=lambda i: not i.close):
        if intent.close:
            selected.append(intent)
            continue
        # Conservative buffer covers fees without relying on a rebate.
        required = intent.notional * D("1.01")
        used = pair_used.get(intent.pair, ZERO)
        if required <= headroom and used + required <= settings.pair_margin:
            selected.append(intent)
            pair_used[intent.pair] = used + required
            headroom -= required
    return selected


def candidate_universe(contracts: List[dict], tickers: List[dict], settings: Settings,
                       limit: int, allowlist: Optional[List[str]] = None) -> Dict[str, dict]:
    """Select a bounded, subscribed pool from Gate's USDT futures endpoints."""
    ticker_map = {t["contract"]: t for t in tickers}
    candidates = []
    for contract in contracts:
        name = contract.get("name", "")
        pair = name.replace("_", "-")
        if not name.endswith("_USDT") or contract.get("in_delisting", True):
            continue
        if allowlist and pair not in allowlist:
            continue
        try:
            ticker = ticker_map[name]
            last = D(str(ticker["last"]))
            minimum = D(str(contract["quanto_multiplier"])) * D(str(contract["order_size_min"]))
            volume = D(str(ticker["volume_24h_quote"]))
            if not all(x.is_finite() for x in (last, minimum, volume)) or min(last, minimum) <= 0:
                continue
            if minimum * last > settings.order_quote or volume < settings.min_volume:
                continue
            candidates.append((volume, pair, contract))
        except (KeyError, ValueError, ArithmeticError):
            continue
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return {pair: contract for _, pair, contract in candidates[:limit]}
