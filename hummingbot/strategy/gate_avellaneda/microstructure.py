"""Sequence-checked Gate L2/prints monitoring, without network or order calls.

Weighted-mid is a proxy, not a fitted Stoikov micro-price. Unmatched aggregate
depth reductions estimate liquidity removal, not identified cancellations.
"""

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Literal

D = Decimal
ZERO = D("0")
ONE = D("1")


@dataclass(frozen=True)
class MicroSettings:
    enabled: bool = True
    mode: Literal["observe", "protect"] = "protect"
    micro_weight: Decimal = D("0.25")
    max_age: float = 2
    warmup_seconds: float = 5
    min_samples: int = 20
    window_seconds: float = 5
    baseline_seconds: float = 120
    ema_seconds: float = 1
    confirm_seconds: float = 2
    recover_seconds: float = 5
    retire_seconds: float = 60
    match_grace_seconds: float = 0.5
    imbalance_threshold: Decimal = D("0.6")
    flow_threshold: Decimal = D("0.5")
    min_flow_quote: Decimal = D("25")
    depth_floor_ratio: Decimal = D("0.35")
    unmatched_loss_ratio: Decimal = D("0.5")
    caution_scale: Decimal = D("0.5")
    reprice_bps: Decimal = D("2")
    min_reprice_seconds: float = 2

    def __post_init__(self):
        if self.mode not in ("observe", "protect") or self.min_samples < 2:
            raise ValueError("Invalid microstructure mode/sample count")
        if any(not v.is_finite() for v in vars(self).values() if isinstance(v, D)):
            raise ValueError("Microstructure settings must be finite")
        durations = (self.max_age, self.warmup_seconds, self.window_seconds, self.baseline_seconds,
                     self.ema_seconds, self.confirm_seconds, self.recover_seconds, self.retire_seconds,
                     self.match_grace_seconds, self.min_reprice_seconds)
        if any(not math.isfinite(v) or v <= 0 for v in durations):
            raise ValueError("Microstructure durations must be positive and finite")
        if not ZERO <= self.micro_weight <= ONE or not ZERO <= self.caution_scale <= ONE:
            raise ValueError("Microstructure weights must be between zero and one")
        if any(not ZERO < v < ONE for v in (self.imbalance_threshold, self.flow_threshold,
                                            self.depth_floor_ratio, self.unmatched_loss_ratio)):
            raise ValueError("Microstructure thresholds must be between zero and one")
        if min(self.min_flow_quote, self.reprice_bps) <= 0:
            raise ValueError("Flow minimum and reprice threshold must be positive")
        if self.baseline_seconds <= self.retire_seconds + self.confirm_seconds + self.recover_seconds:
            raise ValueError("Depth baseline must outlast retirement and hysteresis")


@dataclass(frozen=True)
class MicroSignal:
    ready: bool
    reason: str
    observed_at: float = 0
    bid: Decimal = ZERO
    ask: Decimal = ZERO
    bid_depth: Decimal = ZERO
    ask_depth: Decimal = ZERO
    weighted_mid: Decimal = ZERO
    reference: Decimal = ZERO
    imbalance: Decimal = ZERO
    flow: Decimal = ZERO
    flow_quote: Decimal = ZERO
    bid_loss: Decimal = ZERO
    ask_loss: Decimal = ZERO
    buy_scale: Decimal = ONE
    sell_scale: Decimal = ONE
    paused: bool = False
    pause_age: float = 0
    retire: bool = False


class BookState:
    def __init__(self):
        self.bids, self.asks = {}, {}
        self.update_id = None
        self.synced = False
        self.reason = "waiting for sequence-checked WebSocket book"
        self.pending = deque(maxlen=1000)
        self.received_at = self.event_at = self.started_at = 0
        self.samples = 0
        self.smoothed_imbalance = ZERO
        self.depths = deque(maxlen=4096)
        self.reductions = deque(maxlen=20000)
        self.trades = deque(maxlen=20000)
        self.trade_ids = set()
        self.confirming = {}
        self.confirmed = set()
        self.pause_since = self.clear_since = self.assessed_id = None


class GateMarketSignalFeed:
    level = 100

    def __init__(self, multipliers: Dict[str, Decimal], settings: MicroSettings):
        self.settings, self.multipliers = settings, multipliers
        if any(not q.is_finite() or q <= 0 for q in multipliers.values()):
            raise ValueError("Contract multipliers must be positive and finite")
        self.states = {pair: BookState() for pair in multipliers}

    def invalidate(self, pair, reason):
        self.states[pair] = BookState()
        self.states[pair].reason = reason

    def reset(self):
        for pair in self.states:
            self.invalidate(pair, "WebSocket reconnect; resynchronizing")

    def needs_snapshot(self, pair):
        state = self.states[pair]
        return state.update_id is None or not state.synced

    def _levels(self, pair, rows):
        result = {}
        for row in rows:
            price, amount = D(str(row["p"])), D(str(row["s"])) * self.multipliers[pair]
            if not price.is_finite() or not amount.is_finite() or price <= 0 or amount < 0 or price in result:
                raise ValueError("Invalid/duplicate depth level")
            result[price] = amount
        return result

    def _bound_book(self, state):
        state.bids = dict(sorted(((p, q) for p, q in state.bids.items() if q > 0), reverse=True)[:self.level])
        state.asks = dict(sorted((p, q) for p, q in state.asks.items() if q > 0)[:self.level])
        if not state.bids or not state.asks or max(state.bids) >= min(state.asks):
            raise ValueError("Empty or crossed book")

    def seed_snapshot(self, pair, raw, received_at):
        if pair not in self.states:
            return False
        try:
            update_id = int(raw["id"])
            if update_id < 0:
                raise ValueError("Invalid snapshot ID")
            state = self.states[pair]
            if state.synced:
                return False
            pending = list(state.pending)
            self.invalidate(pair, "waiting for WebSocket bridge after REST snapshot")
            state = self.states[pair]
            state.bids = self._levels(pair, raw["bids"])
            state.asks = self._levels(pair, raw["asks"])
            self._bound_book(state)
            state.update_id = update_id
            for data, arrived in pending:
                if int(data["u"]) > update_id and not self.observe_depth(pair, data, arrived):
                    break
            return True
        except (KeyError, ValueError, TypeError, ArithmeticError):
            self.invalidate(pair, "invalid REST book snapshot")
            return False

    def observe_depth(self, pair, raw, received_at):
        if pair not in self.states:
            return False
        try:
            event_at = float(raw["t"]) / 1000
            first, last = int(raw["U"]), int(raw["u"])
            state = self.states[pair]
            if state.update_id is not None and last <= state.update_id:
                return True  # Duplicates cannot refresh age or warmup.
            if (not math.isfinite(received_at) or not math.isfinite(event_at) or min(first, last) < 0 or last < first
                    or received_at - event_at > self.settings.max_age or event_at > received_at + 1
                    or str(raw.get("l", self.level)) != str(self.level)):
                raise ValueError("Delayed/invalid WebSocket book")
            if state.synced and event_at - state.event_at > self.settings.max_age:
                self.invalidate(pair, "WebSocket quiet/stale interval; resynchronizing")
                self.states[pair].pending.append((raw, received_at))
                return False
            bids, asks = self._levels(pair, raw["b"]), self._levels(pair, raw["a"])
            if raw.get("full", False):
                if not state.synced or first > state.update_id + 1:
                    self.invalidate(pair, "warming full WebSocket snapshot")
                    state = self.states[pair]
                state.bids, state.asks = bids, asks
            elif state.update_id is None:
                if len(state.pending) == state.pending.maxlen:
                    self.invalidate(pair, "WebSocket bootstrap buffer overflow")
                    state = self.states[pair]
                state.pending.append((raw, received_at))
                return False
            elif first > state.update_id + 1:
                self.invalidate(pair, "WebSocket update gap; new REST snapshot required")
                self.states[pair].pending.append((raw, received_at))
                return False
            else:
                mid = (max(state.bids) + min(state.asks)) / 2
                self._prune_events(state, event_at)
                for side, changes, levels in (("bid", bids, state.bids), ("ask", asks, state.asks)):
                    for price, amount in changes.items():
                        lost = max(ZERO, levels.get(price, ZERO) - amount)
                        if lost and abs(price / mid - ONE) <= D("0.001"):
                            if len(state.reductions) == state.reductions.maxlen:
                                raise ValueError("Depth event buffer overflow")
                            state.reductions.append((event_at, side, price, lost))
                        levels[price] = amount
            self._bound_book(state)
            if state.samples and event_at < state.event_at:
                raise ValueError("Book event time moved backwards")
            state.update_id, state.synced, state.reason = last, True, ""
            state.received_at = received_at
            bid, ask = max(state.bids), min(state.asks)
            imbalance = (state.bids[bid] - state.asks[ask]) / (state.bids[bid] + state.asks[ask])
            if state.samples == 0:
                state.started_at = event_at
                state.smoothed_imbalance = imbalance
            else:
                weight = D(str(1 - math.exp(-(event_at - state.event_at) / self.settings.ema_seconds)))
                state.smoothed_imbalance += weight * (imbalance - state.smoothed_imbalance)
            state.event_at = event_at
            state.samples += 1
            bid_depth, ask_depth = self._depths(state)
            state.depths.append((event_at, bid_depth, ask_depth))
            self._prune_events(state, event_at)
            return True
        except (KeyError, ValueError, TypeError, ArithmeticError):
            self.invalidate(pair, "invalid/delayed WebSocket depth; resynchronizing")
            return False

    def observe_trade(self, pair, trade_id, event_at, price, amount, buy, received_at, internal=False):
        if pair not in self.states or internal:
            return False
        state = self.states[pair]
        try:
            price, amount = D(str(price)), D(str(amount))
            if (not math.isfinite(event_at) or not math.isfinite(received_at)
                    or received_at - event_at > self.settings.max_age or event_at > received_at + 1
                    or not price.is_finite() or not amount.is_finite() or min(price, amount) <= 0):
                raise ValueError("Invalid/delayed trade")
            self._prune_events(state, received_at)
            trade_id = str(trade_id)
            if not trade_id or trade_id == "None" or not isinstance(buy, bool):
                raise ValueError("Invalid trade identity/direction")
            if trade_id in state.trade_ids:
                return True
            if len(state.trades) == state.trades.maxlen:
                raise ValueError("Trade event buffer overflow")
            state.trades.append((event_at, trade_id, buy, price, amount))
            state.trade_ids.add(trade_id)
            return True
        except (ValueError, TypeError, ArithmeticError):
            self.invalidate(pair, "invalid/delayed trade stream; resynchronizing")
            return False

    def _prune_events(self, state, now):
        cutoff = now - self.settings.window_seconds - self.settings.match_grace_seconds
        # Cross-channel delivery and event times can differ slightly.
        while state.trades and state.trades[0][0] < cutoff:
            state.trade_ids.discard(state.trades.popleft()[1])
        while state.reductions and state.reductions[0][0] < cutoff:
            state.reductions.popleft()
        while state.depths and state.depths[0][0] < now - self.settings.baseline_seconds:
            state.depths.popleft()

    @staticmethod
    def _depths(state):
        mid = (max(state.bids) + min(state.asks)) / 2
        return tuple(sum((p * q for p, q in levels.items() if abs(p / mid - ONE) <= D("0.001")), ZERO)
                     for levels in (state.bids, state.asks))

    def signal(self, pair, now):
        state, settings = self.states[pair], self.settings
        if not state.synced:
            return MicroSignal(False, state.reason)
        if (not math.isfinite(now) or now - state.received_at > settings.max_age
                or now - state.event_at > settings.max_age or state.received_at > now + 1):
            return MicroSignal(False, "stale WebSocket book")
        bid, ask = max(state.bids), min(state.asks)
        mid = (bid + ask) / 2
        imbalance = (state.bids[bid] - state.asks[ask]) / (state.bids[bid] + state.asks[ask])
        weighted_mid = mid + imbalance * (ask - bid) / 2
        reference = mid + settings.micro_weight * state.smoothed_imbalance * (ask - bid) / 2
        bid_depth, ask_depth = self._depths(state)
        if state.samples < settings.min_samples or state.event_at - state.started_at < settings.warmup_seconds:
            return MicroSignal(False, "microstructure warming up")
        self._prune_events(state, now)
        peak_bid = max(d[1] for d in state.depths)
        peak_ask = max(d[2] for d in state.depths)
        trades = [t for t in state.trades if now - settings.window_seconds <= t[0] <= now]
        flow_quote = sum((p * q for _, _, _, p, q in trades), ZERO)
        flow = (sum(((ONE if buy else -ONE) * p * q for _, _, buy, p, q in trades), ZERO)
                / flow_quote if flow_quote else ZERO)
        removed, executed = defaultdict(lambda: ZERO), defaultdict(lambda: ZERO)
        for timestamp, side, price, amount in state.reductions:
            if now - settings.window_seconds <= timestamp <= now - settings.match_grace_seconds:
                removed[(side, price)] += amount
        for timestamp, _, buy, price, amount in state.trades:
            if now - settings.window_seconds - settings.match_grace_seconds <= timestamp <= now:
                executed[("ask" if buy else "bid", price)] += amount
        unmatched = {side: sum((price * max(ZERO, amount - executed[(s, price)])
                               for (s, price), amount in removed.items() if s == side), ZERO)
                     for side in ("bid", "ask")}
        bid_loss = min(ONE, unmatched["bid"] / peak_bid) if peak_bid else ZERO
        ask_loss = min(ONE, unmatched["ask"] / peak_ask) if peak_ask else ZERO
        enough_flow = flow_quote >= settings.min_flow_quote
        bid_conflict = enough_flow and imbalance >= settings.imbalance_threshold and flow <= -settings.flow_threshold
        ask_conflict = enough_flow and imbalance <= -settings.imbalance_threshold and flow >= settings.flow_threshold
        raw_flags = set()
        if peak_bid and bid_depth / peak_bid <= settings.depth_floor_ratio:
            raw_flags.add("bid liquidity collapse")
        if peak_ask and ask_depth / peak_ask <= settings.depth_floor_ratio:
            raw_flags.add("ask liquidity collapse")
        if bid_conflict:
            raw_flags.add("bid-heavy book conflicts with sell flow")
        if ask_conflict:
            raw_flags.add("ask-heavy book conflicts with buy flow")
        if bid_conflict and bid_loss >= settings.unmatched_loss_ratio:
            raw_flags.add("unmatched bid loss with sell flow")
        if ask_conflict and ask_loss >= settings.unmatched_loss_ratio:
            raw_flags.add("unmatched ask loss with buy flow")
        if enough_flow and imbalance <= -settings.imbalance_threshold and flow <= -settings.flow_threshold:
            raw_flags.add("sell pressure")
        if enough_flow and imbalance >= settings.imbalance_threshold and flow >= settings.flow_threshold:
            raw_flags.add("buy pressure")
        # Only new sequence-checked book IDs advance confirmation/recovery.
        if state.assessed_id != state.update_id:
            timestamp = state.event_at
            state.confirming = {flag: state.confirming.get(flag, timestamp) for flag in raw_flags}
            state.confirmed = {flag for flag, since in state.confirming.items()
                               if timestamp - since >= settings.confirm_seconds}
            severe = any("collapse" in f or "unmatched" in f for f in state.confirmed)
            if severe:
                if state.pause_since is None:
                    state.pause_since = timestamp
                state.clear_since = None
            elif state.pause_since is not None:
                if state.clear_since is None:
                    state.clear_since = timestamp
                if timestamp - state.clear_since >= settings.recover_seconds:
                    state.pause_since = state.clear_since = None
            state.assessed_id = state.update_id
        paused = state.pause_since is not None
        pause_age = max(0, state.event_at - state.pause_since) if paused else 0
        buy_scale = (settings.caution_scale if any(f in state.confirmed for f in
                     ("sell pressure", "bid-heavy book conflicts with sell flow")) else ONE)
        sell_scale = (settings.caution_scale if any(f in state.confirmed for f in
                      ("buy pressure", "ask-heavy book conflicts with buy flow")) else ONE)
        reason = "; ".join(sorted(state.confirmed)) or ("recovery hysteresis" if paused else "normal")
        return MicroSignal(True, reason, state.event_at, bid, ask, bid_depth, ask_depth, weighted_mid,
                           reference, imbalance, flow, flow_quote, bid_loss, ask_loss,
                           buy_scale, sell_scale, paused, pause_age,
                           paused and state.clear_since is None and pause_age >= settings.retire_seconds)
