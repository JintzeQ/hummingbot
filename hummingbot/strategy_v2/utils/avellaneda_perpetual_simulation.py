"""Offline perpetual paper broker driven by the production Avellaneda engine.

No exchange connector is imported. Market paths and assumed contra volume are
shared by all modes; order events, fee cash flows and inventory are independent.
Top-of-book replay cannot establish real fills: the queue/volume model is an
explicit user assumption, not recovered exchange trade history.
"""

import csv
import heapq
import math
import random
from dataclasses import dataclass, replace
from decimal import Decimal as D
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hummingbot.strategy_v2.utils.avellaneda_perpetual import (
    AvellanedaEngine,
    AvellanedaSettings,
    Quote,
    Snapshot,
    round_step,
)

ZERO = D("0")
CAPTURE = Path(__file__).resolve().parents[3] / "examples/avellaneda_perpetual_validation/gate_btc_capture_20260930.csv"
MODE_LABELS = {"fixed_1": "固定 gamma = 1", "calibrated_fixed": "校準後固定 gamma", "adaptive": "自適應 gamma"}


class SimulationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["synthetic", "capture"] = "synthetic"
    scenario: Literal["range", "uptrend", "downtrend", "shock"] = "range"
    seed: int = Field(42, ge=0, le=2147483647)
    duration_seconds: int = Field(1800, ge=60, le=7200)
    initial_capital: D = Field(D("100"), ge=20, le=10000)
    max_position_quote: D = Field(D("40"), ge=5, le=10000)
    capital_budget_quote: D = Field(D("50"), ge=5, le=10000)
    order_amount_quote: D = Field(D("10"), ge=1, le=10000)
    min_spread: D = Field(D("0.0004"), ge=D("0.0001"), le=D("0.01"))
    maker_fee: D = Field(D("0.0002"), ge=0, le=D("0.01"))
    taker_fee: D = Field(D("0.0005"), ge=0, le=D("0.01"))
    rebate_rate: D = Field(D("0.6"), ge=0, le=1)
    rebate_delay_seconds: float = Field(3600, ge=0, le=86400)
    order_latency_seconds: float = Field(1, ge=0, le=10)
    cancel_latency_seconds: float = Field(2, ge=0, le=10)
    position_latency_seconds: float = Field(1, ge=0, le=10)
    queue_ahead_lots: int = Field(2, ge=0, le=100)
    assumed_volume_lots: int = Field(3, ge=1, le=100)
    market_slippage_bps: D = Field(D("1"), ge=0, le=50)
    volatility_bps: float = Field(0.75, ge=0.05, le=3)
    funding_rate_8h: D = Field(D("0"), ge=D("-0.01"), le=D("0.01"))
    max_session_loss_quote: D = Field(D("1"), gt=0, le=1000)

    @model_validator(mode="after")
    def check_settings(self):
        for name in self.__class__.model_fields:
            value = getattr(self, name)
            if isinstance(value, (float, D)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.capital_budget_quote + D("5") > self.initial_capital:
            raise ValueError("策略預算加 5 USDT 保留金必須小於等於本金")
        if self.max_session_loss_quote >= self.initial_capital:
            raise ValueError("停止虧損必須小於本金")
        # Delegate model relationships to the same settings used in production.
        self.strategy()
        return self

    def strategy(self, **changes):
        values = {name: getattr(self, name) for name in (
            "max_position_quote", "capital_budget_quote", "order_amount_quote", "min_spread",
            "maker_fee", "taker_fee", "rebate_rate", "max_session_loss_quote")}
        values.update(changes)
        return AvellanedaSettings(**values)


@dataclass(frozen=True)
class MarketTick:
    snapshot: Snapshot
    volume: D  # assumed executable contra volume per side when the book trades through
    funding: bool = False


def synthetic_tape(settings):
    rng = random.Random(settings.seed)
    # Historical synthetic clock, so funding boundaries are reproducible.
    start = 1790725400.0
    calibration_end = 925
    log_offset = 0.0
    tape = []
    for i in range(calibration_end + settings.duration_seconds + 1):
        elapsed = max(0, i - calibration_end)
        sigma = settings.volatility_bps / 10000
        drift = 0.0
        if elapsed and settings.scenario in ("uptrend", "downtrend"):
            drift = (1 if settings.scenario == "uptrend" else -1) * sigma * 0.18
        if settings.scenario == "range" or not elapsed:
            drift -= 0.015 * log_offset
        log_offset += drift + rng.gauss(0, sigma)
        if settings.scenario == "shock" and elapsed == settings.duration_seconds // 2:
            log_offset -= 0.025
        mid = D(str(83000 * math.exp(log_offset)))
        bid = round_step(mid - D("0.1"), D("0.1"))
        ask = bid + D("0.2")
        timestamp = start + i
        snap = Snapshot(timestamp, bid, ask, ZERO, settings.initial_capital, settings.initial_capital,
                        D("0.1"), D("0.0001"), D("0.0001"), D("1"))
        lots = rng.randint(1, settings.assumed_volume_lots)
        tape.append(MarketTick(snap, lots * snap.amount_step,
                               int(timestamp) // 28800 != int(timestamp - 1) // 28800))
    return tape


def capture_tape(settings, path=CAPTURE):
    rng = random.Random(settings.seed)
    tape = []
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["trading_pair"] != "BTC-USDT":
                raise ValueError("回放只支援此範例的 BTC-USDT")
            snap = Snapshot(float(row["timestamp"]), D(row["bid"]), D(row["ask"]), ZERO,
                            settings.initial_capital, settings.initial_capital,
                            D(row["price_tick"]), D(row["amount_step"]), D(row["min_amount"]),
                            D(row["min_notional"]), book_age=float(row["book_age"]))
            if not snap.valid or not 0 <= snap.book_age <= 10:
                raise ValueError("回放有無效或過期行情")
            if tape and snap.timestamp <= tape[-1].snapshot.timestamp:
                raise ValueError("回放時間必須嚴格遞增")
            # A crossed funding boundary with missing observations is not an exact
            # funding snapshot. Do not invent that missing settlement event.
            boundary = bool(tape and snap.timestamp - tape[-1].snapshot.timestamp <= 10 and
                            int(snap.timestamp) // 28800 != int(tape[-1].snapshot.timestamp) // 28800)
            tape.append(MarketTick(snap, rng.randint(1, settings.assumed_volume_lots) * snap.amount_step, boundary))
    if not tape:
        raise ValueError("回放檔案沒有行情")
    return tape


def calibrate_tape(tape, settings):
    engine = AvellanedaEngine(settings.strategy(dry_run=True, gamma_mode="adaptive"))
    for index, tick in enumerate(tape):
        engine.step(tick.snapshot)
        if engine.gamma.profile:
            return index, engine.gamma.profile
    raise ValueError(f"無法校準 gamma：{engine.calibration_reason}")


@dataclass
class PaperOrder:
    quote: Quote
    remaining: D
    queue_ahead: D
    active: bool = False
    cancel_pending: bool = False


class PaperBroker:
    """One-way linear USDT futures cash ledger, without leverage or live APIs."""

    def __init__(self, settings, engine):
        self.settings, self.engine = settings, engine
        self.wallet = settings.initial_capital
        self.position = ZERO
        self.visible_position = ZERO
        self.entry = ZERO
        self.realized = ZERO
        self.fees = ZERO
        self.rebates_credited = ZERO
        self.rebates_pending = ZERO
        self.funding = ZERO
        self.orders = {}
        self.events = []
        self.sequence = 0
        self.order_count = 0
        self.rejected = 0
        self.cancelled = 0
        self.fills = []
        self.event_log = []
        self.max_abs_position_quote = ZERO

    def schedule(self, timestamp, kind, payload):
        self.sequence += 1
        heapq.heappush(self.events, (timestamp, self.sequence, kind, payload))

    def unrealized(self, mid):
        return (mid - self.entry) * self.position

    def equity(self, mid):
        return self.wallet + self.unrealized(mid)

    def reserved(self, excluding=""):
        return sum((o.remaining * o.quote.price * (self.settings.maker_fee + (0 if o.quote.close else 1))
                    for oid, o in self.orders.items() if oid != excluding), ZERO)

    def snapshot(self, market):
        available = max(ZERO, min(self.wallet, self.equity(market.mid)) - abs(self.position) * market.mid - self.reserved())
        return replace(market, position=self.visible_position, available=available, equity=self.equity(market.mid))

    def dispatch(self, commands, market):
        for command in commands:
            if command.kind == "create":
                self.order_count += 1
                oid = f"paper-{self.order_count}"
                q = command.quote
                self.engine.register(oid, q, market.timestamp)
                self.orders[oid] = PaperOrder(q, q.amount, self.settings.queue_ahead_lots * market.amount_step)
                self.schedule(market.timestamp + self.settings.order_latency_seconds, "activate", oid)
            else:
                order = self.orders.get(command.order_id)
                if order and not order.cancel_pending:
                    order.cancel_pending = True
                    self.schedule(market.timestamp + self.settings.cancel_latency_seconds, "cancel", command.order_id)

    def terminal(self, oid, now, failed=False):
        if oid in self.orders:
            del self.orders[oid]
            self.engine.terminal(oid, failed=failed, timestamp=now)
            if self.engine.halt_reason:
                self.schedule(now + self.settings.position_latency_seconds, "rest", None)

    def process_events(self, market):
        while self.events and self.events[0][0] <= market.timestamp:
            _, _, kind, payload = heapq.heappop(self.events)
            order = self.orders.get(payload) if kind in ("activate", "cancel") else None
            if kind == "position":
                self.visible_position = payload
            elif kind == "rebate":
                self.wallet += payload
                self.rebates_credited += payload
                self.rebates_pending -= payload
            elif kind == "rest":
                if self.engine.halt_reason and not self.orders:
                    self.visible_position = self.position
                    self.engine.accept_rest_position(self.position)
            elif kind == "cancel" and order:
                self.cancelled += 1
                self.terminal(payload, market.timestamp)
            elif kind == "activate" and order:
                q = order.quote
                crosses = q.price >= market.ask if q.side == "buy" else q.price <= market.bid
                if not q.market and crosses:
                    self.rejected += 1
                    self.event_log.append(dict(timestamp=market.timestamp, kind="post_only_rejected", order_id=payload))
                    self.terminal(payload, market.timestamp, failed=True)
                else:
                    order.active = True
                    if q.market:
                        bps = self.settings.market_slippage_bps / 10000
                        price = market.ask * (1 + bps) if q.side == "buy" else market.bid * (1 - bps)
                        price = round_step(price, market.price_tick, up=q.side == "buy")
                        self.fill(payload, order.remaining, price, market, "market_close")

    def fill(self, oid, amount, price, market, reason):
        order = self.orders.get(oid)
        if not order:
            return
        q = order.quote
        amount = min(amount, order.remaining)
        sign = 1 if q.side == "buy" else -1
        if q.close:
            if self.position * sign >= 0:
                self.rejected += 1
                self.terminal(oid, market.timestamp, failed=True)
                return
            amount = min(amount, abs(self.position))
        amount = round_step(amount, market.amount_step)
        if amount <= 0:
            return
        before = self.position
        delta = amount * sign
        closing = min(abs(before), amount) if before * delta < 0 else ZERO
        realized = closing * (price - self.entry) * (1 if before > 0 else -1)
        after = before + delta
        fee = amount * price * (self.settings.taker_fee if q.market else self.settings.maker_fee)
        next_wallet = self.wallet + realized - fee
        next_entry = self.entry
        if before == 0 or before * after < 0:
            next_entry = price
        elif before * delta > 0:
            next_entry = (abs(before) * self.entry + amount * price) / abs(after)
        elif after == 0:
            next_entry = ZERO
        # Never fund an opening fill with an unpaid rebate. Reductions still work
        # during a loss/margin shortfall so the engine can flatten safely.
        next_equity = next_wallet + (market.mid - next_entry) * after
        if abs(after) > abs(before) and abs(after) * market.mid + self.reserved(oid) > min(next_wallet, next_equity):
            self.rejected += 1
            self.terminal(oid, market.timestamp, failed=True)
            return
        self.wallet, self.position, self.entry = next_wallet, after, next_entry
        self.realized += realized
        self.fees += fee
        rebate = fee * self.settings.rebate_rate
        self.rebates_pending += rebate
        self.schedule(market.timestamp + self.settings.rebate_delay_seconds, "rebate", rebate)
        trade_id = f"fill-{len(self.fills) + 1}"
        self.engine.filled(oid, amount, market.timestamp, trade_id)
        self.engine.record_fill_metrics(oid, trade_id, market.timestamp, amount, price, str(fee))
        self.schedule(market.timestamp + self.settings.position_latency_seconds, "position", after)
        order.remaining -= amount
        self.fills.append(dict(timestamp=market.timestamp, order_id=oid, trade_id=trade_id,
                               side=q.side, amount=float(amount), price=float(price), notional=float(amount * price),
                               liquidity="taker" if q.market else "maker", reduce_only=q.close,
                               fee=float(fee), rebate_earned=float(rebate), realized=float(realized),
                               position=float(after), during_cancel=order.cancel_pending, reason=reason,
                               markouts={}))
        self.max_abs_position_quote = max(self.max_abs_position_quote, abs(after) * market.mid)
        if order.remaining <= 0 or q.close and self.position == 0 or q.market:
            self.terminal(oid, market.timestamp)

    def match(self, tick):
        market = tick.snapshot
        # A missing top-book interval supplies no trade or queue evidence. Never
        # interpolate fills or accumulate assumed volume across that interval.
        volume = {"buy": tick.volume, "sell": tick.volume}
        for oid, order in list(self.orders.items()):
            if not order.active or order.quote.market:
                continue
            q = order.quote
            # Touching the price alone is insufficient in this conservative model.
            through = market.ask < q.price if q.side == "buy" else market.bid > q.price
            if through and volume[q.side] > 0:
                ahead = min(order.queue_ahead, volume[q.side])
                order.queue_ahead -= ahead
                volume[q.side] -= ahead
                amount = min(order.remaining, volume[q.side])
                if amount > 0:
                    self.fill(oid, amount, q.price, market, "assumed_trade_through")
                    volume[q.side] -= amount

    def settle_funding(self, market):
        payment = -self.position * market.mid * self.settings.funding_rate_8h
        self.wallet += payment
        self.funding += payment
        self.event_log.append(dict(timestamp=market.timestamp, kind="funding", payment=float(payment)))

    def markouts(self, market):
        for fill in reversed(self.fills):
            age = market.timestamp - fill["timestamp"]
            if age > 70 and len(fill["markouts"]) == 3:
                break
            for horizon in (5, 30, 60):
                key = str(horizon)
                if age >= horizon and key not in fill["markouts"]:
                    value = (float(market.mid) - fill["price"]) * fill["amount"] * (1 if fill["side"] == "buy" else -1)
                    fill["markouts"][key] = dict(value=value, observed_delay=age, late=age - horizon > 10)


def run_mode(tape, calibration_index, profile, settings, mode):
    config = settings.strategy(dry_run=False, gamma_mode="adaptive" if mode == "adaptive" else "fixed",
                               gamma_calibration=profile, risk_factor=1 if mode == "fixed_1" else profile.gamma_base)
    engine = AvellanedaEngine(config)
    # Historical estimator warmup is shared and causally precedes trading.
    for tick in tape[max(0, calibration_index - config.volatility_samples + 1):calibration_index + 1]:
        engine.observe(tick.snapshot)
    broker = PaperBroker(settings, engine)
    samples = []
    first = tape[calibration_index + 1].snapshot.timestamp
    last_market = None
    rest_scheduled = False

    def advance(tick, matching=True, phase="trading"):
        nonlocal rest_scheduled
        market = tick.snapshot
        broker.process_events(market)
        if matching:
            broker.match(tick)
        if tick.funding:
            broker.settle_funding(market)
        # Zero delay credit/position events can happen on the fill's same tick.
        broker.process_events(market)
        if engine.halt_reason and engine.rest_ack_required and not broker.orders and not rest_scheduled:
            broker.schedule(market.timestamp + settings.position_latency_seconds, "rest", None)
            rest_scheduled = True
        if not engine.rest_ack_required:
            rest_scheduled = False
        commands = engine.step(broker.snapshot(market))
        broker.dispatch(commands, market)
        broker.process_events(market)
        broker.markouts(market)
        mid = market.mid
        broker.max_abs_position_quote = max(broker.max_abs_position_quote, abs(broker.position) * mid)
        active = [o.quote for o in broker.orders.values() if o.active and not o.quote.market]
        buy = next((float(q.price) for q in active if q.side == "buy"), None)
        sell = next((float(q.price) for q in active if q.side == "sell"), None)
        samples.append(dict(timestamp=market.timestamp, elapsed=market.timestamp - first, mid=float(mid),
                            bid=buy, ask=sell, position_quote=float(broker.position * mid),
                            position=float(broker.position), gamma=engine.gamma.current,
                            equity=float(broker.equity(mid)), wallet=float(broker.wallet),
                            pending_rebate=float(broker.rebates_pending), fees=float(broker.fees),
                            rebates_credited=float(broker.rebates_credited), funding=float(broker.funding),
                            realized=float(broker.realized), unrealized=float(broker.unrealized(mid)),
                            status=engine.status, phase=phase))

    for tick in tape[calibration_index + 1:]:
        if tick.snapshot.timestamp - first > settings.duration_seconds:
            break
        gap = last_market is not None and tick.snapshot.timestamp - last_market.timestamp > config.stale_seconds
        if gap:
            # Cancel requests occur at the last known book + stale timeout; ack
            # callbacks settle at the resumed book, with no assumed intervening fills.
            stale = replace(last_market, timestamp=last_market.timestamp + config.stale_seconds + 0.01,
                            book_age=config.stale_seconds + 0.01)
            broker.process_events(stale)
            broker.dispatch(engine.step(broker.snapshot(stale)), stale)
        advance(tick, matching=not gap)
        last_market = tick.snapshot
    if last_market is None:
        raise ValueError("校準後沒有足夠行情可模擬")
    # Apply the real engine's halt/cancel/REST/reduce-only sequence at a frozen
    # final book. This is explicit synthetic shutdown, separate from the tape.
    engine.halt("simulation end")
    for offset in range(1, 61):
        market = replace(last_market, timestamp=last_market.timestamp + offset, book_age=0)
        advance(MarketTick(market, ZERO), matching=False, phase="shutdown")
        if broker.position == 0 and not broker.orders and not engine.rest_ack_required:
            break
    final = samples[-1]
    drawdown = peak = ZERO
    for sample in samples:
        peak = max(peak, D(str(sample["equity"])))
        drawdown = max(drawdown, peak - D(str(sample["equity"])))
    exact_markouts = [f["markouts"]["30"]["value"] for f in broker.fills
                      if "30" in f["markouts"] and not f["markouts"]["30"]["late"]]
    summary = dict(cash_pnl=final["equity"] - float(settings.initial_capital),
                   economic_pnl=final["equity"] + final["pending_rebate"] - float(settings.initial_capital),
                   equity=final["equity"], realized=final["realized"], unrealized=final["unrealized"],
                   fees=float(broker.fees), rebates_credited=float(broker.rebates_credited),
                   rebates_pending=float(broker.rebates_pending), funding=float(broker.funding),
                   volume=sum(f["notional"] for f in broker.fills), fill_count=len(broker.fills),
                   maker_fills=sum(f["liquidity"] == "maker" for f in broker.fills),
                   taker_fills=sum(f["liquidity"] == "taker" for f in broker.fills),
                   cancelled=broker.cancelled, rejected=broker.rejected,
                   cancel_race_fills=sum(f["during_cancel"] for f in broker.fills),
                   max_position_quote=float(broker.max_abs_position_quote), max_drawdown=float(drawdown),
                   mean_markout_30s=sum(exact_markouts) / len(exact_markouts) if exact_markouts else None,
                   markout_30s_count=len(exact_markouts), halt_reason=engine.halt_reason,
                   final_position=float(broker.position), unresolved_orders=len(broker.orders),
                   shutdown_complete=broker.position == 0 and not broker.orders and not engine.rest_ack_required)
    # Up to ~1,200 chart samples; preserve fills, position changes and phase/halt
    # changes so excursions and risk transitions survive downsampling.
    stride = max(1, math.ceil(len(samples) / 1000))
    chart = [s for i, s in enumerate(samples) if i % stride == 0 or i == len(samples) - 1 or
             i and (s["position"] != samples[i - 1]["position"] or s["phase"] != samples[i - 1]["phase"] or
                    s["status"] != samples[i - 1]["status"])]
    return dict(mode=mode, label=MODE_LABELS[mode], summary=summary, samples=chart,
                fills=broker.fills, events=broker.event_log)


def simulate(settings=None):
    settings = settings or SimulationSettings()
    tape = capture_tape(settings) if settings.source == "capture" else synthetic_tape(settings)
    index, profile = calibrate_tape(tape, settings)
    if index + 1 >= len(tape):
        raise ValueError("行情只有校準期，沒有後續模擬期")
    results = [run_mode(tape, index, profile, settings, mode) for mode in MODE_LABELS]
    available = tape[-1].snapshot.timestamp - tape[index + 1].snapshot.timestamp
    gaps = [dict(after=a.snapshot.timestamp, before=b.snapshot.timestamp,
                 seconds=b.snapshot.timestamp - a.snapshot.timestamp)
            for a, b in zip(tape[index + 1:], tape[index + 2:])
            if b.snapshot.timestamp - a.snapshot.timestamp > 10 and
            b.snapshot.timestamp - tape[index + 1].snapshot.timestamp <= settings.duration_seconds]
    return dict(version=1, settings=settings.model_dump(mode="json"),
                calibration=profile.model_dump(mode="json"), results=results,
                metadata=dict(trading_pair="BTC-USDT", leverage=1,
                              source="Gate BTC top-of-book capture, 2026-09-30" if settings.source == "capture" else "seeded synthetic BTC-like path",
                              clock="historical UTC; elapsed seconds on charts",
                              calibration_samples=index + 1, calibration_seconds=profile.observation_seconds,
                              trading_seconds=min(available, settings.duration_seconds), gaps=gaps,
                              fill_model="strict top-of-book trade-through + assumed queue and contra volume; no touch fills",
                              volume_model="uniform 1..assumed_volume_lots per observed tick and per side; shared between modes",
                              latency_model="events take effect on the next observed tick at or after their scheduled time",
                              rebate_model="gross fees debit wallet; pending rebate credits only after configured delay",
                              funding_model="configured rate at observed UTC 8-hour boundaries; no accrued or imputed gap funding",
                              shutdown_model="up to 60 seconds at frozen last book; taker close and slippage included",
                              limitations=["成交、排隊量、滑價與延遲都是假設，沒有真實成交或深度資料。",
                                           "合成行情不代表 Gate 未來價格；真實行情回放的成交仍是假設。",
                                           "待入帳反傭不計入可用保證金；economic_pnl 假設反傭最終全額支付。",
                                           "此模擬不涵蓋強平、ADL、風險級距、斷線或交易所異常。",
                                           "單一路徑或種子不能證明正期望。"],
                              no_exchange_connection=True))
