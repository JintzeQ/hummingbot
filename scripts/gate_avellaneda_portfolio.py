"""Two-slot Gate USDT perpetual maker; dry-run is the default.

Subscribe to a bounded candidate pool once at startup. Only occupied slots are
quoted; replacements are drawn from that live pool, not from a fixed two-pair
list. A restart rebuilds the pool from the current futures volume ranking.
"""

import asyncio
import json
import math
import os
from collections import Counter, deque
from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar, Dict, List
from urllib.request import urlopen

from pydantic import Field, model_validator

from hummingbot.client.config.config_data_types import BaseClientModel
from hummingbot.connector.derivative.gate_io_perpetual import gate_io_perpetual_constants as C
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, TradeType
from hummingbot.core.data_type.order_candidate import PerpetualOrderCandidate
from hummingbot.core.network_iterator import NetworkStatus
from hummingbot.strategy.__utils__.trailing_indicators.instant_volatility import InstantVolatilityIndicator
from hummingbot.strategy.__utils__.trailing_indicators.trading_intensity import TradingIntensityIndicator
from hummingbot.strategy.gate_avellaneda.core import (
    Intent, Market, Portfolio, Settings, ZERO, allocate, avellaneda_quotes, candidate_universe,
)
from hummingbot.strategy.order_book_asset_price_delegate import OrderBookAssetPriceDelegate
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class GateAvellanedaPortfolioConfig(BaseClientModel):
    script_file_name: str = os.path.basename(__file__)
    dry_run: bool = True
    risk: Settings = Field(default_factory=Settings)
    candidate_limit: int = Field(default=20, ge=2, le=40)
    candidate_pairs: List[str] = Field(default_factory=list)
    sample_ticks: int = Field(default=60, ge=20, le=600)
    account_refresh_seconds: float = Field(default=3, ge=1, le=10)
    book_refresh_seconds: float = Field(default=10, ge=3, le=30)
    monitor_seconds: float = Field(default=60, ge=10)
    quote_refresh_seconds: float = Field(default=15, ge=5)
    exit_timeout_seconds: float = Field(default=60, ge=15)
    allow_market_exit: bool = True
    maker_fee_floor: Decimal = Field(default=Decimal("0.0002"), ge=0, lt=1)

    @model_validator(mode="after")
    def validate_refresh_age(self):
        if self.risk.max_age <= self.book_refresh_seconds + self.account_refresh_seconds:
            raise ValueError("max_age must exceed the book refresh interval plus the account refresh interval")
        return self


@dataclass
class TrackedOrder:
    intent: Intent
    remaining: Decimal
    submitted_at: float
    cancel_at: float = 0


class GateAvellanedaPortfolio(ScriptStrategyBase):
    markets: ClassVar[Dict[str, set]] = {}
    _initial_contracts: ClassVar[dict] = {}
    _initial_tickers: ClassVar[dict] = {}
    connector_name = "gate_io_perpetual"

    @classmethod
    def init_markets(cls, config: GateAvellanedaPortfolioConfig):
        # Script init_markets is synchronous in this Hummingbot version. There
        # are just two bounded, public HTTP requests, before the clock starts.
        root = "https://api.gateio.ws/api/v4/futures/usdt/"
        with urlopen(root + "contracts", timeout=10) as response:
            contracts = json.load(response)
        with urlopen(root + "tickers", timeout=10) as response:
            tickers = json.load(response)
        cls._initial_contracts = candidate_universe(
            contracts, tickers, config.risk, config.candidate_limit, config.candidate_pairs,
        )
        if not cls._initial_contracts:
            raise ValueError("No affordable USDT perpetual candidate meets the configured universe filters")
        cls._initial_tickers = {t["contract"].replace("_", "-"): t for t in tickers}
        cls.markets = {cls.connector_name: set(cls._initial_contracts)}

    def __init__(self, connectors, config: GateAvellanedaPortfolioConfig):
        super().__init__(connectors, config)
        self.config = config
        self.portfolio = Portfolio(config.risk)
        self.contracts = dict(self._initial_contracts)
        self.tickers = dict(self._initial_tickers)
        self.books = {}
        self.indicators = {}
        self.history = {p: deque(maxlen=config.sample_ticks) for p in self.contracts}
        self.orders: Dict[str, TrackedOrder] = {}
        # Retain ownership through delayed REST cancellation visibility and
        # any final fill delivered after a terminal callback.
        self.terminal_orders: Dict[str, TrackedOrder] = {}
        self.positions = {}
        self.position_quote = {}
        self.position_marks = {}
        self.unrealised_pnl = {}
        self.cashflows = {}
        self.initial_book_ids = set()
        self.run_started_at = None
        # Live starts flat. Track signed fills independently of potentially
        # delayed REST positions; a flat response alone cannot release a slot.
        self.expected_positions = {}
        self.exchange_open_pairs = set()
        self.exchange_open_ids = set()
        self.available = ZERO
        self.equity = ZERO
        self.account_at = 0
        self.account_epoch = 0
        self.account_dirty = True
        self.revision = 0
        self.used_revision = -1
        self.initialized = False
        self.halt_reason = ""
        self.refresh_task = None
        self.next_account = self.next_books = self.next_public = self.next_monitor = 0
        self.next_quote = {}
        self.last_quotes = {}
        self.stopping = False

    @property
    def connector(self):
        return self.connectors[self.connector_name]

    def _account_fresh(self):
        return (self.initialized and not self.account_dirty
                and self.current_timestamp - self.account_at <= self.config.risk.max_age)

    def on_tick(self):
        if self.stopping:
            return
        now = self.current_timestamp
        if self.refresh_task is None or self.refresh_task.done():
            if now >= self.next_account:
                self.refresh_task = asyncio.create_task(self._refresh())
                self.next_account = now + self.config.account_refresh_seconds
        if self.connector.network_status != NetworkStatus.CONNECTED or self.halt_reason:
            self._cancel_all()
            return
        self._sample()
        snapshots = self._snapshots()
        fresh = self._account_fresh()
        if not fresh:
            # Once an account snapshot expires, outstanding orders are also
            # withdrawn. After a fill, wait for REST reconciliation before quoting.
            if now - self.account_at > self.config.risk.max_age:
                self._cancel_all()
            return
        for pair in list(self.portfolio.slots):
            if self.portfolio.stop_loss(pair, self._pair_pnl(pair), now):
                self.logger().warning(self.portfolio.slots[pair].reason + f"; exiting {pair} at market")
                self._cancel_pair(pair)
                # An already-flat loss exit can be replaced on this tick.
                self.next_monitor = 0
        if now >= self.next_monitor:
            before = {(p, s.state) for p, s in self.portfolio.slots.items()}
            self.portfolio.evaluate(snapshots, now, fresh, self.positions,
                                    self.exchange_open_pairs | {o.intent.pair for o in self.orders.values()},
                                    allow_entries=self.equity > self.config.risk.reserve)
            after = {(p, s.state) for p, s in self.portfolio.slots.items()}
            if before != after:
                self.logger().info(f"Portfolio slots: {sorted(after)}")
            self.next_monitor = now + self.config.monitor_seconds
        if self.equity <= self.config.risk.reserve:
            for pair in self.portfolio.slots:
                self.portfolio.retire(pair, "equity reached reserve floor", now)
        intents = []
        for pair, slot in list(self.portfolio.slots.items()):
            market = snapshots.get(pair)
            if slot.state == "retiring":
                # Do not submit a close against an account snapshot which still
                # includes an opening order that is being cancelled.
                existing = [o for o in self.orders.values() if o.intent.pair == pair]
                if any(not o.intent.close for o in existing):
                    self._cancel_pair(pair)
                    continue
                if existing:
                    if ((slot.force_market and any(not o.intent.market for o in existing))
                            or (not slot.force_market and now >= self.next_quote.get(pair, 0))):
                        self._cancel_pair(pair)
                    continue
                if pair in self.exchange_open_pairs:
                    self._cancel_pair(pair)
                    continue
                exit_intent = self._exit_intent(pair, market, slot.retiring_since, slot.force_market)
                if exit_intent:
                    intents.append(exit_intent)
                continue
            if market is None or market.rejection(self.config.risk, now):
                self._cancel_pair(pair)
                continue
            if now < self.next_quote.get(pair, 0):
                continue
            if any(o.intent.pair == pair for o in self.orders.values()):
                self._cancel_pair(pair)
                continue
            if pair in self.exchange_open_pairs:
                self._cancel_pair(pair)
                continue
            intents.extend(avellaneda_quotes(market, self.positions.get(pair, ZERO), self.config.risk))
        if not intents:
            return
        outstanding = {}
        pending = list(self.orders.values())
        pending.extend(o for oid, o in self.terminal_orders.items() if oid in self.exchange_open_ids)
        for order in pending:
            if not order.intent.close:
                pair = order.intent.pair
                outstanding[pair] = outstanding.get(pair, ZERO) + order.remaining * order.intent.price * Decimal("1.01")
        selected = allocate(intents, self.position_quote, outstanding, self.available, self.equity, self.config.risk)
        if self.config.dry_run:
            self.last_quotes = {p: [i for i in selected if i.pair == p] for p in self.portfolio.slots}
            for pair, quotes in self.last_quotes.items():
                if quotes:
                    self.logger().info(f"DRY RUN {pair}: {quotes}")
                    self.next_quote[pair] = now + self.config.quote_refresh_seconds
            return
        # One submission batch per reconciled account revision. This prevents
        # the second pair/tick reusing collateral before the exchange updates it.
        if self.used_revision == self.revision:
            return
        self.used_revision = self.revision
        for intent in selected:
            self._submit(intent)

    def _sample(self):
        for pair in self.contracts:
            try:
                book = self.connector.get_order_book(pair)
                if pair not in self.indicators:
                    base, quote = pair.rsplit("-", 1)
                    info = MarketTradingPairTuple(self.connector, pair, base, quote)
                    delegate = OrderBookAssetPriceDelegate(info)
                    self.indicators[pair] = (
                        InstantVolatilityIndicator(sampling_length=self.config.sample_ticks),
                        TradingIntensityIndicator(book, delegate, sampling_length=self.config.sample_ticks),
                    )
                bid, ask = self.connector.get_price(pair, False), self.connector.get_price(pair, True)
                mid = (bid + ask) / 2
                if mid.is_finite() and mid > 0:
                    vol, intensity = self.indicators[pair]
                    vol.add_sample(mid)
                    intensity.calculate(self.current_timestamp)
                    self.history[pair].append(mid)
            except (KeyError, ValueError, ArithmeticError):
                continue

    def _snapshots(self):
        snapshots = {}
        for pair, (raw, observed_at) in self.books.items():
            try:
                contract = self.contracts[pair]
                multiplier = Decimal(str(contract["quanto_multiplier"]))
                bids, asks = raw["bids"], raw["asks"]
                bid, ask = Decimal(str(bids[0]["p"])), Decimal(str(asks[0]["p"]))
                mid = (bid + ask) / 2
                if not mid.is_finite() or mid <= 0:
                    continue
                def depth(rows):
                    return sum((Decimal(str(r["p"])) * abs(Decimal(str(r["s"]))) * multiplier
                                for r in rows if abs(Decimal(str(r["p"])) / mid - 1) <= Decimal("0.001")), ZERO)
                vol, intensity = self.indicators[pair]
                alpha, kappa = intensity.current_value
                history = self.history[pair]
                trend = history[-1] / history[0] - 1 if history else ZERO
                base, quote = pair.rsplit("-", 1)
                fee = self.connector.get_fee(base, quote, OrderType.LIMIT_MAKER, TradeType.BUY,
                                             self.config.risk.order_quote / mid, bid).percent
                ticker = self.tickers[pair]
                snapshots[pair] = Market(
                    pair, bid, ask, Decimal(str(vol.current_value)), Decimal(str(kappa)), Decimal(str(alpha)),
                    Decimal(str(ticker["volume_24h_quote"])), depth(bids), depth(asks),
                    Decimal(str(ticker["funding_rate_indicative"])), trend, max(self.config.maker_fee_floor, fee),
                    multiplier, Decimal(str(contract["order_price_round"])),
                    multiplier * Decimal(str(contract["order_size_min"])), observed_at,
                    vol.is_sampling_buffer_full and intensity.is_sampling_buffer_full,
                    not contract.get("in_delisting", True),
                )
            except (KeyError, IndexError, ValueError, ArithmeticError):
                continue
        return snapshots

    async def _refresh(self):
        try:
            if self.run_started_at is None:
                self.run_started_at = self.current_timestamp
            epoch = self.account_epoch
            positions, account, open_orders, cashflows = await asyncio.gather(
                self.connector._api_get(path_url=C.POSITION_INFORMATION_URL, is_auth_required=True),
                self.connector._api_get(path_url=C.USER_BALANCES_PATH_URL, is_auth_required=True),
                self._open_orders(),
                self._account_book(),
            )
            if epoch != self.account_epoch:
                return
            nonzero = [p for p in positions if Decimal(str(p["size"])) != 0]
            if not self.initialized and not self.config.dry_run:
                if nonzero or open_orders:
                    raise ValueError("Start requires a flat futures account with no exchange open orders; reconcile existing exposure first")
                await self._setup_live()
            unknown = [o for o in open_orders if o.get("text") not in self.orders
                       and o.get("text") not in self.terminal_orders]
            if unknown and not self.config.dry_run:
                raise ValueError("Unowned exchange orders detected; new submissions paused")
            converted = {}
            position_quote = {}
            position_marks = {}
            unrealised_pnl = {}
            for position in nonzero:
                pair = position["contract"].replace("_", "-")
                if pair not in self.contracts or position.get("mode") != "single":
                    raise ValueError("An unmanaged or hedge-mode position exists; manual reconciliation required")
                converted[pair] = Decimal(str(position["size"])) * Decimal(str(self.contracts[pair]["quanto_multiplier"]))
                mark = Decimal(str(position.get("mark_price", self.contracts[pair].get("mark_price", "0"))))
                if not mark.is_finite() or mark <= 0:
                    raise ValueError("Cannot value an existing position")
                position_quote[pair] = abs(converted[pair] * mark)
                position_marks[pair] = mark
                unrealised_pnl[pair] = Decimal(str(position["unrealised_pnl"]))
                if not unrealised_pnl[pair].is_finite():
                    raise ValueError("Invalid per-pair unrealized PnL")
            if epoch != self.account_epoch:
                return
            if not self.config.dry_run:
                expected = {p: a for p, a in self.expected_positions.items() if a != 0}
                if converted != expected:
                    self.account_dirty = True
                    self._cancel_all()
                    self.logger().warning("REST positions disagree with confirmed fills; waiting for reconciliation")
                    return
            self.positions = converted
            self.position_quote = position_quote
            self.position_marks = position_marks
            self.unrealised_pnl = unrealised_pnl
            if not self.initialized:
                # Ignore historical records already present before any bot orders.
                self.initial_book_ids = set(cashflows)
            self.cashflows = cashflows
            self.available = Decimal(str(account["available"]))
            self.equity = Decimal(str(account["total"])) + Decimal(str(account.get("unrealised_pnl", "0")))
            if not self.available.is_finite() or not self.equity.is_finite():
                raise ValueError("Invalid account balances")
            self.exchange_open_pairs = {o["contract"].replace("_", "-") for o in open_orders}
            self.exchange_open_ids = {o.get("text") for o in open_orders}
            self.account_at = self.current_timestamp
            self.account_dirty = False
            self.initialized = True
            self.revision += 1
            if self.current_timestamp >= self.next_public:
                contracts, tickers = await asyncio.gather(
                    self.connector._api_get(path_url="futures/usdt/contracts", limit_id=C.NETWORK_CHECK_PATH_URL),
                    self.connector._api_get(path_url=C.TICKER_PATH_URL),
                )
                contract_map = {c["name"].replace("_", "-"): c for c in contracts}
                for pair in self.contracts:
                    self.contracts[pair] = contract_map.get(pair, dict(self.contracts[pair], in_delisting=True))
                self.tickers = {t["contract"].replace("_", "-"): t for t in tickers}
                self.next_public = self.current_timestamp + self.config.monitor_seconds
            if self.current_timestamp >= self.next_books:
                semaphore = asyncio.Semaphore(4)

                async def fetch_book(pair):
                    async with semaphore:
                        try:
                            raw = await self.connector._api_get(
                                path_url=C.ORDER_BOOK_PATH_URL, params={"contract": pair.replace("-", "_"), "limit": 20},
                            )
                            self.books[pair] = (raw, self.current_timestamp)
                        except Exception as exc:
                            self.logger().warning(f"Book refresh failed for {pair}: {exc}")

                await asyncio.gather(*(fetch_book(pair) for pair in self.contracts))
                self.next_books = self.current_timestamp + self.config.book_refresh_seconds
        except asyncio.CancelledError:
            raise
        except ValueError as exc:
            self.halt_reason = str(exc)
            self.account_dirty = True
            self._cancel_all()
            self.logger().error(f"Portfolio halted: {exc}")
        except Exception as exc:
            self.account_dirty = True
            self._cancel_all()
            self.logger().warning(f"Account/market refresh failed; quoting paused: {exc}")

    def _pair_pnl(self, pair):
        """Gate cash settlements since selection + current mark-price PnL.

        Fee/funding changes are signed USDT cashflows. Do not also add the
        position's realised_pnl, which includes the same settlements.
        """
        selected_at = self.portfolio.slots[pair].selected_at
        settled = sum((change for record_id, (p, timestamp, change) in self.cashflows.items()
                       if p == pair and timestamp >= selected_at and record_id not in self.initial_book_ids), ZERO)
        return settled + self.unrealised_pnl.get(pair, ZERO)

    async def _account_book(self):
        # Re-read this run's fixed time range rather than advancing a timestamp
        # cursor: late fee/settlement records must not disappear between polls.
        # IDs deduplicate records; offsets apply to a fixed 'to' on every page.
        result = {}
        start, end = int(self.run_started_at), int(self.current_timestamp)
        for offset in range(0, 10000, 1000):
            page = await self.connector._api_get(
                path_url="futures/usdt/account_book", is_auth_required=True,
                limit_id=C.USER_BALANCES_PATH_URL,
                params={"from": start, "to": end, "limit": 1000, "offset": offset},
            )
            for record in page:
                pair = record.get("contract", "").replace("_", "-")
                kind = record["type"]
                if kind in ("pnl", "fee", "fund") and not pair:
                    raise ValueError("Cash settlement record has no contract; cannot attribute pair losses")
                if pair not in self.contracts or kind not in ("pnl", "fee", "fund", "point_fee", "bonus_offset"):
                    continue
                change = Decimal(str(record["change"]))
                timestamp = float(record["time"])
                record_id = str(record["id"])
                if not change.is_finite() or not math.isfinite(timestamp) or not record_id or record["id"] is None:
                    raise ValueError("Invalid per-pair account-book record")
                if kind in ("point_fee", "bonus_offset") and change != 0:
                    raise ValueError("Per-pair loss accounting requires USDT fees; POINT/bonus charges are unsupported")
                if record_id in result:
                    raise RuntimeError("Account-book pages changed during pagination; retrying snapshot")
                result[record_id] = (pair, timestamp, change)
            if len(page) < 1000:
                return result
        raise ValueError("Account-book pagination limit reached; cannot monitor pair losses safely")

    async def _open_orders(self):
        # Gate paginates orders. Do not infer a flat/order-free account from only
        # the first page, even when this strategy normally has at most four orders.
        result = []
        for offset in range(0, 1000, 100):
            page = await self.connector._api_get(path_url=C.USER_ORDERS_PATH_URL, is_auth_required=True,
                                                 params={"status": "open", "limit": 100, "offset": offset})
            result.extend(page)
            if len(page) < 100:
                return result
        raise ValueError("Open-order pagination limit reached; cannot reconcile safely")

    async def _setup_live(self):
        first = next(iter(self.contracts))
        success, message = await self.connector._trading_pair_position_mode_set(PositionMode.ONEWAY, first)
        if not success:
            raise ValueError(f"Could not confirm One-way mode: {message}")
        self.connector._perpetual_trading.set_position_mode(PositionMode.ONEWAY)
        self.connector._position_mode = PositionMode.ONEWAY
        for pair in self.contracts:
            success, message = await self.connector._set_trading_pair_leverage(pair, 1)
            if not success:
                raise ValueError(f"Could not confirm 1x leverage for {pair}: {message}")
            self.connector._perpetual_trading.set_leverage(pair, 1)

    def _exit_intent(self, pair, market, retiring_since, force_market=False):
        amount = self.positions.get(pair, ZERO)
        if amount == 0:
            return None
        if force_market:
            # A loss stop overrides allow_market_exit and the maker timeout.
            # Use a fresh REST mark for budget valuation even if the candidate
            # book/indicators have become stale or unsuitable for market making.
            price = self.position_marks.get(pair, ZERO)
            if not self._account_fresh() or price <= 0:
                return None
            return Intent(pair, amount < 0, abs(amount), price, close=True, market=True)
        if market is None or self.current_timestamp - market.observed_at > self.config.risk.max_age:
            return None
        buy = amount < 0
        price = market.bid if buy else market.ask
        use_market = (self.config.allow_market_exit
                      and self.current_timestamp - retiring_since >= self.config.exit_timeout_seconds)
        return Intent(pair, buy, abs(amount), price, close=True, market=use_market)

    def _submit(self, intent):
        candidate = PerpetualOrderCandidate(
            trading_pair=intent.pair, is_maker=not intent.market,
            order_type=OrderType.MARKET if intent.market else OrderType.LIMIT_MAKER,
            order_side=TradeType.BUY if intent.buy else TradeType.SELL,
            amount=intent.amount, price=intent.price, leverage=Decimal(1), position_close=intent.close,
        )
        adjusted = self.connector.budget_checker.adjust_candidate(candidate, all_or_none=True)
        if adjusted.amount <= 0:
            return
        if adjusted.amount > intent.amount:
            raise ValueError("Budget checker unexpectedly increased an order")
        intent = Intent(intent.pair, intent.buy, adjusted.amount, intent.price, intent.close, intent.market)
        method = self.buy if intent.buy else self.sell
        order_id = method(self.connector_name, intent.pair, intent.amount, candidate.order_type,
                          price=intent.price, position_action=PositionAction.CLOSE if intent.close else PositionAction.OPEN)
        self.orders[order_id] = TrackedOrder(intent, intent.amount, self.current_timestamp)
        self.next_quote[intent.pair] = self.current_timestamp + self.config.quote_refresh_seconds

    def _cancel_pair(self, pair):
        if self.config.dry_run:
            return
        pending = dict(self.orders)
        pending.update((oid, o) for oid, o in self.terminal_orders.items() if oid in self.exchange_open_ids)
        for order_id, order in pending.items():
            if order.intent.pair != pair:
                continue
            if order.cancel_at and self.current_timestamp - order.cancel_at < 5:
                continue
            self.cancel(self.connector_name, pair, order_id)
            order.cancel_at = self.current_timestamp

    def _cancel_all(self):
        pairs = {o.intent.pair for o in self.orders.values()}
        pairs.update(o.intent.pair for oid, o in self.terminal_orders.items() if oid in self.exchange_open_ids)
        for pair in pairs:
            self._cancel_pair(pair)

    def did_fill_order(self, event):
        order = self.orders.get(event.order_id) or self.terminal_orders.get(event.order_id)
        if order is not None:
            order.remaining = max(ZERO, order.remaining - event.amount)
            pair = order.intent.pair
            change = event.amount if order.intent.buy else -event.amount
            self.expected_positions[pair] = self.expected_positions.get(pair, ZERO) + change
            self.account_epoch += 1
            self.account_dirty = True
            # Reprice both sides only after the new position is reconciled.
            self.next_quote[order.intent.pair] = 0

    def _terminal(self, event):
        order = self.orders.pop(event.order_id, None)
        if order is not None:
            self.terminal_orders[event.order_id] = order
            self.account_epoch += 1
            self.account_dirty = True
            if order.intent.pair in self.portfolio.excluded:
                self.next_monitor = 0

    did_cancel_order = _terminal
    did_fail_order = _terminal
    did_expire_order = _terminal
    did_complete_buy_order = _terminal
    did_complete_sell_order = _terminal

    async def on_stop(self):
        self.stopping = True
        if self.refresh_task and not self.refresh_task.done():
            self.refresh_task.cancel()
            await asyncio.gather(self.refresh_task, return_exceptions=True)
        self._cancel_all()
        # Hummingbot stopping removes the strategy clock. Do not claim that a
        # cancellation request or a sent close order is a confirmed flat account.
        if self.positions or self.orders or self.exchange_open_pairs or any(self.expected_positions.values()):
            self.logger().warning("Strategy stopped with possible positions/orders; check Gate and reconcile before restarting")

    def format_status(self):
        lines = [f"Gate Avellaneda portfolio: {'DRY RUN (no orders)' if self.config.dry_run else 'LIVE'}",
                 f"Budget {self.config.risk.capital} USDT / reserve {self.config.risk.reserve}",
                 f"Subscribed candidates: {len(self.contracts)}; account fresh: {self._account_fresh()}"]
        if self.halt_reason:
            lines.append(f"HALTED: {self.halt_reason}")
        if not self.portfolio.slots:
            lines.append("No qualifying pair yet; warming indicators or waiting for suitable markets")
        snapshots = self._snapshots()
        reasons = Counter(m.rejection(self.config.risk, self.current_timestamp) or "eligible"
                          for m in snapshots.values())
        reasons["no usable book/indicator snapshot"] = len(self.contracts) - len(snapshots)
        lines.append("Candidates: " + ", ".join(f"{reason}={count}" for reason, count in sorted(reasons.items()) if count))
        for pair, slot in self.portfolio.slots.items():
            lines.append(f"{pair}: {slot.state}, position={self.positions.get(pair, ZERO)}, "
                         f"net_PnL={self._pair_pnl(pair)} USDT, reason={slot.reason or '-'}")
            for quote in self.last_quotes.get(pair, []):
                lines.append(f"  {'buy' if quote.buy else 'sell'} {quote.amount} @ {quote.price}, close={quote.close}")
        if self.portfolio.excluded:
            lines.append("Loss-stopped pairs excluded this run: " + ", ".join(sorted(self.portfolio.excluded)))
        return "\n".join(lines)
