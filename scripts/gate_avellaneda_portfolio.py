"""Two-slot Gate USDT perpetual maker; dry-run is the default.

Subscribe to a bounded candidate pool once at startup. Only occupied slots are
quoted; replacements are drawn from that live pool, not from a fixed two-pair
list. A restart includes checkpoint-owned markets before filling the pool from
the current futures volume ranking.
"""

import asyncio
import json
import math
import os
import re
import sqlite3
from collections import Counter, deque
from dataclasses import asdict, dataclass
from decimal import Decimal
from types import SimpleNamespace
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
from hummingbot.strategy.gate_avellaneda.adaptive import (
    AdaptiveSettings, InventoryTracker, LossCircuit, ReturnHistory, entry_rejection,
    exposure_bounds, exposure_scale, limit_exposure, quote_plan,
)
from hummingbot.strategy.gate_avellaneda.kline_volatility import KlineSettings, KlineVolatility, calibrated_half_spread
from hummingbot.strategy.gate_avellaneda.gamma import GammaController, GammaSettings, GammaState
from hummingbot.strategy.gate_avellaneda.account_risk import (
    AccountBookSnapshot, AccountRisk, AccountRiskSettings, RiskStateStore, finite, finite_time, mode_path,
)
from hummingbot.strategy.gate_avellaneda.verification import QualityRecorder, TelemetrySettings, plain
from hummingbot.strategy.gate_avellaneda.recovery import RecoveryLedger, RecoverySettings, intent_from_json, recovery_pairs
from hummingbot.strategy.gate_avellaneda.cash_journal import CashJournal
from hummingbot.strategy.gate_avellaneda.execution import DeadmanSettings, ExecutionSettings
from hummingbot.strategy.gate_avellaneda.quality_control import QualityControl, QualitySettings
from hummingbot.strategy.order_book_asset_price_delegate import OrderBookAssetPriceDelegate
from hummingbot.strategy.gate_avellaneda.microstructure import GateMarketSignalFeed, MicroSettings
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class GateAvellanedaPortfolioConfig(BaseClientModel):
    script_file_name: str = os.path.basename(__file__)
    dry_run: bool = True
    risk: Settings = Field(default_factory=Settings)
    micro: MicroSettings = Field(default_factory=MicroSettings)
    adaptive: AdaptiveSettings = Field(default_factory=AdaptiveSettings)
    gamma_control: GammaSettings = Field(default_factory=GammaSettings)
    account_risk: AccountRiskSettings = Field(default_factory=AccountRiskSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    recovery: RecoverySettings = Field(default_factory=RecoverySettings)
    deadman: DeadmanSettings = Field(default_factory=DeadmanSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    quality_control: QualitySettings = Field(default_factory=QualitySettings)
    kline_volatility: KlineSettings = Field(default_factory=KlineSettings)
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
        if self.kline_volatility.enabled and (not self.adaptive.enabled or not self.telemetry.enabled):
            raise ValueError("K-line calibration requires adaptive pricing and telemetry")
        if self.quality_control.enabled and (not self.adaptive.enabled or not self.telemetry.enabled):
            raise ValueError("Execution quality requires adaptive pricing and observed-fill telemetry")
        if self.recovery.enabled and (not self.account_risk.enabled or not self.account_risk.persist):
            raise ValueError("Recovery requires persistent account risk")
        if not self.dry_run and self.account_risk.enabled and not self.account_risk.persist:
            raise ValueError("Live account risk requires persistent state")
        if self.risk.max_age <= self.book_refresh_seconds + self.account_refresh_seconds:
            raise ValueError("max_age must exceed the book refresh interval plus the account refresh interval")
        if (self.adaptive.enabled and self.gamma_control.enabled
                and not self.gamma_control.minimum <= self.risk.gamma <= self.gamma_control.maximum):
            raise ValueError("Base risk.gamma must fit gamma_control bounds")
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
        if config.recovery.enabled:
            payload = RiskStateStore.read_file(mode_path(config.account_risk.state_path, config.dry_run))
            if payload is not None:
                if payload["scope"] != "gate_io_perpetual/usdt/classic" or payload["dry_run"] is not config.dry_run:
                    raise ValueError("Recovery state scope/mode mismatch")
                needed = recovery_pairs(payload)
                available = {c["name"].replace("_", "-"): c for c in contracts}
                if not needed.issubset(available) or len(needed) > config.candidate_limit:
                    raise ValueError("Cannot subscribe to all checkpoint-owned contracts")
                pool = {p: available[p] for p in sorted(needed)}
                for pair, contract in cls._initial_contracts.items():
                    if len(pool) < config.candidate_limit:
                        pool.setdefault(pair, contract)
                cls._initial_contracts = pool
        if not cls._initial_contracts:
            raise ValueError("No affordable USDT perpetual candidate meets the configured universe filters")
        cls._initial_tickers = {t["contract"].replace("_", "-"): t for t in tickers}
        cls.markets = {cls.connector_name: set(cls._initial_contracts)}

    def __init__(self, connectors, config: GateAvellanedaPortfolioConfig):
        super().__init__(connectors, config)
        self.config = config
        self.inventory = InventoryTracker(config.adaptive)
        self.loss_circuit = LossCircuit(config.adaptive)
        self.return_history = ReturnHistory(config.adaptive)
        self.gamma = GammaController(config.risk, config.adaptive,
                                     config.gamma_control if config.adaptive.enabled else GammaSettings(enabled=False))
        self.last_gamma_values = {}
        self.plans = {}
        self.last_control_limits = {}
        self.last_quality_limits = {}
        self.kline_volatility = KlineVolatility(config.kline_volatility)
        self.kline_task = None
        self.next_kline = 0
        self.last_calibration_limits = {}
        self.entry_allowed = True
        self.selection_rejections = {}
        self.portfolio = Portfolio(config.risk)
        self.contracts = dict(self._initial_contracts)
        self.tickers = dict(self._initial_tickers)
        self.books = {}
        self.signal_feed = GateMarketSignalFeed(
            {p: Decimal(str(c["quanto_multiplier"])) for p, c in self.contracts.items()}, config.micro,
        )
        self.signals = {}
        self.quote_references = {}
        self.last_submission = {}
        self.last_micro_limits = {}
        self.next_signal_log = 0
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
        self.cash_journal = None
        self.cash_error = ""
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
        self.account_risk = AccountRisk(config.account_risk)
        self.quality = QualityControl(config.quality_control)
        self.recorder = QualityRecorder(config.telemetry, config.dry_run, on_markout=self.quality.observe)
        self.heartbeat_task = None
        self.heartbeat_deadlines = {}
        self.next_heartbeat = 0
        self.heartbeat_error = ""
        self.recovery = RecoveryLedger(config.recovery)
        self._replaying_fills = False
        self._recovery_pending_settlements = set()
        self._recovery_block_open = False
        self._restart_order_ids = set()
        self._recovery_verified = False
        self._recovery_correlation_pairs = set()
        self.risk_store = None
        self.persistence_error = ""
        self._saved_payload = None
        self._global_exit_started = False
        self.last_markets = {}
        self.fill_ids = set()
        self.fill_id_queue = deque()
        try:
            if config.micro.enabled and getattr(self.connector, "_gate_market_signal_feed", None) is not None:
                raise ValueError("Market signal feed already owned by another strategy")
            if config.account_risk.enabled and config.account_risk.persist:
                self.risk_store = RiskStateStore(mode_path(config.account_risk.state_path, config.dry_run))
                payload = self.risk_store.load()
                if payload is not None:
                    self._restore_checkpoint(payload)
            if config.micro.enabled:
                self.connector._gate_market_signal_feed = self.signal_feed
        except Exception:
            if self.risk_store is not None:
                self.risk_store.close()
            raise

    @property
    def connector(self):
        return self.connectors[self.connector_name]

    def _account_fresh(self):
        return (self.initialized and not self.account_dirty
                and self.current_timestamp - self.account_at <= self.config.risk.max_age)

    def _opening_ready(self):
        return (self.account_risk.allow_open and not self.persistence_error and not self.recorder.error and not self.cash_error
                and not self.recovery.recovering and not self._recovery_block_open)

    def _restore_checkpoint(self, payload):
        if payload["dry_run"] is not self.config.dry_run or payload["scope"] != "gate_io_perpetual/usdt/classic":
            raise ValueError("Persistent risk scope/mode mismatch")
        self.account_risk.restore(payload["account"])
        self.quality.restore(payload.get("quality", []))
        self.run_started_at = finite_time(payload["run_started_at"])
        if self.account_risk.anchor is None:
            raise ValueError("Persistent state has no account baseline")
        self.initial_book_ids = set(self.account_risk.anchor["book_ids"])
        excluded = payload["excluded"]
        if not isinstance(excluded, list) or not all(isinstance(pair, str) and pair.endswith("-USDT") for pair in excluded):
            raise ValueError("Invalid persistent pair exclusions")
        self.portfolio.excluded = set(excluded)
        valid_pair = lambda pair: isinstance(pair, str) and pair.endswith("-USDT")
        if not isinstance(payload["cooldowns"], dict) or not all(valid_pair(p) for p in payload["cooldowns"]):
            raise ValueError("Invalid persistent cooldowns")
        self.portfolio.cooldowns = {p: finite_time(t) for p, t in payload["cooldowns"].items()}
        circuit = payload["circuit"]
        if (not isinstance(circuit["seen"], list) or not all(valid_pair(p) for p in circuit["seen"])
                or not isinstance(circuit["events"], list)
                or not all(isinstance(row, list) and len(row) == 2 and valid_pair(row[0]) for row in circuit["events"])):
            raise ValueError("Invalid persistent loss circuit")
        self.loss_circuit.events = deque((p, finite_time(t)) for p, t in circuit["events"])
        self.loss_circuit.seen = set(circuit["seen"])
        self.loss_circuit.until = finite_time(circuit["until"])
        self.loss_circuit.healthy_since = self.loss_circuit.last_check = None
        if self.config.recovery.enabled and "recovery" in payload:
            self.expected_positions = self.recovery.restore(payload["recovery"], self.portfolio, self.inventory)
            if not recovery_pairs(payload).issubset(self.contracts):
                raise ValueError("Checkpoint-owned contracts are absent from startup subscriptions")
            for client_id, row in self.recovery.orders.items():
                intent = intent_from_json(row["intent"])
                order = TrackedOrder(intent, intent.amount - self.recovery.filled(client_id), row["submitted_at"])
                self.terminal_orders[client_id] = order
            self.entry_allowed = False
            self._restart_order_ids = set(self.recovery.orders)
            for pair, state in payload["recovery"].get("gamma_states", {}).items():
                if pair not in self.portfolio.slots:
                    raise ValueError("Persistent Gamma has no owning slot")
                components = {name: finite(value) for name, value in state["components"].items()}
                if any(name not in ("volatility", "depth", "micro", "loss") or not ZERO <= value <= 1
                       for name, value in components.items()):
                    raise ValueError("Invalid persistent Gamma stress")
                bounds = self.config.gamma_control
                value = max(bounds.minimum, min(bounds.maximum, finite(state["value"])))
                target = max(bounds.minimum, min(bounds.maximum, finite(state["target"])))
                self.gamma.states[pair] = GammaState(value, target, min(finite_time(state["updated_at"]), self.account_risk.last_at), components)
            if self.config.dry_run:
                if self.recovery.orders or any(self.expected_positions.values()):
                    raise ValueError("Dry-run checkpoint unexpectedly contains execution ownership")
                self.recovery.phase, self.recovery.reason = "ready", "dry-run state restored; no execution recovery"
            elif self.config.adaptive.enabled and self.config.adaptive.require_correlation and len(self.portfolio.slots) > 1:
                self._recovery_correlation_pairs = set(self.portfolio.slots)
            if self.config.candidate_pairs:
                for pair in self.portfolio.slots:
                    if pair not in self.config.candidate_pairs:
                        self.portfolio.retire(pair, "recovered contract removed from configured universe", self.account_risk.last_at)

    def _checkpoint(self):
        if self.risk_store is None or self.account_risk.anchor is None:
            return not self.persistence_error
        payload = dict(scope="gate_io_perpetual/usdt/classic", dry_run=self.config.dry_run,
                       run_started_at=self.run_started_at, account=self.account_risk.dump(),
                       excluded=sorted(self.portfolio.excluded), cooldowns=self.portfolio.cooldowns.copy(),
                       circuit=dict(events=list(self.loss_circuit.events), seen=sorted(self.loss_circuit.seen),
                                    until=self.loss_circuit.until))
        payload["quality"] = self.quality.dump(self.current_timestamp)
        if self.config.recovery.enabled:
            payload["recovery"] = self.recovery.dump(self.expected_positions, self.portfolio, self.inventory)
            payload["recovery"]["gamma_states"] = {p: plain(asdict(state)) for p, state in self.gamma.states.items()
                                                   if p in self.portfolio.slots}
        try:
            if self.cash_journal is not None:
                for slot in self.portfolio.slots.values():
                    self.cash_journal.begin_cycle(slot.pair, slot.selected_at)
                for row in self.recovery.orders.values():
                    slot = self.portfolio.slots.get(row["intent"]["pair"])
                    if slot is not None:
                        for trade_id in row["fills"]:
                            self.cash_journal.own_trade(slot.pair, trade_id, slot.selected_at)
            if payload != self._saved_payload or self.persistence_error:
                self.risk_store.save(payload)
                self._saved_payload = payload
            self.persistence_error = ""
            return True
        except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
            self.persistence_error = f"risk checkpoint failed: {exc}"
            self._cancel_opening_orders()
            self.logger().error(self.persistence_error)
            return False

    def _global_exit(self, now):
        if not self.account_risk.latched:
            return
        for pair in set(self.portfolio.slots) | {p for p, a in self.positions.items() if a != 0}:
            if pair not in self.portfolio.slots:
                from hummingbot.strategy.gate_avellaneda.core import Slot
                self.portfolio.slots[pair] = Slot(pair, selected_at=now)
            self.portfolio.retire(pair, "account stop: " + self.account_risk.reason, now)
            self.portfolio.slots[pair].force_market = True
        self._cancel_opening_orders()
        if not self._global_exit_started:
            self.logger().warning(f"Account stop latched: {self.account_risk.reason}; exiting all managed positions")
            self.recorder.record("account_stop", now, reason=self.account_risk.reason, risk=self.account_risk.dump())
            self._global_exit_started = True

    def on_tick(self):
        if self.stopping:
            return
        now = self.current_timestamp
        if self.refresh_task is None or self.refresh_task.done():
            if now >= self.next_account:
                self.refresh_task = asyncio.create_task(self._refresh())
                self.next_account = now + self.config.account_refresh_seconds
        if self.connector.network_status != NetworkStatus.CONNECTED or self.halt_reason:
            self.recorder.sample(now, {}, 0)
            self._cancel_all()
            return
        self._kline_tick(now)
        self._heartbeat_tick(now)
        self._sample()
        snapshots = self._snapshots()
        self.last_markets = snapshots
        self.recorder.sample(now, snapshots, self.config.micro.max_age if self.config.micro.enabled else self.config.risk.max_age)
        if not self._opening_ready():
            self._cancel_opening_orders()
        fresh = self._account_fresh()
        if self.recovery.recovering and not self._recovery_verified:
            return
        if not fresh:
            if self.config.adaptive.enabled:
                self.loss_circuit.allow_open(now, False, self.config.risk.max_age)
            # Once an account snapshot expires, outstanding orders are also
            # withdrawn. After a fill, wait for REST reconciliation before quoting.
            if now - self.account_at > self.config.risk.max_age:
                self._cancel_all()
            return
        for pair in list(self.portfolio.slots):
            if self.portfolio.stop_loss(pair, self._pair_pnl(pair), now):
                self.logger().warning(self.portfolio.slots[pair].reason + f"; exiting {pair} at market")
                self._cancel_pair(pair)
                if self.config.adaptive.enabled:
                    self.loss_circuit.record(pair, now)
                # An already-flat loss exit can be replaced on this tick.
                self.next_monitor = 0
                self.recorder.record("pair_stop", now, pair=pair, pnl=self._pair_pnl(pair))
        self._global_exit(now)
        self._checkpoint()
        previous_entry_allowed = self.entry_allowed
        self.entry_allowed = self._opening_ready()
        if self.config.adaptive.enabled:
            for pair, market in snapshots.items():
                signal = self.signals.get(pair) if self.config.micro.enabled and self.config.micro.mode == "protect" else None
                self.gamma.update(market, self._pair_pnl(pair) if pair in self.portfolio.slots else ZERO, now, signal)
            healthy = any(self._entry_check(m) is None for m in snapshots.values())
            allowed = self.loss_circuit.allow_open(now, healthy, self.config.micro.max_age if self.config.micro.enabled
                                                   else self.config.risk.max_age)
            self.entry_allowed = self.entry_allowed and allowed
            if not self.entry_allowed:
                self._cancel_opening_orders()
            for pair in self.portfolio.slots:
                age = self.inventory.age(pair, self.positions.get(pair, ZERO), now)
                market = snapshots.get(pair)
                evidence = self._quality_snapshot(market).get(str(self.positions.get(pair, ZERO) > 0), {}) if market else {}
                if (age >= self.config.quality_control.soft_exit_age_seconds and self._pair_pnl(pair) < 0
                        and evidence.get("ready") and evidence.get("block")):
                    self.portfolio.retire(pair, "persistent adverse execution quality with losing inventory", now)
                    self._cancel_opening_orders(pair)
                if age >= self.config.adaptive.max_hold_seconds:
                    self.portfolio.retire(pair, "maximum inventory holding time", now)
        if previous_entry_allowed != self.entry_allowed:
            self.next_monitor = 0
        self._micro_telemetry(now)
        if now >= self.next_monitor:
            before = {(p, s.state) for p, s in self.portfolio.slots.items()}
            self.portfolio.evaluate(snapshots, now, fresh, self.positions,
                                    self.exchange_open_pairs | {o.intent.pair for o in self.orders.values()}
                                    | ({row["intent"]["pair"] for row in self.recovery.orders.values()}
                                       if self.config.recovery.enabled else set()),
                                    allow_entries=self.equity > self.config.risk.reserve and self.entry_allowed,
                                    entry_check=self._entry_check if self.config.adaptive.enabled else None,
                                    candidate_filter=self._candidate_filter if self.config.adaptive.enabled else None,
                                    on_release=self._release_slot,
                                    candidate_score=self._candidate_score if self.config.adaptive.enabled else None)
            after = {(p, s.state) for p, s in self.portfolio.slots.items()}
            if before != after:
                self.logger().info(f"Portfolio slots: {sorted(after)}")
            self.next_monitor = now + self.config.monitor_seconds
        if self.equity <= self.config.risk.reserve:
            for pair in self.portfolio.slots:
                self.portfolio.retire(pair, "equity reached reserve floor", now)
        intents = []
        signed_quote, pending_exposure, pair_caps = self._exposure_inputs(snapshots)
        joint_breach = False
        if self.config.adaptive.enabled:
            longs, shorts = exposure_bounds(signed_quote, pending_exposure)
            pairs = set(longs) | set(shorts)
            joint_breach = (sum((max(longs.get(p, ZERO), shorts.get(p, ZERO)) for p in pairs), ZERO)
                            > self.config.adaptive.max_gross_quote
                            or max(sum(longs.values(), ZERO), sum(shorts.values(), ZERO))
                            > self.config.adaptive.max_directional_quote
                            or any(max(longs.get(p, ZERO), shorts.get(p, ZERO)) > pair_caps.get(p, ZERO) for p in pairs))
            if joint_breach:
                self._cancel_opening_orders()
        for pair, slot in list(self.portfolio.slots.items()):
            market = snapshots.get(pair)
            signal = self.signals.get(pair)
            protecting = self.config.micro.enabled and self.config.micro.mode == "protect"
            if protecting and signal and signal.ready and signal.retire and slot.state == "active":
                self.portfolio.retire(pair, "persistent microstructure anomaly: " + signal.reason, now)
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
            if market is None:
                self._cancel_pair(pair)
                continue
            if self.config.adaptive.enabled:
                if market.close_rejection(self.config.risk, now):
                    self._cancel_pair(pair)
                    continue
                plan = quote_plan(
                    market, self.positions.get(pair, ZERO),
                    self.inventory.age(pair, self.positions.get(pair, ZERO), now),
                    self.config.risk, self.config.adaptive, now, signal if protecting else None,
                    allow_open=self.entry_allowed and not joint_breach and self._recovered_pair_open_ready(pair),
                    blocked_side=self.inventory.blocked_side(pair, now),
                    gamma=self.gamma.value(pair), quality=self._quality_snapshot(market),
                    exit_cost_bps=self._exit_cost_bps(market), calibration=self._calibration_snapshot(pair),
                )
                self.plans[pair] = plan
                extra = self._calibration_extra(market)
                if extra > self.last_calibration_limits.get(pair, ZERO):
                    self._cancel_opening_orders(pair)
                    self.next_quote[pair] = 0
                quality_now = self._quality_snapshot(market)
                previous_quality = self.last_quality_limits.get(pair, {})
                if any((row.get("block", False) and not previous_quality.get(side, {}).get("block", False))
                       or Decimal(row.get("penalty_bps", "0")) > Decimal(previous_quality.get(side, {}).get("penalty_bps", "0"))
                       or Decimal(row.get("size_scale", "1")) < Decimal(previous_quality.get(side, {}).get("size_scale", "1"))
                       for side, row in quality_now.items()):
                    self._cancel_opening_orders(pair)
                    self.next_quote[pair] = 0
                self.recorder.decision(now, market, self.positions.get(pair, ZERO),
                                       self.inventory.age(pair, self.positions.get(pair, ZERO), now),
                                       self.config.risk, self.config.adaptive, signal if protecting else None,
                                       self.entry_allowed and not joint_breach and self._recovered_pair_open_ready(pair), self.inventory.blocked_side(pair, now),
                                       self.gamma.value(pair), plan, quality=self._quality_snapshot(market),
                                       exit_cost_bps=self._exit_cost_bps(market),
                                       calibration=self._calibration_snapshot(pair),
                                       shadow=self._calibration_shadow(market, protecting, joint_breach))
                opening = any(not q.close for q in plan.intents)
                existing_close = any(o.intent.pair == pair and o.intent.close for o in self.orders.values())
                if not opening:
                    self._cancel_opening_orders(pair)
                    if not existing_close:
                        self.next_quote[pair] = 0
                limits = self._control_limits(pair, plan)
                previous_limits = self.last_control_limits.get(pair)
                if previous_limits and (limits[0] < previous_limits[0] or limits[1] < previous_limits[1]
                                        or limits[2] > previous_limits[2] or limits[3] < previous_limits[3]
                                        or limits[4] > previous_limits[4] or limits[5] > previous_limits[5]
                                        or (limits[6] is not None and limits[6] != previous_limits[6])):
                    if opening:
                        self._cancel_pair(pair)
                        self.next_quote[pair] = 0
                if self._gamma_reprice_needed(pair, plan, market, now):
                    self.next_quote[pair] = 0
            elif market.rejection(self.config.risk, now):
                self._cancel_pair(pair)
                continue
            if protecting and signal and signal.ready:
                previous = self.last_micro_limits.get(pair, (Decimal(1), Decimal(1), False))
                stricter = (signal.buy_scale < previous[0] or signal.sell_scale < previous[1]
                            or (signal.paused and not previous[2]))
                if stricter:
                    if self.config.adaptive.enabled and not opening:
                        self._cancel_opening_orders(pair)
                        if not existing_close:
                            self.next_quote[pair] = 0
                    else:
                        self._cancel_pair(pair)
                        self.next_quote[pair] = 0
                previous_reference = self.quote_references.get(pair)
                threshold = max(2 * market.tick, market.mid * self.config.micro.reprice_bps / Decimal(10000))
                if (previous_reference is not None and abs((plan.reference if self.config.adaptive.enabled else signal.reference) - previous_reference) >= threshold
                        and now - self.last_submission.get(pair, 0) >= self.config.micro.min_reprice_seconds):
                    self.next_quote[pair] = 0
            if now < self.next_quote.get(pair, 0):
                continue
            if any(o.intent.pair == pair for o in self.orders.values()):
                if self.config.adaptive.enabled and self._keep_quotes(pair, plan.intents, market, now):
                    self.next_quote[pair] = now + self.config.quote_refresh_seconds
                    self.recorder.record("quote_retained", now, pair=pair)
                    continue
                self._cancel_pair(pair)
                continue
            if pair in self.exchange_open_pairs:
                self._cancel_pair(pair)
                continue
            quotes = plan.intents if self.config.adaptive.enabled else avellaneda_quotes(
                market, self.positions.get(pair, ZERO), self.config.risk,
                reference_price=signal.reference if protecting else None,
                buy_scale=signal.buy_scale if protecting else Decimal(1),
                sell_scale=signal.sell_scale if protecting else Decimal(1),
            )
            if protecting and signal.paused:
                quotes = [q for q in quotes if q.close]
            intents.extend(quotes)
        if not intents:
            self._checkpoint()
            return
        checkpoint_ok = self._checkpoint()
        if not self.entry_allowed or not self._opening_ready() or not checkpoint_ok:
            intents = [intent for intent in intents if intent.close]
        outstanding = {}
        pending = list(self.orders.values())
        pending.extend(o for oid, o in self.terminal_orders.items() if oid in self.exchange_open_ids)
        for order in pending:
            if not order.intent.close:
                pair = order.intent.pair
                outstanding[pair] = outstanding.get(pair, ZERO) + order.remaining * order.intent.price * Decimal("1.01")
        if self.config.adaptive.enabled:
            intents = limit_exposure(intents, signed_quote, pending_exposure, snapshots, pair_caps, self.config.adaptive)
        selected = allocate(intents, self.position_quote, outstanding, self.available, self.equity, self.config.risk)
        if not self.recorder.record("allocation", now, revision=self.revision, intents=selected,
                                    opening_allowed=self.entry_allowed and self._opening_ready()):
            selected = [intent for intent in selected if intent.close]
        if self.config.dry_run:
            self.last_quotes = {p: [i for i in selected if i.pair == p] for p in self.portfolio.slots}
            for pair, quotes in self.last_quotes.items():
                if quotes:
                    self.logger().info(f"DRY RUN {pair}: {quotes}")
                    self.next_quote[pair] = now + self.config.quote_refresh_seconds
                    self._remember_quote_signal(pair)
            return
        # One submission batch per reconciled account revision. This prevents
        # the second pair/tick reusing collateral before the exchange updates it.
        selected = [intent for intent in selected if intent.close or self._heartbeat_ready(intent.pair)]
        if not selected or self.used_revision == self.revision:
            return
        self.used_revision = self.revision
        for intent in selected:
            self._submit(intent)

    def _sample(self):
        for pair in self.contracts:
            try:
                book = self.connector.get_order_book(pair)
                if pair not in self.indicators:
                    delegate = OrderBookAssetPriceDelegate(self.connector, pair)
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
                    if self.config.adaptive.enabled:
                        if self.config.micro.enabled:
                            observed = self.signal_feed.signal(pair, self.current_timestamp)
                            if observed.ready:
                                self.return_history.observe(pair, self.current_timestamp, (observed.bid + observed.ask) / 2)
                        elif pair in self.books:
                            raw, observed_at = self.books[pair]
                            if self.current_timestamp - observed_at <= self.config.risk.max_age:
                                price = (Decimal(str(raw["bids"][0]["p"])) + Decimal(str(raw["asks"][0]["p"]))) / 2
                                self.return_history.observe(pair, self.current_timestamp, price)
            except (KeyError, ValueError, ArithmeticError):
                continue

    def _snapshots(self):
        snapshots = {}
        for pair in self.contracts:
            try:
                contract = self.contracts[pair]
                multiplier = Decimal(str(contract["quanto_multiplier"]))
                if self.config.micro.enabled:
                    signal = self.signal_feed.signal(pair, self.current_timestamp)
                    self.signals[pair] = signal
                    if not signal.ready:
                        continue
                    bid, ask = signal.bid, signal.ask
                    bid_depth, ask_depth = signal.bid_depth, signal.ask_depth
                    observed_at = signal.observed_at
                else:
                    raw, observed_at = self.books[pair]
                    bids, asks = raw["bids"], raw["asks"]
                    bid, ask = Decimal(str(bids[0]["p"])), Decimal(str(asks[0]["p"]))
                mid = (bid + ask) / 2
                if not mid.is_finite() or mid <= 0:
                    continue
                if not self.config.micro.enabled:
                    def depth(rows):
                        return sum((Decimal(str(r["p"])) * abs(Decimal(str(r["s"]))) * multiplier
                                    for r in rows if abs(Decimal(str(r["p"])) / mid - 1) <= Decimal("0.001")), ZERO)
                    bid_depth, ask_depth = depth(bids), depth(asks)
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
                    Decimal(str(ticker["volume_24h_quote"])), bid_depth, ask_depth,
                    Decimal(str(ticker["funding_rate_indicative"])), trend, max(self.config.maker_fee_floor, fee),
                    multiplier, Decimal(str(contract["order_price_round"])),
                    multiplier * Decimal(str(contract["order_size_min"])), observed_at,
                    vol.is_sampling_buffer_full and intensity.is_sampling_buffer_full,
                    not contract.get("in_delisting", True),
                    funding_interval=float(contract.get("funding_interval", 28800)),
                    funding_next_at=float(contract.get("funding_next_apply", 0)),
                    taker_fee=max(Decimal("0.0005"), Decimal(str(contract.get("taker_fee_rate", "0.0005"))),
                                  self.connector.get_fee(base, quote, OrderType.MARKET, TradeType.BUY,
                                                         self.config.risk.order_quote / mid, ask).percent),
                )
            except (KeyError, IndexError, ValueError, ArithmeticError):
                continue
        return snapshots

    def _quality_regime(self, market):
        if market is None:
            return "calm"
        signal = self.signals.get(market.pair)
        return "stressed" if ((signal is not None and (signal.paused or signal.retire or signal.book_flow_conflict))
                              or market.volatility / market.mid > self.config.adaptive.volatility_reference) else "calm"

    def _quality_snapshot(self, market):
        return self.quality.snapshot(market.pair, self._quality_regime(market), self.current_timestamp)

    def _kline_tick(self, now):
        if (self.config.kline_volatility.enabled and now >= self.next_kline
                and (self.kline_task is None or self.kline_task.done())):
            self.next_kline = now + self.config.kline_volatility.refresh_seconds
            self.kline_task = asyncio.create_task(self._refresh_klines())

    async def _refresh_klines(self):
        # Public data never holds the private account refresh or CLOSE path.
        cfg = self.config.kline_volatility
        semaphore = asyncio.Semaphore(4)

        async def fetch(pair):
            async with semaphore:
                try:
                    rows = await asyncio.wait_for(self.connector._api_get(
                        path_url=C.CANDLESTICKS_PATH_URL,
                        params={"contract": pair.replace("-", "_"), "interval": cfg.interval,
                                "limit": cfg.lookback_bars + 1, "timezone": "utc0"},
                        is_auth_required=False, limit_id=C.CANDLESTICKS_PATH_URL),
                        timeout=cfg.request_timeout_seconds)
                    value = self.kline_volatility.update(pair, rows, self.current_timestamp)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    value = self.kline_volatility.fail(pair, self.current_timestamp, str(exc))
                self.recorder.record("kline_calibration", self.current_timestamp, observation=value, settings=cfg)

        await asyncio.gather(*(fetch(pair) for pair in self.contracts))

    def _calibration_snapshot(self, pair):
        return self.kline_volatility.snapshot(pair, self.current_timestamp)

    def _calibration_extra(self, market):
        base = self.config.adaptive.spread_multiplier * (
            self.gamma.value(market.pair) * market.volatility / market.mid / 2
            + (1 + 1 / (market.kappa * market.mid)).ln())
        return calibrated_half_spread(base, self._calibration_snapshot(market.pair)) - base

    def _calibration_shadow(self, market, protecting, joint_breach):
        calibration = self._calibration_snapshot(market.pair)
        if not calibration or calibration["mode"] != "observe" or not calibration["ready"]:
            return None
        pair, now = market.pair, self.current_timestamp
        return quote_plan(market, self.positions.get(pair, ZERO),
                          self.inventory.age(pair, self.positions.get(pair, ZERO), now),
                          self.config.risk, self.config.adaptive, now,
                          self.signals.get(pair) if protecting else None,
                          allow_open=self.entry_allowed and not joint_breach and self._recovered_pair_open_ready(pair),
                          blocked_side=self.inventory.blocked_side(pair, now), gamma=self.gamma.value(pair),
                          quality=self._quality_snapshot(market), exit_cost_bps=self._exit_cost_bps(market),
                          calibration=dict(calibration, mode="protect"))

    def _exit_cost_bps(self, market):
        cfg = self.config.quality_control
        if not cfg.enabled or cfg.mode != "protect":
            return ZERO
        return cfg.emergency_exit_probability * (max(ZERO, market.taker_fee - market.maker_fee) * 10000
                                                 + cfg.exit_slippage_bps)

    def _candidate_score(self, market):
        # Bounded tie-breakers; observed markouts are not a profit forecast.
        quality = self._quality_snapshot(market)
        evidence = [Decimal(row["mean_net_bps"]) for row in quality.values() if row.get("ready")]
        plan = quote_plan(market, ZERO, 0, self.config.risk, self.config.adaptive, self.current_timestamp,
                          self.signals.get(market.pair) if self.config.micro.enabled and self.config.micro.mode == "protect" else None,
                          gamma=self.gamma.value(market.pair), quality=quality, exit_cost_bps=self._exit_cost_bps(market),
                          calibration=self._calibration_snapshot(market.pair))
        margins = [(abs(q.price - plan.reference) / plan.reference - 2 * market.maker_fee) * 10000
                   for q in plan.intents if not q.close]
        margin = min(margins) - self._exit_cost_bps(market) if margins else ZERO
        return market.score() + float(max(Decimal(-10), min(Decimal(10), margin))) / 10 + (
            float(max(Decimal(-10), min(Decimal(10), min(evidence)))) / 10 if evidence else 0)

    def _keep_quotes(self, pair, proposed, market, now):
        cfg = self.config.execution
        if not cfg.retain_quotes:
            return False
        working = [o for o in self.orders.values() if o.intent.pair == pair]
        targets = {(q.buy, q.close): q for q in proposed}
        if not working or len(working) != len(targets):
            return False
        threshold = max(2 * market.tick, market.mid * cfg.reprice_bps / 10000)
        for order in working:
            target = targets.get((order.intent.buy, order.intent.close))
            if (target is None or order.cancel_at or order.intent.market or target.market
                    or not 0 <= now - order.submitted_at < cfg.max_quote_age_seconds
                    or abs(order.intent.price - target.price) >= threshold
                    or not target.amount * (1 - cfg.size_tolerance) <= order.remaining <= target.amount):
                return False
        return True

    def _heartbeat_ready(self, pair):
        return (self.config.dry_run or not self.config.deadman.enabled
                or self.heartbeat_deadlines.get(pair, 0) > self.current_timestamp + self.config.deadman.renew_seconds)

    def _heartbeat_tick(self, now):
        if self.config.dry_run or not self.config.deadman.enabled:
            return
        pairs = set(self.portfolio.slots) | {o.intent.pair for o in self.orders.values()}
        for pair in pairs:
            if not self._heartbeat_ready(pair):
                self._cancel_opening_orders(pair)
        if (not pairs or not self._account_fresh() or not self._opening_ready()
                or (self.heartbeat_task is not None and not self.heartbeat_task.done())):
            return
        if now >= self.next_heartbeat or any(p not in self.heartbeat_deadlines for p in pairs):
            self.next_heartbeat = now + self.config.deadman.renew_seconds
            self.heartbeat_task = asyncio.create_task(self._renew_heartbeat(pairs))

    async def _renew_heartbeat(self, pairs):
        if self.config.dry_run or not self.config.deadman.enabled:
            return
        cfg = self.config.deadman
        for pair in sorted(pairs):
            sent_at = self.current_timestamp
            try:
                result = await asyncio.wait_for(self.connector._api_post(
                    path_url=C.COUNTDOWN_CANCEL_PATH_URL,
                    data={"timeout": cfg.timeout_seconds, "contract": pair.replace("-", "_")},
                    is_auth_required=True), timeout=cfg.renew_seconds)
                deadline = float(result["triggerTime"]) / 1000
                if (not math.isfinite(deadline) or deadline <= self.current_timestamp + cfg.renew_seconds
                        or deadline > sent_at + cfg.timeout_seconds + 5):
                    raise ValueError("Invalid exchange cancellation countdown acknowledgement")
                if not self._heartbeat_ready(pair):
                    self.next_quote[pair] = 0
                self.heartbeat_deadlines[pair] = deadline
                self.heartbeat_error = ""
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.heartbeat_deadlines.pop(pair, None)
                self.heartbeat_error = f"exchange countdown unavailable: {exc}"
                self._cancel_opening_orders(pair)
                self.recorder.record("countdown_error", self.current_timestamp, pair=pair, reason=str(exc))

    def _entry_check(self, market):
        signal = self.signals.get(market.pair) if self.config.micro.enabled and self.config.micro.mode == "protect" else None
        if signal is not None and signal.paused:
            return "microstructure opening paused"
        return entry_rejection(market, self.config.risk, self.config.adaptive, self.current_timestamp, signal,
                               gamma=self.gamma.value(market.pair), quality=self._quality_snapshot(market),
                               exit_cost_bps=self._exit_cost_bps(market),
                               calibration=self._calibration_snapshot(market.pair))

    def _gamma_reprice_needed(self, pair, plan, market, now):
        if (plan.gamma == self.last_gamma_values.get(pair, plan.gamma)
                or now - self.last_submission.get(pair, now) < self.config.gamma_control.min_reprice_seconds):
            return False
        threshold = max(2 * market.tick, market.mid * self.config.gamma_control.reprice_bps / Decimal(10000))
        targets = {(intent.buy, intent.close): intent.price for intent in plan.intents}
        working = self.last_quotes.get(pair, ()) if self.config.dry_run else (
            o.intent for o in self.orders.values() if o.intent.pair == pair)
        return any(not intent.market and (intent.buy, intent.close) in targets
                   and abs(targets[intent.buy, intent.close] - intent.price) >= threshold for intent in working)

    def _candidate_filter(self, pair, selected):
        eligible = self.return_history.eligible(pair, selected, self.current_timestamp)
        if not eligible:
            self.selection_rejections[pair] = "correlation unavailable or too high"
        else:
            self.selection_rejections.pop(pair, None)
        return eligible

    def _recovered_pair_open_ready(self, pair):
        if pair not in self._recovery_correlation_pairs:
            return True
        others = [p for p, slot in self.portfolio.slots.items() if p != pair and slot.state == "active"]
        if self.return_history.eligible(pair, others, self.current_timestamp):
            self._recovery_correlation_pairs.discard(pair)
            self.selection_rejections.pop(pair, None)
            return True
        self.selection_rejections[pair] = "recovery correlation warmup or too high"
        return False

    def _exposure_inputs(self, snapshots):
        prices = {p: max(self.position_marks.get(p, ZERO), snapshots[p].mid if p in snapshots else ZERO)
                  for p in self.contracts}
        signed = {p: amount * prices[p] for p, amount in self.positions.items()}
        owned = dict(self.orders)
        owned.update((oid, o) for oid, o in self.terminal_orders.items() if oid in self.exchange_open_ids)
        pending = [Intent(o.intent.pair, o.intent.buy, o.remaining, max(o.intent.price, prices[o.intent.pair]),
                          o.intent.close, o.intent.market) for o in owned.values()]
        caps = {}
        for pair in self.contracts:
            market = snapshots.get(pair)
            valid = market and all(v.is_finite() for v in (market.mid, market.volatility, market.bid_depth, market.ask_depth))
            scale = exposure_scale(market, self.config.risk, self.config.adaptive) if valid and market.mid > 0 else Decimal(1)
            caps[pair] = self.config.risk.max_position_quote * scale
        return signed, pending, caps

    def _control_limits(self, pair, plan):
        return (plan.position_cap, plan.order_quote, {"normal": 0, "caution": 1, "reduce_only": 2, "paused": 3}[plan.stage],
                plan.confidence, plan.buy_premium, plan.sell_premium, self.inventory.blocked_side(pair, self.current_timestamp))

    def _cancel_opening_orders(self, pair=None):
        if self.config.dry_run:
            return
        pending = dict(self.orders)
        pending.update((oid, o) for oid, o in self.terminal_orders.items() if oid in self.exchange_open_ids)
        for order_id, order in pending.items():
            if (order.intent.close or (pair is not None and order.intent.pair != pair)
                    or (order.cancel_at and self.current_timestamp - order.cancel_at < 5)):
                continue
            self.cancel(self.connector_name, order.intent.pair, order_id)
            order.cancel_at = self.current_timestamp
            self.recorder.record("cancel_requested", self.current_timestamp, order_id=order_id, pair=order.intent.pair)

    def _remember_quote_signal(self, pair):
        self.last_submission[pair] = self.current_timestamp
        signal = self.signals.get(pair)
        if self.config.micro.enabled and signal and signal.ready:
            self.quote_references[pair] = self.plans[pair].reference if self.config.adaptive.enabled and pair in self.plans else signal.reference
            self.last_micro_limits[pair] = (signal.buy_scale, signal.sell_scale, signal.paused)
        if self.config.adaptive.enabled and pair in self.plans:
            self.last_control_limits[pair] = self._control_limits(pair, self.plans[pair])
            market = self.last_markets.get(pair)
            if market is not None:
                self.last_quality_limits[pair] = self._quality_snapshot(market)
                self.last_calibration_limits[pair] = self._calibration_extra(market)
            self.last_gamma_values[pair] = self.plans[pair].gamma

    def _micro_telemetry(self, now):
        if not self.config.micro.enabled or now < self.next_signal_log:
            return
        for pair in self.portfolio.slots:
            s = self.signals.get(pair)
            if s:
                self.logger().info("MICRO " + json.dumps(dict(
                    pair=pair, mode=self.config.micro.mode, ready=s.ready, reason=s.reason,
                    weighted_mid=str(s.weighted_mid), reference=str(s.reference), imbalance=str(s.imbalance),
                    flow=str(s.flow), flow_quote=str(s.flow_quote), bid_loss=str(s.bid_loss), ask_loss=str(s.ask_loss),
                    paused=s.paused, buy_scale=str(s.buy_scale), sell_scale=str(s.sell_scale),
                )))
        self.next_signal_log = now + 10

    async def _order_detail(self, client_id):
        row = self.recovery.orders[client_id]
        try:
            return await self.connector._api_get(
                path_url=C.ORDER_STATUS_PATH_URL.format(id=row["exchange_id"] or client_id),
                is_auth_required=True, limit_id=C.ORDER_STATUS_LIMIT_ID)
        except IOError as exc:
            if re.search(r'["\']label["\']\s*:\s*["\']ORDER_NOT_FOUND["\']', str(exc)):
                return None
            raise

    async def _order_fills(self, exchange_id, pair):
        result = []
        limit = 1000
        for offset in range(0, self.config.recovery.max_fills_per_order, limit):
            page = await self.connector._api_get(
                path_url=C.MY_TRADES_PATH_URL, is_auth_required=True, limit_id=C.MY_TRADES_PATH_URL,
                params={"order": exchange_id, "contract": pair.replace("-", "_"), "limit": limit, "offset": offset})
            result.extend(page)
            if len(page) < limit:
                return result
        raise ValueError("Order fill pagination limit reached; recovery cannot prove completeness")

    async def _sync_recovery_orders(self, client_ids, open_orders, cancel_old):
        """Verify order identity before cancellation; replay actual fills only.

        A terminal record is pruned only after order totals, fill IDs and a later
        independent REST position snapshot all agree. New market CLOSE orders
        issued during startup verification are excluded from the old-ID set.
        """
        client_ids = [cid for cid in client_ids if cid in self.recovery.orders]
        semaphore = asyncio.Semaphore(4)
        user = self.account_risk.anchor["user"]

        async def fetch(client_id):
            async with semaphore:
                raw = await self._order_detail(client_id)
                row = self.recovery.orders[client_id]
                intent = intent_from_json(row["intent"])
                if raw is None:
                    if self.recovery.filled(client_id) != 0:
                        raise ValueError("A previously filled strategy order is missing from Gate history")
                    if (any(o.get("text") == client_id for o in open_orders)
                            or self.current_timestamp - row["submitted_at"] < self.config.recovery.not_found_grace_seconds):
                        return client_id, None, {}, False
                    return client_id, None, {}, True
                multiplier = Decimal(str(self.contracts[intent.pair]["quanto_multiplier"]))
                expected_fill = self.recovery.validate_order(client_id, raw, multiplier, user)
                rows = await self._order_fills(row["exchange_id"], intent.pair)
                fills = self.recovery.trade_rows(client_id, rows, multiplier, self.current_timestamp)
                if sum((v["amount"] for v in fills.values()), ZERO) != expected_fill:
                    raise RuntimeError("Order status and trade totals disagree; waiting for settlement")
                return client_id, raw, fills, raw["status"] == "finished"

        results = await asyncio.gather(*(fetch(client_id) for client_id in client_ids))
        # All identities are checked before touching any exchange order.
        fills = [(cid, tid, value) for cid, raw, trades, done in results for tid, value in trades.items()
                 if tid not in self.recovery.orders[cid]["fills"]]
        previous_ages = self.inventory.opened_at.copy()
        self._replaying_fills = True
        try:
            for cid, tid, value in sorted(fills, key=lambda item: (item[2]["timestamp"], item[1])):
                intent = intent_from_json(self.recovery.orders[cid]["intent"])
                fee = None if value["fee"] is None else SimpleNamespace(
                    percent=ZERO, percent_token="USDT", flat_fees=[SimpleNamespace(token="USDT", amount=value["fee"])])
                event = SimpleNamespace(order_id=cid, exchange_trade_id=tid, amount=value["amount"],
                                        price=value["price"], timestamp=value["timestamp"], trade_fee=fee)
                self.did_fill_order(event)
                # Missing older callbacks cannot make an existing holding
                # appear younger. Conservative ages include the offline gap.
                if self.expected_positions.get(intent.pair, ZERO):
                    self.inventory.opened_at[intent.pair] = min(
                        self.inventory.opened_at.get(intent.pair, value["timestamp"]),
                        previous_ages.get(intent.pair, value["timestamp"]), value["timestamp"])
        finally:
            self._replaying_fills = False
        pending = False
        terminal = set()
        for cid, raw, trades, done in results:
            row = self.recovery.orders[cid]
            intent = intent_from_json(row["intent"])
            if done:
                row["terminal"] = True
                if any(o.get("text") == cid for o in open_orders):
                    pending = True
                else:
                    terminal.add(cid)
                continue
            pending = True
            if raw is not None and (cancel_old or row["terminal"]) and not (intent.close and intent.market):
                self._checkpoint()
                await self.connector._api_delete(
                    path_url=C.ORDER_STATUS_PATH_URL.format(id=row["exchange_id"]),
                    is_auth_required=True, limit_id=C.ORDER_DELETE_LIMIT_ID)
                self.recorder.record("recovery_cancel_requested", self.current_timestamp, order_id=cid)
        if self.recovery.recovering:
            self._recovery_pending_settlements = terminal
            if pending:
                self.recovery.wait("waiting for old order cancellation, market CLOSE completion, or submission grace")
        else:
            self._recovery_pending_settlements = terminal
            self._recovery_block_open = pending
        self._checkpoint()
        return not pending

    async def _cancel_old_submission_tasks(self, client_ids=None):
        tasks = getattr(self.connector, "_gate_portfolio_order_tasks", {})
        cancelled = []
        for client_id in self._restart_order_ids if client_ids is None else client_ids:
            task = tasks.get(client_id)
            if task is not None and not task.done():
                task.cancel()
                cancelled.append(task)
        if cancelled:
            await asyncio.gather(*cancelled, return_exceptions=True)

    async def _setup_recovered(self, positions, account):
        if account.get("in_dual_mode", False) or account.get("position_mode", "single") != "single":
            raise ValueError("Recovery requires the original one-way account mode")
        self.connector._perpetual_trading.set_position_mode(PositionMode.ONEWAY)
        by_pair = {p["contract"].replace("_", "-"): p for p in positions}
        for pair in self.contracts:
            if pair in by_pair:
                if Decimal(str(by_pair[pair].get("leverage", "0"))) != 1:
                    raise ValueError("Recovered position leverage is not confirmed 1x; no automatic leverage change")
            elif not self.account_risk.latched:
                success, message = await self.connector._set_trading_pair_leverage(pair, 1)
                if not success:
                    raise ValueError(f"Could not confirm 1x leverage for recovered flat contract {pair}: {message}")
            self.connector._perpetual_trading.set_leverage(pair, 1)

    async def _refresh(self):
        try:
            if self.run_started_at is None:
                self.run_started_at = self.current_timestamp
            epoch = self.account_epoch
            positions, account, open_orders = await asyncio.gather(
                self.connector._api_get(path_url=C.POSITION_INFORMATION_URL, is_auth_required=True),
                self.connector._api_get(path_url=C.USER_BALANCES_PATH_URL, is_auth_required=True),
                self._open_orders(),
            )
            if epoch != self.account_epoch:
                return
            if self.config.account_risk.enabled:
                self.account_risk.validate_account(account)
            if self.cash_journal is None and self.config.account_risk.enabled and self.config.account_risk.persist:
                try:
                    self.cash_journal = CashJournal(str(mode_path(self.config.account_risk.state_path, self.config.dry_run)) + ".cash.sqlite",
                                                    account["user"], self.run_started_at)
                except Exception as exc:
                    self.cash_error = f"cash journal unavailable: {exc}"
            cashflows = None
            try:
                if self.config.account_risk.enabled and self.config.account_risk.persist and self.cash_journal is None:
                    raise RuntimeError(self.cash_error)
                cashflows = await self._account_book()
                self.cash_error = ""
            except Exception as exc:
                self.cash_error = f"cash reconciliation pending: {exc}"
                self.logger().warning(self.cash_error)
                # Position/order ownership is still independently reconciled
                # below. A cash-history failure never authorizes new risk.
                cashflows = self.cashflows if isinstance(self.cashflows, AccountBookSnapshot) else AccountBookSnapshot()
            recovering = self.config.recovery.enabled and self.recovery.recovering and not self.config.dry_run
            if recovering:
                known = set(self.recovery.orders) | set(self.orders)
                if any(o.get("text") not in known for o in open_orders):
                    raise ValueError("Recovery found an unowned exchange order; manual reconciliation required")
                await self._cancel_old_submission_tasks()
                self._recovery_verified = False
                try:
                    if not await self._sync_recovery_orders(self._restart_order_ids, open_orders, True):
                        self.account_dirty = True
                        return
                except RuntimeError as exc:
                    self.recovery.wait(str(exc))
                    self.account_dirty = True
                    return
                if epoch != self.account_epoch:
                    self.recovery.wait("offline fills replayed; waiting for a new private position snapshot")
                    self.account_dirty = True
                    return
            elif self.config.recovery.enabled and self.initialized and not self.config.dry_run:
                terminal_ids = [cid for cid, row in self.recovery.orders.items() if row["terminal"]]
                self._recovery_block_open = bool(terminal_ids)
                if terminal_ids:
                    try:
                        await self._sync_recovery_orders(terminal_ids, open_orders, False)
                    except (IOError, RuntimeError) as exc:
                        self._recovery_block_open = True
                        self.logger().warning(f"Terminal order settlement pending: {exc}")
                    if epoch != self.account_epoch:
                        self.account_dirty = True
                        return
            nonzero = [p for p in positions if Decimal(str(p["size"])) != 0]
            if not self.initialized and not self.config.dry_run:
                if not recovering and (nonzero or open_orders):
                    raise ValueError("Start requires a flat futures account with no exchange open orders; reconcile existing exposure first")
                if not recovering and not self.account_risk.latched:
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
                    if recovering:
                        self.recovery.wait("exchange positions differ from persisted and replayed strategy fills; manual review if persistent")
                    return
                if recovering and not self.initialized:
                    await self._setup_recovered(nonzero, account)
                    if epoch != self.account_epoch:
                        return
            self.positions = converted
            if self.config.adaptive.enabled:
                self.inventory.reconcile(converted, self.current_timestamp)
            self.position_quote = position_quote
            self.position_marks = position_marks
            self.unrealised_pnl = unrealised_pnl
            if (self.account_risk.anchor is None if self.config.account_risk.enabled else not self.initialized) and not self.cash_error:
                # Ignore historical records already present before any bot orders.
                self.initial_book_ids = set(cashflows.records) if self.config.account_risk.enabled else set(cashflows)
            self.cashflows = cashflows
            self.available = Decimal(str(account["available"]))
            self.equity = Decimal(str(account["total"])) + Decimal(str(account.get("unrealised_pnl", "0")))
            if not self.available.is_finite() or not self.equity.is_finite():
                raise ValueError("Invalid account balances")
            if self.config.account_risk.enabled and (self.account_risk.anchor is not None or not self.cash_error):
                self.account_risk.observe(account, cashflows.records, sum(unrealised_pnl.values(), ZERO),
                                          self.current_timestamp, self.config.risk.reserve,
                                          cash_totals=getattr(cashflows, "cash_totals", None) if self.account_risk.anchor is not None else None)
                if not self.cash_error:
                    try:
                        if self.cash_journal is not None and self.account_risk.reconciliation_error:
                            self.cash_journal.request_rescan()
                        self._check_closed_losses()
                    except Exception as exc:
                        self.cash_error = f"closed-cycle reconciliation pending: {exc}"
                self._checkpoint()
                if not self._opening_ready():
                    self._cancel_opening_orders()
            if self.config.recovery.enabled:
                if recovering:
                    self._recovery_verified = True  # Position and order ownership are proven; CLOSE may proceed.
                    if self.account_risk.reconciliation_error or self.persistence_error or self.cash_error:
                        self.recovery.wait(self.cash_error or self.account_risk.reconciliation_error or self.persistence_error)
                    else:
                        self._recovery_verified = True
                        fingerprint = json.dumps(dict(positions={p: str(a) for p, a in sorted(converted.items())},
                                                       open_ids=sorted(o.get("text", "") for o in open_orders),
                                                       cash_ids=getattr(cashflows, "revision", sorted(cashflows.records))), sort_keys=True)
                        if self.recovery.confirm(fingerprint, self.current_timestamp):
                            self.recovery.phase, self.recovery.reason = "ready", "exchange reconciliation complete"
                            self.recorder.record("recovery_complete", self.current_timestamp, positions=converted)
                if not self.recovery.recovering or self._recovery_verified:
                    durable = self._checkpoint()
                    for client_id in self._recovery_pending_settlements:
                        if durable:
                            self.recovery.orders.pop(client_id, None)
                        self.orders.pop(client_id, None)
                        self.terminal_orders.pop(client_id, None)
                        self._restart_order_ids.discard(client_id)
                        if hasattr(self.connector, "stop_tracking_order"):
                            self.connector.stop_tracking_order(client_id)
                    self._recovery_pending_settlements.clear()
                self._checkpoint()
            self.recorder.record("account", self.current_timestamp, revision=self.revision + 1,
                                 equity=self.equity, available=self.available, positions=self.positions,
                                 risk=self.account_risk.dump(), cash_totals=self.account_risk.cash_totals,
                                 reconciliation_error=self.account_risk.reconciliation_error)
            self.exchange_open_pairs = {o["contract"].replace("_", "-") for o in open_orders}
            self.exchange_open_ids = {o.get("text") for o in open_orders}
            self.account_at = self.current_timestamp
            self.account_dirty = False
            self.initialized = True
            self.revision += 1
            if self.current_timestamp >= self.next_public:
                try:
                    contracts, tickers = await asyncio.gather(
                        self.connector._api_get(path_url="futures/usdt/contracts", limit_id=C.NETWORK_CHECK_PATH_URL),
                        self.connector._api_get(path_url=C.TICKER_PATH_URL),
                    )
                    contract_map = {c["name"].replace("_", "-"): c for c in contracts}
                    for pair in self.contracts:
                        self.contracts[pair] = contract_map.get(pair, dict(self.contracts[pair], in_delisting=True))
                    self.tickers = {t["contract"].replace("_", "-"): t for t in tickers}
                    self.next_public = self.current_timestamp + self.config.monitor_seconds
                except Exception as exc:
                    # Public screening failure must not invalidate authenticated
                    # positions and block an already-triggered market loss exit.
                    self.logger().warning(f"Public screening refresh failed: {exc}")
                    self.tickers = {}  # No new entry economics until recovery.
            resync_needed = self.config.micro.enabled and any(self.signal_feed.needs_snapshot(p) for p in self.contracts)
            if resync_needed or self.current_timestamp >= self.next_books:
                semaphore = asyncio.Semaphore(4)

                async def fetch_book(pair):
                    if self.config.micro.enabled and not self.signal_feed.needs_snapshot(pair):
                        return
                    async with semaphore:
                        try:
                            raw = await self.connector._api_get(
                                path_url=C.ORDER_BOOK_PATH_URL, params={"contract": pair.replace("-", "_"), "limit": 100, "with_id": "true"},
                            )
                            self.books[pair] = (raw, self.current_timestamp)
                            if self.config.micro.enabled:
                                self.signal_feed.seed_snapshot(pair, raw, self.current_timestamp)
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
        if self.cash_journal is not None:
            try:
                return self.cash_journal.pair_total(pair, selected_at, self.initial_book_ids) + self.unrealised_pnl.get(pair, ZERO)
            except Exception as exc:
                self.cash_error = f"pair cash lookup failed: {exc}"
        settled = sum((change for record_id, (p, timestamp, change) in self.cashflows.items()
                       if p == pair and timestamp >= selected_at and record_id not in self.initial_book_ids), ZERO)
        return settled + self.unrealised_pnl.get(pair, ZERO)

    def _release_slot(self, slot, now):
        if self.cash_journal is None:
            return not self.cash_error
        try:
            self.cash_journal.end_cycle(slot.pair, slot.selected_at, now)
            return True
        except Exception as exc:
            self.cash_error = f"cannot archive slot settlement: {exc}"
            return False

    def _check_closed_losses(self):
        if self.cash_journal is None:
            return
        for pair, start, ended in self.cash_journal.closed_cycles():
            if pair in self.portfolio.excluded:
                continue
            pnl = self.cash_journal.pair_total(pair, start, self.initial_book_ids)
            if pnl <= -self.config.risk.pair_loss_limit:
                if pair in self.portfolio.slots:
                    self.portfolio.stop_loss(pair, pnl, self.current_timestamp)
                    self._cancel_pair(pair)
                self.portfolio.excluded.add(pair)
                if self.config.adaptive.enabled:
                    self.loss_circuit.record(pair, self.current_timestamp)
                self.next_monitor = 0
                self.recorder.record("late_pair_stop", self.current_timestamp, pair=pair, selected_at=start, pnl=pnl)

    async def _account_book(self):
        if self.cash_journal is None:
            return await self._legacy_account_book()
        # Periodic backfill also captures small late records below the account
        # reconciliation tolerance and outside the overlap window.
        if (self.current_timestamp - self.cash_journal.get("last_full_scan", self.run_started_at) >= 3600
                and self.cash_journal.get("pending") is None):
            self.cash_journal.request_rescan()
        for _ in range(self.cash_journal.pages_per_poll):
            interval = self.cash_journal.page_range(self.current_timestamp)
            start, end, offset = interval
            page = await self.connector._api_get(
                path_url="futures/usdt/account_book", is_auth_required=True, limit_id=C.USER_BALANCES_PATH_URL,
                params={"from": start, "to": end, "limit": 1000, "offset": offset})
            if self.cash_journal.ingest(page, interval):
                if end >= int(self.current_timestamp):
                    with self.cash_journal.db:
                        if self.cash_journal.get("rescan", False) or self.cash_journal.get("last_full_scan") is None:
                            self.cash_journal.put("last_full_scan", self.current_timestamp)
                        self.cash_journal.put("rescan", False)
                    return self.cash_journal.snapshot(self.initial_book_ids, bootstrap=self.account_risk.anchor is None)
        raise RuntimeError("Cash history catch-up continues on the next poll; new risk paused")

    async def _legacy_account_book(self):
        # Re-read this run's fixed time range rather than advancing a timestamp
        # cursor: late fee/settlement records must not disappear between polls.
        # IDs deduplicate records; offsets apply to a fixed 'to' on every page.
        result = AccountBookSnapshot()
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
                if self.config.account_risk.enabled:
                    change = Decimal(str(record["change"]))
                    timestamp = finite_time(record["time"])
                    record_id = str(record["id"])
                    if not change.is_finite() or not record_id or record["id"] is None:
                        raise ValueError("Invalid account-book record")
                    if record_id in result.records:
                        raise RuntimeError("Account-book pages changed during pagination; retrying snapshot")
                    result.records[record_id] = dict(pair=pair, kind=kind, time=timestamp, change=change)
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
        if market is None or market.close_rejection(self.config.risk, self.current_timestamp):
            return None
        buy = amount < 0
        price = market.bid if buy else market.ask
        use_market = (self.config.allow_market_exit
                      and self.current_timestamp - retiring_since >= self.config.exit_timeout_seconds)
        return Intent(pair, buy, abs(amount), price, close=True, market=use_market)

    def _submit(self, intent):
        if not intent.close and not self._heartbeat_ready(intent.pair):
            return
        if not intent.close and (not self.entry_allowed or not self._opening_ready() or not self._checkpoint()):
            return
        if not self.recorder.record("submit_intent", self.current_timestamp, intent=intent, revision=self.revision):
            if not intent.close:
                return
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
        reserved_id = None
        if self.config.recovery.enabled:
            try:
                reserved_id = self.recovery.new_id()
                self.recovery.prepare(reserved_id, intent, self.current_timestamp)
                if not self._checkpoint():
                    self.recovery.orders.pop(reserved_id, None)
                    reserved_id = None
                    if not intent.close:
                        return
                    self.logger().error("CLOSE allowed without a durable recovery journal; later restart may need manual reconciliation")
            except ValueError as exc:
                self.logger().error(str(exc))
                self._recovery_block_open = True
                reserved_id = None
                if not intent.close:
                    return
            if reserved_id is not None:
                try:
                    self.connector.reserve_portfolio_order_id(reserved_id)
                except Exception:
                    self.recovery.orders[reserved_id]["terminal"] = True
                    self.terminal_orders[reserved_id] = TrackedOrder(intent, intent.amount, self.current_timestamp)
                    self._recovery_block_open = True
                    self._checkpoint()
                    raise
        method = self.buy if intent.buy else self.sell
        try:
            order_id = method(self.connector_name, intent.pair, intent.amount, candidate.order_type,
                              price=intent.price, position_action=PositionAction.CLOSE if intent.close else PositionAction.OPEN)
        except Exception as exc:
            if reserved_id is not None:
                if getattr(self.connector, "_gate_portfolio_client_order_id", None) == reserved_id:
                    del self.connector._gate_portfolio_client_order_id
                self.recovery.orders[reserved_id]["terminal"] = True
                self.terminal_orders[reserved_id] = TrackedOrder(intent, intent.amount, self.current_timestamp)
                self._recovery_block_open = True
                self._checkpoint()
            else:
                self.halt_reason = "Unjournaled submission failed; manual order reconciliation required"
            self.logger().error(f"Order submission requires reconciliation: {exc}")
            return
        self.orders[order_id] = TrackedOrder(intent, intent.amount, self.current_timestamp)
        if reserved_id is not None and order_id != reserved_id:
            self.halt_reason = "Connector did not honor the persisted client order ID"
            self._cancel_all()
            return
        self.recorder.record("submitted", self.current_timestamp, order_id=order_id, intent=intent,
                             calibration=self._calibration_snapshot(intent.pair))
        self.next_quote[intent.pair] = self.current_timestamp + self.config.quote_refresh_seconds
        self._remember_quote_signal(intent.pair)

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
            self.recorder.record("cancel_requested", self.current_timestamp, order_id=order_id, pair=pair)

    def _cancel_all(self):
        pairs = {o.intent.pair for o in self.orders.values()}
        pairs.update(o.intent.pair for oid, o in self.terminal_orders.items() if oid in self.exchange_open_ids)
        for pair in pairs:
            self._cancel_pair(pair)

    def did_fill_order(self, event):
        order = self.orders.get(event.order_id) or self.terminal_orders.get(event.order_id)
        if order is not None:
            trade_id = str(getattr(event, "exchange_trade_id", "") or "")
            if self.config.recovery.enabled:
                if self.recovery.recovering and event.order_id in self._restart_order_ids and not self._replaying_fills:
                    self.account_epoch += 1
                    self.account_dirty = True
                    return  # Historical REST replay owns restored executions.
                if event.order_id.startswith("t-ga") and event.order_id not in self.recovery.orders:
                    return  # Already proved settled and persisted before pruning.
                if event.order_id in self.recovery.orders:
                    try:
                        if not self.recovery.fill(event.order_id, trade_id, event.amount,
                                                  getattr(event, "price", order.intent.price),
                                                  getattr(event, "timestamp", self.current_timestamp)):
                            return
                    except ValueError as exc:
                        self.recovery.orders[event.order_id]["terminal"] = True
                        self._recovery_block_open = self.account_dirty = True
                        self.next_account = 0
                        self.logger().error(f"Fill requires REST reconciliation: {exc}")
                        return
            key = (event.order_id, trade_id)
            if trade_id and key in self.fill_ids:
                return
            if trade_id:
                self.fill_ids.add(key)
                self.fill_id_queue.append(key)
                if len(self.fill_id_queue) > 10000:
                    self.fill_ids.discard(self.fill_id_queue.popleft())
            slot = self.portfolio.slots.get(order.intent.pair)
            if self.cash_journal is not None and slot is not None:
                try:
                    self.cash_journal.own_trade(slot.pair, trade_id, slot.selected_at)
                except Exception as exc:
                    self.cash_error = f"cannot persist cash trade ownership: {exc}"
            self.recorder.fill(self.current_timestamp, order.intent.pair, event.order_id, trade_id,
                               event.amount, getattr(event, "price", order.intent.price), order.intent.buy,
                               order.intent.close, order.intent.market, getattr(event, "timestamp", self.current_timestamp),
                               self.last_markets.get(order.intent.pair), getattr(event, "trade_fee", None),
                               slot.reason if slot is not None and order.intent.close else "",
                               regime=self._quality_regime(self.last_markets.get(order.intent.pair)))
            order.remaining = max(ZERO, order.remaining - event.amount)
            pair = order.intent.pair
            change = event.amount if order.intent.buy else -event.amount
            before = self.expected_positions.get(pair, ZERO)
            self.expected_positions[pair] = before + change
            if self.config.adaptive.enabled:
                timestamp = getattr(event, "timestamp", self.current_timestamp)
                timestamp = min(timestamp, self.current_timestamp) if math.isfinite(timestamp) and timestamp >= 0 else self.current_timestamp
                self.inventory.fill(pair, event.order_id, order.intent.buy, before, before + change,
                                    order.intent.close, timestamp)
            self.account_epoch += 1
            self.account_dirty = True
            # Reprice both sides only after the new position is reconciled.
            self.next_quote[order.intent.pair] = 0
            if not self.stopping:
                self._checkpoint()

    def _terminal(self, event):
        order = self.orders.pop(event.order_id, None)
        if order is not None:
            self.recorder.record("terminal", self.current_timestamp, order_id=event.order_id,
                                 event_type=type(event).__name__, remaining=order.remaining)
            self.terminal_orders[event.order_id] = order
            if event.order_id in self.recovery.orders:
                self.recovery.orders[event.order_id]["terminal"] = True
            self.account_epoch += 1
            self.account_dirty = True
            if order.intent.pair in self.portfolio.excluded:
                self.next_monitor = 0
            if not self.stopping:
                self._checkpoint()

    did_cancel_order = _terminal
    did_fail_order = _terminal
    did_expire_order = _terminal
    did_complete_buy_order = _terminal
    did_complete_sell_order = _terminal

    async def on_stop(self):
        if self.stopping:
            return
        self.stopping = True
        if self.heartbeat_task and not self.heartbeat_task.done():
            self.heartbeat_task.cancel()
            await asyncio.gather(self.heartbeat_task, return_exceptions=True)
        if self.kline_task and not self.kline_task.done():
            self.kline_task.cancel()
            await asyncio.gather(self.kline_task, return_exceptions=True)
        # Leave the exchange countdown armed after stopping.
        if self.refresh_task and not self.refresh_task.done():
            self.refresh_task.cancel()
            await asyncio.gather(self.refresh_task, return_exceptions=True)
        await self._cancel_old_submission_tasks(set(self.recovery.orders))
        self._cancel_all()
        self._checkpoint()
        self.recorder.record("strategy_stop", self.current_timestamp, positions=self.positions,
                             pending_markouts=len(self.recorder.pending))
        if self.risk_store is not None:
            self.risk_store.close()
        if self.cash_journal is not None:
            self.cash_journal.close()
        if getattr(self.connector, "_gate_market_signal_feed", None) is self.signal_feed:
            delattr(self.connector, "_gate_market_signal_feed")
        # Hummingbot stopping removes the strategy clock. Do not claim that a
        # cancellation request or a sent close order is a confirmed flat account.
        if self.positions or self.orders or self.exchange_open_pairs or any(self.expected_positions.values()):
            self.logger().warning("Strategy stopped with possible positions/orders; preserve the checkpoint for restart reconciliation")

    def format_status(self):
        lines = [f"Gate Avellaneda portfolio: {'DRY RUN (no orders)' if self.config.dry_run else 'LIVE'}",
                 f"Budget {self.config.risk.capital} USDT / reserve {self.config.risk.reserve}",
                 f"Subscribed candidates: {len(self.contracts)}; account fresh: {self._account_fresh()}"]
        if self.config.account_risk.enabled:
            lines.append(f"Account risk: session={self.account_risk.pnl}, daily={self.account_risk.daily_pnl}, "
                         f"peak={self.account_risk.peak}, drawdown={self.account_risk.drawdown}, "
                         f"latched={self.account_risk.latched}, reason={self.account_risk.reason or '-'}, "
                         f"reconciliation={self.account_risk.reconciliation_error or 'OK'}")
        if self.config.recovery.enabled:
            lines.append(f"Recovery: phase={self.recovery.phase}, reason={self.recovery.reason or '-'}, "
                         f"unsettled_orders={len(self.recovery.orders)}, terminal_settlement_block={self._recovery_block_open}, "
                         f"correlation_wait={sorted(self._recovery_correlation_pairs)}")
        if self.persistence_error or self.recorder.error or self.cash_error or self.heartbeat_error:
            lines.append(f"Opening paused: {self.persistence_error or self.recorder.error or self.cash_error or self.heartbeat_error}")
        lines.append(f"Quality: {self.config.quality_control.mode}; retained quotes={self.config.execution.retain_quotes}; "
                     f"exchange countdown={self.config.deadman.enabled}")
        if self.config.adaptive.enabled:
            lines.append(f"Adaptive controls: opening allowed={self.entry_allowed}, circuit_until={self.loss_circuit.until}, "
                         f"gross cap={self.config.adaptive.max_gross_quote}, directional cap={self.config.adaptive.max_directional_quote}")
            if self.selection_rejections:
                lines.append("Selection checks: " + ", ".join(f"{p}={r}" for p, r in sorted(self.selection_rejections.items())))
        if self.halt_reason:
            lines.append(f"HALTED: {self.halt_reason}")
        if not self.portfolio.slots:
            lines.append("No qualifying pair yet; warming indicators or waiting for suitable markets")
        snapshots = self._snapshots()
        reasons = Counter((self._entry_check(m) if self.config.adaptive.enabled
                           else m.rejection(self.config.risk, self.current_timestamp)) or "eligible"
                          for m in snapshots.values())
        reasons["no usable book/indicator snapshot"] = len(self.contracts) - len(snapshots)
        lines.append("Candidates: " + ", ".join(f"{reason}={count}" for reason, count in sorted(reasons.items()) if count))
        for pair, slot in self.portfolio.slots.items():
            lines.append(f"{pair}: {slot.state}, position={self.positions.get(pair, ZERO)}, "
                         f"net_PnL={self._pair_pnl(pair)} USDT, reason={slot.reason or '-'}")
            if self.config.adaptive.enabled:
                state = self.gamma.states.get(pair)
                lines.append(f"  gamma_mode={'adaptive' if self.config.gamma_control.enabled else 'fixed'}, "
                             f"gamma={self.gamma.value(pair)}, gamma_target={state.target if state else self.config.risk.gamma}, "
                             f"quoted_gamma={self.last_gamma_values.get(pair, '-')}, "
                             f"gamma_stress={state.components if state else {}}")
            plan = self.plans.get(pair)
            if self.config.adaptive.enabled and plan:
                lines.append(f"  inventory_stage={plan.stage}, age={self.inventory.age(pair, self.positions.get(pair, ZERO), self.current_timestamp):.1f}s, "
                             f"position_cap={plan.position_cap}, order_quote={plan.order_quote}, signal_confidence={plan.confidence}, "
                             f"effective_reference={plan.reference}, buy_premium={plan.buy_premium}, sell_premium={plan.sell_premium}, "
                             f"entry_reason={plan.entry_reason or '-'}")
            calibration = self._calibration_snapshot(pair)
            if calibration:
                lines.append(f"  K-line={calibration['mode']}, ready={calibration['ready']}, "
                             f"returns={calibration['samples']}, daily_sigma={calibration['daily_volatility'] * 100}%, "
                             f"suggested_half_bps={calibration['suggested_half_spread'] * 10000}, "
                             f"reason={calibration['reason'] or '-'}")
            signal = self.signals.get(pair)
            if self.config.micro.enabled and signal:
                lines.append(f"  micro={self.config.micro.mode}, ready={signal.ready}, reference={signal.reference}, "
                             f"flow={signal.flow}, unmatched_bid={signal.bid_loss}, unmatched_ask={signal.ask_loss}, "
                             f"paused={signal.paused}, reason={signal.reason}")
            for quote in self.last_quotes.get(pair, []):
                lines.append(f"  {'buy' if quote.buy else 'sell'} {quote.amount} @ {quote.price}, close={quote.close}")
        if self.portfolio.excluded:
            lines.append("Loss-stopped pairs excluded this run: " + ", ".join(sorted(self.portfolio.excluded)))
        return "\n".join(lines)
