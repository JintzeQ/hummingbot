"""Account-wide cash accounting and durable, manually released loss latches.

Scope: a dedicated classic Gate USDT futures account. No exchange I/O here.
"""
import fcntl
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, model_validator

ZERO = Decimal(0)
TRADING_TYPES = frozenset({"pnl", "fee", "fund"})
CASH_TYPES = TRADING_TYPES | {"dnw", "refr"}


class AccountRiskSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    persist: bool = True
    state_path: str = "data/gate_avellaneda_risk_state.json"
    session_loss_limit: Decimal = Field(default=Decimal(10), gt=0)
    daily_loss_limit: Decimal = Field(default=Decimal(10), gt=0)
    max_drawdown: Decimal = Field(default=Decimal(10), gt=0)
    reconciliation_tolerance: Decimal = Field(default=Decimal("0.01"), ge=0, le=1)
    timezone: str = "Asia/Taipei"

    @model_validator(mode="after")
    def validate_settings(self):
        ZoneInfo(self.timezone)
        if not self.state_path.strip():
            raise ValueError("Account risk state_path must be nonempty")
        return self


class AccountBookSnapshot(dict):
    """Legacy pair settlements plus all-account records for independent risk."""
    def __init__(self):
        super().__init__()
        self.records = {}


def finite(value):
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError("Nonfinite risk state/account value")
    return number


def finite_time(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("Invalid risk timestamp")
    return number


def mode_path(path, dry_run):
    target = Path(path)
    return target.with_name(target.stem + ".dry_run" + target.suffix) if dry_run else target


class RiskStateStore:
    """Single writer, checksum, atomic replacement and fsync before new risk.

    A corrupt existing file is never silently replaced. The checksum detects
    damage, not malicious edits. The lock is held until strategy shutdown.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = open(str(self.path) + ".lock", "a+")
        os.chmod(self.lock.name, 0o600)
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise ValueError("Risk state is already owned by another process")

    @staticmethod
    def encoded(payload):
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    def load(self):
        return self.read_file(self.path)

    @classmethod
    def read_file(cls, path):
        """Read/checksum before market subscriptions; writer locks in ctor."""
        path = Path(path)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text())
            payload = envelope["payload"]
            digest = hashlib.sha256(cls.encoded(payload)).hexdigest()
            if envelope["version"] != 1 or envelope["sha256"] != digest:
                raise ValueError("Risk state version/checksum mismatch")
            return payload
        except Exception as exc:
            raise ValueError(f"Risk state cannot be restored: {exc}") from exc

    def save(self, payload):
        if self.lock.closed:
            raise ValueError("Risk state writer is closed")
        raw = self.encoded(payload)
        envelope = {"version": 1, "payload": payload, "sha256": hashlib.sha256(raw).hexdigest()}
        temporary = None
        try:
            fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
            with os.fdopen(fd, "w") as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump(envelope, stream, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                os.unlink(temporary)

    def close(self):
        if not self.lock.closed:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()


class AccountRisk:
    def __init__(self, settings):
        self.settings = settings
        self.anchor = None
        self.pnl = self.daily_pnl = self.peak = self.drawdown = ZERO
        self.day = ""
        self.day_start_pnl = ZERO
        self.last_at = 0.0
        self.latched = False
        self.reason = ""
        self.reconciliation_error = "not initialized"
        self.cash_totals = {}

    @property
    def allow_open(self):
        return not self.settings.enabled or (self.anchor is not None and not self.latched and not self.reconciliation_error)

    def validate_account(self, account):
        # margin_mode=0 alone is also used by a unified classic-spot account;
        # require classic cash/history consistency, no credit/cross settlement.
        if (str(account.get("currency", "")).upper() != "USDT"
                or account.get("margin_mode") != 0 or account.get("enable_credit", False)
                or finite(account.get("history", {}).get("cross_settle", 0)) != 0):
            raise ValueError("Account risk requires a dedicated classic USDT futures account (margin_mode=0, no credit)")
        user = str(account.get("user", ""))
        if not user or user == "None":
            raise ValueError("Account response missing user identity")
        if self.anchor is not None and user != self.anchor["user"]:
            raise ValueError("Persistent risk state belongs to another Gate account")
        history = account.get("history")
        if not isinstance(history, dict) or not CASH_TYPES.issubset(history):
            raise ValueError("Account response missing classic cash history")
        if any(finite(account.get(key, 0)) != 0 for key in ("point", "bonus")):
            raise ValueError("POINT/bonus balances are unsupported by USDT account risk")
        if any(finite(history.get(key, 0)) != 0 for key in
               ("point_fee", "point_dnw", "point_refr", "bonus_offset", "bonus_dnw")):
            raise ValueError("POINT/bonus cash history requires manual USDT reconciliation")
        return user

    def observe(self, account, records, position_upnl, now, reserve, cash_totals=None):
        if not self.settings.enabled:
            return
        user = self.validate_account(account)
        now = finite_time(now)
        if self.anchor is not None and now < self.last_at:
            raise ValueError("Clock moved backwards relative to persistent account risk")
        wallet, upnl = finite(account["total"]), finite(account["unrealised_pnl"])
        history = {kind: finite(account["history"][kind]) for kind in CASH_TYPES}
        day = datetime.fromtimestamp(now, ZoneInfo(self.settings.timezone)).date().isoformat()
        if self.anchor is None:
            self.anchor = {"user": user, "wallet": str(wallet), "upnl": str(upnl), "started_at": now,
                           "book_ids": sorted(records), "history": {k: str(v) for k, v in history.items()}}
            self.day = day
        initial_ids = set(self.anchor["book_ids"])
        self.cash_totals = {kind: sum((finite(r["change"]) for rid, r in records.items()
                                      if rid not in initial_ids and r["kind"] == kind), ZERO)
                            for kind in CASH_TYPES}
        if cash_totals is not None:
            self.cash_totals = {kind: finite(cash_totals[kind]) for kind in CASH_TYPES}
        # History provides a second source for late/missing account-book rows.
        deltas = {k: history[k] - finite(self.anchor["history"][k]) for k in CASH_TYPES}
        ledger_settled = sum((self.cash_totals[k] for k in TRADING_TYPES), ZERO)
        history_settled = sum((deltas[k] for k in TRADING_TYPES), ZERO)
        conservative_upnl = min(upnl, finite(position_upnl))
        next_pnl = min(ledger_settled, history_settled) + conservative_upnl - finite(self.anchor["upnl"])
        if day != self.day:
            # Include an unobserved midnight/offline gap conservatively instead
            # of erasing it on the first snapshot after midnight.
            self.day, self.day_start_pnl = day, self.pnl
        self.pnl = next_pnl
        self.daily_pnl = self.pnl - self.day_start_pnl
        self.peak = max(self.peak, self.pnl)
        self.drawdown = self.peak - self.pnl
        self.last_at = now
        tolerance = self.settings.reconciliation_tolerance
        errors = []
        if abs(sum(history.values(), ZERO) - wallet) > tolerance:
            errors.append("classic history does not reconcile to wallet")
        if abs(wallet - finite(self.anchor["wallet"]) - sum(self.cash_totals.values(), ZERO)) > tolerance:
            errors.append("wallet/account-book mismatch")
        if any(abs(deltas[k] - self.cash_totals[k]) > tolerance for k in CASH_TYPES):
            errors.append("cash history/account-book mismatch")
        if abs(upnl - finite(position_upnl)) > tolerance:
            errors.append("position/account unrealized PnL mismatch")
        if any(r["kind"] not in CASH_TYPES and finite(r["change"]) != 0 for r in records.values()):
            errors.append("unsupported non-USDT account-book flow")
        self.reconciliation_error = "; ".join(errors)
        triggers = [(self.pnl <= -self.settings.session_loss_limit, "session loss limit"),
                    (self.daily_pnl <= -self.settings.daily_loss_limit, "daily loss limit"),
                    (self.drawdown >= self.settings.max_drawdown, "profit drawdown limit"),
                    (wallet + conservative_upnl <= reserve, "account reserve floor")]
        if not self.latched:
            for hit, reason in triggers:
                if hit:
                    self.latched, self.reason = True, reason
                    break

    def dump(self):
        return {"anchor": self.anchor, "pnl": str(self.pnl), "daily_pnl": str(self.daily_pnl),
                "peak": str(self.peak), "drawdown": str(self.drawdown), "day": self.day,
                "day_start_pnl": str(self.day_start_pnl), "last_at": self.last_at,
                "latched": self.latched, "reason": self.reason}

    def restore(self, payload):
        anchor = payload["anchor"]
        if anchor is not None:
            if (not isinstance(anchor["user"], str) or not anchor["user"]
                    or not isinstance(anchor["book_ids"], list) or len(anchor["book_ids"]) > 10000
                    or not all(isinstance(x, str) and x for x in anchor["book_ids"])
                    or len(set(anchor["book_ids"])) != len(anchor["book_ids"])):
                raise ValueError("Invalid persistent risk anchor")
            finite(anchor["wallet"])
            finite(anchor["upnl"])
            finite_time(anchor["started_at"])
            for kind in CASH_TYPES:
                finite(anchor["history"][kind])
            datetime.strptime(payload["day"], "%Y-%m-%d")
        for key in ("pnl", "daily_pnl", "peak", "drawdown", "day_start_pnl"):
            setattr(self, key, finite(payload[key]))
        self.last_at = finite_time(payload["last_at"])
        if type(payload["latched"]) is not bool or not isinstance(payload["reason"], str):
            raise ValueError("Invalid persistent risk latch")
        if (self.peak < max(ZERO, self.pnl) or self.drawdown != self.peak - self.pnl
                or self.daily_pnl != self.pnl - self.day_start_pnl
                or (anchor is not None and self.last_at < anchor["started_at"])
                or (payload["latched"] and not payload["reason"])):
            raise ValueError("Inconsistent persistent risk state")
        self.anchor, self.day = anchor, payload["day"]
        self.latched, self.reason = payload["latched"], payload["reason"]
        # A restored snapshot never authorizes new orders before a fresh poll.
        self.reconciliation_error = "awaiting fresh account reconciliation"
