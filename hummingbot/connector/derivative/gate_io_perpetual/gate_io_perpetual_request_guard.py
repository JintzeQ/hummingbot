"""Opt-in request protection; standard-library only, also used by the paper broker.

Counts attempts at the REST boundary, including rejected requests. The daily
budget restricts new quotes; cancellation and emergency reduction retain priority.
"""

import json
import math
import os
import time
from collections import deque
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlsplit


class GateRequestDeferred(IOError):
    """No order was accepted: a local gate or an explicit HTTP 429 blocked it."""


def scope(method, url):
    path = urlsplit(url).path
    path = path.split("/api/v4/")[-1].lstrip("/")
    parts = path.split("/")
    if len(parts) >= 4 and parts[2] == "orders":
        path = "/".join(parts[:3])  # All individual cancellations share one bucket.
    return f"{method}:{path}"


class GateRequestGuard:
    def __init__(self, per_second=6, safety_per_second=2, daily=50000, safety_daily=1000,
                 journal=None, clock=time.time):
        if not 0 < safety_per_second < per_second or not 0 < safety_daily < daily:
            raise ValueError("Request limits must leave positive quote and safety budgets")
        self.per_second, self.safety_per_second = per_second, safety_per_second
        self.daily, self.safety_daily = daily, safety_daily
        self.clock = clock
        self.attempts = deque()
        self.cooldowns = {}
        self.strikes = {}
        self.deferred_orders = set()
        self.deferred_cancels = set()
        self.blocked = 0
        self.rate_limited = 0
        self.total_attempts = 0
        self.last_reason = "ready"
        self.generation = 0
        self.pending_quotes = {}
        self.quote_ttl = 10.0
        self.quote_validator = None
        self.journal_error = False
        self.journal = Path(journal) if journal else None
        self._stream = None
        self._lock = None
        if self.journal:
            self._open_journal()

    def _open_journal(self):
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        self._lock = self.journal.with_suffix(".lock").open("a+b")
        # OS locks are released after a crash. A second local process must not
        # trade against an independent counter for the same dedicated account.
        try:
            if os.name == "nt":
                import msvcrt
                self._lock.seek(0)
                if not self._lock.read(1):
                    self._lock.write(b"0")
                    self._lock.flush()
                self._lock.seek(0)
                msvcrt.locking(self._lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            now = self.clock()
            if self.journal.exists():
                with self.journal.open() as stream:
                    for line in stream:
                        stamp = json.loads(line)
                        if not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
                            raise ValueError("Invalid request journal")
                        if stamp > now + 5:
                            raise ValueError("Clock moved backwards; inspect request journal")
                        if stamp > now - 86400:
                            self.attempts.append(stamp)
            self.attempts = deque(sorted(self.attempts))
            state = self.journal.with_suffix(".cooldown.json")
            if state.exists():
                saved = json.loads(state.read_text())
                for key, deadline in saved["cooldowns"].items():
                    if not isinstance(deadline, (float, int)) or not math.isfinite(deadline):
                        raise ValueError("Invalid cooldown journal")
                    self.cooldowns[key] = deadline
                self.strikes = saved["strikes"]
            # Compact only at startup while holding the account's OS lock.
            temp = self.journal.with_suffix(".tmp")
            with temp.open("w") as stream:
                stream.writelines(f"{stamp}\n" for stamp in self.attempts)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.journal)
            self._stream = self.journal.open("a")
        except Exception:
            self.close()
            raise

    def close(self):
        if self._stream:
            self._stream.close()
            self._stream = None
        if self._lock:
            self._lock.close()
            self._lock = None

    def _prune(self, now):
        if self.attempts and now < self.attempts[-1]:
            raise RuntimeError("Request clock moved backwards; trading disabled")
        while self.attempts and self.attempts[0] <= now - 86400:
            self.attempts.popleft()

    def _recent(self, now):
        count = 0
        for stamp in reversed(self.attempts):
            if stamp <= now - 1:
                break
            count += 1
        return count

    def quote_reason(self, now=None):
        now = self.clock() if now is None else now
        self._prune(now)
        if self.journal_error:
            return "request journal unavailable"
        if len(self.attempts) >= self.daily - self.safety_daily:
            return "rolling 24h quote request budget exhausted"
        if any(deadline > now for deadline in self.cooldowns.values()):
            return "Gate rate-limit cooldown"
        if self._recent(now) >= self.per_second - self.safety_per_second:
            return "local quote requests/second limit"
        return ""

    def allowed(self, method, url, trading=False, safety=False, now=None):
        now = self.clock() if now is None else now
        self._prune(now)
        reason = ""
        if self.cooldowns.get(scope(method, url), 0) > now:
            reason = "Gate endpoint cooldown"
        elif trading and self._recent(now) >= self.per_second:
            reason = "local trading requests/second limit"
        elif trading and not safety:
            reason = self.quote_reason(now)
        self.last_reason = reason or "ready"
        return not reason

    def before_request(self, method, url, data=None, now=None):
        now = self.clock() if now is None else now
        key = scope(method, url)
        trading = key in ("POST:futures/usdt/orders", "PUT:futures/usdt/orders", "DELETE:futures/usdt/orders")
        if isinstance(data, str):
            data = json.loads(data) if data else {}
        data = data or {}
        safety = method == "DELETE" or (data.get("reduce_only") is True and str(data.get("price")) == "0")
        order_id = data.get("text")
        if trading and not safety and order_id in self.pending_quotes:
            generation, dispatched = self.pending_quotes[order_id]
            if (generation != self.generation or now - dispatched > self.quote_ttl
                    or self.quote_validator and not self.quote_validator(order_id)):
                self.blocked += 1
                raise GateRequestDeferred("Queued quote is obsolete; calculate a fresh quote")
        if not self.allowed(method, url, trading, safety, now):
            self.blocked += 1
            raise GateRequestDeferred(self.last_reason)
        if trading:
            if self.journal and self._stream is None:
                raise RuntimeError("Request journal is closed; trading disabled")
            if self._stream and not self.journal_error:
                try:
                    self._stream.write(f"{now}\n")
                    self._stream.flush()
                    os.fsync(self._stream.fileno())
                except OSError:
                    self.journal_error = True
                    if not safety:
                        raise RuntimeError("Request journal failed; new quotes disabled")
            self.attempts.append(now)
            self.total_attempts += 1

    def after_response(self, method, url, status, headers, now=None):
        now = self.clock() if now is None else now
        key = scope(method, url)
        headers = {k.lower(): str(v) for k, v in headers.items()}
        if status == 429:
            self.rate_limited += 1
            self.strikes[key] = self.strikes.get(key, 0) + 1
        elif status < 400:
            self.strikes[key] = 0
        exhausted = headers.get("x-gate-ratelimit-requests-remain") == "0"
        if status != 429 and not exhausted:
            return
        delays = []
        try:
            reset = float(headers["x-gate-ratelimit-reset-timestamp"])
            if not math.isfinite(reset):
                raise ValueError("Invalid rate reset timestamp")
            while reset > 1e11:
                reset /= 1000
            server_now = parsedate_to_datetime(headers["date"]).timestamp() if "date" in headers else now
            if math.isfinite(reset):
                delays.append(max(0, reset - server_now))
        except (KeyError, ValueError, TypeError, OverflowError):
            pass
        retry = headers.get("retry-after")
        if retry:
            try:
                delay = float(retry)
            except ValueError:
                try:
                    delay = parsedate_to_datetime(retry).timestamp() - now
                except (ValueError, TypeError, OverflowError):
                    delay = 0
            if math.isfinite(delay) and delay > 0:
                delays.append(delay)
        delay = max(delays) if delays and max(delays) > 0 else min(60, 5 * 2 ** min(4, max(0, self.strikes.get(key, 1) - 1)))
        # Small margin avoids sending exactly on a server bucket boundary.
        self.cooldowns[key] = max(self.cooldowns.get(key, 0), now + delay + 0.25)
        self.generation += 1  # Previously queued quote prices must never resume after this cooldown.
        if self.journal:
            state = self.journal.with_suffix(".cooldown.json")
            temp = state.with_suffix(".tmp")
            try:
                with temp.open("w") as stream:
                    json.dump(dict(cooldowns=self.cooldowns, strikes=self.strikes), stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp, state)
            except OSError:
                self.journal_error = True
        if status == 429:
            raise GateRequestDeferred("Gate HTTP 429; retry after server reset")

    def stats(self, now=None):
        now = self.clock() if now is None else now
        self._prune(now)
        return dict(requests_1s=self._recent(now),
                    requests_24h=len(self.attempts), session_requests=self.total_attempts,
                    quote_daily_limit=self.daily - self.safety_daily, safety_daily_reserve=self.safety_daily,
                    cooldown_seconds=max(0, max(self.cooldowns.values(), default=0) - now),
                    blocked=self.blocked, http_429=self.rate_limited, journal_ok=not self.journal_error)
