"""Transactional, incremental cash history. Never infer completeness from a cap."""
import json
import os
import sqlite3
from decimal import Decimal
from pathlib import Path

from .account_risk import AccountBookSnapshot, CASH_TYPES, TRADING_TYPES, finite, finite_time

ZERO = Decimal(0)


class CashJournal:
    overlap = 300
    window = 3600
    pages_per_poll = 10

    def __init__(self, path, user, started_at):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        os.chmod(path, 0o600)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS cash (id TEXT PRIMARY KEY, pair TEXT, at REAL, kind TEXT, amount TEXT, trade_id TEXT)")
        self.db.execute("CREATE INDEX IF NOT EXISTS cash_pair_at ON cash(pair, at)")
        self.db.execute("CREATE TABLE IF NOT EXISTS cycles (pair TEXT, start REAL, ended REAL, PRIMARY KEY(pair,start))")
        self.db.execute("CREATE TABLE IF NOT EXISTS owners (pair TEXT, trade_id TEXT, start REAL, PRIMARY KEY(pair,trade_id))")
        identity = [str(user), finite_time(started_at)]
        previous = self.get("identity")
        if previous is not None and previous != identity:
            self.db.close()
            raise ValueError("Cash journal identity/session mismatch; retain or archive state and cash journal together")
        with self.db:
            self.put("identity", identity)
        self.start = int(started_at)
        self.closed = False

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value, allow_nan=False)))

    def request_rescan(self):
        if self.get("rescan", False):
            return
        with self.db:
            self.put("cursor", self.start)
            self.put("pending", None)
            self.put("rescan", True)

    def page_range(self, now):
        pending = self.get("pending")
        if pending is not None:
            return pending
        cursor = self.get("cursor", self.start)
        return [max(self.start, int(cursor) - self.overlap), min(int(now), int(cursor) + self.window), 0]

    def ingest(self, rows, interval, page_size=1000):
        start, end, offset = interval
        parsed, seen = [], set()
        for raw in rows:
            rid = str(raw.get("id", ""))
            pair = raw.get("contract", "").replace("_", "-")
            kind, amount, at = raw["type"], finite(raw["change"]), finite_time(raw["time"])
            if not rid or rid == "None" or rid in seen or not start <= at < end + 1:
                raise RuntimeError("Invalid/duplicate/out-of-range cash page; retrying")
            if kind in TRADING_TYPES and not pair:
                raise ValueError("Cash settlement has no contract")
            seen.add(rid)
            parsed.append((rid, pair, at, kind, str(amount), str(raw.get("trade_id", ""))))
        totals = {k: finite(v) for k, v in self.get("totals", {}).items()}
        revision = self.get("revision", 0)
        with self.db:
            for row in parsed:
                previous = self.db.execute("SELECT id,pair,at,kind,amount,trade_id FROM cash WHERE id=?", (row[0],)).fetchone()
                if previous == row:
                    continue
                if previous:
                    totals[previous[3]] = totals.get(previous[3], ZERO) - finite(previous[4])
                totals[row[3]] = totals.get(row[3], ZERO) + finite(row[4])
                self.db.execute("INSERT OR REPLACE INTO cash VALUES (?,?,?,?,?,?)", row)
                revision += 1
            self.put("totals", {k: str(v) for k, v in totals.items()})
            self.put("revision", revision)
            if len(rows) < page_size:
                self.put("cursor", end)
                self.put("pending", None)
            else:
                self.put("pending", [start, end, offset + len(rows)])
        return len(rows) < page_size

    def snapshot(self, initial_ids=(), bootstrap=False):
        result = AccountBookSnapshot()
        result.revision = self.get("revision", 0)
        result.cash_totals = {k: finite(self.get("totals", {}).get(k, 0)) for k in CASH_TYPES}
        for rid in initial_ids:
            row = self.db.execute("SELECT kind,amount FROM cash WHERE id=?", (rid,)).fetchone()
            if row and row[0] in CASH_TYPES:
                result.cash_totals[row[0]] -= finite(row[1])
        # Only bootstrap needs the actual initial IDs. Subsequent risk checks use
        # cumulative Decimal totals, not a growing in-memory history collection.
        if bootstrap:
            rows = self.db.execute("SELECT id,pair,at,kind,amount FROM cash")
        else:
            rows = self.db.execute("SELECT id,pair,at,kind,amount FROM cash WHERE kind NOT IN ('pnl','fee','fund','dnw','refr')")
        for rid, pair, at, kind, amount in rows:
            result.records[rid] = dict(pair=pair, time=at, kind=kind, change=finite(amount))
            if kind in TRADING_TYPES:
                result[rid] = pair, at, finite(amount)
        return result

    def begin_cycle(self, pair, start):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO cycles VALUES (?,?,NULL)", (pair, start))

    def end_cycle(self, pair, start, now):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO cycles VALUES (?,?,?)", (pair, start, now))
            self.db.execute("UPDATE cycles SET ended=? WHERE pair=? AND start=?", (now, pair, start))

    def own_trade(self, pair, trade_id, start):
        if not trade_id:
            return
        with self.db:
            previous = self.db.execute("SELECT start FROM owners WHERE pair=? AND trade_id=?", (pair, trade_id)).fetchone()
            if previous and previous[0] != start:
                raise ValueError("Cash trade ownership changed")
            self.db.execute("INSERT OR IGNORE INTO owners VALUES (?,?,?)", (pair, trade_id, start))

    def pair_total(self, pair, start, initial_ids=()):
        next_cycle = self.db.execute("SELECT min(start) FROM cycles WHERE pair=? AND start>?", (pair, start)).fetchone()[0]
        end = next_cycle if next_cycle is not None else float("inf")
        rows = self.db.execute("""SELECT c.id,c.amount FROM cash c LEFT JOIN owners o
            ON c.pair=o.pair AND c.trade_id=o.trade_id WHERE c.pair=? AND c.kind IN ('pnl','fee','fund')
            AND (o.start=? OR (o.start IS NULL AND c.at>=? AND c.at<?))""", (pair, start, start, end))
        initial = set(initial_ids)
        return sum((finite(amount) for rid, amount in rows if rid not in initial), ZERO)

    def closed_cycles(self):
        return list(self.db.execute("SELECT pair,start,ended FROM cycles WHERE ended IS NOT NULL"))

    def close(self):
        if not self.closed:
            self.db.close()
            self.closed = True
