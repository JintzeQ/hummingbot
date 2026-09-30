"""Capture public Gate perpetual top-of-book observations; never use credentials."""

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from decimal import Decimal
from pathlib import Path

BASE = "https://api.gateio.ws/api/v4/futures/usdt/"


def public_get(path):
    with urllib.request.urlopen(BASE + path, timeout=5) as response:
        return json.load(response)


def collect(path, pair, duration, capital, append=False):
    contract = pair.replace("-", "_")
    fields = ("timestamp", "bid", "ask", "price_tick", "amount_step", "min_amount",
              "min_notional", "available", "equity", "book_age", "trading_pair")
    started = time.monotonic()
    rule, last_rule, count = None, -30.0, 0
    path.parent.mkdir(parents=True, exist_ok=True)
    if append:
        with path.open(newline="") as previous:
            reader = csv.DictReader(previous)
            if tuple(reader.fieldnames or ()) != fields or any(row["trading_pair"] != pair for row in reader):
                raise ValueError("Cannot append to a different capture schema or trading pair")
    # Existing observations are never overwritten. Calibration resets across
    # stale gaps; replay waits for a fresh estimator warmup after each gap.
    with path.open("a" if append else "x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        if not append:
            writer.writeheader()
        while time.monotonic() - started <= duration:
            tick_started = time.monotonic()
            try:
                if rule is None or tick_started - started - last_rule >= 30:
                    rule = public_get("contracts/" + urllib.parse.quote(contract, safe=""))
                    last_rule = tick_started - started
                    if rule.get("in_delisting"):
                        raise ValueError("Contract is delisting")
                params = urllib.parse.urlencode({"contract": contract, "limit": 1, "with_id": "true"})
                book = public_get("order_book?" + params)
                now = time.time()
                bid, ask = Decimal(book["bids"][0]["p"]), Decimal(book["asks"][0]["p"])
                if not 0 < bid < ask:
                    raise ValueError("Invalid top of book")
                writer.writerow(dict(timestamp=now, bid=bid, ask=ask,
                                     price_tick=rule["order_price_round"], amount_step=rule["quanto_multiplier"],
                                     min_amount=Decimal(str(rule["order_size_min"])) * Decimal(rule["quanto_multiplier"]),
                                     min_notional="1", available=capital, equity=capital,
                                     book_age=max(0.0, now - float(book["update"])), trading_pair=pair))
                stream.flush()
                count += 1
            except Exception as exc:
                print(f"Observation skipped: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(max(0.0, 1.0 - (time.monotonic() - tick_started)))
    print(f"Captured {count} public observations in {path}; balances are hypothetical {capital} USDT.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pair", default="BTC-USDT")
    parser.add_argument("--seconds", type=float, default=1200)
    parser.add_argument("--capital", type=Decimal, default=Decimal("100"))
    parser.add_argument("--append", action="store_true", help="Extend an existing same-pair CSV; gaps remain explicit")
    args = parser.parse_args()
    if args.seconds < (60 if args.append else 900) or not args.capital.is_finite() or args.capital <= 0:
        parser.error("Capture at least 900 seconds (60 when appending) with positive hypothetical capital")
    collect(args.output, args.pair, args.seconds, args.capital, args.append)
