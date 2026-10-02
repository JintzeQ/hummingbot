"""Read-only verification: python -m hummingbot.strategy.gate_avellaneda.verify."""
import argparse
import importlib
import json
from pathlib import Path
from urllib.request import urlopen

from .core import Settings, candidate_universe
from .verification import audit_report


def preflight(runtime=False, gate_public=False):
    checks = {}
    if runtime:
        try:
            common = importlib.import_module("hummingbot.core.data_type.common")
            importlib.import_module("hummingbot.connector.derivative.gate_io_perpetual.gate_io_perpetual_derivative")
            adapter = importlib.import_module("scripts.gate_avellaneda_portfolio")
            adapter.GateAvellanedaPortfolioConfig()
            assert common.OrderType.LIMIT_MAKER and common.PositionAction.CLOSE
            checks["compiled_runtime"] = {"ok": True}
        except Exception as exc:
            checks["compiled_runtime"] = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
    if gate_public:
        try:
            root = "https://api.gateio.ws/api/v4/futures/usdt/"
            with urlopen(root + "contracts", timeout=5) as response:
                contracts = json.load(response)
            with urlopen(root + "tickers", timeout=5) as response:
                tickers = json.load(response)
            eligible = candidate_universe(contracts, tickers, Settings(), 20, [])
            if not eligible:
                raise ValueError("No affordable supported contract candidates")
            checks["gate_public"] = {"ok": True, "candidates": sorted(eligible)}
        except Exception as exc:
            checks["gate_public"] = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
    return {"ok": all(c["ok"] for c in checks.values()), "checks": checks,
            "live_ready": False,
            "note": "Public/import checks do not validate signed account/WS/order execution or establish profitability."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true", help="Check actual installed Hummingbot imports")
    parser.add_argument("--gate-public", action="store_true", help="GET public contracts/tickers; no credentials or orders")
    parser.add_argument("--report", nargs="+", type=Path, help="Explicit JSONL paths, including retained rotated files")
    args = parser.parse_args(argv)
    result = preflight(args.runtime, args.gate_public)
    if args.report:
        try:
            records = []
            for path in args.report:
                for line in path.read_text().splitlines():
                    if line.strip():
                        records.append(json.loads(line))
            result["audit"] = audit_report(records)
            result["ok"] = result["ok"] and result["audit"]["replay_mismatches"] == 0
        except Exception as exc:
            result["ok"] = False
            result["report_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
