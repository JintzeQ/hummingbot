"""Run the offline Avellaneda paper trading dashboard or export a portable report.

Examples (from repository root):
  python tools/avellaneda_perpetual_simulator.py
  python tools/avellaneda_perpetual_simulator.py --export report.html --json result.json
  python tools/avellaneda_perpetual_simulator.py --source capture --export replay.html

Only pydantic is needed. No API keys, connector, Cython build or web CDN is used.
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hummingbot.strategy_v2.utils.avellaneda_perpetual_simulation import SimulationSettings, simulate  # noqa: E402

UI = Path(__file__).resolve().parent / "avellaneda_simulator/index.html"


def report_html(result):
    # Embed data, not executable user content. The portable report retains chart
    # selection/scrubbing/export and disables server-only reruns.
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    return UI.read_text().replace("<!-- SIMULATION_DATA -->", f'<script type="application/json" id="embedded-result">{payload}</script>')


def write_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


class SimulatorHandler(BaseHTTPRequestHandler):
    # Exact routes only: this server cannot list/read arbitrary repository files.
    def reply(self, status, body, content_type="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.reply(200, UI.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/defaults":
            self.reply(200, SimulationSettings().model_dump_json())
        elif self.path == "/favicon.ico":
            self.reply(204, b"", "image/x-icon")
        else:
            self.reply(404, '{"error":"沒有這個路徑"}')

    def do_POST(self):
        if self.path != "/api/simulate":
            self.reply(404, '{"error":"沒有這個路徑"}')
            return
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if host not in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}") or (
                origin is not None and origin not in (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}")):
            self.reply(403, '{"error":"只接受本機同源請求"}')
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                raise ValueError("請求大小必須介於 1 與 16384 bytes")
            settings = SimulationSettings.model_validate_json(self.rfile.read(length))
            result = simulate(settings)
            self.reply(200, json.dumps(result, ensure_ascii=False, allow_nan=False))
        except (ValueError, ValidationError) as exc:
            self.reply(400, json.dumps(dict(error=str(exc)), ensure_ascii=False))

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--source", choices=("synthetic", "capture"), default="synthetic")
    parser.add_argument("--scenario", choices=("range", "uptrend", "downtrend", "shock"), default="range")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=int, default=1800, help="Trading seconds after calibration")
    parser.add_argument("--settings", type=Path, help="JSON settings, same schema as the dashboard API")
    parser.add_argument("--export", type=Path, help="Export a portable HTML report instead of serving")
    parser.add_argument("--json", type=Path, help="Export full result JSON instead of serving")
    args = parser.parse_args()
    if args.export or args.json:
        try:
            settings = (SimulationSettings.model_validate_json(args.settings.read_text()) if args.settings else
                        SimulationSettings(source=args.source, scenario=args.scenario, seed=args.seed,
                                           duration_seconds=args.duration))
            for path in (args.export, args.json):
                if path and path.exists():
                    raise ValueError(f"檔案已存在：{path}；請指定新檔名")
            result = simulate(settings)
            if args.export:
                write_new(args.export, report_html(result))
                print(f"互動報告：{args.export}")
            if args.json:
                write_new(args.json, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
                print(f"完整帳務與成交：{args.json}")
            for mode in result["results"]:
                s = mode["summary"]
                print(f'{mode["label"]}: 現金損益 {s["cash_pnl"]:+.6f} USDT，'
                      f'待入帳反傭 {s["rebates_pending"]:.6f} USDT，成交 {s["fill_count"]}')
        except ValueError as exc:
            parser.error(str(exc))
    else:
        if args.settings or args.source != "synthetic" or args.scenario != "range" or args.seed != 42 or args.duration != 1800:
            parser.error("伺服器模式請在網頁設定參數；命令列行情參數供 --export / --json 使用")
        if not 1024 <= args.port <= 65535:
            parser.error("port 必須介於 1024 與 65535")
        server = HTTPServer(("127.0.0.1", args.port), SimulatorHandler)
        print(f"Avellaneda 離線模擬器：http://127.0.0.1:{args.port}（Ctrl+C 結束）", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
