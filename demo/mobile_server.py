"""Serve the on-device demo (demo/mobile/) to phones on the same network, and save the
device benchmarks they send back.

- Static files from demo/mobile/, with no caching, so a re-exported model is never stale.
- Cross-origin isolation headers (COOP same-origin, COEP credentialless), which Chrome needs
  before WebAssembly may use threads (`?threads=4` on the page). Without them the page runs
  single-threaded, as before.
- POST /bench: the page's benchmark result (JSON), written to
  outputs/mobile_bench/bench_<UTC time>.json and summarised on stdout.
- GET /report?...: the page's load/inference events, printed (the old 404 log lines).

usage: python demo/mobile_server.py [PORT]   (run from the repository root)
"""
import json
import os
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, "demo", "mobile")
OUT = os.path.join(ROOT, "outputs", "mobile_bench")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SITE, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        super().end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/report":
            fields = {k: v[0] for k, v in parse_qs(url.query).items()}
            print(f"[report] {json.dumps(fields)}", flush=True)
            self.send_response(204)
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/bench":
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            result = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "not JSON")
            return
        result["client_ip"] = self.client_address[0]
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, f"bench_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=1)
        m = result.get("inference_ms", {})
        print(f"[bench] saved {os.path.relpath(path, ROOT)}: inference p50 {m.get('p50')} ms, "
              f"p90 {m.get('p90')} ms, threads {result.get('runtime', {}).get('threads')}, "
              f"{result.get('device', {}).get('ua', '')[:80]}", flush=True)
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt, *args):  # quieter: only errors and our own lines
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"serving {SITE} on 0.0.0.0:{port}; benchmarks go to {OUT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
