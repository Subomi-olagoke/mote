#!/usr/bin/env python3
"""web/serve.py — serve the repo with cross-origin isolation.

SharedArrayBuffer (which wasm threads are built on) only exists on pages that
send these two headers; python3 -m http.server does not, so this is that plus
the headers. Serves the repo root so /web/ and /models/ both resolve.

    python3 web/serve.py [port]        # default 8000, then open /web/
"""
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler


class Isolated(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def log_message(self, *args):
        pass   # quiet


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    print(f"serving on http://127.0.0.1:{port}/web/ (cross-origin isolated)")
    HTTPServer(("127.0.0.1", port), Isolated).serve_forever()
