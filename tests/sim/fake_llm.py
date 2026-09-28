"""Fake LLM upstream for tests/sim: GET anything → 200; POST /v1/chat/completions with
{"sleep": N, "tag": "x"} holds the request open for N seconds. Events go to $SIM_EVENTS."""
import json
import os
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port, name = int(sys.argv[1]), sys.argv[2]
EVENTS = os.environ["SIM_EVENTS"]


def event(what, tag=""):
    with open(EVENTS, "a") as f:
        f.write(json.dumps({"t": time.time(), "model": name, "event": what, "tag": tag}) + "\n")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, body=b'{"ok":true}'):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        tag = body.get("tag", "")
        event("begin", tag)
        time.sleep(float(body.get("sleep", 0)))
        event("end", tag)
        self.reply()


signal.signal(signal.SIGTERM, lambda *a: (event("stopped"), os._exit(0)))
event("started")
ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
