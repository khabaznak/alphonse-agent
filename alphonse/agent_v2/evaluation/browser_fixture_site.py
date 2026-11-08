"""Loopback-only disposable comparison site; synthetic data only."""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

from alphonse.agent_v2.evaluation.browser_submission_probe import Fixture


class Site(ThreadingHTTPServer):
    def __init__(self, address):
        super().__init__(address, Handler)
        self.fixtures = {}
        self.lock = threading.RLock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _send(self, data, content_type="application/json"):
        data = data.encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlsplit(self.path)
        key = parse_qs(parsed.query).get("run", [""])[0]
        if not key.isalnum() or len(key) > 30:
            self.send_error(400)
            return
        with self.server.lock:
            fixture = self.server.fixtures.setdefault(key, Fixture())
            if parsed.path == "/oracle":
                self._send(json.dumps({"verified": fixture.oracle(), "submissions": len(fixture.submissions)}))
            elif parsed.path == "/form":
                if fixture.submissions:
                    body = "<h1>Result: accepted</h1>" if fixture.oracle() else "<h1>Result: rejected</h1>"
                else:
                    body = (f'<h1>Synthetic form</h1><form method="post" action="/form?run={key}">'
                            '<label>Name <input name="name" required></label>'
                            '<label>Number <input name="number" required></label>'
                            '<button type="submit">Submit</button></form>')
                self._send("<!doctype html><title>Synthetic form</title>" + body, "text/html; charset=utf-8")
            else:
                self.send_error(404)

    def do_POST(self):
        parsed = urlsplit(self.path)
        key = parse_qs(parsed.query).get("run", [""])[0]
        if not key.isalnum() or len(key) > 30:
            self.send_error(400)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > 8192:
            self.send_error(413)
            return
        raw = self.rfile.read(length).decode()
        with self.server.lock:
            fixture = self.server.fixtures.setdefault(key, Fixture())
            if parsed.path == "/form":
                values = parse_qs(raw)
                if fixture.submissions:
                    self.send_response(303)
                    self.send_header("Location", "/form?run=" + key)
                    self.end_headers()
                    return
                before = fixture.observe()
                for target, field in (("1_1", "name"), ("1_2", "number")):
                    fixture.fill_if_unchanged(target, values.get(field, [""])[0], before)
                    before = fixture.observe()
                fixture.submit_if_unchanged("1_3", before)
                self.send_response(303)
                self.send_header("Location", "/form?run=" + key)
                self.end_headers()
            elif parsed.path == "/api":
                payload = json.loads(raw)
                op, args = payload.get("op"), payload.get("args", [])
                if op not in {"session_identity", "observe", "fill_if_unchanged", "verify_fill", "submit_if_unchanged", "verify_submission", "oracle", "snapshot"}:
                    self.send_error(400)
                    return
                self._send(json.dumps({"value": getattr(fixture, op)(*args),
                                       "submissions": len(fixture.submissions)}))
            else:
                self.send_error(404)


class SiteTransport:
    """Same fixture as the visible form, used by the synthetic V3 service."""
    def __init__(self, base, run):
        self.base, self.run = base, run
        self.observations = []
        self.submissions = []

    def _call(self, op, *args):
        request = Request(self.base + "/api?run=" + self.run,
                          data=json.dumps({"op": op, "args": args}).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            result = json.load(response)
        self.submissions = [None] * result["submissions"]
        return result["value"]

    def session_identity(self, actor, task): return self._call("session_identity", actor, task)
    def observe(self):
        result = self._call("observe")
        self.observations.append(result)
        return result
    def fill_if_unchanged(self, *args): return self._call("fill_if_unchanged", *args)
    def verify_fill(self, *args): return self._call("verify_fill", *args)
    def submit_if_unchanged(self, *args): return self._call("submit_if_unchanged", *args)
    def verify_submission(self, *args): return self._call("verify_submission", *args)
    def oracle(self): return self._call("oracle")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()
    server = Site(("127.0.0.1", args.port))
    print(f"fixture http://127.0.0.1:{server.server_port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
