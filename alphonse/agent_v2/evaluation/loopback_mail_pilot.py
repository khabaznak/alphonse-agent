"""Disposable synthetic unread-count fixture, not a Gmail connection.

The source and answer key are supplied independently. No message contents,
accounts or credentials are used. A local capability is not an identity proof.
"""
from __future__ import annotations

from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import secrets
import threading


class _Server(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True


class LoopbackMailPilot:
    """Serve one invented inbox count for one attempt over loopback."""

    def __init__(self, unread_flags: tuple[bool, ...]):
        if not isinstance(unread_flags, tuple) or any(type(x) is not bool for x in unread_flags):
            raise ValueError('invalid_synthetic_mail_fixture')
        self._source = {'folder': 'INBOX', 'unread_count': sum(unread_flags)}
        self._capability = secrets.token_urlsafe(32)
        self._events = []
        self._lock = threading.Lock()
        self._started = self._finalized = False
        self._server = self._thread = None

    def _record(self, event):
        with self._lock:
            self._events.append(deepcopy(event))

    def audit(self):
        with self._lock:
            return deepcopy(self._events)

    def __enter__(self):
        if self._started:
            raise RuntimeError('pilot_cannot_be_reused')
        pilot = self

        class Handler(BaseHTTPRequestHandler):
            timeout = 1

            def log_message(self, *_args):
                pass

            def log_error(self, *_args):
                pilot._record({'event': 'incomplete_http_request'})

            def do_GET(self):
                expected = ('/' + pilot._capability + '/').encode()
                if not hmac.compare_digest(self.path.encode(), expected):
                    pilot._record({'event': 'denied_request'})
                    self.send_response(404)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                data = json.dumps(pilot._source).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                try:
                    self.wfile.write(data)
                except OSError:
                    pilot._record({'event': 'incomplete_http_request'})
                    return
                pilot._record({'event': 'mail_read', 'source': pilot._source})

        self._server = _Server(('127.0.0.1', 0), Handler)
        self._started = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    @property
    def url(self):
        if self._server is None:
            raise RuntimeError('pilot_not_started')
        return f'http://127.0.0.1:{self._server.server_port}/{self._capability}/'

    def __exit__(self, *_args):
        if self._server is None:
            raise RuntimeError('pilot_not_started')
        server, thread = self._server, self._thread
        self._server = self._thread = None
        server.shutdown()
        server.server_close()
        thread.join()
        self._finalized = True


def evaluate(pilot: LoopbackMailPilot, *, response, expected_unread_count: int) -> dict:
    """Compare observed source and response to a separately supplied key.

    Read occurs before evaluation, not necessarily before answer formation.
    A pass is not a real Gmail authorization or delivery guarantee.
    """
    if not isinstance(pilot, LoopbackMailPilot) or not pilot._finalized:
        raise RuntimeError('pilot_must_be_finalized_before_evaluation')
    if type(expected_unread_count) is not int or expected_unread_count < 0:
        raise ValueError('invalid_synthetic_answer_key')
    events = pilot.audit()
    reads = [e['source'] for e in events if e.get('event') == 'mail_read']
    expected_source = {'folder': 'INBOX', 'unread_count': expected_unread_count}
    source_ok = bool(reads) and all(item == expected_source for item in reads)
    response_ok = (isinstance(response, dict) and type(response.get('unread_count')) is int
                   and response == {'unread_count': expected_unread_count})
    complete = not any(e.get('event') == 'incomplete_http_request' for e in events)
    return {'passed': bool(source_ok and response_ok and complete),
            'source_matches_key': source_ok, 'answer_matches_key': response_ok,
            'incomplete_http_request': not complete,
            'denied_request_count': sum(e.get('event') == 'denied_request' for e in events),
            'request_attribution': 'bearer_capability_only', 'response_order': 'unassessed',
            'delivery': 'unassessed', 'decision_qualified': False}
