"""Synthetic, read-only prescription-memory source for an offline pilot.

This is a development fixture, not a live benchmark or a security sandbox.
Only caller-supplied invented data may be served. The evaluator supplies an
independent answer key after the server has stopped; no route serves that key
or the access audit. A bearer capability limits access but does not establish
process identity or prevent another same-host process from acquiring the URL.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import secrets
import threading
from typing import Any


@dataclass(frozen=True)
class PrescriptionSources:
    """Invented source records; never put real household records here."""

    conversation: dict[str, Any]
    prescription: dict[str, Any]
    treatment_status: dict[str, Any]


class _Server(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True


class LoopbackPrescriptionPilot:
    """Fresh single-attempt source service, shared by any HTTP-capable arm."""

    def __init__(self, sources: PrescriptionSources):
        if not isinstance(sources, PrescriptionSources):
            raise ValueError('invalid_synthetic_sources')
        for value in (sources.conversation, sources.prescription, sources.treatment_status):
            if not isinstance(value, dict) or not value:
                raise ValueError('invalid_synthetic_sources')
        # Reject values that cannot be served deterministically; disallow NaNs.
        try:
            json.dumps([sources.conversation, sources.prescription, sources.treatment_status], allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError('invalid_synthetic_sources') from exc
        self._sources = deepcopy(sources)
        self._capability = secrets.token_urlsafe(32)
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._server = None
        self._thread = None
        self._started = False
        self._finalized = False

    def _record(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(deepcopy(event))

    def audit(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._events)

    def __enter__(self):
        if self._started:
            raise RuntimeError('pilot_cannot_be_reused')
        pilot = self

        class Handler(BaseHTTPRequestHandler):
            timeout = 1

            def log_message(self, *_args):
                return

            def log_error(self, *_args):
                pilot._record({'event': 'incomplete_http_request'})

            def do_GET(self):
                prefix = '/' + pilot._capability
                paths = {
                    prefix + '/': ('index', {'sources': ['conversation', 'prescription', 'treatment-status']}),
                    prefix + '/conversation': ('conversation', pilot._sources.conversation),
                    prefix + '/prescription': ('prescription', pilot._sources.prescription),
                    prefix + '/treatment-status': ('treatment-status', pilot._sources.treatment_status),
                }
                match = next((v for k, v in paths.items()
                              if hmac.compare_digest(self.path.encode('utf-8'), k.encode('utf-8'))), None)
                if match is None:
                    # Preserve the attempt without retaining a potentially
                    # sensitive path or bearer capability. This cannot tell a
                    # benign wrong route from an unauthorized access attempt.
                    pilot._record({'event': 'denied_request'})
                    self.send_response(404)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                name, value = match
                body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Referrer-Policy', 'no-referrer')
                self.end_headers()
                try:
                    self.wfile.write(body)
                except OSError:
                    pilot._record({'event': 'incomplete_http_request'})
                    return
                pilot._record({'event': 'read', 'source': name, 'value': value})

        server = _Server(('127.0.0.1', 0), Handler)
        self._server = server
        self._started = True
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
        return self

    @property
    def url(self) -> str:
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


def _same_json_value(actual: Any, expected: Any) -> bool:
    """Compare JSON values without Python's bool/int or int/float coercions."""
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return (actual.keys() == expected.keys()
                and all(_same_json_value(actual[key], value) for key, value in expected.items()))
    if isinstance(actual, list):
        return (len(actual) == len(expected)
                and all(_same_json_value(a, b) for a, b in zip(actual, expected)))
    return actual == expected


def evaluate(pilot: LoopbackPrescriptionPilot, *, response: Any, expected_response: dict,
             expected_sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Grade exact response and separately specified source values after shutdown.

    This checks that matching reads preceded evaluator invocation, not when the
    model formed its answer or who made the reads. It does not assess natural
    language, medical safety, real user delivery or live-model performance.
    """
    if not pilot._finalized:
        raise RuntimeError('pilot_must_be_finalized_before_evaluation')
    required = {'index', 'conversation', 'prescription', 'treatment-status'}
    if (not isinstance(expected_response, dict) or not isinstance(expected_sources, dict)
            or set(expected_sources) != required or any(not isinstance(v, dict) for v in expected_sources.values())):
        raise ValueError('invalid_independent_expectations')
    events = pilot.audit()
    reads = [e for e in events if e.get('event') == 'read']
    observed = {e['source'] for e in reads if e['source'] in expected_sources
                and _same_json_value(e.get('value'), expected_sources[e['source']])}
    mismatch = any(e['source'] not in expected_sources or not _same_json_value(e.get('value'), expected_sources[e['source']])
                   for e in reads)
    incomplete = any(e.get('event') == 'incomplete_http_request' for e in events)
    answer_ok = _same_json_value(response, expected_response)
    return {'passed': bool(answer_ok and required <= observed and not mismatch and not incomplete),
            'answer_matches_key': answer_ok, 'observed_required_sources': sorted(observed),
            'source_mismatch': mismatch, 'incomplete_http_request': incomplete,
            'denied_request_count': sum(e.get('event') == 'denied_request' for e in events),
            'unauthorized_access_attempts': 'unassessed',
            'request_attribution': 'bearer_capability_only', 'delivery': 'unassessed',
            'response_order': 'unassessed', 'decision_qualified': False}
