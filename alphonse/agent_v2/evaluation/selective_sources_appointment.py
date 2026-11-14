"""Public synthetic appointment/contacts source-selection development case.

The answer key and read expectations are consulted only after the capability
service shuts down. This is not a held-out case or a real household calendar.
"""
from __future__ import annotations

from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import secrets
import threading
from typing import Any, Callable

from .loopback_prescription_pilot import _same_json_value

GOAL = ('When is the current service appointment and where is it? Return a JSON object '
        'with time and address as strings.')
SOURCE_DESCRIPTION = (
    'Read a synthetic source: index lists available sources; conversation contains earlier '
    'interactions; calendar contains appointments; contacts contains contact details; '
    'household-note contains household notes.'
)
ALLOWED_SOURCES = ('index', 'conversation', 'calendar', 'contacts', 'household-note')
CASE_IDS = ('plumber', 'electrician')


def sources(case_id: str) -> dict[str, dict[str, Any]]:
    if case_id not in CASE_IDS:
        raise ValueError('unknown_appointment_case')
    return {
        'conversation': {'current_appointment': 'visit-11' if case_id == 'plumber' else 'visit-22'},
        'calendar': {'appointments': [
            {'id': 'visit-11', 'time': '2026-04-20 10:00', 'contact': 'service-a'},
            {'id': 'visit-22', 'time': '2026-04-21 14:30', 'contact': 'service-b'},
        ]},
        'contacts': {'contacts': [
            {'id': 'service-a', 'address': '28 Elm Street'},
            {'id': 'service-b', 'address': '49 Pine Street'},
        ]},
        'household-note': {'shopping': 'bread and milk'},
    }


def _expected(case_id: str) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    if case_id not in CASE_IDS:
        raise ValueError('unknown_appointment_case')
    first = case_id == 'plumber'
    return ({'time': '2026-04-20 10:00' if first else '2026-04-21 14:30',
             'address': '28 Elm Street' if first else '49 Pine Street'}, {
        'conversation': {'current_appointment': 'visit-11' if first else 'visit-22'},
        'calendar': {'appointments': [
            {'id': 'visit-11', 'time': '2026-04-20 10:00', 'contact': 'service-a'},
            {'id': 'visit-22', 'time': '2026-04-21 14:30', 'contact': 'service-b'},
        ]},
        'contacts': {'contacts': [
            {'id': 'service-a', 'address': '28 Elm Street'},
            {'id': 'service-b', 'address': '49 Pine Street'},
        ]},
    })


class _Server(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True


class LoopbackAppointmentPilot:
    """Serve fresh invented records on capability-scoped loopback routes."""

    def __init__(self, case_id: str):
        self._records = deepcopy(sources(case_id))
        self._capability = secrets.token_urlsafe(32)
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._server = self._thread = None
        self._started = self._finalized = False

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
                prefix = '/' + pilot._capability + '/'
                routes = {prefix: ('index', {'sources': list(ALLOWED_SOURCES[1:])})}
                routes.update({prefix + name: (name, value) for name, value in pilot._records.items()})
                match = next((v for k, v in routes.items()
                              if hmac.compare_digest(self.path.encode(), k.encode())), None)
                if match is None:
                    pilot._record({'event': 'denied_request'})
                    self.send_response(404)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                name, value = match
                body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
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

        self._server = _Server(('127.0.0.1', 0), Handler)
        self._started = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
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


def evaluate(pilot: LoopbackAppointmentPilot, *, case_id: str, response: Any, status: Any) -> dict[str, Any]:
    """Score answer and matching source reads separately after shutdown.

    Bearer audit does not establish reader identity or answer-formation order.
    Question utility and real delivery are not assessed.
    """
    if not pilot._finalized:
        raise RuntimeError('pilot_must_be_finalized_before_evaluation')
    expected_answer, expected_sources = _expected(case_id)
    try:
        answer = json.loads(response) if isinstance(response, str) else None
    except (TypeError, ValueError):
        answer = None
    events = pilot.audit()
    reads = [e for e in events if e.get('event') == 'read']
    observed = {e['source'] for e in reads if e.get('source') in expected_sources
                and _same_json_value(e.get('value'), expected_sources[e['source']])}
    mismatch = any(e.get('source') in expected_sources
                   and not _same_json_value(e.get('value'), expected_sources[e['source']]) for e in reads)
    return {
        'answer_matches_key': _same_json_value(answer, expected_answer),
        'required_sources_covered': observed == set(expected_sources) and not mismatch,
        'observed_required_sources': sorted(observed),
        'household_note_read': any(e.get('source') == 'household-note' for e in reads),
        'index_read': any(e.get('source') == 'index' for e in reads),
        'source_mismatch': mismatch,
        'incomplete_http_request': any(e.get('event') == 'incomplete_http_request' for e in events),
        'candidate_status': status,
        'question_content': 'unassessed', 'request_attribution': 'bearer_capability_only',
        'response_order': 'unassessed', 'delivery': 'unassessed', 'decision_qualified': False,
    }


def run(case_id: str, run_candidate: Callable[[str], Any]) -> dict[str, Any]:
    pilot = LoopbackAppointmentPilot(case_id)
    with pilot:
        outcome = run_candidate(pilot.url)
    return evaluate(pilot, case_id=case_id, response=outcome.response, status=outcome.status)
