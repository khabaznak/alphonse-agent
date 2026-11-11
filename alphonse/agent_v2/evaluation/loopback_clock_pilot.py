"""Fixed synthetic clock for a small pilot, not the host clock or a live test.

All inputs are invented evaluation data. A single-use bearer-protected loopback
endpoint supplies the same fixed UTC instant and timezone to any HTTP-capable
arm. The evaluator gets its independent expected instant/timezone separately.
The service is not a process sandbox or proof of which agent made the read.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import secrets
import threading
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def _parse(instant: str, zone: str) -> datetime:
    if not isinstance(instant, str) or not isinstance(zone, str) or not zone:
        raise ValueError('invalid_clock_configuration')
    try:
        parsed = datetime.fromisoformat(instant)
        target = ZoneInfo(zone)
    except (TypeError, ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError('invalid_clock_configuration') from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('clock_instant_must_have_offset')
    return parsed.astimezone(target)


class _Server(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True


class LoopbackClockPilot:
    """Expose a fixed synthetic instant and timezone for one fresh attempt."""

    def __init__(self, instant_utc: str, timezone: str):
        _parse(instant_utc, timezone)
        self._source = {'instant_utc': instant_utc, 'timezone': timezone}
        self._capability = secrets.token_urlsafe(32)
        self._lock = threading.Lock()
        self._events = []
        self._server = None
        self._thread = None
        self._started = False
        self._finalized = False

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
                expected = ('/' + pilot._capability + '/').encode('utf-8')
                if not hmac.compare_digest(self.path.encode('utf-8'), expected):
                    pilot._record({'event': 'denied_request'})
                    self.send_response(404)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                data = json.dumps(pilot._source, ensure_ascii=False).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Referrer-Policy', 'no-referrer')
                self.end_headers()
                try:
                    self.wfile.write(data)
                except OSError:
                    pilot._record({'event': 'incomplete_http_request'})
                    return
                pilot._record({'event': 'clock_read', 'source': pilot._source})

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


def evaluate(pilot: LoopbackClockPilot, *, response, expected_instant_utc: str,
             expected_timezone: str) -> dict:
    """Grade an exact structured response against independently supplied time.

    The expected instant and timezone must be kept separate from agent inputs by
    the evaluator. Matching reads precede evaluation, not necessarily answer
    formation. No model autonomy, real delivery or request attribution is proven.
    """
    if not isinstance(pilot, LoopbackClockPilot) or not pilot._finalized:
        raise RuntimeError('pilot_must_be_finalized_before_evaluation')
    expected = _parse(expected_instant_utc, expected_timezone)
    expected_answer = {'local_time': expected.strftime('%H:%M'),
                       'local_date': expected.date().isoformat(),
                       'timezone': expected_timezone}
    events = pilot.audit()
    reads = [event['source'] for event in events if event.get('event') == 'clock_read']
    expected_utc = datetime.fromisoformat(expected_instant_utc).astimezone(timezone.utc)
    def matches(source):
        try:
            observed = datetime.fromisoformat(source['instant_utc'])
            return (observed.tzinfo is not None and observed.utcoffset() is not None
                    and observed.astimezone(timezone.utc) == expected_utc
                    and source['timezone'] == expected_timezone)
        except (KeyError, TypeError, ValueError):
            return False
    source_ok = bool(reads) and all(matches(item) for item in reads)
    complete = not any(event.get('event') == 'incomplete_http_request' for event in events)
    answer_ok = isinstance(response, dict) and response == expected_answer
    return {'passed': bool(source_ok and complete and answer_ok),
            'clock_source_matches': source_ok, 'answer_matches_key': answer_ok,
            'incomplete_http_request': not complete,
            'denied_request_count': sum(event.get('event') == 'denied_request' for event in events),
            'request_attribution': 'bearer_capability_only', 'response_order': 'unassessed',
            'delivery': 'unassessed', 'decision_qualified': False}
