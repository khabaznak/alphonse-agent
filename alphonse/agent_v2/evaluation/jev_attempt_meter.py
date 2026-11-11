"""Per-HTTP-attempt accounting for a synthetic Jev pilot.

This wrapper is an injected transport for TypeSafeSystemOneClient, so each
client retry passes through the same ledger. It stores no request, response,
credentials, endpoint, question text or state. It is NOT a hard dollar or token
cap, an OS/network isolation boundary, or authorization for a live call: TypeSafe
does not expose a request output-token limit in the documented System One
schema, and actual framing, provider price and hidden/failed usage are unknown.
Use only with an explicitly approved synthetic pilot and an appropriately
isolated, single-request transport. The wrapped callable is not certified.
"""
from __future__ import annotations

import json
import re
import sqlite3
from time import monotonic
from typing import Any, Callable

from alphonse.agent_v2.evaluation.token_usage_diagnostics import usage_diagnostics
from alphonse.agent_v2.evaluation.jev_pilot_payload import serialize_payload
from alphonse.agent_v2.system_one import DEFAULT_SYSTEM_ONE_URL, SystemOneRequestError, system_one_error_code


class JevAttemptLedger:
    """Persist every attempt before dispatch; no automatic retry or refund.

    A started row with no terminal row represents an unknown outcome after a
    crash. The call-count ceiling applies only to this ledger and wrapper.
    """

    def __init__(self, path: str, *, max_attempts: int, max_payload_bytes: int):
        if not isinstance(path, str) or not path or path == ':memory:':
            raise ValueError('jev_ledger_requires_file')
        if any(type(v) is not int or v <= 0 for v in (max_attempts, max_payload_bytes)):
            raise ValueError('jev_invalid_limits')
        self.path, self.max_attempts, self.max_payload_bytes = path, max_attempts, max_payload_bytes
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS jev_pilot_limits (id INTEGER PRIMARY KEY CHECK(id=1), max_attempts INTEGER NOT NULL, max_payload_bytes INTEGER NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS jev_pilot_attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, payload_bytes INTEGER NOT NULL, question_count INTEGER NOT NULL, state TEXT NOT NULL, duration_ms REAL, input_tokens INTEGER, output_tokens INTEGER, input_estimated_tokens INTEGER, error_code TEXT)')
            columns = {row[1] for row in db.execute('PRAGMA table_info(jev_pilot_attempts)')}
            if 'response_model' not in columns:
                db.execute('ALTER TABLE jev_pilot_attempts ADD COLUMN response_model TEXT')
            db.execute('INSERT OR IGNORE INTO jev_pilot_limits VALUES (1, ?, ?)', (max_attempts, max_payload_bytes))
            if db.execute('SELECT max_attempts, max_payload_bytes FROM jev_pilot_limits WHERE id=1').fetchone() != (max_attempts, max_payload_bytes):
                raise ValueError('jev_limits_changed')

    def start(self, *, payload_bytes: int, question_count: int, estimated_input_tokens: int) -> int:
        if (type(payload_bytes) is not int or payload_bytes <= 0 or
                type(question_count) is not int or question_count <= 0 or
                type(estimated_input_tokens) is not int or estimated_input_tokens < 0):
            raise ValueError('jev_invalid_attempt')
        with sqlite3.connect(self.path, isolation_level=None) as db:
            db.execute('BEGIN IMMEDIATE')
            count = db.execute('SELECT COUNT(*) FROM jev_pilot_attempts').fetchone()[0]
            if count >= self.max_attempts or payload_bytes > self.max_payload_bytes:
                db.execute('ROLLBACK')
                raise ValueError('jev_pilot_attempt_limit_exceeded')
            cursor = db.execute('INSERT INTO jev_pilot_attempts (payload_bytes, question_count, state, input_estimated_tokens) VALUES (?, ?, ?, ?)',
                                (payload_bytes, question_count, 'started', estimated_input_tokens))
            db.execute('COMMIT')
            return cursor.lastrowid

    def finish(self, attempt_id: int, *, state: str, duration_ms: float,
               input_tokens: int | None = None, output_tokens: int | None = None,
               error_code: str | None = None, response_model: str | None = None) -> None:
        with sqlite3.connect(self.path, isolation_level=None) as db:
            cursor = db.execute('UPDATE jev_pilot_attempts SET state=?, duration_ms=?, input_tokens=?, output_tokens=?, error_code=?, response_model=? WHERE id=? AND state=?',
                                (state, duration_ms, input_tokens, output_tokens, error_code, response_model, attempt_id, 'started'))
            if cursor.rowcount != 1:
                raise ValueError('jev_attempt_not_pending')

    def snapshot(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute('SELECT id, payload_bytes, question_count, state, duration_ms, input_tokens, output_tokens, input_estimated_tokens, error_code, response_model FROM jev_pilot_attempts ORDER BY id').fetchall()
        return [{**dict(row), 'input_estimate_method': 'utf8_bytes_divided_by_4',
                 'input_estimate_calibration': 'unverified_for_model_and_language',
                 'usage_cost': 'unknown'} for row in rows]


class MeteredJevTransport:
    """Count retries and provider usage without trusting error or response text."""

    def __init__(self, transport: Callable, ledger: JevAttemptLedger, *, cancellation_checker: Callable | None = None):
        if (not callable(transport) or not isinstance(ledger, JevAttemptLedger)
                or (cancellation_checker is not None and not callable(cancellation_checker))):
            raise ValueError('jev_transport_dependencies_required')
        self.transport, self.ledger = transport, ledger
        self.cancellation_checker = cancellation_checker

    def _cancelled(self) -> bool:
        if self.cancellation_checker is None:
            return False
        try:
            result = self.cancellation_checker()
            return result if type(result) is bool else True
        except Exception:
            return True

    def __call__(self, url: str, api_key: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        if url != DEFAULT_SYSTEM_ONE_URL:
            raise ValueError('jev_pilot_endpoint_denied')
        if not isinstance(payload, dict) or not isinstance(payload.get('questions'), dict):
            raise ValueError('jev_pilot_invalid_payload')
        # This executes on every client retry, not only on the enclosing V3
        # decision. A malformed or failing checker is treated as cancellation.
        if self._cancelled():
            raise ValueError('jev_pilot_cancelled_before_dispatch')
        # The pilot HTTP sender uses this identical serialization. Other
        # injected transports must not be assumed to send these bytes.
        data = serialize_payload(payload)
        text = data.decode('utf-8')
        # The caller may own mutable nested dictionaries. Give the transport a
        # detached snapshot whose serialization is the metered request body.
        captured_payload = json.loads(data)
        estimated = usage_diagnostics(input_text=text, output_text=None)['input']['tokens']
        attempt_id = self.ledger.start(payload_bytes=len(data),
            question_count=len(payload['questions']), estimated_input_tokens=estimated)
        started = monotonic()
        if self._cancelled():
            self.ledger.finish(attempt_id, state='cancelled_before_dispatch', duration_ms=0)
            raise ValueError('jev_pilot_cancelled_before_dispatch')
        failure = None
        response = None
        try:
            response = self.transport(url, api_key, captured_payload, timeout)
        except Exception as exc:
            # Copy only sanitized fields before leaving the exception handler;
            # the exception itself can retain a request with credentials.
            status = exc.http_status if isinstance(exc, SystemOneRequestError) and exc.http_status in {400, 413, 415, 422} else None
            provider_code = exc.provider_code if status is not None else None
            failure = (system_one_error_code(exc), isinstance(exc, RuntimeError), status, provider_code)
        duration = round((monotonic() - started) * 1000, 3)
        cancelled_after_dispatch = self._cancelled()
        if failure is not None:
            code, runtime, request_status, provider_code = failure
            self.ledger.finish(attempt_id, state='failed', duration_ms=duration, error_code=code)
            if cancelled_after_dispatch:
                raise ValueError('jev_pilot_cancelled_after_dispatch')
            # Preserve known retryable error classes/codes, never exception text
            # or chained request objects that might contain credentials.
            if request_status is not None:
                raise SystemOneRequestError(request_status, provider_code)
            if runtime and re.fullmatch(r'system_one_(?:connection_failed|temporarily_unavailable:[0-9]{3}|http_error:[0-9]{3})', code):
                raise RuntimeError(code)
            if code in {'system_one_api_key_invalid', 'system_one_api_key_required',
                        'system_one_request_invalid', 'system_one_response_invalid',
                        'system_one_context_budget_exceeded'}:
                raise ValueError(code)
            raise ValueError('jev_pilot_transport_failed')
        raw = response.get('usage') if isinstance(response, dict) else None
        raw = raw if isinstance(raw, dict) else {}
        usage = {key: raw.get(key) if type(raw.get(key)) is int and raw[key] >= 0 else None
                 for key in ('input_tokens', 'output_tokens')}
        # The returned model can differ from a requested alias. Persist only a
        # bounded identifier; missing or malformed model is unknown, not the alias.
        model = response.get('model') if isinstance(response, dict) else None
        model = (model if isinstance(model, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}', model)
                 and not (isinstance(api_key, str) and api_key and api_key in model) else None)
        state = ('returned_after_cancellation' if cancelled_after_dispatch and isinstance(response, dict)
                 else 'returned' if isinstance(response, dict) else 'invalid')
        self.ledger.finish(attempt_id, state=state,
                           duration_ms=duration, input_tokens=usage['input_tokens'], output_tokens=usage['output_tokens'],
                           response_model=model)
        if cancelled_after_dispatch:
            raise ValueError('jev_pilot_cancelled_after_dispatch')
        if not isinstance(response, dict):
            raise ValueError('system_one_response_invalid')
        return response
