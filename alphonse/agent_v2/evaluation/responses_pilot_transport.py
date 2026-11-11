"""Disabled-by-default Responses API pilot transport and durable reservations.

This is experimental evaluation code, not a permission boundary or a proven
provider spend cap. It is not connected to the production engine or the existing
offline-only bridges. No credentials are read from the environment. The caller
must separately verify approved scope, isolation, prices, complete token counting,
and a non-bypassable network boundary before enabling actual provider traffic.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable

from alphonse.agent_v2.core.inference import InferenceRequest, InferenceResult

_URL = 'https://api.openai.com/v1/responses'


class PilotProviderResultError(ValueError):
    """A sanitized failure with any individually valid reported token counts.

    The provider payload and response text are deliberately not retained.
    Counts are telemetry, not authority to settle a candidate reservation.
    """
    def __init__(self, code: str, usage: Any = None):
        super().__init__(code)
        raw = usage if isinstance(usage, dict) else {}
        self.provider_usage = {key: raw.get(key) if type(raw.get(key)) is int and raw[key] >= 0 else None
                               for key in ('input_tokens', 'output_tokens')}


def requests_sender(url: str, body: dict[str, Any], api_key: str) -> dict[str, Any]:
    """Single non-redirecting HTTP request; never called until a pilot enables it.

    This network implementation is not an authorization, isolation or billing
    boundary. A network failure leaves the ledger reservation intact because the
    provider may have processed the request. No response body is logged.
    """
    if url != _URL:
        raise ValueError('pilot_endpoint_denied')
    import requests
    failed = False
    value = None
    try:
        with requests.Session() as session:
            session.trust_env = False
            session.mount('https://', requests.adapters.HTTPAdapter(max_retries=0))
            with session.post(url, json=body, headers={'Authorization': 'Bearer ' + api_key},
                              allow_redirects=False, timeout=(10, 120), stream=True) as response:
                if response.status_code != 200:
                    raise ValueError('pilot_provider_http_error')
                chunks, size = [], 0
                for chunk in response.iter_content(chunk_size=8192):
                    size += len(chunk)
                    if size > 1048576:
                        raise ValueError('pilot_provider_response_too_large')
                    chunks.append(chunk)
                value = json.loads(b''.join(chunks))
    except Exception:
        # Request exceptions may retain an Authorization header, and JSON
        # exceptions may retain provider text. Leave the handler before
        # raising so neither appears in the new exception's chain.
        failed = True
    if failed:
        raise ValueError('pilot_provider_request_failed')
    if not isinstance(value, dict):
        raise ValueError('pilot_provider_response_invalid')
    return value


@dataclass(frozen=True)
class PilotLimits:
    max_calls: int
    max_input_tokens: int
    max_output_tokens: int
    max_cost_units: int
    input_cost_units_per_token: int
    output_cost_units_per_token: int

    def __post_init__(self):
        if any(type(x) is not int or x <= 0 for x in vars(self).values()):
            raise ValueError('invalid_pilot_limits')


class PilotLedger:
    """Reserve worst-case token and caller-supplied price units before dispatch.

    Reservations are never automatically refunded, including on failure,
    unknown provider usage or restart. The cost units and prices are supplied
    by the operator, not retrieved or validated here. SQLite serializes calls
    using the same ledger; it cannot stop other clients bypassing this ledger.
    """
    def __init__(self, path: str, limits: PilotLimits):
        if not isinstance(path, str) or not path or path == ':memory:':
            raise ValueError('file_backed_ledger_required')
        self.path, self.limits = path, limits
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS pilot_budget (id INTEGER PRIMARY KEY CHECK(id=1), limits TEXT NOT NULL, calls INTEGER NOT NULL, input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, cost_units INTEGER NOT NULL)')
            encoded = json.dumps(vars(limits), sort_keys=True)
            db.execute('INSERT OR IGNORE INTO pilot_budget VALUES (1, ?, 0, 0, 0, 0)', (encoded,))
            if db.execute('SELECT limits FROM pilot_budget WHERE id=1').fetchone()[0] != encoded:
                raise ValueError('pilot_limits_changed')

    def reserve(self, input_tokens: int, output_tokens: int) -> None:
        if any(type(x) is not int or x <= 0 for x in (input_tokens, output_tokens)):
            raise ValueError('invalid_reservation')
        cost = input_tokens * self.limits.input_cost_units_per_token + output_tokens * self.limits.output_cost_units_per_token
        with sqlite3.connect(self.path, isolation_level=None) as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT calls, input_tokens, output_tokens, cost_units FROM pilot_budget WHERE id=1').fetchone()
            values = (row[0] + 1, row[1] + input_tokens, row[2] + output_tokens, row[3] + cost)
            limits = (self.limits.max_calls, self.limits.max_input_tokens,
                      self.limits.max_output_tokens, self.limits.max_cost_units)
            if any(x > y for x, y in zip(values, limits)):
                db.execute('ROLLBACK')
                raise ValueError('pilot_budget_exhausted')
            db.execute('UPDATE pilot_budget SET calls=?, input_tokens=?, output_tokens=?, cost_units=? WHERE id=1', values)
            db.execute('COMMIT')

    def snapshot(self) -> dict[str, int]:
        with sqlite3.connect(self.path) as db:
            row = db.execute('SELECT calls, input_tokens, output_tokens, cost_units FROM pilot_budget WHERE id=1').fetchone()
        return dict(zip(('calls', 'input_tokens', 'output_tokens', 'cost_units'), row))


class ResponsesPilotTransport:
    """Minimal Responses request with explicit caps; live networking is opt-in.

    The injected counter must bound *all* provider-billed input including API
    framing; no such counter is provided or certified here. A mock sender and
    counter permit offline tests. Production authorization and isolation are
    external prerequisites, not established by setting `enabled=True`.
    """
    offline = False

    def __init__(self, *, model: str, api_key: str, ledger: PilotLedger,
                 count_complete_input: Callable[[dict[str, Any]], int],
                 sender: Callable[..., Any], enabled: bool = False):
        if not isinstance(model, str) or not model.strip() or not isinstance(api_key, str) or not api_key.strip():
            raise ValueError('pilot_model_and_key_required')
        if not isinstance(ledger, PilotLedger) or not callable(count_complete_input) or not callable(sender):
            raise ValueError('pilot_transport_dependencies_required')
        if type(enabled) is not bool:
            raise ValueError('invalid_enabled_flag')
        self._model, self._key, self._ledger = model, api_key, ledger
        self._counter, self._sender, self._enabled = count_complete_input, sender, enabled

    def _generate(self, request: InferenceRequest, *, max_input_tokens: int,
                  max_output_tokens: int, mode: str) -> InferenceResult:
        if not self._enabled:
            raise ValueError('pilot_live_transport_disabled')
        if not isinstance(request, InferenceRequest) or not isinstance(request.prompt, str):
            raise ValueError('invalid_pilot_request')
        if request.tools or request.images or request.output_schema:
            raise ValueError('pilot_request_capability_unsupported')
        if type(max_input_tokens) is not int or max_input_tokens <= 0 or type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError('invalid_pilot_request_limits')
        if request.cancel_checker and request.cancel_checker():
            raise ValueError('pilot_cancelled_before_reservation')
        body = {'model': self._model, 'input': request.prompt,
                'max_output_tokens': max_output_tokens, 'store': False}
        if mode == 'json':
            body['text'] = {'format': {'type': 'json_object'}}
        tokens = self._counter(body)
        if type(tokens) is not int or tokens <= 0 or tokens > max_input_tokens:
            raise ValueError('pilot_input_limit_or_count_invalid')
        self._ledger.reserve(max_input_tokens, max_output_tokens)
        if request.cancel_checker and request.cancel_checker():
            raise ValueError('pilot_cancelled_after_reservation')
        # The sender is injected; arbitrary callers can supply unsafe senders.
        # The optional requests_sender above posts once without redirects, but
        # does not establish complete input counting or a non-bypassable gateway.
        sender_failed = False
        payload = None
        try:
            payload = self._sender(_URL, body, self._key)
        except Exception:
            sender_failed = True
        if sender_failed:
            raise ValueError('pilot_sender_failed')
        if not isinstance(payload, dict):
            raise PilotProviderResultError('pilot_response_not_completed')
        raw_usage = payload.get('usage')
        if payload.get('status') != 'completed':
            raise PilotProviderResultError('pilot_response_not_completed', raw_usage)
        if payload.get('model') != self._model:
            raise PilotProviderResultError('pilot_response_model_mismatch', raw_usage)
        usage = raw_usage
        if not isinstance(usage, dict):
            raise PilotProviderResultError('pilot_provider_usage_missing')
        input_used, output_used = usage.get('input_tokens'), usage.get('output_tokens')
        if (type(input_used) is not int or type(output_used) is not int or
                input_used < 0 or output_used < 0 or input_used > max_input_tokens or output_used > max_output_tokens):
            raise PilotProviderResultError('pilot_provider_usage_invalid_or_overrun', usage)
        output = payload.get('output')
        if not isinstance(output, list):
            raise PilotProviderResultError('pilot_output_unavailable', usage)
        texts = [part.get('text') for item in output if isinstance(item, dict)
                 and item.get('type') == 'message' and item.get('role') == 'assistant'
                 for part in (item.get('content') if isinstance(item.get('content'), list) else [])
                 if isinstance(part, dict) and part.get('type') == 'output_text']
        if len(texts) != 1 or not isinstance(texts[0], str):
            raise PilotProviderResultError('pilot_output_unavailable', usage)
        if request.cancel_checker and request.cancel_checker():
            raise PilotProviderResultError('pilot_cancelled_after_dispatch', usage)
        usage_out = {'input_tokens': input_used, 'output_tokens': output_used}
        if mode == 'json':
            invalid_json = False
            value = None
            try:
                value = json.loads(texts[0])
            except (ValueError, TypeError):
                invalid_json = True
            if invalid_json:
                raise PilotProviderResultError('pilot_output_invalid_json', usage)
            if not isinstance(value, dict):
                raise PilotProviderResultError('pilot_output_not_object', usage)
            return InferenceResult(json_value=value, usage=usage_out, model_profile=request.model_profile)
        return InferenceResult(content=texts[0], usage=usage_out, model_profile=request.model_profile)

    def generate_json(self, request: InferenceRequest, *, max_input_tokens: int, max_output_tokens: int) -> InferenceResult:
        return self._generate(request, max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens, mode='json')

    def generate_markdown(self, request: InferenceRequest, *, max_input_tokens: int, max_output_tokens: int) -> InferenceResult:
        return self._generate(request, max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens, mode='text')
