"""Offline V3-to-Responses request adapter for synthetic pilot development.

This is not a live transport or an authorization boundary.  The injected
transport must declare itself an offline test double; that declaration does
not prevent network access.  A real run requires independently verified
isolation, billing limits, TypeSafe scope and explicit approval.
"""
from __future__ import annotations

from dataclasses import replace
import json
from time import monotonic
from typing import Any, Callable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from alphonse.agent_v2.core.inference import InferenceRequest, InferenceResult
from alphonse.agent_v2.core.inference.openai_codex import (
    _json_envelope, _markdown_envelope, _tool_planning_envelope,
)
from .token_usage_diagnostics import usage_diagnostics
from .responses_pilot_transport import PilotProviderResultError


def _check_schema(schema: Any) -> None:
    """Forbid external and local references: validation must not fetch data."""
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if '$ref' in value or '$dynamicRef' in value:
                raise ValueError('pilot_schema_references_unsupported')
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    if not isinstance(schema, dict):
        raise ValueError('pilot_schema_invalid')
    visit(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError:
        raise ValueError('pilot_schema_invalid') from None


class OfflineV3ResponsesProvider:
    """Adapt V3 envelopes to a bounded offline Responses-shaped transport.

    V3's existing Codex envelope is reused, including tool descriptions; JSON
    schemas are supplied as prompt context and independently checked locally.
    This does not reproduce constrained decoding or prove model equivalence.
    Input limits apply to the entire rendered envelope, not only the V3 prompt.
    Usage is passed through from the transport; missing usage is never estimated
    here or counted as zero.  The transport must enforce its own total budgets.
    """

    supports_image_input = False

    def __init__(self, transport: Any, *, max_input_tokens: int,
                 max_output_tokens: int, emit: Callable[[dict], None] | None = None):
        if getattr(transport, 'offline', None) is not True:
            raise ValueError('pilot_live_transport_not_authorized')
        if any(type(value) is not int or value <= 0 for value in (max_input_tokens, max_output_tokens)):
            raise ValueError('pilot_invalid_inference_limit')
        self.transport = transport
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.emit = emit

    def _prepare(self, request: InferenceRequest, kind: str) -> InferenceRequest:
        if not isinstance(request, InferenceRequest) or request.images:
            raise ValueError('pilot_request_unsupported')
        if request.output_schema is not None:
            _check_schema(request.output_schema)
        render = {'json': _json_envelope, 'markdown': _markdown_envelope,
                  'tool': _tool_planning_envelope}[kind]
        envelope = json.loads(render(request))
        if request.output_schema is not None:
            envelope['output_schema'] = request.output_schema
            envelope['schema_instruction'] = 'Return a JSON object conforming to output_schema.'
        prompt = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        if len(prompt.encode('utf-8')) > self.max_input_tokens:
            raise ValueError('pilot_v3_envelope_exceeds_input_limit')
        # The complete tool/schema content is in prompt, before reservation.
        # Keep cancellation and model profile but never send duplicate tools.
        return replace(request, prompt=prompt, tools=(), output_schema=None)

    def _call(self, request: InferenceRequest, kind: str) -> InferenceResult:
        prepared = self._prepare(request, kind)
        if prepared.cancel_checker and prepared.cancel_checker():
            raise ValueError('pilot_inference_cancelled')
        method = self.transport.generate_markdown if kind == 'markdown' else self.transport.generate_json
        started = monotonic()
        status = 'failed'
        usage: dict[str, Any] = {}
        observed_output = None
        try:
            result = method(prepared, max_input_tokens=self.max_input_tokens,
                            max_output_tokens=self.max_output_tokens)
            if not isinstance(result, InferenceResult):
                raise ValueError('pilot_invalid_provider_result')
            usage = result.usage if isinstance(result.usage, dict) else {}
            observed_output = (result.raw_response if isinstance(result.raw_response, str)
                               else result.content if result.content else None)
            for key, cap in (('input_tokens', self.max_input_tokens), ('output_tokens', self.max_output_tokens)):
                value = usage.get(key)
                if value is not None and (type(value) is not int or value < 0 or value > cap):
                    raise ValueError('pilot_provider_usage_invalid_or_over_limit')
            if prepared.cancel_checker and prepared.cancel_checker():
                raise ValueError('pilot_inference_cancelled')
            if kind != 'markdown':
                if not isinstance(result.json_value, dict):
                    raise ValueError('pilot_invalid_json_result')
                if request.output_schema is not None:
                    try:
                        valid = Draft202012Validator(request.output_schema).is_valid(result.json_value)
                    except Exception:
                        raise ValueError('pilot_schema_validation_failed') from None
                    if not valid:
                        raise ValueError('pilot_schema_validation_failed')
                if kind == 'tool':
                    result = replace(result, tool_call=result.json_value)
            status = 'returned'
            return result
        except PilotProviderResultError as exc:
            # Preserve individually valid provider-reported counts even when
            # the response is incomplete or rejected. No response text is kept.
            usage = dict(exc.provider_usage)
            raise
        finally:
            if self.emit:
                event = {'kind': 'system_two', 'status': status,
                         'duration_ms': round((monotonic() - started) * 1000, 3),
                         'provider_usage': {
                             key: usage.get(key) if type(usage.get(key)) is int and usage[key] >= 0 else None
                             for key in ('input_tokens', 'output_tokens')}}
                event['usage_diagnostics'] = usage_diagnostics(input_text=prepared.prompt,
                    output_text=observed_output, provider_usage=usage)
                try:
                    self.emit(event)
                except Exception:
                    pass  # Optional telemetry must not change agent execution.

    def generate_json(self, request: InferenceRequest) -> InferenceResult:
        return self._call(request, 'json')

    def generate_markdown(self, request: InferenceRequest) -> InferenceResult:
        return self._call(request, 'markdown')

    def plan_tool_call(self, request: InferenceRequest) -> InferenceResult:
        return self._call(request, 'tool')
