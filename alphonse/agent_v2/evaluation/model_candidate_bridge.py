"""Provider-neutral candidate decision bridge for offline integration work.

No provider, credentials or network client is created here. The transport must
be injected and explicitly identify itself as an offline test double; that
declaration is not an OS/network sandbox. This bridge does *not* authorize live
calls or prove a spending limit. A live implementation additionally needs an
independently enforced, non-bypassable metered gateway, input and output caps,
synthetic-data isolation and the experiment's explicit approval.

Unlike the existing scripted fixture adapter, usage is never invented: the
candidate receives only provider-reported usage, or unknown values. The bridge
does not establish that a provider's usage counters are accurate or complete.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from time import monotonic
from typing import Any, Callable, Protocol

from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest, InferenceResult
from .token_usage_diagnostics import usage_diagnostics

_INSTRUCTIONS = (
    'Return one JSON decision. To read an authorized source use '
    '{"kind":"tool","tool_id":"native.fixture_read","arguments":{"path":"..."}}. '
    'To answer use {"kind":"answer","text":"..."}; to request essential missing '
    'information use {"kind":"question","text":"..."}. '
    'Use the available synthetic source paths and do not claim evidence that is absent.\n\n'
)

_SOURCE_INSTRUCTIONS = (
    'Return one JSON decision. To read an authorized source use '
    '{"kind":"tool","tool_id":"native.fixture_read","arguments":{"source":"..."}}. '
    'Available synthetic sources: {sources}. '
    'To answer use {"kind":"answer","text":"..."}; to request essential missing '
    'information use {"kind":"question","text":"..."}. '
    'Use the available synthetic sources and do not claim evidence that is absent.\n\n'
)
_ALLOWED_SOURCES = frozenset({'index', 'conversation', 'prescription', 'treatment-status'})


def _diagnostics(bridge, input_text, output_text, provider_usage=None):
    return usage_diagnostics(input_text=input_text, output_text=output_text,
        provider_usage=provider_usage, bytes_per_token=bridge.estimate_bytes_per_token,
        tokens_per_word=bridge.estimate_tokens_per_word,
        selected_method=bridge.estimate_method, calibration=bridge.estimate_calibration)


class OfflineBoundedTransport(Protocol):
    """A test-only transport; offline declaration is not a security boundary."""
    offline: bool

    def generate_json(self, request: InferenceRequest, *, max_input_tokens: int,
                      max_output_tokens: int) -> InferenceResult: ...


@dataclass(frozen=True)
class BoundedV3Provider:
    """Expose a bounded offline transport through V3's provider interface.

    Transport-side bounds must include every prompt, schema, tool descriptor,
    image and framing byte. This adapter cannot enforce that contract. It is
    intentionally restricted to declared offline doubles until an actual safe
    provider gateway and separate live authorization exist.
    """
    transport: Any
    max_input_tokens: int
    max_output_tokens: int
    emit: Callable[[dict[str, Any]], None]
    supports_image_input: bool = False
    estimate_bytes_per_token: float = 4.0
    estimate_tokens_per_word: float = 1.33
    estimate_method: str = 'utf8_bytes'
    estimate_calibration: str | None = None

    def __post_init__(self):
        if getattr(self.transport, 'offline', None) is not True:
            raise ValueError('live_transport_not_authorized')
        if any(type(value) is not int or value <= 0 for value in
               (self.max_input_tokens, self.max_output_tokens)):
            raise ValueError('invalid_inference_limit')
        _diagnostics(self, None, None)

    def _invoke(self, method: str, request: InferenceRequest) -> InferenceResult:
        if request.images:
            raise ValueError('image_input_not_supported')
        # This is an early check only. The transport still must count the full
        # request and decline before dispatch if it exceeds its actual bound.
        if len(request.prompt.encode('utf-8')) > self.max_input_tokens:
            raise ValueError('provider_prompt_exceeds_input_limit')
        start = monotonic()
        try:
            result = getattr(self.transport, method)(request, max_input_tokens=self.max_input_tokens,
                                                     max_output_tokens=self.max_output_tokens)
        except Exception:
            self.emit({'kind': 'system_two', 'status': 'failed', 'provider_usage': None,
                       'usage_diagnostics': _diagnostics(self, request.prompt, None),
                       'duration_ms': round((monotonic() - start) * 1000, 3)})
            raise
        if not isinstance(result, InferenceResult):
            self.emit({'kind': 'system_two', 'status': 'invalid', 'provider_usage': None,
                       'duration_ms': round((monotonic() - start) * 1000, 3)})
            raise ValueError('invalid_provider_result')
        raw_usage = result.usage if isinstance(result.usage, dict) else {}
        usage = {key: raw_usage.get(key) if type(raw_usage.get(key)) is int and raw_usage[key] >= 0 else None
                 for key in ('input_tokens', 'output_tokens')}
        self.emit({'kind': 'system_two', 'status': 'returned', 'provider_usage': usage,
                   'usage_diagnostics': _diagnostics(self, request.prompt,
                       result.raw_response if isinstance(result.raw_response, str) else result.content or None, usage),
                   'duration_ms': round((monotonic() - start) * 1000, 3)})
        if ((usage['input_tokens'] is not None and usage['input_tokens'] > self.max_input_tokens)
                or (usage['output_tokens'] is not None and usage['output_tokens'] > self.max_output_tokens)):
            raise ValueError('provider_reported_usage_over_limit')
        return result

    def generate_markdown(self, request: InferenceRequest) -> InferenceResult:
        return self._invoke('generate_markdown', request)

    def generate_json(self, request: InferenceRequest) -> InferenceResult:
        return self._invoke('generate_json', request)

    def plan_tool_call(self, request: InferenceRequest) -> InferenceResult:
        return self._invoke('plan_tool_call', request)


@dataclass(frozen=True)
class CandidateDecisionBridge:
    # Construction rejects transports that do not declare themselves offline.
    # This declaration is not an operating-system or network boundary.
    offline = True
    transport: OfflineBoundedTransport
    max_input_tokens: int
    emit: Callable[[dict[str, Any]], None]
    estimate_bytes_per_token: float = 4.0
    estimate_tokens_per_word: float = 1.33
    estimate_method: str = 'utf8_bytes'
    estimate_calibration: str | None = None
    fixture_argument: str = 'path'
    allowed_source_names: tuple[str, ...] = ('index', 'conversation', 'prescription', 'treatment-status')

    def __post_init__(self):
        if getattr(self.transport, 'offline', None) is not True:
            raise ValueError('live_transport_not_authorized')
        if type(self.max_input_tokens) is not int or self.max_input_tokens <= 0:
            raise ValueError('invalid_input_limit')
        if not isinstance(self.fixture_argument, str) or self.fixture_argument not in {'path', 'source'}:
            raise ValueError('invalid_fixture_argument')
        if (not isinstance(self.allowed_source_names, tuple) or not self.allowed_source_names
                or any(not isinstance(name, str) or name not in _ALLOWED_SOURCES for name in self.allowed_source_names)
                or len(set(self.allowed_source_names)) != len(self.allowed_source_names)):
            raise ValueError('invalid_fixture_sources')
        _diagnostics(self, None, None)

    def __call__(self, projection: str, max_output_tokens: int) -> dict[str, Any]:
        if not isinstance(projection, str) or type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError('invalid_candidate_request')
        # The input cap is an instruction to the bounded transport, not proof
        # of billing protection: transport framing and schemas also count.
        instructions = (_SOURCE_INSTRUCTIONS.replace('{sources}', json.dumps(self.allowed_source_names))
                        if self.fixture_argument == 'source' else _INSTRUCTIONS)
        prompt = instructions + projection
        if len(prompt.encode('utf-8')) > self.max_input_tokens:
            raise ValueError('candidate_projection_exceeds_input_limit')
        request = InferenceRequest(prompt=prompt, purpose=InferencePurpose.TACTICAL_ACTION,
                                   metadata={'evaluation': 'synthetic_candidate'})
        start = monotonic()
        try:
            result = self.transport.generate_json(request, max_input_tokens=self.max_input_tokens,
                                                  max_output_tokens=max_output_tokens)
        except Exception:
            self.emit({'kind': 'system_two', 'status': 'failed',
                       'duration_ms': round((monotonic() - start) * 1000, 3),
                       'provider_usage': None,
                       'usage_diagnostics': _diagnostics(self, prompt, None)})
            raise
        if not isinstance(result, InferenceResult) or not isinstance(result.json_value, dict):
            self.emit({'kind': 'system_two', 'status': 'invalid',
                       'duration_ms': round((monotonic() - start) * 1000, 3),
                       'provider_usage': None})
            raise ValueError('invalid_provider_decision')
        # Never accept a model-generated 'usage' field. Only the transport's
        # separate provider usage metadata may settle candidate reservations.
        decision = dict(result.json_value)
        decision.pop('usage', None)
        raw_usage = result.usage if isinstance(result.usage, dict) else {}
        usage = {}
        for key in ('input_tokens', 'output_tokens'):
            value = raw_usage.get(key)
            usage[key] = value if type(value) is int and value >= 0 else None
        decision['usage'] = usage
        self.emit({'kind': 'system_two', 'status': 'returned',
                   'duration_ms': round((monotonic() - start) * 1000, 3),
                   'provider_usage': dict(usage),
                   'usage_diagnostics': _diagnostics(self, prompt,
                       result.raw_response if isinstance(result.raw_response, str) else result.content or None, usage)})
        return decision
