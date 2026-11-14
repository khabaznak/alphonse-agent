"""Synthetic candidate decisions through Alphonse's Codex CLI provider.

Not a production host or an authentication boundary. The supplied provider
must independently enforce subscription-only ChatGPT authentication; the
comparison runner is responsible for that preflight and isolation. Nothing in
this module invokes Codex at import time.
"""
from __future__ import annotations

import json
from time import monotonic
from typing import Any, Callable

from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest, InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.openai_codex import _json_envelope
from .codex_subscription_provider import CodexSubscriptionProvider
from .token_usage_diagnostics import usage_diagnostics

_ALLOWED_SOURCES = frozenset({'index', 'conversation', 'prescription', 'treatment-status'})


def _action_schema(sources: tuple[str, ...]) -> dict:
    """Constrain the synthetic host's wire format, not access or correctness."""
    return {'type': 'object', 'properties': {
        'kind': {'type': 'string', 'enum': ['tool', 'answer', 'question']},
        'tool_id': {'type': ['string', 'null']},
        'arguments': {'anyOf': [
            {'type': 'object', 'properties': {'source': {'type': 'string', 'enum': list(sources)}},
             'required': ['source'], 'additionalProperties': False},
            {'type': 'null'}]},
        'text': {'type': ['string', 'null']},
    }, 'required': ['kind', 'tool_id', 'arguments', 'text'], 'additionalProperties': False}


def _valid_action(decision: dict, sources: tuple[str, ...]) -> bool:
    """Check the action discriminator; the host still authorizes actual reads."""
    if set(decision) != {'kind', 'tool_id', 'arguments', 'text'}:
        return False
    kind = decision.get('kind')
    if kind == 'tool':
        args = decision.get('arguments')
        return (decision.get('tool_id') == 'native.fixture_read' and isinstance(args, dict)
                and set(args) == {'source'} and isinstance(args['source'], str) and args['source'] in sources
                and decision.get('text') is None)
    if kind in ('answer', 'question'):
        return (isinstance(decision.get('text'), str) and bool(decision['text'].strip())
                and decision.get('tool_id') is None and decision.get('arguments') is None)
    return False


class CodexSubscriptionChooser:
    """Compose real candidate decisions with the existing Codex provider.

    Codex does not enforce a provider-side output token limit through this
    interface. Input/output byte checks are local conservative bounds on visible
    text, not a guarantee about hidden tokens or subscription cost. The actual
    provider usage stays unknown when unavailable; estimates are telemetry only.
    """
    def __init__(self, *, provider: CodexSubscriptionProvider, model: str,
                 max_input_tokens: int, emit: Callable[[dict[str, Any]], None],
                 cancellation_checker: Callable[[], bool] | None = None,
                 allowed_source_names: tuple[str, ...] = ('index',),
                 source_description: str = 'Read an authorized synthetic source.'):
        if type(provider) is not CodexSubscriptionProvider:
            raise ValueError('codex_subscription_provider_required')
        if not isinstance(model, str) or not model.strip() or len(model) > 128:
            raise ValueError('codex_model_required')
        if provider.model != model:
            raise ValueError('codex_subscription_model_mismatch')
        if type(max_input_tokens) is not int or max_input_tokens <= 0 or not callable(emit):
            raise ValueError('codex_chooser_configuration_invalid')
        if cancellation_checker is not None and not callable(cancellation_checker):
            raise ValueError('codex_chooser_configuration_invalid')
        if (not isinstance(allowed_source_names, tuple) or not allowed_source_names or
                any(not isinstance(x, str) or x not in _ALLOWED_SOURCES for x in allowed_source_names) or
                len(set(allowed_source_names)) != len(allowed_source_names)):
            raise ValueError('invalid_fixture_sources')
        if not isinstance(source_description, str) or not source_description.strip() or len(source_description) > 1000:
            raise ValueError('invalid_source_description')
        self.provider, self.model = provider, model
        self.max_input_tokens, self.emit = max_input_tokens, emit
        self.cancellation_checker, self.allowed_source_names = cancellation_checker, allowed_source_names
        self.source_description = source_description

    def _cancelled(self) -> bool:
        if self.cancellation_checker is None:
            return False
        try:
            return self.cancellation_checker() is not False
        except Exception:
            return True

    def __call__(self, projection: str, max_output_tokens: int) -> dict[str, Any]:
        if not isinstance(projection, str) or type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError('invalid_candidate_request')
        if self._cancelled():
            raise ValueError('candidate_cancelled_before_codex')
        # The registered fixture tool belongs to the Alphonse host, not to the
        # Codex subprocess. The JSON provider envelope does not render
        # InferenceRequest.tools, so describe the *actual host capability* here;
        # the host independently rechecks tool identity, arguments and access.
        tool = {
            'tool_id': 'native.fixture_read',
            'description': self.source_description,
            'arguments': {'source': {'type': 'string', 'enum': list(self.allowed_source_names)}},
        }
        instructions = (
            'You are choosing one action for Alphonse, not executing the action in Codex. '
            'Alphonse has the following host tool available: ' + json.dumps(tool, ensure_ascii=False) + '. '
            'To use it, return a JSON object with kind="tool", tool_id="native.fixture_read", '
            'and arguments containing a listed source. Alphonse executes that action and supplies '
            'the observation on the next decision; it is not a tool you need to call from Codex. '
            'If available evidence suffices, return {"kind":"answer","text":"..."}. '
            'Always include kind, tool_id, arguments, and text; use null for fields not needed by the chosen action. '
            'Before answering, consider whether another available authorized source could materially '
            'change the conclusion, for example by clarifying timing, current status, or conflicting facts. '
            'Inspect relevant sources when needed; do not assume missing context from a partial record. '
            'Choose sources using the request, available source descriptions, and observations already read: '
            'prefer a source that can resolve a specific missing fact or material uncertainty. '
            'Do not read a source whose description is clearly unrelated just because it is available. '
            'If its relevance is uncertain and it could resolve a material '
            'uncertainty, inspect it before answering. '
            'Use a question only when essential information is genuinely missing after available '
            'source descriptions and relevant observations have been considered. Do not ask the user to enable an '
            'already listed tool, and do not claim an observation that was not supplied.\n\n'
        )
        request = InferenceRequest(prompt=instructions + projection,
            purpose=InferencePurpose.SIMPLE_LOOP_ACTION,
            model_profile=ModelProfile(provider='openai_codex', model=self.model,
                                       profile_id='synthetic-subscription-comparison'),
            cancel_checker=self.cancellation_checker, output_schema=_action_schema(self.allowed_source_names),
            metadata={'evaluation': 'synthetic_subscription_comparison'})
        # Include the same JSON envelope this provider actually renders. This
        # still cannot account for provider-side framing or hidden tokens.
        observed_input = _json_envelope(request)
        if len(observed_input.encode('utf-8')) > self.max_input_tokens:
            raise ValueError('candidate_codex_input_exceeds_reservation')
        started = monotonic()
        failed = False
        try:
            result = self.provider.generate_json(request)
        except Exception:
            failed = True
            self.emit({'kind': 'system_two', 'status': 'failed', 'provider_usage': None,
                       'usage_diagnostics': usage_diagnostics(input_text=observed_input, output_text=None),
                       'duration_ms': round((monotonic() - started) * 1000, 3)})
        if failed:
            # Raise outside the handler so a provider error containing source
            # data or credentials is not retained in exception context.
            raise ValueError('candidate_codex_unavailable')
        if not isinstance(result, InferenceResult) or not isinstance(result.json_value, dict):
            raw_usage = result.usage if isinstance(result, InferenceResult) and isinstance(result.usage, dict) else {}
            known = {key: raw_usage.get(key) if type(raw_usage.get(key)) is int and raw_usage[key] >= 0 else None
                     for key in ('input_tokens', 'output_tokens')}
            self.emit({'kind': 'system_two', 'status': 'invalid', 'provider_usage': known,
                       'usage_diagnostics': usage_diagnostics(input_text=observed_input, output_text=None,
                                                             provider_usage=known),
                       'duration_ms': round((monotonic() - started) * 1000, 3)})
            raise ValueError('candidate_codex_invalid_decision')
        raw = result.raw_response if isinstance(result.raw_response, str) else None
        if raw is None:
            # A mock or future provider can return parsed JSON without raw text.
            # Do not report a fabricated observed output estimate.
            output_check = json.dumps(result.json_value, ensure_ascii=False)
        else:
            output_check = raw
        usage = result.usage if isinstance(result.usage, dict) else {}
        provider_usage = {key: usage.get(key) if type(usage.get(key)) is int and usage[key] >= 0 else None
                          for key in ('input_tokens', 'output_tokens')}
        status = 'cancelled' if self._cancelled() else ('over_output_bound' if len(output_check.encode('utf-8')) > max_output_tokens else 'returned')
        self.emit({'kind': 'system_two', 'status': status, 'provider_usage': provider_usage,
                   'usage_diagnostics': usage_diagnostics(input_text=observed_input, output_text=raw,
                                                         provider_usage=provider_usage),
                   'duration_ms': round((monotonic() - started) * 1000, 3),
                   'limits': 'visible_text_bytes_not_provider_token_enforcement'})
        if status != 'returned':
            raise ValueError('candidate_codex_' + status)
        decision = dict(result.json_value)
        decision.pop('usage', None)  # Model text cannot manufacture provider billing metadata.
        if not _valid_action(decision, self.allowed_source_names):
            # Preserve no model text; the candidate may use its existing
            # bounded format-repair path, never dispatch an invalid action.
            decision = {'kind': 'invalid'}
        decision['usage'] = provider_usage
        return decision
