"""Disposable candidate chooser using the configured inference router.

This adapter is not registered in the household runtime. Tool descriptions
are supplied by the fixture host; they are not an authorization decision.
The candidate and invocation service must independently enforce access and
argument validation. Visible byte bounds are not provider token limits.
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Callable

from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.inference.models import InferencePurpose, InferenceRequest, InferenceResult
from alphonse.agent_v2.core.inference.openai_codex import _json_envelope
from alphonse.agent_v2.core.inference.router import InferenceRouter


class RouterCandidateChooser:
    """Choose a candidate action against an explicitly supplied tool catalog."""

    def __init__(self, *, router: InferenceRouter, tools: tuple[ToolDescriptor, ...],
                 max_input_tokens: int, cancellation_checker: Callable[[], bool] | None = None):
        if not isinstance(router, InferenceRouter) or type(max_input_tokens) is not int or max_input_tokens <= 0:
            raise ValueError('candidate_router_configuration_invalid')
        if cancellation_checker is not None and not callable(cancellation_checker):
            raise ValueError('candidate_router_configuration_invalid')
        if not isinstance(tools, tuple):
            raise ValueError('candidate_router_tools_invalid')
        catalog = []
        seen = set()
        for tool in tools:
            if (not isinstance(tool, ToolDescriptor) or tool.kind is not ToolKind.NATIVE
                    or tool.read_only is not True or not isinstance(tool.tool_id, str)
                    or not tool.tool_id.strip() or tool.tool_id in seen
                    or not isinstance(tool.description, str) or not isinstance(tool.argument_schema, dict)):
                raise ValueError('candidate_router_tools_invalid')
            seen.add(tool.tool_id)
            catalog.append({'tool_id': tool.tool_id, 'description': tool.description,
                            'argument_schema': deepcopy(tool.argument_schema)})
        try:
            self._catalog_json = json.dumps(catalog, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError('candidate_router_tools_invalid') from exc
        self._tool_ids = frozenset(seen)
        self.router = router
        self.max_input_tokens = max_input_tokens
        self.cancellation_checker = cancellation_checker

    def _cancelled(self) -> bool:
        if self.cancellation_checker is None:
            return False
        try:
            return self.cancellation_checker() is not False
        except Exception:
            return True

    def __call__(self, projection: str, max_output_tokens: int) -> dict:
        if not isinstance(projection, str) or type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError('candidate_router_request_invalid')
        if self._cancelled():
            raise ValueError('candidate_router_cancelled')
        instructions = (
            'Choose one action for Alphonse; do not execute tools yourself. '
            'Available host read tools: ' + self._catalog_json + '. '
            'Use kind tool with tool_id and arguments to read; answer or question with text to communicate. '
            'Return kind, tool_id, arguments and text; use null for fields not needed. '
            'Inspect sources relevant to missing facts and consider uncertainty; do not invent observations.\n\n'
        )
        schema = {'type': 'object', 'properties': {
            'kind': {'type': 'string', 'enum': ['tool', 'answer', 'question']},
            'tool_id': {'type': ['string', 'null']},
            'arguments': {'type': ['object', 'null']},
            'text': {'type': ['string', 'null']},
        }, 'required': ['kind', 'tool_id', 'arguments', 'text'], 'additionalProperties': False}
        request = InferenceRequest(prompt=instructions + projection,
            purpose=InferencePurpose.SIMPLE_LOOP_ACTION, output_schema=schema,
            cancel_checker=self.cancellation_checker,
            metadata={'evaluation': 'synthetic_router_candidate'})
        # This matches the existing Codex JSON transport, but cannot attest
        # framing of another provider or hidden provider tokens.
        if len(_json_envelope(request).encode('utf-8')) > self.max_input_tokens:
            raise ValueError('candidate_router_input_exceeds_reservation')
        failed = False
        try:
            result = self.router.generate_json(request)
        except Exception:
            failed = True
        if failed:
            raise ValueError('candidate_router_unavailable')
        if self._cancelled():
            raise ValueError('candidate_router_cancelled')
        if not isinstance(result, InferenceResult) or not isinstance(result.json_value, dict):
            raise ValueError('candidate_router_result_invalid')
        try:
            rendered = json.dumps(result.json_value, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError):
            raise ValueError('candidate_router_result_invalid') from None
        raw = result.raw_response if isinstance(result.raw_response, str) else ''
        if max(len(rendered.encode('utf-8')), len(raw.encode('utf-8'))) > max_output_tokens:
            raise ValueError('candidate_router_output_exceeds_reservation')
        decision = dict(result.json_value)
        # Ignore any model-authored usage. Only provider metadata can settle.
        decision.pop('usage', None)
        kind = decision.get('kind')
        valid = set(decision) == {'kind', 'tool_id', 'arguments', 'text'}
        if kind == 'tool':
            valid = (valid and isinstance(decision.get('tool_id'), str)
                     and decision['tool_id'] in self._tool_ids
                     and isinstance(decision.get('arguments'), dict) and decision.get('text') is None)
        elif kind in ('answer', 'question'):
            valid = (valid and isinstance(decision.get('text'), str) and bool(decision['text'].strip())
                     and decision.get('tool_id') is None and decision.get('arguments') is None)
        else:
            valid = False
        if not valid:
            decision = {'kind': 'invalid'}
        usage = result.usage if isinstance(result.usage, dict) else {}
        decision['usage'] = {key: usage.get(key) if type(usage.get(key)) is int and usage[key] >= 0 else None
                             for key in ('input_tokens', 'output_tokens')}
        return decision
