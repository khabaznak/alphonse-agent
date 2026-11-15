"""Shared next-action policy for candidate hosts, not an authority boundary."""
from __future__ import annotations

import json
from typing import Any

from alphonse.agent_v2.core.inference.models import InferencePurpose, InferenceRequest


def action_schema(*, fixture_sources: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Build the shared wire schema with an optional fixture source constraint."""
    arguments: dict[str, Any] = {'type': ['object', 'null']}
    if fixture_sources is not None:
        arguments = {'anyOf': [
            {'type': 'object', 'properties': {'source': {'type': 'string', 'enum': list(fixture_sources)}},
             'required': ['source'], 'additionalProperties': False}, {'type': 'null'}]}
    return {'type': 'object', 'properties': {
        'kind': {'type': 'string', 'enum': ['tool', 'answer', 'question']},
        'tool_id': {'type': ['string', 'null']}, 'arguments': arguments,
        'text': {'type': ['string', 'null']},
    }, 'required': ['kind', 'tool_id', 'arguments', 'text'], 'additionalProperties': False}


def validate_source_progress(progress: Any) -> dict[str, list[str]]:
    """Validate bounded host-provided observation labels, never decide access."""
    if not isinstance(progress, dict) or set(progress) != {'observed_sources', 'unread_sources'}:
        raise ValueError('invalid_source_progress')
    values = [progress[key] for key in ('observed_sources', 'unread_sources')]
    if any(not isinstance(group, list) or len(group) > 128 for group in values):
        raise ValueError('invalid_source_progress')
    names = values[0] + values[1]
    if (any(not isinstance(name, str) or not name.strip() or len(name) > 256 for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('invalid_source_progress')
    return {'observed_sources': list(values[0]), 'unread_sources': list(values[1])}


def make_request(*, projection: str, catalog: list[dict[str, Any]], cancellation_checker: Any,
                 output_schema: dict[str, Any], task_id: str = '', user: str = '', project_id: str = '',
                 model_profile: Any = None, metadata: dict[str, Any] | None = None,
                 source_progress: dict[str, list[str]] | None = None) -> InferenceRequest:
    """Use identical source-selection and uncertainty guidance for every host.

    Optional source progress is host-provided observation metadata, never an
    oracle or an authorization decision. Hosts decide whether it is available.
    """
    instructions = (
        'You are choosing one action for Alphonse, not executing the action in Codex. '
        'Alphonse has the following host tools available: ' + json.dumps(catalog, ensure_ascii=False) + '. '
        'To read, return kind="tool" with a listed tool_id and arguments; Alphonse executes '
        'the action and supplies the observation on the next decision. These are not tools '
        'you need to call from Codex. If available evidence suffices, return an answer. '
        'Always include kind, tool_id, arguments, and text; use null for fields not needed. '
        'Before answering, consider whether another available authorized source could materially '
        'change the conclusion, for example by clarifying timing, current status, or conflicting facts. '
        'Inspect relevant sources when needed; do not assume missing context from a partial record. '
        'Choose sources using the request, available source descriptions, and observations already read: '
        'prefer a source that can resolve a specific missing fact or material uncertainty. '
        'For references to a current, named, or user-specific item, establish which item '
        'the request refers to from available evidence rather than assuming a match. '
        'A record being the only one found does not establish that it is the requested item. '
        'Resolve the item identity before applying a found or missing result to it; '
        'if available evidence cannot resolve it, report uncertainty or ask a useful question. '
        'Do not read a source whose description is clearly unrelated just because it is available. '
        'If its relevance is uncertain and it could resolve a material uncertainty, inspect it before answering. '
        'Use a question only when essential information is genuinely missing after available '
        'source descriptions and relevant observations have been considered. Do not ask the user to enable an '
        'already listed tool, and do not claim an observation that was not supplied.\n\n'
    )
    tool_ids = {item.get('tool_id') for item in catalog if isinstance(item, dict)}
    if {'native.search_memory', 'native.read_memory_source'} <= tool_ids:
        instructions += (
            'For questions about past conversations, use search to discover candidate sources. '
            'A complete preview may provide the needed evidence; if it is shortened or missing '
            'material context, read the relevant source before answering. Select sources that '
            'establish facts requested by the user; '
            'exclude incidental details that do not help answer the question. If no relevant '
            'source is available, say what is unknown.\n\n')
    if source_progress is not None:
        source_progress = validate_source_progress(source_progress)
        instructions += (
            'The following source progress is based only on completed host observations; '
            'it is not a list of required sources or a judgment of relevance. '
            'Compare the facts needed to answer with what the observations actually establish. '
            'If an essential fact is unsupported, consider whether an unread source can establish it. '
            + json.dumps(source_progress, ensure_ascii=False) + '\n\n')
    return InferenceRequest(prompt=instructions + projection, purpose=InferencePurpose.SIMPLE_LOOP_ACTION,
        output_schema=output_schema, cancel_checker=cancellation_checker, task_id=task_id, user=user,
        project_id=project_id, model_profile=model_profile, metadata=metadata or {})


def normalize_action(value: dict[str, Any], allowed_tool_ids: frozenset[str]) -> dict[str, Any]:
    """Reject malformed actions; host invocation validates tool arguments and authority."""
    decision = dict(value)
    decision.pop('usage', None)  # Never trust model-authored accounting.
    valid = set(decision) == {'kind', 'tool_id', 'arguments', 'text'}
    kind = decision.get('kind')
    if kind == 'tool':
        valid = (valid and isinstance(decision.get('tool_id'), str)
                 and decision['tool_id'] in allowed_tool_ids and isinstance(decision.get('arguments'), dict)
                 and decision.get('text') is None)
    elif kind in ('answer', 'question'):
        valid = (valid and isinstance(decision.get('text'), str) and bool(decision['text'].strip())
                 and decision.get('tool_id') is None and decision.get('arguments') is None)
    else:
        valid = False
    return decision if valid else {'kind': 'invalid'}
