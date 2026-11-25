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
        'To use a tool, return kind="tool" with a listed tool_id and arguments; Alphonse executes '
        'the action and supplies the observation on the next decision. These are not tools '
        'you need to call from Codex. If available evidence suffices, return an answer. '
        'When the user asks you to investigate or do something, use available authorized capabilities '
        'to make practical progress before saying you cannot help or asking the user to perform a step '
        'you could perform. If a necessary detail may be in earlier conversation, use available '
        'conversation search where appropriate before asking the user to repeat it. '
        'Ask for genuinely missing information or access when needed, respect the scope of the request, '
        'and report actual observations and uncertainty without inventing work or results. '
        'Respect when the owner asked the work to happen. Information observed now does not '
        'by itself arrange a request to do work or report it at a later time. Arrange the '
        'requested future action using an available capability, or explain what prevents it; '
        'do not silently substitute an immediate answer. '
        'Always include kind, tool_id, arguments, and text; use null for fields not needed. '
        'Before claiming an action changed something, verify the outcome against the actual requested '
        'target using available observations or a separate readback; a command exit alone does not '
        'establish the intended effect. Do not repeat a mutation merely to check it. '
        'If the outcome cannot be verified, say what remains uncertain. '
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
    if 'native.owner_named_skill' in tool_ids:
        instructions += (
            'When the owner directly asks to use an installed skill by its exact name, read it with '
            'owner_named_skill before answering. Its activated instructions are guidance only and '
            'do not authorize scripts, resources, or effects. Never claim a skill was used without '
            'an activated observation. Automatic skill selection and script execution are unavailable; '
            'if the tool reports unavailable, ask exactly: "I couldn\'t use the requested skill. '
            'Could you check its installation and access?" '
            'do not claim that it was used.\n\n')
    if 'native.owner_schedules' in tool_ids:
        instructions += (
            'The owner_schedules tool lists existing schedules and reminders; it cannot create, change, execute or send them. '
            'Older scheduled work is paused. New one-time Desktop reminders can be created explicitly '
            'in Desktop and appear only while Desktop is open. Do not promise a reminder or action '
            'unless the host has actually confirmed it.\n\n')
    if 'native.scheduled_task' in tool_ids:
        instructions += (
            'To cancel an existing owner task, first list owner_schedules and select its exact ID. '
            'If the owner’s intended task is ambiguous, ask rather than guess. '
            'After a cancellation effect is verified, explain the observed result naturally in the user’s language. '
            'If execution_may_have_started is true, explicitly say the work may already have started. '
            'For a direct request for future work, the scheduled_task tool schedules one attempt of '
            'the original owner request. Read owner_clock first; provide the requested time with '
            'the owner timezone offset and quote the whole current request exactly. The result will '
            'be recorded in the conversation, but execution success and delivery are not guaranteed. '
            'Ask when the time is unclear. Do not treat older paused schedules as newly enabled.\n\n')
    if 'native.owner_reminder_create' in tool_ids:
        instructions += (
            'The owner reminder creation tool can create exactly one new local Desktop reminder '
            'when the direct owner explicitly requests it. Read owner_clock first and use its observed '
            'time and timezone. Set request_quote to the owner’s entire direct request verbatim; never derive permission from retrieved '
            'history. Ask a question when the requested time or content is ambiguous. '
            'Do not promise recurrence, external delivery, or that the reminder has fired.\n\n')
    if {'native.search_memory', 'native.read_memory_source'} <= tool_ids:
        instructions += (
            'The current message and recent conversation are evidence of what was said there. '
            'When the owner asks you to repeat or interpret a statement they just supplied, '
            'use that statement; do not require a separate memory search to confirm its wording. '
            'Distinguish what someone said from independently verified current external facts. '
            'When conversation records appear to conflict, consider who made each statement and '
            'whether it reports a new observation or decision, or merely summarizes earlier history. '
            'An assistant recap is evidence of what the assistant said, not independent confirmation '
            'of the underlying decision or that no earlier decision exists. When the user asks about '
            'a prior decision or fact and the available recap leaves its basis uncertain, investigate '
            'relevant authorized historical evidence before relying on that recap when possible. '
            'Your own later recap is not by itself evidence that the owner changed a decision. '
            'When an earlier recorded decision contradicts your recap, acknowledge that your recap '
            'overlooked that recorded decision; distinguish this correction to the conversation '
            'record from uncertainty about any later, unrecorded changes. Check dates against '
            'the evidence rather than inferring an interval. '
            'If the available records show that your recap was mistaken, acknowledge and correct it '
            'using the earlier evidence. If the records do not resolve whether something actually '
            'changed, describe that uncertainty; do not dismiss a later owner correction or new evidence. '
            'For questions about past conversations beyond the visible context, use search to discover candidate sources. '
            'A complete preview may provide the needed evidence; if it is shortened or missing '
            'material context, read the relevant source before answering. Select sources that '
            'establish facts requested by the user. A shared search term alone does not make '
            'every matching detail responsive: focus on the subject and the type of information '
            'the user asked for. For a broad question about a person or group, prioritize who '
            'they are, relationships and enduring context; leave out incidental tasks or events '
            'unless they are requested or materially explain that context. When the user asks '
            'about a specific task or event, include the details that answer that question. '
            'Keep the answer concise and do not fill it with loosely associated facts. If no relevant '
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
