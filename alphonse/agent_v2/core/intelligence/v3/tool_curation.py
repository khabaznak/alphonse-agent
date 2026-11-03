"""Request-level relevance hints. Runtime exposure and execution remain authoritative."""
from __future__ import annotations

import hashlib
import json
import re

from alphonse.agent_v2.core.memory.contracts import conversation_item
from alphonse.agent_v2.system_one import system_one_error_code

STATE_VERSION = 1
STATE_BUDGET_BYTES = 48000
BASELINE_TOOLS = {'native.respond', 'native.ask_question'}
RECALL_TOOLS = {'native.search_memory', 'native.read_memory_source'}


def request_tool_state(task, context) -> dict:
    """Build named facts without a generated plan or serialized system prompt.

    Preserve whole records. Missing/oversized records are explicit and cause the
    caller to defer exclusion to System Two, rather than classify clipped data.
    Skill resources use the same ownership checks as execution guidance.
    """
    state = {'schema_version': STATE_VERSION, 'request': {'text': task.goal},
             'conversation': [], 'selected_skills': [], 'known_context': [], 'evidence': [],
             'coverage': {'resource_paths_are_not_resource_contents': True}, 'omissions': []}

    def add(section, value, identity):
        state[section].append(value)
        if len(json.dumps(state, ensure_ascii=False).encode('utf-8')) > STATE_BUDGET_BYTES:
            state[section].pop()
            state['omissions'].append({'source_id': identity, 'reason': 'over_budget'})

    recent = getattr(getattr(context, 'conversation_store', None), 'list_recent', None)
    seen = set()
    if callable(recent) and task.user:
        try:
            events = list(recent(owner_user_id=task.user, limit=30))
            visible = [item for event in events if (item := conversation_item(event, str(task.user)))
                       and item.visible_to(str(task.user)) and item.source.role in {'user', 'assistant'}]
            visible.sort(key=lambda item: item.source.sequence or 0)
            # Never cut through a user/assistant exchange at the beginning.
            start = max(0, len(visible) - 8)
            while start > 0 and visible[start].source.role != 'user':
                start -= 1
            state['coverage']['earlier_conversation_available'] = start > 0 or len(events) >= 30
            for item in visible[start:]:
                role = item.source.role
                seen.add((role, item.content))
                add('conversation', {'speaker': 'Requester' if role == 'user' else 'Alphonse',
                    'text': item.content, 'source_id': item.source.source_id,
                    'sequence': item.source.sequence, 'observed_at': item.source.observed_at, 'provenance': item.source.provenance}, item.source.source_id)
        except Exception:
            state['omissions'].append({'source_id': 'conversation', 'reason': 'unavailable'})
    # The originating exchange may be older than the short tail. Only use
    # source bindings already revalidated under the current requester.
    retry = task.metadata.get('v3_retry_continuity') or {}
    reader = getattr(getattr(context, 'conversation_store', None), 'read', None)
    if retry.get('status') == 'selected' and retry.get('access_status') == 'available' and callable(reader):
        for identity in retry.get('source_ids') or []:
            try:
                event = reader(event_id=identity.removeprefix('conversation:'), requester_user_id=str(task.user))
                item = conversation_item(event, str(task.user))
                if not item or not item.visible_to(str(task.user)) or item.source.source_id != identity:
                    raise ValueError('source_unavailable')
                if (item.source.role, item.content) not in seen:
                    seen.add((item.source.role, item.content))
                    add('conversation', {'speaker': 'Requester' if item.source.role == 'user' else 'Alphonse',
                        'text': item.content, 'source_id': identity, 'sequence': item.source.sequence, 'observed_at': item.source.observed_at,
                        'provenance': item.source.provenance}, identity)
            except Exception:
                state['omissions'].append({'source_id': identity, 'reason': 'unavailable'})
    state['conversation'].sort(key=lambda record: record.get('sequence') or 0)
    for index, record in enumerate(task.conversation_records or []):
        role = 'user' if record.get('speaker') == task.user else 'assistant'
        text = str(record.get('content') or '')
        if not text or (role, text) in seen or role == 'user' and text == task.goal:
            continue
        seen.add((role, text))
        add('conversation', {'speaker': 'Requester' if role == 'user' else str(record.get('speaker') or 'Alphonse'),
            'text': text, 'source_id': f'task:{task.task_id}:conversation:{index}'}, f'conversation:{index}')
    if not state['conversation'] and str(task.recent_conversation_md or '').strip() not in {'', '- (none)'}:
        add('known_context', {'kind': 'legacy_conversation', 'text': task.recent_conversation_md}, 'legacy_conversation')

    selection = task.metadata.get('v3_skill_selection') or {}
    getter = getattr(getattr(context, 'skill_store', None), 'get', None)
    for identity in dict.fromkeys(selection.get('selected_skill_ids') or []):
        try:
            skill = getter(identity) if callable(getter) else None
            if skill is None:
                raise ValueError('skill_unavailable')
            entry = {'source_id': identity, 'name': skill.name, 'description': skill.description,
                     'package_directory': skill.directory,
                     'workflow': re.split(r'(?m)^## Artifact: ', skill.instructions)[0].rstrip(), 'resources': []}
            resources = getattr(context, 'artifact_resources', None)
            if resources is not None:
                for record in resources.store.list():
                    if record.skill_id == identity and resources.can_use(record, str(task.user or '')):
                        entry['resources'].append({'manifest': resources.describe(record, str(task.user or '')),
                            'operating_instructions': resources.skills.artifact_instructions(identity, record.artifact_id)})
            add('selected_skills', entry, identity)
        except Exception:
            state['omissions'].append({'source_id': identity, 'reason': 'unavailable'})

    from .context import selected_project_context
    project = selected_project_context(task, context)
    memory = str(task.memory_context_md or '').split('\n\n# Current Request\n', 1)[0].strip()
    for kind, text in [('selected_project', project), ('household_memory', memory),
                       ('task_facts', task.facts_md), ('memory_facts', task.memory_facts_md)]:
        if str(text or '').strip() not in {'', '- (none)'}:
            add('known_context', {'kind': kind, 'text': str(text)}, kind)
    # Successful and failed observations both matter; neither is a permission grant.
    journal = list(task.evidence_journal or [])
    for history in task.metadata.get('v3_phase_history') or []:
        journal.extend((history.get('evidence') or {}).get('entries') or [])
    evidence_seen = set()
    for entry in journal[-8:]:
        ref = str(entry.get('evidence_ref') or '')
        if ref and ref in evidence_seen:
            continue
        evidence_seen.add(ref)
        add('evidence', {key: entry[key] for key in ('evidence_ref', 'tool_id', 'status', 'result', 'error') if key in entry}, ref)
    coverage = task.metadata.get('conversation_projection') or {}
    state['coverage'].update({key: coverage[key] for key in ('requires_recall', 'history_unavailable') if key in coverage})
    retry = task.metadata.get('v3_retry_continuity') or {}
    state['coverage']['recall_status'] = retry.get('status', 'not_performed')
    return state


def curate_request_tools(task, tools, context):
    """One relevance pass per input snapshot; no phase-plan-dependent recheck.

    Keep uncertain tools, preserve interaction/recall, and defer empty, incomplete,
    invalid, or unavailable classifications to the LLM with all exposed tools.
    """
    tools = tuple(tool for tool in tools if str(getattr(tool.kind, 'value', tool.kind)) != 'artifact')
    if not tools:
        return ()
    if context.is_cancelled():
        from alphonse.agent_v2.system_one import SystemOneUnavailableError
        raise SystemOneUnavailableError('v3_tool_curation_cancelled')
    state = request_tool_state(task, context)
    settings = getattr(context.system_one, 'settings', None)
    policy = {name: getattr(settings, name, None) for name in ('model', 'tool_selection_threshold', 'no_threshold')}
    signature = hashlib.sha256(json.dumps([state, policy, [tool.tool_id for tool in tools]],
        ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
    cached = task.metadata.get('v3_request_tool_curation') or {}
    if cached.get('signature') == signature:
        retained = set(cached['retained_tool_ids'])
        return tuple(tool for tool in tools if tool.tool_id in retained)
    metadata = {'schema_version': STATE_VERSION, 'signature': signature,
                'candidate_tool_ids': [tool.tool_id for tool in tools], 'thresholds': policy, 'state_bytes': len(json.dumps(state, ensure_ascii=False).encode()),
                'state_fields': list(state), 'omissions': list(state['omissions'])}
    retained = set(metadata['candidate_tool_ids'])
    selector = getattr(context.system_one, 'curate_request_tools', None)
    if state['omissions'] or metadata['state_bytes'] > STATE_BUDGET_BYTES:
        metadata['status'] = 'incomplete_state_llm_fallback'
    elif not callable(selector):
        metadata['status'] = 'unavailable_llm_fallback'
    else:
        try:
            result = selector(state=state, tools=tools)
            selected = set(result.selected_tool_ids) | set(result.ambiguous_tool_ids)
            candidates = set(metadata['candidate_tool_ids'])
            if selected - candidates:
                raise ValueError('system_one_tool_selection_unknown_id')
            metadata.update(result.to_metadata())
            metadata['status'] = 'used' if selected else 'empty_selection_llm_fallback'
            if selected:
                retained = selected | (BASELINE_TOOLS & candidates)
                if state['coverage'].get('requires_recall') or state['coverage'].get('recall_status') in {'ambiguous', 'unavailable', 'over_budget'}:
                    retained |= RECALL_TOOLS & candidates
        except Exception as exc:
            metadata.update(status='error_llm_fallback', error_code=system_one_error_code(exc), exception_type=type(exc).__name__)
    if context.is_cancelled():
        from alphonse.agent_v2.system_one import SystemOneUnavailableError
        raise SystemOneUnavailableError('v3_tool_curation_cancelled')
    metadata['retained_tool_ids'] = [tool.tool_id for tool in tools if tool.tool_id in retained]
    task.metadata['v3_request_tool_curation'] = metadata
    history = task.metadata.get('system_one_request_tool_selections')
    if not isinstance(history, list):
        history = []
        task.metadata['system_one_request_tool_selections'] = history
    history.append(dict(metadata))
    del history[:-20]
    context.emit_telemetry({'event': 'system_one_request_tool_selection', 'task_id': task.task_id, **metadata})
    return tuple(tool for tool in tools if tool.tool_id in retained)
