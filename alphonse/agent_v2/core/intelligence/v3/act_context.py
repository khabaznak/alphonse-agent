"""Bounded, source-linked mission review; never serialize the tactical checkpoint."""
from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy

from alphonse.agent_v2.system_one import SystemOneUnavailableError

# Conservative UTF-8 byte budgets leave room for questions/provider framing below
# Jev's per-question context limit without assuming a chars-per-token ratio.
ACT_STATE_MAX_BYTES = 24000


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False).encode('utf-8'))


def mission_review_state(task, state, outcome, review, available_tool_ids, *, request_sources=()):
    """Keep the request, steering and supporting observations complete and unique.

    Keep every observation referenced by a satisfied semantic criterion; other
    observations are optional context. Required content never silently clips.
    Oversized required input stops review explicitly, rather than manufacturing a
    positive judgment from a partial source or increasing the model context limit.
    """
    def stop(reason, value=None):
        task.metadata['v3_act_context'] = {'schema_version': 1, 'status': 'blocked',
            'reason': reason, 'budget_bytes': ACT_STATE_MAX_BYTES,
            'state_bytes': _size(value) if value is not None else None}
        raise SystemOneUnavailableError(reason)

    contract = task.ensure_acceptance_contract()
    criteria = [{key: row[key] for key in ('id', 'statement', 'status', 'required', 'verification', 'evidence_refs') if key in row}
                for row in contract.get('criteria', []) if row.get('superseded') is not True]
    entries = {}
    for entry in state.evidence.entries:
        if isinstance(entry, dict) and entry.get('evidence_ref'):
            entries[entry['evidence_ref']] = entry
    ordered = list(entries.values())
    essential = set()
    missing = []
    for row in criteria:
        if row.get('status') != 'satisfied' or row.get('required', True) is False or row.get('verification') == 'system':
            continue
        refs = set(row.get('evidence_refs') or [])
        for ref in refs:
            if ref in entries and entries[ref].get('status') == 'success':
                essential.add(ref)
            else:
                missing.append(ref)
    if missing:
        stop('act_context_required_evidence_unavailable')
    followups = [{key: entry[key] for key in ('evidence_ref', 'kind', 'result') if key in entry}
                 for entry in task.metadata.get('v3_resumption_evidence') or []]
    messages = [{'speaker': row.get('speaker'), 'content': row.get('content')}
                for row in task.conversation_records or [] if row.get('content') and row.get('content') != task.goal]
    prepared = task.metadata.get('prepared_user_response') or {}
    from .task_understanding import snapshot
    value = {
        "task_understanding": snapshot(task),
        'schema_version': 1, 'goal': task.goal, 'task_id': task.task_id,
        'check_verdict': ('success' if review.status.value == 'phase_verified_task_complete' else
                          'wip' if review.status.value == 'phase_verified_task_incomplete' else 'failure'),
        'check_reason': review.reason, 'acceptance_contract': {'criteria': criteria},
        'strategic_plan': {'objective': state.phase.objective, 'subgoals': [
            {'subgoal_id': subgoal.subgoal_id, 'objective': subgoal.objective,
             'depends_on': list(subgoal.depends_on),
             'completion': {'kind': subgoal.completion.kind, 'field': subgoal.completion.field,
                            'expected': subgoal.completion.expected, 'output_type': subgoal.completion.output_type},
             'allowed_side_effects': [effect.value for effect in subgoal.allowed_side_effects]}
            for subgoal in state.phase.subgoals]},
        'phase_outcome': outcome.to_dict(), 'completed_subgoal_ids': list(state.completed_subgoal_ids),
        'authenticated_followups': followups, 'request_sources': list(request_sources),
        'current_task_messages': messages, 'user_constraints': task.metadata.get('user_constraints', {}),
        'prepared_response': {'message': prepared.get('message', ''), 'delivery_status': 'pending_final_review'},
        'execution_evidence': [], 'execution_summary': dict(Counter(str(entry.get('status') or 'unknown') for entry in ordered)),
        'context_coverage': {'omitted_result_refs': [], 'omitted_evidence_refs': [], 'evidence_is_not_permission': True},
    }

    def observation(entry, include_result):
        row = {key: deepcopy(entry[key]) for key in ('evidence_ref', 'phase_id', 'subgoal_id', 'tool_id', 'status', 'error') if key in entry}
        if include_result and 'result' in entry:
            row['result'] = deepcopy(entry['result'])
        return row

    # The prepared response is included once, rather than both as a tool result
    # and in a separate response block. Preserve its evidence identity.
    response_action = str(prepared.get('tool_call_id') or '')
    for entry in ordered:
        if entry['evidence_ref'] not in essential:
            continue
        row = observation(entry, entry.get('tool_id') != 'native.respond' or not prepared.get('message') or entry['evidence_ref'] != 'tactical-action:' + response_action)
        if entry.get('tool_id') == 'native.respond' and prepared.get('message'):
            if not response_action or entry['evidence_ref'] != 'tactical-action:' + response_action:
                row = observation(entry, True)
            else:
                row['result_reference'] = 'prepared_response.message'
        value['execution_evidence'].append(row)
    if _size(value) > ACT_STATE_MAX_BYTES:
        stop('act_context_required_input_exceeds_budget', value)
    # Whole optional observations only. A missing excerpt is never evidence that
    # the tool failed, the record is absent or a requested effect occurred.
    for entry in reversed(ordered):
        if entry['evidence_ref'] in essential:
            continue
        row = observation(entry, True)
        value['execution_evidence'].append(row)
        if _size(value) > ACT_STATE_MAX_BYTES - 2000:
            value['execution_evidence'].pop()
            row = observation(entry, False)
            row['result_omitted'] = True
            value['execution_evidence'].append(row)
            value['context_coverage']['omitted_result_refs'].append(entry['evidence_ref'])
            if _size(value) > ACT_STATE_MAX_BYTES - 2000:
                value['execution_evidence'].pop()
                value['context_coverage']['omitted_evidence_refs'].append(entry['evidence_ref'])
    # The catalog is a relevance hint, never support for an executed effect.
    value['available_tool_ids'] = list(available_tool_ids)
    if _size(value) > ACT_STATE_MAX_BYTES:
        stop('act_context_required_input_exceeds_budget', value)
    task.metadata['v3_act_context'] = {
        'schema_version': 1, 'status': 'ready', 'state_bytes': _size(value), 'budget_bytes': ACT_STATE_MAX_BYTES,
        'evidence_refs': [row['evidence_ref'] for row in value['execution_evidence']],
        'omitted_result_refs': list(value['context_coverage']['omitted_result_refs']),
        'omitted_evidence_refs': list(value['context_coverage']['omitted_evidence_refs']),
    }
    return value
