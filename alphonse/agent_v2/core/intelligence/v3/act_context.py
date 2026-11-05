"""Bounded, source-linked mission review; never serialize the tactical checkpoint."""
from __future__ import annotations

import json
from copy import deepcopy

from alphonse.agent_v2.system_one import SystemOneUnavailableError

from .task_understanding import snapshot

# Conservative UTF-8 byte budgets leave room for questions/provider framing below
# Jev's per-question context limit without assuming a chars-per-token ratio.
ACT_STATE_MAX_BYTES = 24000


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False).encode('utf-8'))


def mission_review_state(task, state, *, request_sources=()):
    """Keep the request, steering and supporting observations complete and unique.

    Keep every observation referenced by a satisfied semantic criterion; other
    observations are optional context. Required content never silently clips.
    Oversized required input stops review explicitly, rather than manufacturing a
    positive judgment from a partial source or increasing the model context limit.
    """
    def stop(reason, value=None):
        task.metadata['v3_act_context'] = {'schema_version': 2, 'status': 'blocked',
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
    positions = {ref: index for index, ref in enumerate(entries)}
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
    prepared = task.metadata.get('prepared_user_response') or {}
    value = {
        'schema_version': 2,
        'task_understanding': snapshot(task),
        'request_sources': list(request_sources),
        'required_criteria': [{key: row[key] for key in ('id', 'statement', 'evidence_refs') if key in row}
                              for row in criteria if row.get('required', True)],
        'user_constraints': task.metadata.get('user_constraints', {}),
        'prepared_response': {'message': prepared.get('message', ''), 'delivery_status': 'pending_final_review'},
        'execution_evidence': [],
        'context_coverage': {'omitted_result_refs': [], 'omitted_evidence_refs': [], 'evidence_is_not_permission': True},
    }

    # Keep the prepared reply in one place while preserving its evidence identity.
    response_action = str(prepared.get('tool_call_id') or '')
    prepared_ref = f'tactical-action:{response_action}' if response_action and prepared.get('message') else None

    def observation(entry, include_result):
        row = {key: deepcopy(entry[key]) for key in ('evidence_ref', 'phase_id', 'subgoal_id', 'tool_id', 'status', 'error') if key in entry}
        if entry.get('tool_id') in {'native.respond', 'native.ask_question'}:
            row['observation_kind'] = 'conversation'
        elif include_result and _has_operational_result(entry):
            row['observation_kind'] = 'operational_result'
        else:
            row['observation_kind'] = 'attempt'
        if include_result and entry.get('tool_id') == 'native.respond' and row['evidence_ref'] == prepared_ref:
            row['result_reference'] = 'prepared_response.message'
        elif include_result and 'result' in entry:
            row['result'] = deepcopy(entry['result'])
        return row

    value['execution_evidence'] = [
        observation(entry, True) for entry in ordered if entry['evidence_ref'] in essential
    ]
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
    if _size(value) > ACT_STATE_MAX_BYTES:
        stop('act_context_required_input_exceeds_budget', value)
    # Prioritize linked evidence and recent optional results for budget selection.
    # Present selected observations in ledger order for conflict and read-back review.
    value['execution_evidence'].sort(key=lambda row: positions[row['evidence_ref']])
    task.metadata['v3_act_context'] = {
        'schema_version': 2, 'status': 'ready', 'state_bytes': _size(value), 'budget_bytes': ACT_STATE_MAX_BYTES,
        'evidence_refs': [row['evidence_ref'] for row in value['execution_evidence']],
        'omitted_result_refs': list(value['context_coverage']['omitted_result_refs']),
        'omitted_evidence_refs': list(value['context_coverage']['omitted_evidence_refs']),
    }
    return value


def _has_operational_result(entry):
    if entry.get('status') != 'success':
        return False
    result = entry.get('result')
    if not isinstance(result, dict):
        return isinstance(result, (str, list)) and bool(result)
    if 'exit_code' in result:
        stdout = result.get('stdout')
        return result['exit_code'] == 0 and not result.get('timed_out') and isinstance(stdout, str) and bool(stdout.strip())
    metadata = {'status', 'success', 'ok', 'duration_ms', 'command', 'arguments'}
    return any(key not in metadata and value not in (None, '', [], {}) for key, value in result.items())
