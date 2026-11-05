"""Bounded, source-linked mission review; never serialize the tactical checkpoint."""
from __future__ import annotations

import json
from pathlib import PurePosixPath

from alphonse.agent_v2.system_one import SystemOneUnavailableError

from .act_evidence import link_readbacks, observation
from .task_understanding import input_receipt, snapshot

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
    def stop(reason, value=None, *, unavailable_refs=()):
        task.metadata['v3_act_context'] = {'schema_version': 3, 'status': 'blocked',
            'reason': reason, 'budget_bytes': ACT_STATE_MAX_BYTES,
            'state_bytes': _size(value) if value is not None else None,
            'unavailable_evidence_refs': list(unavailable_refs)}
        raise SystemOneUnavailableError(reason)

    contract = task.ensure_acceptance_contract()
    criteria = [{key: row[key] for key in ('id', 'statement', 'status', 'required', 'verification', 'evidence_refs') if key in row}
                for row in contract.get('criteria', []) if row.get('superseded') is not True]
    entries = {}
    for entry in state.evidence.entries:
        if isinstance(entry, dict) and entry.get('evidence_ref'):
            entries[entry['evidence_ref']] = entry
    for action in state.actions:
        ref = f"tactical-action:{action.action_id}"
        if action.status in {"planned", "running"} and ref not in entries:
            entries[ref] = {**action.to_dict(), "evidence_ref": ref, "phase_id": state.phase.phase_id}
    essential = set()
    missing = []
    criterion_ids = {}
    for row in criteria:
        if row.get('status') != 'satisfied' or row.get('required', True) is False or row.get('verification') == 'system':
            continue
        refs = dict.fromkeys(row.get('evidence_refs') or [])
        for ref in refs:
            if ref in entries and entries[ref].get('status') == 'success':
                essential.add(ref)
                criterion_ids.setdefault(ref, []).append(row['id'])
            else:
                missing.append(ref)
    if missing:
        stop('act_context_required_evidence_unavailable', unavailable_refs=missing)
    prepared = task.metadata.get('prepared_user_response') or {}
    understanding = snapshot(task)
    value = {
        'schema_version': 3,
        'task_understanding': understanding,
        'request_sources': list(request_sources),
        'required_criteria': [{key: row[key] for key in ('id', 'statement', 'evidence_refs') if key in row}
                              for row in criteria if row.get('required', True)],
        'user_constraints': task.metadata.get('user_constraints', {}),
        'prepared_response': {'message': prepared.get('message', ''), 'delivery_status': 'pending_final_review'},
        'execution_evidence': [],
        'context_coverage': {'omitted_result_refs': [], 'omitted_evidence_refs': [], 'evidence_is_not_permission': True},
    }

    # Capture the reviewed scope for each phase. A requested or allowed target
    # helps correlate observations. Scope alone is never evidence of execution.
    scopes = {state.phase.phase_id: list(state.phase.mutation_scope.allowed_paths)}
    for phase in task.metadata.get("v3_phase_history") or []:
        if not isinstance(phase, dict):
            continue
        plan = phase.get("phase") or {}
        scopes.setdefault(plan.get("phase_id"), (plan.get("mutation_scope") or {}).get("allowed_paths") or [])

    response_ref = 'tactical-action:' + str(prepared.get('tool_call_id') or '')
    rows = {}
    for entry in entries.values():
        ref = entry['evidence_ref']
        row = observation(entry, criterion_ids.get(ref, []))
        if entry.get('tool_id') == 'native.respond' and prepared.get('message') and ref == response_ref:
            row.pop('result', None)
            row['result_reference'] = 'prepared_response.message'
        # Scope applies to the exact project-relative argument or absolute path.
        # A same-named file in another directory is not a scope match.
        target = row['operation'].get('target', {})
        allowed = scopes.get(entry.get('phase_id') or state.phase.phase_id, [])
        path = target.get('path')
        if path:
            cwd = (entry['result'].get('cwd') if isinstance(entry.get('result'), dict) else None)
            raw_cwd = (entry.get('arguments') or {}).get('cwd')
            if path in allowed or (isinstance(cwd, str)
                                   and raw_cwd == cwd and any(
                                       not PurePosixPath(p).is_absolute() and '..' not in PurePosixPath(p).parts
                                       and str(PurePosixPath(cwd, p)) == path for p in allowed)):
                row['target_in_planned_scope'] = True
        elif target.get('kind') == 'project_path' and target.get('id') in allowed:
            row['target_in_planned_scope'] = True
        rows[ref] = row
    link_readbacks(rows.values())

    # A cited read-back and its linked invocation form one required unit.
    for ref in tuple(essential):
        verification = rows[ref].get('verification') or {}
        if linked := verification.get('preceding_mutation_ref'):
            essential.add(linked)
    value['execution_evidence'] = [row for ref, row in rows.items() if ref in essential]
    coverage = value['context_coverage']
    coverage['budget_bytes'] = ACT_STATE_MAX_BYTES
    coverage['unavailable_evidence_refs'] = [
        ref for ref, row in rows.items() if row['operational_outcome'] in {'denied', 'unavailable', 'invocation_outcome_unavailable'}
        or row['result_coverage']['status'] == 'unavailable'
    ]
    if _size(value) > ACT_STATE_MAX_BYTES:
        stop('act_context_required_input_exceeds_budget', value)
    included = set(essential)
    for ref, row in reversed(list(rows.items())):
        if ref in essential:
            continue
        value['execution_evidence'].append(row)
        if _size(value) <= ACT_STATE_MAX_BYTES - 2000:
            included.add(ref)
            continue
        value['execution_evidence'].pop()
        coverage['omitted_result_refs'].append(ref)
        shell = observation(entries[ref], [], include_result=False)
        value['execution_evidence'].append(shell)
        if _size(value) <= ACT_STATE_MAX_BYTES - 2000:
            rows[ref] = shell
            included.add(ref)
        else:
            value['execution_evidence'].pop()
            coverage['omitted_result_refs'].pop()
            coverage['omitted_evidence_refs'].append(ref)
    value['execution_evidence'] = [row for ref, row in rows.items() if ref in included]
    for row in value['execution_evidence']:
        if (verification := row.get('verification')) and verification.get('preceding_mutation_ref') not in included:
            if verification.get('preceding_mutation_ref'):
                verification['preceding_mutation_omitted'] = True
    if _size(value) > ACT_STATE_MAX_BYTES:
        stop('act_context_required_input_exceeds_budget', value)
    task.metadata['v3_act_context'] = {
        'schema_version': 3, 'status': 'ready', 'state_bytes': _size(value), 'budget_bytes': ACT_STATE_MAX_BYTES,
        'task_understanding': input_receipt(understanding),
        'evidence_refs': [
            row['evidence_ref'] for row in value['execution_evidence']],
        **{key: list(coverage[key]) for key in (
            'omitted_result_refs', 'omitted_evidence_refs', 'unavailable_evidence_refs')},
    }
    return value
