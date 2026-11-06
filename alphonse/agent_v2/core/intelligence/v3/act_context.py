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


def mission_review_state(task, state, *, request_sources=(), context=None):
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
    plans = {state.phase.phase_id: state.phase.to_dict()}
    for phase in task.metadata.get("v3_phase_history") or []:
        if not isinstance(phase, dict):
            continue
        plan = phase.get("phase") or {}
        plans.setdefault(plan.get("phase_id"), plan)

    response_ref = 'tactical-action:' + str(prepared.get('tool_call_id') or '')
    rows = {}
    declared_mutations = set()
    for entry in entries.values():
        ref = entry['evidence_ref']
        row = observation(entry, criterion_ids.get(ref, []), task_id=str(task.task_id), actor_user_id=str(task.user), input_digest=str(understanding['input_digest']))
        if entry.get('tool_id') == 'native.respond' and prepared.get('message') and ref == response_ref:
            row.pop('result', None)
            row['result_reference'] = 'prepared_response.message'
        # Scope applies to the exact project-relative argument or absolute path.
        # A same-named file in another directory is not a scope match.
        target = row['operation'].get('target', {})
        plan = plans.get(entry.get("phase_id") or state.phase.phase_id, {})
        allowed = (plan.get("mutation_scope") or {}).get("allowed_paths", [])
        declared_effects = next((subgoal.get("allowed_side_effects", []) for subgoal in plan.get("subgoals", [])
                                 if subgoal.get("subgoal_id") == entry.get("subgoal_id")), [])
        if any(effect not in {"read_only", "user_response"} for effect in declared_effects):
            declared_mutations.add(ref)
        spec = (entry.get('arguments') or {}).get('verification') or {}
        source = spec.get('source') if isinstance(spec, dict) else None
        if entry.get('tool_id') == 'native.bash' and source in {'skill_data_json', 'workspace_text'}:
            from alphonse.agent_v2.core.tools.registry.native.bash_verification import binding_still_authorized, workspace_still_authorized
            result = entry.get('result') if isinstance(entry.get('result'), dict) else {}
            proof = result.get('verification') if isinstance(result.get('verification'), dict) else {}
            check = binding_still_authorized if source == 'skill_data_json' else workspace_still_authorized
            if not check(task=task, arguments=entry.get('arguments') or {}, context=context, target=proof.get('target')):
                # Redact even expired or invalid receipts after access is revoked.
                row.pop('result', None)
                row.pop('verification', None)
                row['operation'].pop('independent_readback', None)
                row['operation'].pop('target', None)
                row['operation']['kind'] = 'unknown'
                row['effect_status'] = 'unknown'
                row['result_coverage']['status'] = 'unavailable'
        if row['operation'].get('independent_readback'):
            scope = plan.get('mutation_scope') or {}
            authorized = (source == 'workspace_text' and 'project_mutation' in declared_effects
                          and spec.get('path') in allowed and spec.get('require_change') is True)
            authorized = authorized or (source == 'skill_data_json' and bool(scope.get('allow_external_effects'))
                and bool(set(declared_effects) & {'project_mutation', 'external_reversible', 'external_irreversible'}))
            authorized = authorized or (source == 'public_json' and bool(scope.get('allow_external_effects'))
                and bool(set(declared_effects) & {'external_reversible', 'external_irreversible'}))
            if not authorized:
                row['operation'].pop('independent_readback', None)
                row['operation']['kind'] = 'unknown'
                row.pop('verification', None)
                row['effect_status'] = 'unknown'
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
    read_back_mutations = {
        verification.get("preceding_mutation_ref") for row in rows.values()
        if (verification := row.get("verification"))
        and verification.get("preceding_mutation_outcome") == "process_exited_zero"
    }
    coverage['unresolved_mutation_refs'] = [
        ref for ref, row in rows.items()
        if (row["operation"]["kind"] == "mutation"
            or (row["operation"]["kind"] == "unknown" and ref in declared_mutations))
        and row["operational_outcome"] in {"failed", "denied", "invocation_outcome_unavailable"}
    ]
    coverage['unverified_mutation_refs'] = [
        ref for ref, row in rows.items()
        if row["effect_status"] == "attempted_unverified" and ref not in read_back_mutations
    ]
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
        shell = observation(entries[ref], [], include_result=False, task_id=str(task.task_id), actor_user_id=str(task.user), input_digest=str(understanding['input_digest']))
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
            'omitted_result_refs', 'omitted_evidence_refs', 'unavailable_evidence_refs', 'unresolved_mutation_refs', 'unverified_mutation_refs')},
    }
    return value
