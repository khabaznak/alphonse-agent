"""Bounded mission review with disposable evidence, never household records."""
import copy
import json
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.act_context import ACT_STATE_MAX_BYTES, mission_review_state
from alphonse.agent_v2.core.intelligence.v3.contracts import PhasePlan, PhaseSubgoal, PhaseOutcome, PhaseStatus, new_tactical_state
from alphonse.agent_v2.core.intelligence.v3.outer import PhaseReview, PhaseReviewStatus, V3OuterController
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.system_one import SystemOneUnavailableError, SystemOneRequestError, SystemOneReviewResult


def fixture():
    task = TaskState(task_id='fixture', message_id='fixture', user='owner', goal='Read the latest recorded schedule and report it.', status='running')
    task.set_acceptance_contract_from_markdown('1.- [ ] The latest record is retrieved\n2.- [ ] The answer reflects that record')
    for row, ref in zip(task.acceptance_contract['criteria'], ['read', 'reply']):
        row.update(status='satisfied', evidence_refs=['tactical-action:' + ref])
    plan = PhasePlan('lookup', 'Read and report', (PhaseSubgoal('read', 'Read the record', 'record',
        allowed_capabilities=('local_shell',)),), authorized_capabilities=('local_shell',))
    state = new_tactical_state(plan); state.status = PhaseStatus.PHASE_COMPLETE
    state.evidence.append({'evidence_ref': 'tactical-action:read', 'status': 'success', 'tool_id': 'native.bash',
        'result': {'exit_code': 0, 'stdout': '2026-10-30: FixtureBeta, Monday through Thursday'}})
    state.evidence.append({'evidence_ref': 'tactical-action:reply', 'status': 'success', 'tool_id': 'native.respond',
        'result': {'message': 'October 30: FixtureBeta, Monday through Thursday.'}})
    task.metadata['prepared_user_response'] = {'source': 'native.respond', 'tool_call_id': 'reply',
        'message': 'October 30: FixtureBeta, Monday through Thursday.', 'work_evidence_refs': ['tactical-action:read']}
    outcome = PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE)
    review = PhaseReview('lookup', PhaseReviewStatus.PHASE_VERIFIED_TASK_COMPLETE, 'Verified against retrieved records')
    return task, state, outcome, review


def test_large_checkpoint_duplicates_do_not_enter_final_review_and_essential_sources_stay_complete():
    task, state, outcome, review = fixture()
    payload = 'OPTIONAL_SCHEMA ' * 20000
    state.bindings['schema'] = {'value': payload}
    state.evidence.append({'evidence_ref': 'schema', 'status': 'success', 'tool_id': 'native.bash',
        'arguments': {'command': 'DO_NOT_SEND_COMMAND'}, 'result': {'stdout': payload}})
    original = copy.deepcopy(state.to_dict())
    projected = mission_review_state(task, state, outcome, review, ['native.bash'])
    rendered = json.dumps(projected, ensure_ascii=False)
    assert len(rendered.encode()) <= ACT_STATE_MAX_BYTES
    assert 'complete_tool_execution_log' not in projected and 'task_context' not in projected
    assert 'DO_NOT_SEND_COMMAND' not in rendered and payload not in rendered
    assert projected['goal'] == task.goal
    observations = {row['evidence_ref']: row for row in projected['execution_evidence']}
    assert observations['tactical-action:read']['result'] == state.evidence.entries[0]['result']
    assert observations['tactical-action:reply']['result_reference'] == 'prepared_response.message'
    assert rendered.count(task.metadata['prepared_user_response']['message']) == 1
    assert 'schema' in projected['context_coverage']['omitted_result_refs']
    assert state.to_dict() == original


@pytest.mark.parametrize('field', ['goal', 'prepared', 'evidence', 'followup', 'constraints'])
def test_required_context_is_never_silently_truncated(field):
    task, state, outcome, review = fixture()
    huge = 'ESSENTIAL ' * ACT_STATE_MAX_BYTES
    if field == 'goal': task.goal = huge
    elif field == 'prepared': task.metadata['prepared_user_response']['message'] = huge
    elif field == 'evidence': state.evidence.entries[0]['result']['stdout'] = huge
    elif field == 'constraints': task.metadata['user_constraints'] = {'must': huge}
    else: task.metadata['v3_resumption_evidence'] = [{'evidence_ref': 'answer', 'result': {'answer': huge}}]
    with pytest.raises(SystemOneUnavailableError, match='act_context_required_input_exceeds_budget'):
        mission_review_state(task, state, outcome, review, [])


def test_unavailable_supporting_evidence_cannot_be_replaced_by_the_prepared_reply():
    task, state, outcome, review = fixture()
    state.evidence.entries[0]['status'] = 'failed'
    with pytest.raises(SystemOneUnavailableError, match='act_context_required_evidence_unavailable'):
        mission_review_state(task, state, outcome, review, [])


def test_failed_final_review_preserves_draft_and_successful_check_without_completing_task():
    task, state, outcome, _ = fixture()
    task.metadata['v3_phase_review'] = {'status': 'phase_blocked', 'reason': 'Old phase restriction'}
    prepared = copy.deepcopy(task.metadata['prepared_user_response'])
    seen = []
    def recommend(*, state):
        seen.append(state)
        assert len(json.dumps(state).encode()) < ACT_STATE_MAX_BYTES
        raise SystemOneRequestError(400, 'max_tokens_exceeded')
    jev = SimpleNamespace(requires_mission_outcome_review=True, recommend_act=recommend,
        evaluate=lambda **kwargs: SystemOneReviewResult(updates=(), ambiguous_criterion_ids=()))
    context = CoreLoopContext(messages=InMemoryMessageQueue(), system_one=jev)
    with pytest.raises(SystemOneUnavailableError, match='system_one_context_limit_exceeded'):
        V3OuterController().review_and_route(task, state, outcome, context)
    assert len(seen) == 1
    assert task.metadata['prepared_user_response'] == prepared
    assert task.metadata['v3_phase_review']['status'] == 'phase_verified_task_complete'
    assert task.metadata['v3_strategic_decision']['action'] == 'fail'
    assert 'system_one_context_limit_exceeded' in task.metadata['v3_strategic_decision']['reason']
    assert task.status != 'completed'
    assert task.outcome is None or task.outcome.get('status') != 'success'


def test_all_supporting_observations_are_preserved_when_a_criterion_uses_multiple_steps():
    task, state, outcome, review = fixture()
    state.evidence.append({'evidence_ref': 'tactical-action:write', 'status': 'success', 'tool_id': 'native.bash',
        'result': {'exit_code': 0, 'stdout': 'COMMITTED fixture record'}})
    task.acceptance_contract['criteria'][0]['evidence_refs'].append('tactical-action:write')
    projected = mission_review_state(task, state, outcome, review, [])
    rows = {row['evidence_ref']: row for row in projected['execution_evidence']}
    assert rows['tactical-action:read']['result'] == state.evidence.entries[0]['result']
    assert rows['tactical-action:write']['result'] == state.evidence.entries[-1]['result']
    state.evidence.entries[-1]['result']['stdout'] = 'ESSENTIAL ' * ACT_STATE_MAX_BYTES
    with pytest.raises(SystemOneUnavailableError, match='required_input_exceeds_budget'):
        mission_review_state(task, state, outcome, review, [])
