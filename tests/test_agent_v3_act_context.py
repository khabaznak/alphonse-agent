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
    projected = mission_review_state(task, state)
    rendered = json.dumps(projected, ensure_ascii=False)
    assert len(rendered.encode()) <= ACT_STATE_MAX_BYTES
    assert 'complete_tool_execution_log' not in projected and 'task_context' not in projected
    assert 'DO_NOT_SEND_COMMAND' not in rendered and payload not in rendered
    assert projected['task_understanding']['inputs'][0]['content'] == task.goal
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
    else: task.append_conversation_message(task.user, huge)
    with pytest.raises(SystemOneUnavailableError, match='act_context_required_input_exceeds_budget'):
        mission_review_state(task, state)


def test_unavailable_supporting_evidence_cannot_be_replaced_by_the_prepared_reply():
    task, state, outcome, review = fixture()
    state.evidence.entries[0]['status'] = 'failed'
    with pytest.raises(SystemOneUnavailableError, match='act_context_required_evidence_unavailable'):
        mission_review_state(task, state)


def test_failed_final_review_preserves_draft_and_successful_check_without_completing_task():
    task, state, outcome, _ = fixture()
    task.metadata['v3_phase_review'] = {'status': 'phase_blocked', 'reason': 'Old phase restriction'}
    prepared = copy.deepcopy(task.metadata['prepared_user_response'])
    seen = []
    def recommend(*, state):
        seen.append(state)
        assert len(json.dumps(state).encode()) < ACT_STATE_MAX_BYTES
        raise SystemOneRequestError(400, 'max_tokens_exceeded')
    jev = SimpleNamespace(judge_act=recommend,
        evaluate=lambda **kwargs: SystemOneReviewResult(updates=(), ambiguous_criterion_ids=()))
    context = CoreLoopContext(messages=InMemoryMessageQueue(), system_one=jev)
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
    projected = mission_review_state(task, state)
    rows = {row['evidence_ref']: row for row in projected['execution_evidence']}
    assert rows['tactical-action:read']['result'] == state.evidence.entries[0]['result']
    assert rows['tactical-action:write']['result'] == state.evidence.entries[-1]['result']
    state.evidence.entries[-1]['result']['stdout'] = 'ESSENTIAL ' * ACT_STATE_MAX_BYTES
    with pytest.raises(SystemOneUnavailableError, match='required_input_exceeds_budget'):
        mission_review_state(task, state)


def test_review_preserves_order_and_separates_attempts_from_observations():
    task, state, _, _ = fixture()
    state.evidence.entries.insert(0, {'evidence_ref': 'early', 'status': 'failed', 'tool_id': 'native.bash',
        'error': 'No record was available', 'result': {'exit_code': 1, 'stdout': ''}})
    state.evidence.append({'evidence_ref': 'later', 'status': 'success', 'tool_id': 'native.bash',
        'result': {'exit_code': 0, 'stdout': 'Corrected record'}})

    observations = mission_review_state(task, state)['execution_evidence']
    assert [(row['evidence_ref'], row['observation_kind']) for row in observations] == [
        ('early', 'attempt'), ('tactical-action:read', 'operational_result'),
        ('tactical-action:reply', 'conversation'), ('later', 'operational_result'),
    ]


def _shell_receipt(ref, command, *, stdout='', stderr='', exit_code=0, cwd='/fixture', status='success'):
    return {
        'evidence_ref': ref, 'tool_id': 'native.bash', 'status': status, 'phase_id': 'lookup',
        'arguments': {'command': command, 'cwd': cwd},
        'result': {'exit_code': exit_code, 'stdout': stdout, 'stderr': stderr, 'cwd': cwd},
    }


def _projected_rows(task, state, outcome, review):
    projection = mission_review_state(task, state)
    return projection, {row['evidence_ref']: row for row in projection['execution_evidence']}


def test_same_process_result_distinguishes_inspection_and_attempted_mutation_without_command_payload():
    task, state, outcome, review = fixture()
    state.evidence.append(_shell_receipt('inspect', 'cat archive.txt'))
    state.evidence.append(_shell_receipt('mutate', "printf '%s' 'private-write-literal' > archive.txt"))
    state.evidence.append(_shell_receipt('elsewhere', 'touch other.txt'))
    projection, rows = _projected_rows(task, state, outcome, review)
    assert rows['inspect']['result'] == rows['mutate']['result'] == rows['elsewhere']['result']
    assert rows['inspect']['operation']['kind'] == 'inspection'
    assert rows['inspect']['effect_status'] == 'none'
    assert rows['mutate']['operation']['target']['path'] == '/fixture/archive.txt'
    assert rows['elsewhere']['operation']['target']['path'] == '/fixture/other.txt'
    assert rows['mutate']['effect_status'] == 'attempted_unverified'
    assert rows['mutate']['result_coverage']['status'] == 'empty'
    assert 'private-write-literal' not in json.dumps(projection)
    assert rows['tactical-action:reply']['effect_status'] == 'none'


@pytest.mark.parametrize('same_target', [True, False])
def test_fresh_readback_links_resource_effect_and_criterion_through_checkpoint_replay(same_target):
    from alphonse.agent_v2.core.intelligence.v3.act_evidence import has_requested_effect_evidence
    from alphonse.agent_v2.core.intelligence.v3.contracts import TacticalState
    from alphonse.agent_v2.core.intelligence.v3.task_understanding import snapshot

    task, state, outcome, review = fixture()
    task.goal = 'Save the reviewed value to archive.txt'
    task.acceptance_contract['criteria'][0].update(statement='Archive persists reviewed value', evidence_refs=['readback'])
    write = _shell_receipt('write', "printf 'reviewed value' > archive.txt")
    write['invocation'] = {'call_id': 'write', 'task_understanding': {
        'version': 0, 'input_digest': snapshot(task)['input_digest'],
    }}
    state.evidence.append(write)
    state.evidence.append(_shell_receipt('readback', 'cat archive.txt', stdout='reviewed value',
                                        cwd='/fixture' if same_target else '/somewhere-else'))
    first, rows = _projected_rows(task, state, outcome, review)
    assert rows['readback']['criterion_ids'] == ['ac-1']
    assert rows['readback']['verification']['readback_ref'] == 'readback'
    assert has_requested_effect_evidence(first) is same_target
    assert rows['readback']['verification'].get('preceding_mutation_ref') == ('write' if same_target else None)
    assert rows['readback']['verification'].get('effect_ref') == ('write:effect' if same_target else None)
    assert rows['write']['provenance']['task_understanding']['input_digest'] == first['task_understanding']['input_digest']
    restored_task = TaskState.from_dict(json.loads(json.dumps(task.to_checkpoint_dict())))
    restored_state = TacticalState.from_dict(json.loads(json.dumps(state.to_dict())))
    replay, _ = _projected_rows(restored_task, restored_state, outcome, review)
    assert replay == first


@pytest.mark.parametrize('http_status', [403, 404, 503])
def test_caught_http_error_in_a_real_exit_zero_shell_cannot_fulfill_lookup(http_status):
    import shlex
    import sys

    from alphonse.agent_v2.core.intelligence.v3.outer import StrategicAction
    from alphonse.agent_v2.core.tools.registry.native.bash import execute_bash
    from conftest import act_judgments

    task, state, outcome, review = fixture()
    script = (
        "from urllib.error import HTTPError\n"
        f"try: raise HTTPError('https://fixture.invalid', {http_status}, 'fixture failure', None, None)\n"
        "except HTTPError as error: print(error)\n"
    )
    command = shlex.quote(sys.executable) + ' -c ' + shlex.quote(script)
    result = execute_bash({'command': command})
    assert result['exit_code'] == 0
    state.evidence.entries[0].update(arguments={'command': command}, result=result)
    seen = []

    def recommend(*, state):
        seen.extend(state['execution_evidence'])
        return act_judgments()

    jev = SimpleNamespace(judge_act=recommend,
                          evaluate=lambda **_: SystemOneReviewResult((), ()))
    _, decision = V3OuterController().review_and_route(
        task, state, outcome, CoreLoopContext(messages=InMemoryMessageQueue(), system_one=jev)
    )
    row = seen[0]
    assert row['status'] == 'success'
    assert row['operational_outcome'] == ('denied' if http_status == 403 else 'failed')
    assert row['http_status'] == http_status
    assert 'verification' not in row
    assert decision.action == StrategicAction.FAIL
    assert task.metadata["v3_mission_resolution"]["status"] == "unavailable"


def test_verified_native_effect_survives_alongside_failed_mutation_and_unfinished_intent():
    from alphonse.agent_v2.core.intelligence.v3.contracts import TacticalAction

    task, state, outcome, review = fixture()
    state.evidence.append({
        'evidence_ref': 'verified', 'tool_id': 'native.scheduled_task', 'status': 'success',
        'result': {'scheduled_task_id': 'fixture-schedule', 'schedule': {'run_at': '2026-11-05T18:00:00Z'},
                   'verification': {'status': 'verified', 'source': 'schedule_store_readback'},
                   'delivery_channels': ['private-routing-details']},
    })
    state.evidence.append(_shell_receipt('failed', 'rm archive.txt', stderr='Permission denied', exit_code=1, status='failed'))
    state.actions.append(TacticalAction('unfinished', 'read', 'native.bash', {'command': 'touch later.txt'}))
    projection, rows = _projected_rows(task, state, outcome, review)
    assert rows['verified']['effect_status'] == 'verified_persisted_effect'
    assert rows['verified']['verification']['target'] == {'kind': 'schedule', 'id': 'fixture-schedule'}
    assert rows['failed']['effect_status'] == 'failed_attempt_possible_partial_effect'
    assert rows['failed']['operational_outcome'] == 'denied'
    assert rows['tactical-action:unfinished']['operational_outcome'] == 'invocation_outcome_unavailable'
    assert rows['tactical-action:unfinished']['provenance']['source'] == 'checkpoint_action'
    assert set(projection['context_coverage']['unavailable_evidence_refs']) == {'failed', 'tactical-action:unfinished'}
    assert 'private-routing-details' not in json.dumps(projection)


def test_cited_readback_requires_its_linked_write_atomically():
    task, state, outcome, review = fixture()
    task.acceptance_contract['criteria'][0]['evidence_refs'] = ['readback']
    state.evidence.append(_shell_receipt('write', "printf 'reviewed' > archive.txt", stderr='X' * ACT_STATE_MAX_BYTES))
    state.evidence.append(_shell_receipt('readback', 'cat archive.txt', stdout='reviewed'))
    with pytest.raises(SystemOneUnavailableError, match='act_context_required_input_exceeds_budget'):
        mission_review_state(task, state)
    assert task.metadata['v3_act_context']['state_bytes'] > ACT_STATE_MAX_BYTES


def test_context_manifest_reports_source_clipping_access_failure_and_budget_omissions():
    from alphonse.agent_v2.core.context_manifests import manifest_scope
    from alphonse.agent_v2.system_one import JevCriterionDecisionProvider, SystemOneSettings

    task, state, outcome, review = fixture()
    task.intelligence_engine = 'hierarchical_v3'
    state.evidence.append(_shell_receipt('unavailable', 'cat private.txt', stderr='Permission denied'))
    state.evidence.append(_shell_receipt('large', 'cat optional.txt', stdout='x' * ACT_STATE_MAX_BYTES))
    state.evidence.append(_shell_receipt('producer_clip', 'cat other.txt', stdout='prefix\n... [truncated]'))
    state.evidence.append(_shell_receipt('long_command', 'x' * 4095 + '|'))
    projected, rows = _projected_rows(task, state, outcome, review)
    assert rows['producer_clip']['result_coverage']['limitations'] == ['source_truncated']
    assert 'verification' not in rows['producer_clip']
    assert rows['long_command']['operation']['argument_coverage'] == 'command_withheld'
    assert rows['large']['result_coverage']['status'] == 'omitted'
    assert 'unavailable' in projected['context_coverage']['unavailable_evidence_refs']

    def transport(_url, _key, payload, _timeout):
        return {'answers': {
            key: {'type': 'noul', 'noul': .99} for key in payload['questions']
        }}

    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key='fixture'), transport=transport)
    with manifest_scope(task, CoreLoopContext(messages=InMemoryMessageQueue())):
        provider.judge_act(state=projected)
    manifest = task.metadata['decision_context_manifests'][-1]
    assert ('large', 'omitted') in {(row['source_id'], row['status']) for row in manifest['omissions']}
    assert ('unavailable', 'unavailable') in {(row['source_id'], row['status']) for row in manifest['omissions']}
    assert next(row for row in manifest['sources'] if row['source_id'] == 'producer_clip')['limitations'] == ['source_truncated']
    assert manifest['task_understanding']['input_digest'] == projected['task_understanding']['input_digest']
    assert 'prefix' not in json.dumps(manifest)


def test_queued_delivery_receipt_records_the_committed_action_and_recipient():
    from alphonse.agent_v2.core.intelligence.v3.act_evidence import has_requested_effect_evidence

    task, state, outcome, review = fixture()
    task.goal = 'Send the schedule to the fixture recipient'
    task.acceptance_contract['criteria'][0]['evidence_refs'] = ['queued']
    state.evidence.append({
        'evidence_ref': 'queued', 'tool_id': 'native.deliver_message', 'status': 'success',
        'arguments': {'message': 'The schedule is ready.'},
        'result': {'status': 'queued', 'outbox_message_id': 'outbox-1', 'recipient_user_id': 'recipient-1'},
    })
    projected, rows = _projected_rows(task, state, outcome, review)
    row = rows['queued']
    assert row['effect_status'] == 'committed_receipt'
    assert row['effect_receipt'] == {
        'source': 'outbox_enqueue', 'target': {'kind': 'recipient', 'id': 'recipient-1'},
    }
    assert row['submitted_content'] == {'message': 'The schedule is ready.'}
    assert row['result']['status'] == 'queued'
    assert 'verification' not in row
    assert has_requested_effect_evidence(projected)


def test_opaque_skill_write_and_self_reported_readback_are_not_authoritative():
    from alphonse.agent_v2.core.intelligence.v3.act_evidence import has_requested_effect_evidence

    task, state, outcome, review = fixture()
    task.goal = 'Set the fixture target and verify it'
    task.acceptance_contract['criteria'][0].update(statement='The requested target was set', evidence_refs=['readback'])
    state.evidence.append(_shell_receipt('write', 'python fixture.py set', stdout='{"acknowledged": true, "source": "fixture"}'))
    state.evidence.append(_shell_receipt('readback', 'python fixture.py verify', stdout='{"verified": true, "source": "fixture"}'))
    projected, rows = _projected_rows(task, state, outcome, review)
    assert rows['write']['operation']['kind'] == 'unknown'
    assert rows['readback']['operation']['kind'] == 'unknown'
    assert not has_requested_effect_evidence(projected)


def test_receipt_printed_by_an_opaque_shell_cannot_prove_a_new_registration():
    from alphonse.agent_v2.core.intelligence.v3.act_evidence import has_requested_effect_evidence

    task, state, outcome, review = fixture()
    task.goal = 'Register the requester once'
    task.acceptance_contract['criteria'][0].update(statement='Registration occurred', evidence_refs=['submit'])
    state.evidence.append(_shell_receipt('submit', 'python submit.py', stdout='{"registered": true, "receipt": "synthetic"}'))
    projected, rows = _projected_rows(task, state, outcome, review)
    assert rows['submit']['operation']['kind'] == 'unknown'
    assert not has_requested_effect_evidence(projected)


def test_independent_receipt_cannot_override_read_only_phase_scope():
    from datetime import datetime, timezone
    from alphonse.agent_v2.core.intelligence.v3.act_evidence import has_requested_effect_evidence
    from alphonse.agent_v2.core.intelligence.v3.task_understanding import snapshot

    task, state, outcome, review = fixture()
    task.acceptance_contract['criteria'][0]['evidence_refs'] = ['tactical-action:write']
    state.evidence.append({
        'evidence_ref': 'tactical-action:write', 'status': 'success', 'tool_id': 'native.bash',
        'phase_id': 'lookup', 'subgoal_id': 'read',
        'invocation': {'call_id': 'write', 'task_understanding': {'input_digest': snapshot(task)['input_digest']}},
        'arguments': {'command': 'opaque', 'verification': {'source': 'workspace_text', 'path': 'x',
                      'expected': 'value', 'require_change': True}},
        'result': {'exit_code': 0, 'stdout': '', 'stderr': '', 'verification': {
            'status': 'verified', 'source': 'bash_independent_readback',
            'target': {'kind': 'workspace_text', 'id': 'a' * 64}, 'resource': {'path': 'x'}, 'observed': 'value', 'changed': True,
            'actor_user_id': task.user, 'task_id': task.task_id, 'observed_at': datetime.now(timezone.utc).isoformat(),
        }},
    })
    projected, rows = _projected_rows(task, state, outcome, review)
    assert rows['tactical-action:write']['effect_status'] == 'unknown'
    assert not has_requested_effect_evidence(projected)


def test_project_todo_effect_requires_verified_store_readback():
    task, state, outcome, review = fixture()
    for row in [
        {'evidence_ref': 'todo-ok', 'tool_id': 'native.project_todo', 'status': 'success',
         'result': {'todo_id': 'todo-1', 'status': 'pending', 'verification': {'status': 'verified', 'source': 'project_todo_store_readback'}}},
        {'evidence_ref': 'todo-unverified', 'tool_id': 'native.project_todo', 'status': 'success',
         'result': {'todo_id': 'todo-2', 'status': 'pending'}},
        {'evidence_ref': 'todo-failed', 'tool_id': 'native.project_todo', 'status': 'failed',
         'result': {'todo_id': 'todo-3', 'verification': {'status': 'verified', 'source': 'project_todo_store_readback'}}},
    ]:
        state.evidence.append(row)
    _, rows = _projected_rows(task, state, outcome, review)
    assert rows['todo-ok']['effect_status'] == 'verified_persisted_effect'
    assert rows['todo-ok']['verification']['target'] == {'kind': 'project_todo', 'id': 'todo-1'}
    assert rows['todo-unverified']['effect_status'] == 'attempted_unverified'
    assert rows['todo-failed']['effect_status'] != 'verified_persisted_effect'
