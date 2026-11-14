"""Synthetic outcome matrix with controlled Jev judgments; no live/private data."""
from copy import deepcopy

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext, ToolExecutionContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import (
    PhasePlan, PhaseSubgoal, PhaseOutcome, PhaseStatus, SideEffectClass,
    StrategicAction, V3OuterController, new_tactical_state,
)
from alphonse.agent_v2.core.intelligence.v3.delegated_outcome import outcome_assessment
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.core.tools.registry.native.scheduled_task import execute_scheduled_task
from alphonse.agent_v2.outcome_judgments import OUTCOME_QUESTIONS
from alphonse.agent_v2.system_one import JevCriterionDecisionProvider, SystemOneSettings


def judgments(**updates):
    values = {key: .01 for key in OUTCOME_QUESTIONS}
    values.update(mission_outcome_is_fulfilled=.99, capability_claims_are_supported=.99,
                  lookup_claims_are_supported=.99)
    values.update(updates)
    return values


def observation(tool='native.bash', status='success', result=None):
    return {'evidence_ref': 'operation:1', 'subgoal_id': 'work', 'tool_id': tool,
            'status': status, 'result': result if result is not None else {'stdout': 'No matching rows', 'exit_code': 0}}


def run_case(goal, response, answers, rows=(), *, cancel=False, closure=None, transport_error=False, action="complete"):
    task = TaskState(user='fixture', goal=goal)
    # Deliberately weak/self-validating criteria must not define mission success.
    task.set_acceptance_contract_from_markdown('1.- [ ] Explain that the task is done or cannot be done')
    task.metadata['prepared_user_response'] = {'message': response}
    if closure:
        task.metadata['act_directive'] = closure
    phase = PhasePlan('fixture', 'Adapted approach', (PhaseSubgoal(
        'work', 'Inspect then respond', 'response', allowed_capabilities=('local_shell', 'user_response'),
        allowed_side_effects=(SideEffectClass.READ_ONLY, SideEffectClass.USER_RESPONSE),
    ),), criterion_ids=('ac-1',), authorized_capabilities=('local_shell', 'user_response'))
    state = new_tactical_state(phase)
    state.status = PhaseStatus.PHASE_COMPLETE
    state.completed_subgoal_ids = ['work']
    for row in rows:
        state.evidence.append(deepcopy(row))
    state.evidence.append(observation('native.respond', result={'message': response}) | {'evidence_ref': 'reply:1'})
    captured = []
    cancelled = [False]
    delegated = outcome_assessment(answers, rows)

    def transport(url, key, payload, timeout):
        captured.append(payload)
        if 'effect_requested' in payload['questions']:
            if transport_error:
                raise ValueError('synthetic outage')
            cancelled[0] = cancel
            values = {
                'effect_requested': .99 if answers.get('mission_requires_recurring_setup') == .99 else .01,
                'effect_supported': .99 if delegated['fulfilled'] else .01,
                'effect_contradicted': .01,
                'reply_addresses_request': .99 if delegated['fulfilled'] else .01,
                'reply_grounded': .99 if delegated['fulfilled'] else .01,
            }
            return {'answers': {name: {'type': 'noul', 'noul': values[name]} for name in payload['questions']}}
        return {'answers': {name: {'type': 'noul', 'noul': .99} for name in payload['questions']}}
    provider = JevCriterionDecisionProvider(SystemOneSettings(enabled=True, api_key='synthetic', validated_at='now'), transport=transport)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), system_one=provider,
                              cancellation_checker=lambda: cancelled[0])
    review, decision = V3OuterController().review_and_route(task, state, PhaseOutcome('fixture', PhaseStatus.PHASE_COMPLETE), context)
    return task, decision, captured


@pytest.mark.parametrize('goal,response,answers,rows,complete,status', [
    ('Find my invoice', 'Search your mailbox yourself.', judgments(mission_requires_execution=.99, mission_outcome_is_fulfilled=.01), [], False, 'unresolved'),
    ('Find my invoice', 'I cannot search.', judgments(mission_requires_execution=.99, mission_outcome_is_fulfilled=.01, capability_claims_are_supported=.01), [], False, 'unresolved'),
    ('Find my invoice', 'No invoices exist.', judgments(mission_requires_execution=.99, lookup_claims_are_supported=.01), [observation(status='failed', result={'error': 'timeout'})], False, 'unresolved'),
    ('Find my invoice', 'No invoices exist.', judgments(mission_requires_execution=.99), [], False, 'unresolved'),
    ('Find my invoice', 'No invoices exist in the searched mailbox.', judgments(mission_requires_execution=.99), [observation()], True, 'fulfilled'),
    ('Find invoices in both mailboxes', 'First mailbox checked; second is inaccessible.', judgments(mission_requires_execution=.99, mission_outcome_is_fulfilled=.01, work_is_useful_progress=.99), [observation()], False, 'intermediate_progress'),
    ('Find my invoice', 'Mailbox access denied.', judgments(mission_requires_execution=.99, mission_outcome_is_fulfilled=.01, blocker_is_supported=.99), [observation(status='failed', result={'error': 'access_denied'})], False, 'blocked'),
    ('Can you search my invoices?', 'Access is denied in the configured mailbox.', judgments(mission_is_capability_only=.99, blocker_is_supported=.99), [observation(status='failed', result={'error': 'access_denied'})], True, 'fulfilled'),
    ('Can you search my invoices?', 'There are no tools for this.', judgments(mission_is_capability_only=.99, capability_claims_are_supported=.01), [observation(result={'available': True})], False, 'unresolved'),
    ('Explain how to search invoices', 'Here are the manual steps.', judgments(), [], True, 'fulfilled'),
    ('Plan a daily invoice check', 'Here is the plan; it is not active.', judgments(), [], True, 'fulfilled'),
    ('Set up a daily invoice check', 'Here is the plan.', judgments(mission_requires_execution=.99, mission_requires_recurring_setup=.99), [observation()], False, 'unresolved'),
    ('Find my invoice', 'A matching invoice was found.', judgments(mission_requires_execution=.99, lookup_claims_are_supported=.01), [observation()], False, 'unresolved'),
])
def test_original_outcome_controls_completion(goal, response, answers, rows, complete, status):
    task, decision, payloads = run_case(goal, response, answers, rows)
    assert (decision.action == StrategicAction.COMPLETE) is complete
    assert (task.status == 'completed') is complete
    assert outcome_assessment(answers, rows)['status'] == status
    act = next(payload for payload in payloads if 'effect_requested' in payload['questions'])
    assert act['state']['task_understanding']['inputs'][0]['content'] == goal
    assert act['state']['prepared_response']['message'] == response
    from alphonse.agent_v2.system_one import ACT_QUESTIONS
    assert set(act['questions']) == set(ACT_QUESTIONS)
    assert all(question['type'] == 'noul' for question in act['questions'].values())


@pytest.mark.parametrize('field', list(OUTCOME_QUESTIONS))
def test_missing_fixture_judgment_is_unresolved_for_each_field(field):
    result = outcome_assessment(judgments(**{field: None}), [])
    assert result['reason'] == 'missing_or_invalid_outcome_judgment' and not result['fulfilled']


@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), -1, 2])
def test_invalid_fixture_judgment_types_and_ranges_are_unresolved(value):
    result = outcome_assessment(judgments(mission_outcome_is_fulfilled=value), [])
    assert result['reason'] == 'missing_or_invalid_outcome_judgment' and not result['fulfilled']


def test_invalid_fixture_judgment_yields_an_unsupported_synthetic_act_response():
    # The test-only adapter translates a rejected outcome into valid .01 Act
    # judgments. This does not inject malformed answers into the real provider.
    task, decision, _ = run_case('Explain the result', 'Explanation',
                                 judgments(mission_outcome_is_fulfilled=None))
    assert decision.action != StrategicAction.COMPLETE
    assert task.status != 'completed'


@pytest.mark.parametrize('field', ['mission_outcome_is_fulfilled', 'capability_claims_are_supported', 'lookup_claims_are_supported'])
def test_uncertain_support_withholds_completion(field):
    task, decision, _ = run_case('Explain the result', 'Explanation', judgments(**{field: .5}))
    assert decision.action != StrategicAction.COMPLETE and task.status != 'completed'


def test_required_review_outage_never_falls_back_to_criteria_only_success():
    task, decision, _ = run_case('Find my invoice', 'Done', judgments(), transport_error=True)
    assert decision.action != StrategicAction.COMPLETE and task.status != 'completed'


def test_cancellation_during_act_prevents_completion():
    task, decision, _ = run_case('Explain the result', 'Explanation', judgments(), cancel=True)
    assert task.status == 'cancelled' and decision.action == StrategicAction.FAIL


def test_failed_effect_cannot_complete_even_with_positive_judgments():
    task, decision, _ = run_case('Update the record', 'Done', judgments(mission_requires_execution=.99),
        [observation(result={'verification': {'status': 'failed'}})],
        closure={'closure_only': True, 'terminal_outcome': 'success'})
    assert decision.action != StrategicAction.COMPLETE and task.status != 'completed'


def test_recurring_activation_requires_real_store_readback(tmp_path):
    store = ScheduledTaskStore(tmp_path / 'schedules.sqlite')
    task = TaskState(task_id='schedule-fixture', user='fixture', goal='Set up a daily invoice check')
    context = ToolExecutionContext(task=task, messages=InMemoryMessageQueue(), schedule_store=store)
    result = execute_scheduled_task({'name': 'Invoice check', 'prompt': 'Check invoices', 'schedule_kind': 'rrule',
        'rrule': 'FREQ=DAILY', 'dtstart': '2030-01-01T09:00:00+00:00', 'timezone': 'UTC'}, context=context)
    assert store.get_task(result['scheduled_task_id']).schedule == result['schedule']
    answers = judgments(mission_requires_execution=.99, mission_requires_recurring_setup=.99)
    task, decision, _ = run_case('Set up a daily invoice check', 'Active daily check', answers,
                                [observation('native.scheduled_task', result=result)])
    assert task.status == 'completed' and decision.action == StrategicAction.COMPLETE
    for field, value in [('verification', {}), ('status', 'paused'), ('next_run_at', None), ('scheduled_task_id', '')]:
        invalid = dict(result, **{field: value})
        assert not outcome_assessment(answers, [observation('native.scheduled_task', result=invalid)])['fulfilled']
    store.get_task = lambda _id: None
    with pytest.raises(RuntimeError, match='scheduled_task_readback_failed'):
        execute_scheduled_task({'name': 'Missing', 'prompt': 'Missing', 'schedule_kind': 'once',
            'run_at': '2030-01-01T09:00:00+00:00'}, context=context)


def test_supported_blocker_closes_as_failed_without_manufacturing_success():
    answers = judgments(mission_requires_execution=.99, mission_outcome_is_fulfilled=.01, blocker_is_supported=.99)
    task, decision, _ = run_case('Find my invoice', 'Mailbox access denied.', answers,
        [observation(status='failed', result={'error': 'access_denied'})], action='fail_explain')
    assert decision.action == StrategicAction.FAIL and task.status == 'failed'



def test_partial_answer_can_fulfill_explicitly_limited_lookup_request():
    task, decision, _ = run_case('Report whichever mailbox you can access, stating limits',
        'No invoices in the accessible mailbox; the second mailbox was not searched.',
        judgments(mission_requires_execution=.99), [observation()])
    assert decision.action == StrategicAction.COMPLETE and task.status == 'completed'
