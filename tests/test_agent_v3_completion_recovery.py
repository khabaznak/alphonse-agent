"""Completion disagreement and malformed-plan recovery using synthetic evidence."""
from conftest import act_judgments
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.inference import InferencePurpose
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import (
    HierarchicalCAPDProcessor, PhasePlan, PhaseSubgoal, PhaseOutcome, PhaseStatus, SideEffectClass,
    StrategicAction, V3OuterController, new_tactical_state,
)
from alphonse.agent_v2.core.intelligence.v3.planner import plan_phase, V3PhasePlanValidationError
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.ask_question import build_ask_question_tool_definition
from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
from alphonse.agent_v2.core.intelligence.v3.executor import PhaseExecutor
from alphonse.agent_v2.core.intelligence.v3.revealing import ToolRevealPolicy
from alphonse.agent_v2.system_one import SystemOneReviewResult


def completed_lookup(probability=.79, requires_effect=.01):
    task = TaskState(task_id='synthetic-lookup', user='fixture', goal='Which FixtureLang version is installed?')
    task.set_acceptance_contract_from_markdown('1.- [ ] Report the locally verified version')
    task.metadata['prepared_user_response'] = {
        'message': 'FixtureLang 7.2 is installed.', 'tool_call_id': 'reply',
        'work_evidence_refs': ['tactical-action:inspect'],
    }
    phase = PhasePlan('lookup', 'Look up version and answer', (
        PhaseSubgoal('inspect', 'Read version', 'version', allowed_capabilities=('local_shell',)),
        PhaseSubgoal('reply', 'Answer from evidence', 'response', depends_on=('inspect',),
            allowed_capabilities=('user_response',), allowed_side_effects=(SideEffectClass.USER_RESPONSE,)),
    ), criterion_ids=('ac-1',), authorized_capabilities=('local_shell', 'user_response'))
    state = new_tactical_state(phase)
    state.status = PhaseStatus.PHASE_COMPLETE
    state.completed_subgoal_ids = ['inspect', 'reply']
    state.evidence.append({'evidence_ref': 'tactical-action:inspect', 'tool_id': 'native.bash',
        'phase_id': 'lookup', 'subgoal_id': 'inspect', 'status': 'success',
        'arguments': {'command': 'fixturelang --version'},
        'result': {'stdout': 'FixtureLang 7.2', 'exit_code': 0}})
    state.evidence.append({'evidence_ref': 'tactical-action:reply', 'tool_id': 'native.respond',
        'phase_id': 'lookup', 'subgoal_id': 'reply', 'status': 'success', 'result': {'message': 'FixtureLang 7.2 is installed.'}})

    class Jev:
        def evaluate(self, **kwargs):
            return SystemOneReviewResult(({'criterion_id': 'ac-1', 'status': 'satisfied',
                'evidence_refs': ['tactical-action:inspect', 'tactical-action:reply']},), ())
        def judge_act(self, **kwargs):
            return act_judgments(effect_requested=requires_effect, reply_addresses_request=probability)

    requests = []
    verdict = {'next_step': 'complete', 'evidence_refs': ['tactical-action:inspect'],
               'reason': 'The verified output supports the prepared answer.', 'remaining_work': ''}
    def generate(request):
        requests.append(request)
        assert request.purpose == InferencePurpose.STRATEGIC_ACT
        assert request.tools == () and request.output_schema
        assert 'prepared_response' in request.prompt
        return SimpleNamespace(json_value=deepcopy(verdict))
    context = CoreLoopContext(messages=InMemoryMessageQueue(), system_one=Jev(),
                              inference=SimpleNamespace(generate_json=generate))
    return task, state, context, requests, verdict


@pytest.mark.parametrize('probability,band,review_count', [(.2, 'no', 1), (.2001, 'uncertain', 1), (.5, 'uncertain', 1),
    (.7999, 'uncertain', 1), (.8, 'yes', 0)])
def test_uncertain_fulfillment_is_reviewed_without_repeating_completed_work(probability, band, review_count):
    task, state, context, requests, _ = completed_lookup(probability)
    _, decision = V3OuterController().review_and_route(task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context)
    assert len(requests) == review_count
    assert decision.action == StrategicAction.COMPLETE
    assert task.metadata["system_one_act_judgments"]["bands"]["reply_addresses_request"] == band
    assert task.metadata['system_one_act_judgments']['answers']['reply_addresses_request'] == probability
    if review_count:
        assert task.metadata['v3_mission_resolution']['status'] == 'fulfilled'
        assert task.metadata['v3_route'] == 'respond_and_end'
        assert 'act_directive' not in task.metadata


def test_reasoning_identifies_real_missing_work_without_forcing_read_only_tasks_to_mutate():
    task, state, context, requests, verdict = completed_lookup()
    verdict.update(next_step="recover", remaining_work='The version source has not established the requested installation scope.')
    _, decision = V3OuterController().review_and_route(task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context)
    assert decision.action == StrategicAction.REPLAN and len(requests) == 1
    assert task.metadata['v3_mission_resolution']['remaining_work']
    assert 'inspection_resolution' not in task.metadata['act_directive']
    assert task.status != 'completed'


@pytest.mark.parametrize('invalid', [None, {'next_step': 'complete'},
    {'next_step': 'complete', 'evidence_refs': ['invented'], 'reason': 'Done', 'remaining_work': ''},
    {'next_step': 'complete', 'evidence_refs': ['tactical-action:inspect'], 'reason': 'Done', 'remaining_work': 'Still missing work'}])
def test_invalid_reasoning_review_never_promotes_uncertainty_to_success(invalid):
    task, state, context, _, _ = completed_lookup()
    context.inference = SimpleNamespace(generate_json=lambda request: SimpleNamespace(json_value=invalid))
    _, decision = V3OuterController().review_and_route(task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context)
    assert decision.action == StrategicAction.FAIL and task.status == 'failed'
    assert task.metadata['v3_mission_resolution']['status'] == 'invalid'


def test_completion_review_cancellation_wins_over_positive_reasoning():
    task, state, context, _, verdict = completed_lookup()
    cancelled = [False]
    context.cancellation_checker = lambda: cancelled[0]
    def generate(request):
        cancelled[0] = True
        return SimpleNamespace(json_value=verdict)
    context.inference = SimpleNamespace(generate_json=generate)
    _, decision = V3OuterController().review_and_route(task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context)
    assert task.status == 'cancelled' and decision.action == StrategicAction.FAIL


@pytest.mark.parametrize('first', ['invalid_json', 'missing_id', 'nested_subgoal', 'non_object'])
@pytest.mark.parametrize('recovers', [True, False])
def test_plan_encoding_gets_one_explicit_correction_and_preserves_contract(first, recovers):
    task = TaskState(task_id='plan-fixture', user='fixture', goal='Report the supplied fixture value')
    task.set_acceptance_contract_from_markdown('1.- [ ] Report the supplied fixture value')
    original_contract = deepcopy(task.acceptance_contract)
    valid = PhasePlan('reply', 'Report supplied value', (PhaseSubgoal('reply', 'Report supplied value', 'response',
        allowed_capabilities=('user_response',), allowed_side_effects=(SideEffectClass.USER_RESPONSE,)),),
        criterion_ids=('ac-1',), authorized_capabilities=('user_response',)).to_dict()
    invalid = deepcopy(valid)
    if first == 'missing_id': invalid.pop('phase_id')
    elif first == 'nested_subgoal': invalid = invalid['subgoals'][0]
    elif first == 'non_object': invalid = []
    requests = []
    def generate(request):
        requests.append(request)
        if len(requests) == 2:
            assert 'Plan validation feedback' in request.prompt
            assert 'phase_id' in request.prompt
            assert request.tools == ()
        if first == 'invalid_json' and (len(requests) == 1 or not recovers):
            raise ValueError('openai_codex_invalid_json')
        return SimpleNamespace(json_value=valid if len(requests) == 2 and recovers else invalid)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), inference=SimpleNamespace(generate_json=generate))
    if recovers:
        assert plan_phase(task, context).phase_id == 'reply'
    else:
        with pytest.raises(V3PhasePlanValidationError):
            plan_phase(task, context)
    assert len(requests) == 2
    assert task.acceptance_contract == original_contract and not task.hierarchical_state
    assert task.metadata['v3_plan_validation_recovery']['reason'] == 'invalid_plan_encoding'


def test_omitted_observation_cannot_be_used_as_support_by_reasoning_review():
    from alphonse.agent_v2.core.intelligence.v3.act_context import ACT_STATE_MAX_BYTES

    task, state, context, requests, verdict = completed_lookup()
    state.evidence.append({'evidence_ref': 'omitted:1', 'tool_id': 'native.bash', 'status': 'success',
                           'result': {'stdout': 'x' * ACT_STATE_MAX_BYTES, 'exit_code': 0}})
    task.metadata['prepared_user_response']['work_evidence_refs'].append('omitted:1')
    verdict['evidence_refs'] = ['omitted:1']
    _, decision = V3OuterController().review_and_route(task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context)
    assert decision.action == StrategicAction.FAIL
    assert len(requests) == 1
    assert task.metadata['v3_act_context']['omitted_result_refs'] == ['omitted:1']
    assert task.metadata['v3_mission_resolution']['status'] == 'invalid'


def test_failed_reasoning_review_is_unresolved_without_exposing_provider_body():
    task, state, context, _, _ = completed_lookup()

    def fail(request):
        raise RuntimeError('PRIVATE_PROVIDER_BODY')

    context.inference = SimpleNamespace(generate_json=fail)
    _, decision = V3OuterController().review_and_route(task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context)
    assert decision.action == StrategicAction.FAIL
    receipt = task.metadata['v3_mission_resolution']
    assert receipt['status'] == 'unavailable'
    assert receipt['error_code'] == 'mission_resolution_provider_error'
    assert receipt['exception_type'] == 'RuntimeError'
    assert 'PRIVATE_PROVIDER_BODY' not in str(receipt)


# Existing scripted providers delegate the shared interpretation call explicitly.
pytestmark = pytest.mark.usefixtures("task_understanding_provider")


@pytest.mark.parametrize('requested,support,conflict,status', [
    (.5, .99, .01, 'uncertain'),
    (.99, .01, .01, 'unsupported'),
    (.99, .5, .01, 'uncertain'),
    (.99, .99, .5, 'uncertain'),
    (.99, .99, .99, 'contradictory'),
    (.99, .01, .99, 'contradictory'),
    (.99, .8, .2, 'supported'),
    (.01, .01, .99, 'supported'),
])
def test_effect_judgments_have_explicit_routing(requested, support, conflict, status):
    task, state, context, requests, verdict = completed_lookup()
    if status != 'supported':
        verdict.update(next_step='recover', remaining_work='Verify the reminder recipient against the request.')
    if requested > .2:
        task.goal = 'Schedule a reminder for Friday at 09:00'
        task.metadata['prepared_user_response']['message'] = 'The Friday 09:00 reminder is scheduled.'
        state.evidence.entries[0].update(tool_id='native.scheduled_task', result={
            'scheduled_task_id': 'fixture-reminder', 'next_run_at': '2026-11-06T09:00:00Z',
            'verification': {'status': 'verified', 'source': 'schedule_store_readback'},
        })
    context.system_one.judge_act = lambda **_: act_judgments(
        effect_requested=requested, effect_supported=support, effect_contradicted=conflict,
    )
    _, decision = V3OuterController().review_and_route(
        task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context,
    )
    assert task.metadata['system_one_act_judgments']['status'] == status
    assert decision.action == (StrategicAction.COMPLETE if status == 'supported' else StrategicAction.REPLAN)
    assert len(requests) == (status != 'supported')
    if status != 'supported':
        directive = task.metadata['act_directive']
        assert 'tactical-action:inspect' in directive['phase_review']['evidence_refs']
        assert directive['outcome_review']['status'] == status


@pytest.mark.parametrize('result', [
    {'exit_code': 0},
    {'exit_code': 0, 'stdout': ' ', 'stderr': '', 'timed_out': False,
     'cwd': '/tmp', 'timeout_seconds': 15, 'duration_ms': 5},
    {'exit_code': 1, 'stdout': 'A partial result'},
    {'status': 'success', 'ok': True},
])
def test_process_status_and_prepared_reply_cannot_substitute_for_an_effect(result):
    task, state, context, requests, verdict = completed_lookup(requires_effect=.01, probability=.99)
    # A valid current request ref cannot rescue a criterion based on unusable execution.
    verdict["evidence_refs"] = ["request:synthetic-lookup"]
    state.evidence.entries[0]['result'] = result
    _, decision = V3OuterController().review_and_route(
        task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context,
    )
    assert decision.action == StrategicAction.FAIL
    assessment = task.metadata["system_one_act_judgments"]
    assert assessment["status"] == "unsupported"
    assert "unusable execution evidence" in assessment["reason"]
    assert task.metadata["v3_mission_resolution"]["status"] == "invalid"
    assert len(requests) == 1


def test_negative_grounding_blocks_a_reply_even_when_it_addresses_the_request():
    task, state, context, requests, verdict = completed_lookup()
    verdict.update(next_step='recover', remaining_work='Find evidence for the claimed installation scope.')
    context.system_one.judge_act = lambda **_: act_judgments(reply_grounded=.05)
    _, decision = V3OuterController().review_and_route(
        task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context,
    )
    assert decision.action == StrategicAction.REPLAN
    assert task.metadata['system_one_act_judgments']['status'] == 'unsupported'
    assert len(requests) == 1


@pytest.mark.parametrize('legacy_action', ['complete', 'replan'])
def test_legacy_route_labels_do_not_affect_resolution(legacy_action):
    task, state, context, requests, _ = completed_lookup()
    task.metadata['system_one_act_recommendation'] = {'action': legacy_action}
    check = context.system_one.evaluate
    context.system_one.evaluate = lambda **kwargs: replace(
        check(**kwargs), recommended_route=legacy_action, route_confident=True,
    )
    task.metadata['act_directive'] = {'action': legacy_action}
    _, decision = V3OuterController().review_and_route(task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context)
    assert decision.action == StrategicAction.COMPLETE
    assert len(requests) == 1
    assert legacy_action not in task.metadata['v3_mission_resolution']['evidence_refs']


def test_completed_checkpoint_is_reviewed_once_and_resolution_survives_reload(tmp_path):
    task, state, context, requests, _ = completed_lookup()
    store = SQLiteQuestionStore(tmp_path / 'checkpoints.sqlite')
    context.question_store = store
    task.hierarchical_state = state.to_dict()
    task.status = 'running'
    store.save_task_checkpoint(task)
    processor = HierarchicalCAPDProcessor()
    result = processor.process(store.load_task_checkpoint(task.task_id), context)
    assert result.status.value == 'completed'

    restored = store.load_task_checkpoint(task.task_id)
    assert restored.metadata['v3_mission_resolution']['status'] == 'fulfilled'
    assert restored.metadata['v3_act_review_receipt']['evidence_version'] == restored.metadata['v3_mission_resolution']['evidence_version']
    assert restored.hierarchical_state['evidence']['entries'] == state.to_dict()['evidence']['entries']

    def unexpected_judgment(**_):
        pytest.fail('Unchanged state should reuse the Act receipt')

    context.system_one.judge_act = unexpected_judgment
    _, decision = processor.outer.review_and_route(restored, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context)
    assert decision.action == StrategicAction.COMPLETE
    assert len(requests) == 1


def test_recovery_names_missing_operation_and_fences_verified_invocation():
    task, state, context, requests, verdict = completed_lookup()
    verdict.update(next_step='recover', remaining_work='Read the installation scope from the package manifest.')
    _, decision = V3OuterController().review_and_route(task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context)
    assert decision.action == StrategicAction.REPLAN
    assert decision.reason == verdict['remaining_work']
    directive = task.metadata['act_directive']['outcome_review']
    assert directive['verified_evidence_refs'] == ['tactical-action:inspect']

    recovery = PhasePlan('recovery', decision.reason, (PhaseSubgoal(
        'read', decision.reason, 'scope', allowed_capabilities=('local_shell',),
    ),), authorized_capabilities=('local_shell',))
    next_state = new_tactical_state(recovery, cumulative_evidence=state.evidence.entries)
    context.tools = InMemoryToolRegistry()
    context.tools.register(build_bash_tool_definition())
    executor = PhaseExecutor(action_selector=lambda *_: {
        'tool_id': 'native.bash', 'arguments': {'command': 'fixturelang --version'},
    }, reveal_policy=ToolRevealPolicy())
    outcome = executor.run(task, next_state, context)
    assert outcome.blockers == ('act_recovery_repeats_verified_work',)
    assert not next_state.actions

    task.metadata['v3_act_recovery_version'] = directive['evidence_version']
    _, repeated = V3OuterController().review_and_route(task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context)
    assert repeated.action == StrategicAction.FAIL
    assert len(requests) == 1


@pytest.mark.parametrize('status', ['denied', 'unknown_partial', 'missing_verification'])
def test_effect_guards_run_before_reasoning_resolution(status):
    task, state, context, requests, _ = completed_lookup(requires_effect=.99)
    task.goal = 'Update archive.txt and report the verified result.'
    observed = state.evidence.entries[0]
    observed['arguments'] = {'command': "printf 'new' > archive.txt", 'cwd': '/fixture'}
    if status == 'denied':
        observed['result'] = {'exit_code': 1, 'stderr': 'Permission denied'}
    elif status == 'unknown_partial':
        observed['result'] = {'exit_code': 1, 'stderr': 'Write interrupted after applying an unknown amount'}
    else:
        observed['result'] = {'exit_code': 0, 'stdout': 'Write attempted'}
    _, decision = V3OuterController().review_and_route(task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context)
    assert decision.action == StrategicAction.REPLAN
    assert not requests
    assert task.metadata['system_one_act_judgments']['status'] == 'unsupported'


@pytest.mark.parametrize('during_delivery', [False, True])
def test_resolution_question_repairs_effect_and_resumes_once_from_authenticated_answer(tmp_path, during_delivery):
    task, state, context, requests, verdict = completed_lookup()
    verdict.update(next_step='ask_user', remaining_work='Ask for the date the installed version should be checked against.')
    store = SQLiteQuestionStore(tmp_path / 'questions.sqlite')
    context.question_store = store
    context.tools = InMemoryToolRegistry()
    context.tools.register(build_ask_question_tool_definition())
    task.status = 'running'
    task.intelligence_engine = 'hierarchical_v3'
    task.hierarchical_state = state.to_dict()
    store.save_task_checkpoint(task)
    ask_phase = PhasePlan('ask-date', 'Ask for the comparison date', (
        PhaseSubgoal('ask', 'Ask for the comparison date', 'question',
                     allowed_capabilities=('user_interaction',), allowed_side_effects=(SideEffectClass.USER_RESPONSE,)),
    ), authorized_capabilities=('user_interaction',)).to_dict()
    bad_phase = deepcopy(ask_phase)
    bad_phase['subgoals'][0]['allowed_side_effects'] = ['read_only']
    planning = []
    resolve = context.inference.generate_json

    def generate(request):
        if request.purpose == InferencePurpose.STRATEGIC_ACT:
            return resolve(request)
        planning.append(request)
        assert verdict['remaining_work'] in request.prompt
        if len(planning) == 2:
            assert 'user_interaction capability and user_response side effect' in request.prompt
        return SimpleNamespace(json_value=bad_phase if len(planning) == 1 else ask_phase)

    context.inference = SimpleNamespace(generate_json=generate)
    answers = []

    def answer(question_id):
        assert not store.route_answer(respondent_user_id='another-user', question_id=question_id, text='2026-11-04T09:00:00-06:00').handled
        answers.append(store.route_answer(respondent_user_id='fixture', question_id=question_id, text='2026-11-04T09:00:00-06:00'))
        assert not store.route_answer(respondent_user_id='fixture', question_id=question_id, text='2026-11-04T09:00:00-06:00').handled

    def deliver(event):
        if during_delivery:
            answer(event['question']['question_id'])

    context.delivery_sink = deliver
    processor = HierarchicalCAPDProcessor()
    processor.executor = PhaseExecutor(action_selector=lambda *_: {
        'tool_id': 'native.ask_question',
        'arguments': {'question': 'What date and time should I use for the comparison?', 'question_kind': 'datetime'},
    }, reveal_policy=ToolRevealPolicy())
    result = processor.process(store.load_task_checkpoint(task.task_id), context)
    assert result.status.value == 'parked'
    parked = store.load_task_checkpoint(task.task_id)
    if not during_delivery:
        assert parked.hierarchical_state['status'] == 'waiting_user'
        assert parked.metadata['question_interrupt']['kind'] == 'datetime'
        answer(parked.metadata['pending_question_id'])
    assert answers[0].resumed_task is not None
    restored = store.load_task_checkpoint(task.task_id)
    assert restored.metadata['question_interrupt']['status'] == 'answered'
    assert len(restored.metadata['v3_resumption_evidence']) == 1
    assert restored.metadata['v3_mission_resolution']['next_step'] == 'ask_user'
    from alphonse.agent_v2.core.intelligence.v3.processor import _resume_from_authenticated_answer
    from alphonse.agent_v2.core.intelligence.v3.contracts import TacticalState

    assert _resume_from_authenticated_answer(restored, TacticalState.from_dict(restored.hierarchical_state))
    assert 'do not repeat the answered question' in restored.metadata['act_directive']['rationale']
    assert restored.metadata['act_directive']['outcome_review']['verified_evidence_refs'] == ['tactical-action:inspect']
    assert len(requests) == 1 and len(planning) == 2


def test_semantic_conflict_can_resolve_only_with_a_cited_verified_effect():
    task, state, context, requests, verdict = completed_lookup(requires_effect=.99)
    task.goal = 'Schedule a reminder for Friday at 09:00.'
    task.metadata['prepared_user_response']['message'] = 'The reminder is scheduled for Friday at 09:00.'
    state.evidence.entries[0].update(tool_id='native.scheduled_task', result={
        'scheduled_task_id': 'fixture-reminder', 'next_run_at': '2026-11-06T09:00:00Z',
        'verification': {'status': 'verified', 'source': 'schedule_store_readback'},
    })
    context.system_one.judge_act = lambda **_: act_judgments(
        effect_requested=.99, effect_supported=.99, effect_contradicted=.99,
    )
    _, decision = V3OuterController().review_and_route(task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context)
    assert decision.action == StrategicAction.COMPLETE
    assert len(requests) == 1
    assert task.metadata['v3_mission_resolution']['evidence_refs'] == verdict['evidence_refs']


def test_completion_claim_cannot_substitute_an_observation_for_a_missing_operation():
    task, state, context, requests, _ = completed_lookup(requires_effect=.99)
    task.goal = 'Schedule a reminder for Friday at 09:00.'
    _, decision = V3OuterController().review_and_route(task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context)
    assert decision.action == StrategicAction.FAIL
    assert len(requests) == 1
    assert task.metadata['v3_mission_resolution']['status'] == 'invalid'
