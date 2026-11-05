"""Completion disagreement and malformed-plan recovery using synthetic evidence."""
from conftest import act_judgments
from copy import deepcopy
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.inference import InferencePurpose
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import (
    PhasePlan, PhaseSubgoal, PhaseOutcome, PhaseStatus, SideEffectClass,
    StrategicAction, V3OuterController, new_tactical_state,
)
from alphonse.agent_v2.core.intelligence.v3.planner import plan_phase, V3PhasePlanValidationError
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
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
        'subgoal_id': 'inspect', 'status': 'success', 'result': {'stdout': 'FixtureLang 7.2', 'exit_code': 0}})
    state.evidence.append({'evidence_ref': 'tactical-action:reply', 'tool_id': 'native.respond',
        'subgoal_id': 'reply', 'status': 'success', 'result': {'message': 'FixtureLang 7.2 is installed.'}})

    class Jev:
        def evaluate(self, **kwargs):
            return SystemOneReviewResult(({'criterion_id': 'ac-1', 'status': 'satisfied',
                'evidence_refs': ['tactical-action:inspect', 'tactical-action:reply']},), ())
        def judge_act(self, **kwargs):
            return act_judgments(effect_requested=requires_effect, reply_addresses_request=probability)

    requests = []
    verdict = {'fulfilled': True, 'evidence_refs': ['tactical-action:inspect'],
               'reason': 'The verified output supports the prepared answer.', 'remaining_work': ''}
    def generate(request):
        requests.append(request)
        assert request.purpose == InferencePurpose.STRATEGIC_ACT
        assert request.tools == () and request.output_schema
        assert 'FixtureLang 7.2' in request.prompt
        return SimpleNamespace(json_value=deepcopy(verdict))
    context = CoreLoopContext(messages=InMemoryMessageQueue(), system_one=Jev(),
                              inference=SimpleNamespace(generate_json=generate))
    return task, state, context, requests, verdict


@pytest.mark.parametrize('probability,review_count', [(.79, 1), (.5, 1), (.2, 0), (.8, 0), (.99, 0)])
def test_uncertain_fulfillment_is_reviewed_without_repeating_completed_work(probability, review_count):
    task, state, context, requests, _ = completed_lookup(probability)
    _, decision = V3OuterController().review_and_route(task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context)
    assert len(requests) == review_count
    assert decision.action == (StrategicAction.REPLAN if probability == .2 else StrategicAction.COMPLETE)
    assert task.metadata['system_one_act_judgments']['answers']['reply_addresses_request'] == probability
    if review_count:
        assert task.metadata['v3_reply_resolution']['status'] == 'fulfilled'
        assert task.metadata['v3_route'] == 'respond_and_end'
        assert 'act_directive' not in task.metadata


def test_reasoning_identifies_real_missing_work_without_forcing_read_only_tasks_to_mutate():
    task, state, context, requests, verdict = completed_lookup()
    verdict.update(fulfilled=False, remaining_work='The version source has not established the requested installation scope.')
    _, decision = V3OuterController().review_and_route(task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context)
    assert decision.action == StrategicAction.REPLAN and len(requests) == 1
    assert task.metadata['v3_reply_resolution']['remaining_work']
    assert 'inspection_resolution' not in task.metadata['act_directive']
    assert task.status != 'completed'


@pytest.mark.parametrize('invalid', [None, {'fulfilled': True},
    {'fulfilled': True, 'evidence_refs': ['invented'], 'reason': 'Done', 'remaining_work': ''},
    {'fulfilled': True, 'evidence_refs': ['tactical-action:inspect'], 'reason': 'Done', 'remaining_work': 'Still missing work'}])
def test_invalid_reasoning_review_never_promotes_uncertainty_to_success(invalid):
    task, state, context, _, _ = completed_lookup()
    context.inference = SimpleNamespace(generate_json=lambda request: SimpleNamespace(json_value=invalid))
    _, decision = V3OuterController().review_and_route(task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context)
    assert decision.action == StrategicAction.REPLAN and task.status != 'completed'
    assert task.metadata['v3_reply_resolution']['status'] == 'unresolved'


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
    from alphonse.agent_v2.core.intelligence.v3.outer import _resolve_reply_ambiguity
    task, _, context, _, verdict = completed_lookup()
    verdict['evidence_refs'] = ['omitted:1']
    payload = {'execution_evidence': [{'evidence_ref': 'omitted:1', 'status': 'success'}]}
    context.inference = SimpleNamespace(generate_json=lambda request: SimpleNamespace(json_value=verdict))
    assert _resolve_reply_ambiguity(task, payload, context) is None
    assert task.metadata['v3_reply_resolution']['status'] == 'unresolved'


def test_failed_reasoning_review_is_unresolved_without_exposing_provider_body():
    task, state, context, _, _ = completed_lookup()
    def fail(request):
        raise RuntimeError('PRIVATE_PROVIDER_BODY')
    context.inference = SimpleNamespace(generate_json=fail)
    _, decision = V3OuterController().review_and_route(task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context)
    assert decision.action == StrategicAction.REPLAN
    assert task.metadata['v3_reply_resolution'] == {
        'status': 'unresolved', 'provider': 'system_two', 'exception_type': 'RuntimeError',
    }


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
    task, state, context, requests, _ = completed_lookup()
    context.system_one.judge_act = lambda **_: act_judgments(
        effect_requested=requested, effect_supported=support, effect_contradicted=conflict,
    )
    _, decision = V3OuterController().review_and_route(
        task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context,
    )
    assert task.metadata['system_one_act_judgments']['status'] == status
    assert decision.action == (StrategicAction.COMPLETE if status == 'supported' else StrategicAction.REPLAN)
    assert not requests
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
    task, state, context, requests, _ = completed_lookup(requires_effect=.99, probability=.99)
    state.evidence.entries[0]['result'] = result
    _, decision = V3OuterController().review_and_route(
        task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context,
    )
    assert decision.action == StrategicAction.REPLAN
    assert task.metadata['system_one_act_judgments']['status'] == 'unsupported'
    assert task.metadata['act_directive']['outcome_review']['reason'].startswith('No substantive')
    assert not requests


def test_negative_grounding_blocks_a_reply_even_when_it_addresses_the_request():
    task, state, context, requests, _ = completed_lookup()
    context.system_one.judge_act = lambda **_: act_judgments(reply_grounded=.05)
    _, decision = V3OuterController().review_and_route(
        task, state, PhaseOutcome('lookup', PhaseStatus.PHASE_COMPLETE), context,
    )
    assert decision.action == StrategicAction.REPLAN
    assert task.metadata['system_one_act_judgments']['status'] == 'unsupported'
    assert not requests
