"""Disposable reliability cases: no provider, household files or live services."""
import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.contracts import (
    PhasePlan, PhaseSubgoal, PhaseStatus, SideEffectClass, new_tactical_state,
)
from alphonse.agent_v2.core.intelligence.v3.executor import PhaseExecutor
from alphonse.agent_v2.core.intelligence.v3.planner import plan_phase, V3PhasePlanValidationError
from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
from alphonse.agent_v2.core.intelligence.v3.revealing import ToolRevealPolicy, validate_phase_actions
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
from alphonse.agent_v2.core.tools.registry.native.ask_question import build_ask_question_tool_definition
from alphonse.agent_v2.core.tools.registry.native.respond import build_respond_tool_definition
from alphonse.agent_v2.system_one import SystemOneToolRegistrySelection, SystemOneTacticalReview


class Jev:
    def curate_request_tools(self, *, tools, **_):
        return SystemOneToolRegistrySelection(selected_tool_ids=tuple(t.tool_id for t in tools))

    def select_plan_tools(self, *, tools, **_):
        return self.curate_request_tools(tools=tools)

    def evaluate_tactical_progress(self, **_):
        return SystemOneTacticalReview(True, 0.99, True)


class Inference:
    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.requests = []

    def generate_json(self, request):
        self.requests.append(request)
        output = next(self.outputs)
        return SimpleNamespace(json_value=output)


def fixture():
    calls = []
    registry = InMemoryToolRegistry()
    registry.register(ToolDefinition(ToolDescriptor(
        'native.lookup', 'lookup', ToolKind.NATIVE, read_only=True,
        metadata={'v3_capabilities': ['lookup']},
    ), lambda args: calls.append(args) or {'value': 'synthetic'}))
    registry.register(build_respond_tool_definition())
    task = TaskState(task_id='synthetic', user='fixture', goal='Look up the synthetic value', status='running')
    task.set_acceptance_contract_from_markdown('- [ ] The synthetic value is reported')
    task.metadata['v3_admission'] = {'completed': True, 'route': 'task'}
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, system_one=Jev())
    return task, context, calls


def phase():
    return PhasePlan('phase', 'Lookup', (PhaseSubgoal(
        'lookup', 'Lookup', 'value', allowed_capabilities=('lookup',),
    ),), authorized_capabilities=('lookup',))


@pytest.mark.parametrize('bad', ['effect', 'capability', 'dependency', 'encoding'])
def test_phase_gets_precise_bounded_correction_and_preserves_contract(bad):
    task, context, calls = fixture()
    contract = deepcopy(task.acceptance_contract)
    valid = phase().to_dict()
    invalid = deepcopy(valid)
    if bad == 'effect':
        invalid['authorized_capabilities'] = ['user_response']
        invalid['subgoals'][0]['allowed_capabilities'] = ['user_response']
    elif bad == 'capability':
        invalid['subgoals'][0]['allowed_capabilities'] = ['missing']
    elif bad == 'dependency':
        invalid['subgoals'][0]['depends_on'] = ['missing']
    else:
        invalid = ['not a phase']
    context.inference = Inference(invalid, valid)
    assert plan_phase(task, context).phase_id == 'phase'
    assert len(context.inference.requests) == 2
    assert 'Plan validation feedback' in context.inference.requests[1].prompt
    assert 'Do not silently broaden authorization' in context.inference.requests[1].prompt
    assert task.acceptance_contract == contract
    assert calls == []


def test_failed_plan_correction_is_reported_and_prior_completed_work_remains():
    task, context, calls = fixture()
    history = [{'phase_id': 'earlier', 'evidence': {'entries': [
        {'evidence_ref': 'verified-earlier', 'status': 'success', 'result': {'value': 42}},
    ]}}]
    task.metadata['v3_phase_history'] = deepcopy(history)
    context.inference = Inference({}, {})
    result = HierarchicalCAPDProcessor().process(task, context)
    assert result.status.value == 'failed'
    assert 'v3_phase_plan_invalid:phase_id_required' in task.outcome['reason']
    assert task.metadata['v3_phase_history'] == history
    assert len(context.inference.requests) == 2
    assert calls == []


def test_cancelled_phase_correction_remains_cancelled():
    task, context, calls = fixture()
    context.inference = Inference({}, phase().to_dict())
    context.cancellation_checker = lambda: bool(context.inference.requests)
    HierarchicalCAPDProcessor().process(task, context)
    assert task.status == 'cancelled'
    assert len(context.inference.requests) == 1
    assert calls == []


def test_valid_plan_and_incompatible_effects_are_deterministic():
    task, context, _ = fixture()
    validate_phase_actions(phase(), tuple(context.tools.list()))
    invalid = replace(phase(), authorized_capabilities=('user_response',), subgoals=(
        replace(phase().subgoals[0], allowed_capabilities=('user_response',)),
    ))
    with pytest.raises(ValueError, match='phase_subgoal_effect_incompatible'):
        validate_phase_actions(invalid, tuple(context.tools.list()))


def test_tactical_correction_runs_only_corrected_action_and_retains_completed_binding():
    task, context, calls = fixture()
    state = new_tactical_state(phase())
    state.bindings['earlier'] = {'value': {'verified': 42}}
    choices = iter([{'tool_id': 'native.respond', 'arguments': {'message': 'wrong effect'}},
                    {'tool_id': 'native.lookup', 'arguments': {'key': 'fixture'}}])

    def select(current, _subgoal, _tools):
        if current.evidence.entries:
            assert 'tactical_tool_not_revealed' in current.evidence.entries[-1]['error']
            assert 'No tool ran' in current.evidence.entries[-1]['validation_feedback']
        return next(choices)

    outcome = PhaseExecutor(action_selector=select, reveal_policy=ToolRevealPolicy()).run(task, state, context)
    assert outcome.status == PhaseStatus.PHASE_COMPLETE
    assert calls == [{'key': 'fixture'}]
    assert state.bindings['earlier']['value']['verified'] == 42
    assert len(state.actions) == 1


@pytest.mark.parametrize('changed_error', [False, True])
def test_failed_tactical_correction_is_bounded_even_with_changed_ids(changed_error):
    task, context, calls = fixture()
    state = new_tactical_state(phase())
    n = 0

    def select(*_):
        nonlocal n
        n += 1
        return {'action_id': str(n), 'tool_id': 'other' if changed_error and n == 2 else 'hidden', 'arguments': {}}

    outcome = PhaseExecutor(action_selector=select, reveal_policy=ToolRevealPolicy()).run(task, state, context)
    assert outcome.status == PhaseStatus.BLOCKED
    assert 'tactical_planning_error_repeated' in outcome.reason
    assert n == 2
    assert calls == []
    # The same rejected call is recognized after serialization and a renamed phase.
    restored = TaskState.from_dict(task.to_dict())
    fresh = new_tactical_state(replace(phase(), phase_id='renamed'))
    fresh_outcome = PhaseExecutor(action_selector=lambda *_: {'tool_id': 'hidden', 'arguments': {}},
                                 reveal_policy=ToolRevealPolicy()).run(restored, fresh, context)
    assert 'tactical_planning_error_repeated' in fresh_outcome.reason
    assert len(fresh.evidence.entries) == 1


def test_cancellation_during_tactical_selection_prevents_dispatch():
    task, context, calls = fixture()
    selected = []
    context.cancellation_checker = lambda: bool(selected)

    def select(*_):
        selected.append(True)
        return {'tool_id': 'native.lookup', 'arguments': {}}

    state = new_tactical_state(phase())
    outcome = PhaseExecutor(action_selector=select, reveal_policy=ToolRevealPolicy()).run(task, state, context)
    assert outcome.status == PhaseStatus.CANCELLED
    assert calls == []


def test_real_question_delivery_parking_and_answered_resume(tmp_path):
    task, context, calls = fixture()
    context.tools.register(build_ask_question_tool_definition())
    context.question_store = SQLiteQuestionStore(tmp_path / 'questions.sqlite3')
    deliveries = []
    context.delivery_sink = lambda event: deliveries.append(event) or {'delivered': True}
    question_phase = PhasePlan('clarify', 'Find missing source', (PhaseSubgoal(
        'ask', 'Which synthetic source?', 'source', allowed_capabilities=('user_interaction',),
    ),), authorized_capabilities=('user_interaction',))
    validate_phase_actions(question_phase, tuple(context.tools.list()))
    state = new_tactical_state(question_phase)
    executor = PhaseExecutor(action_selector=lambda *_: {
        'tool_id': 'native.ask_question', 'arguments': {'question': 'Which synthetic source?', 'question_kind': 'open_text'},
    }, reveal_policy=ToolRevealPolicy())
    outcome = executor.run(task, state, context)
    assert outcome.status == PhaseStatus.WAITING_USER
    assert len(deliveries) == 1
    question_id = task.metadata['pending_question_id']
    answer = context.question_store.route_answer(respondent_user_id='fixture', question_id=question_id, text='source B')
    resumed = answer.resumed_task
    assert resumed is not None
    assert 'source B' in json.dumps(resumed.metadata['v3_resumption_evidence'])
    # Real processor resume chooses a new phase using the durable answer, never replays the question.
    processor = HierarchicalCAPDProcessor()
    def resumed_plan(current, _context):
        assert 'source B' in json.dumps(current.metadata['v3_phase_history'])
        return new_tactical_state(phase())
    processor._new_state = resumed_plan
    processor.executor = PhaseExecutor(action_selector=lambda *_: {
        'tool_id': 'native.lookup', 'arguments': {'source': 'B'},
    }, reveal_policy=ToolRevealPolicy())

    class Outer:
        def review_and_route(self, current, *_):
            current.status = 'completed'
            current.metadata['v3_route'] = 'end'
            return object(), object()

    processor.outer = Outer()
    processor.process(resumed, context)
    assert resumed.status == 'completed'
    assert calls == [{'source': 'B'}]
    assert len(deliveries) == 1


def test_processor_does_not_replan_or_manufacture_success_after_failed_action_correction():
    task, context, calls = fixture()
    task.hierarchical_state = new_tactical_state(phase()).to_dict()
    processor = HierarchicalCAPDProcessor()
    processor.executor = PhaseExecutor(action_selector=lambda *_: {
        'tool_id': 'hidden', 'arguments': {},
    }, reveal_policy=ToolRevealPolicy())

    class Outer:
        def review_and_route(self, *_):
            pytest.fail('Repeated invalid planning must stop before probabilistic review')

    processor.outer = Outer()
    processor.process(task, context)
    assert task.status == 'failed'
    assert 'tactical_planning_error_repeated' in task.outcome['reason']
    assert len(task.metadata['v3_phase_history']) == 1
    assert calls == []


def test_failed_correction_preserves_actual_completed_subgoal_after_checkpoint(tmp_path):
    task, context, calls = fixture()
    context.question_store = SQLiteQuestionStore(tmp_path / 'checkpoint.sqlite3')
    first = phase().subgoals[0]
    second = replace(first, subgoal_id='second', depends_on=('lookup',), required_output_type='second-value')
    state = new_tactical_state(replace(phase(), subgoals=(first, second)))

    def select(_state, subgoal, _tools):
        return {'tool_id': 'native.lookup' if subgoal.subgoal_id == 'lookup' else 'hidden', 'arguments': {}}

    outcome = PhaseExecutor(action_selector=select, reveal_policy=ToolRevealPolicy()).run(task, state, context)
    assert outcome.status == PhaseStatus.BLOCKED
    restored = context.question_store.load_task_checkpoint(task.task_id)
    assert restored.hierarchical_state['completed_subgoal_ids'] == ['lookup']
    assert restored.hierarchical_state['bindings']['value']['value'] == {'value': 'synthetic'}
    assert len(restored.hierarchical_state['actions']) == 1
    assert len(calls) == 1


pytestmark = pytest.mark.usefixtures("task_understanding_provider")


def test_tactical_uncertainty_uses_next_composition_call_with_evidence_and_no_extra_calls():
    task, context, calls = fixture()
    state = new_tactical_state(phase())
    question = {'question_id': 'result', 'type': 'noul', 'instructions': 'Does the result satisfy the stage?',
                'criteria': {'true': 'The result satisfies it.', 'false': 'The result does not satisfy it.'}}
    context.inference = Inference(
        {'tool_id': 'native.lookup', 'arguments': {'key': 'first'}, 'acceptance_questions': [question]},
        {'tool_id': 'native.lookup', 'arguments': {'key': 'second'}, 'acceptance_questions': [question]},
    )
    reviews = iter([
        SystemOneTacticalReview(False, .6, False, answers={'runtime_action_added_progress': .6,
                                                           'runtime_declared_subgoal_complete': .1}),
        SystemOneTacticalReview(True, .99, True, answers={'runtime_action_added_progress': .99,
                                                          'runtime_declared_subgoal_complete': .99}),
    ])
    context.system_one.evaluate_tactical_progress = lambda **kwargs: next(reviews)
    outcome = PhaseExecutor(reveal_policy=ToolRevealPolicy()).run(task, state, context)
    assert outcome.status == PhaseStatus.PHASE_COMPLETE
    assert calls == [{'key': 'first'}, {'key': 'second'}]
    assert len(context.inference.requests) == 2
    prompt = context.inference.requests[1].prompt
    assert 'Targeted uncertainty recovery' in prompt
    assert state.evidence.entries[0]['evidence_ref'] in prompt
    assert 'synthetic' in prompt
    assert state.evidence.entries[0]['progress_review']['uncertainty_recovery']['status'] == 'consumed'


def test_uncertainty_recovery_cannot_authorize_unrevealed_action():
    task, context, calls = fixture()
    state = new_tactical_state(phase())
    question = {'question_id': 'result', 'type': 'noul', 'instructions': 'Does this satisfy the stage?',
                'criteria': {'true': 'The result satisfies it.', 'false': 'It does not satisfy it.'}}
    first = {'tool_id': 'native.lookup', 'arguments': {'key': 'first'}, 'acceptance_questions': [question]}
    forbidden = {'tool_id': 'native.respond', 'arguments': {'message': 'done'}, 'acceptance_questions': [question]}
    context.inference = Inference(first, forbidden, forbidden)
    context.system_one.evaluate_tactical_progress = lambda **kwargs: SystemOneTacticalReview(
        False, .6, False, answers={'runtime_action_added_progress': .6, 'runtime_declared_subgoal_complete': .1})
    outcome = PhaseExecutor(reveal_policy=ToolRevealPolicy()).run(task, state, context)
    assert outcome.status == PhaseStatus.BLOCKED
    assert calls == [{'key': 'first'}]
    assert not state.completed_subgoal_ids
    assert 'tactical_tool_not_revealed' in outcome.reason


def test_stage_review_recovery_remains_available_after_action_uncertainty():
    task, context, calls = fixture()
    state = new_tactical_state(phase())
    question = {'question_id': 'action_result', 'type': 'noul', 'instructions': 'Is this action result sufficient?',
                'criteria': {'true': 'The result is sufficient.', 'false': 'The result is insufficient.'}}
    context.inference = Inference(*[
        {'tool_id': 'native.lookup', 'arguments': {'key': key}, 'acceptance_questions': [question]}
        for key in ('first', 'second', 'third')])
    reviews = iter([
        SystemOneTacticalReview(False, .6, False, answers={'action_result': .6,
            'runtime_action_added_progress': .9, 'runtime_declared_subgoal_complete': .1}),
        SystemOneTacticalReview(False, .6, False, answers={'action_result': .9,
            'runtime_action_added_progress': .9, 'runtime_declared_subgoal_complete': .6}),
        SystemOneTacticalReview(True, .99, True, answers={'action_result': .99,
            'runtime_action_added_progress': .99, 'runtime_declared_subgoal_complete': .99}),
    ])
    context.system_one.evaluate_tactical_progress = lambda **kwargs: next(reviews)
    outcome = PhaseExecutor(reveal_policy=ToolRevealPolicy()).run(task, state, context)
    assert outcome.status == PhaseStatus.PHASE_COMPLETE
    assert len(calls) == len(context.inference.requests) == 3
    assert 'action_result' in context.inference.requests[1].prompt
    assert 'runtime_declared_subgoal_complete' in context.inference.requests[2].prompt
    markers = [e['progress_review']['uncertainty_recovery']['budget_class'] for e in state.evidence.entries
               if (e.get('progress_review') or {}).get('uncertainty_recovery')]
    assert markers == ['other', 'stage']


def test_system_two_assessment_is_recorded_but_cannot_complete_a_stage():
    task, context, calls = fixture()
    state = new_tactical_state(phase())
    question = {'question_id': 'result', 'type': 'noul', 'instructions': 'Does the action satisfy the stage?',
                'criteria': {'true': 'It satisfies the stage.', 'false': 'It does not satisfy the stage.'}}
    class Provider:
        def __init__(self): self.requests = []
        def generate_json(self, request):
            self.requests.append(request)
            n = len(self.requests)
            if n == 3:
                return SimpleNamespace(json_value=None)
            value = {'tool_id': 'native.lookup', 'arguments': {'key': str(n)}, 'acceptance_questions': [question]}
            if n == 2:
                value['uncertainty_assessment'] = {'question_id': 'runtime_action_added_progress',
                    'answer': 'supported', 'evidence_refs': [state.evidence.entries[0]['evidence_ref']]}
            return SimpleNamespace(json_value=value)
    context.inference = Provider()
    context.system_one.evaluate_tactical_progress = lambda **kwargs: SystemOneTacticalReview(
        False, .6, False, answers={'runtime_action_added_progress': .6, 'runtime_declared_subgoal_complete': .1})
    outcome = PhaseExecutor(reveal_policy=ToolRevealPolicy()).run(task, state, context)
    assert outcome.status == PhaseStatus.BLOCKED
    assert len(calls) == 2
    assert not state.completed_subgoal_ids
    recorded = state.evidence.entries[0]['progress_review']['uncertainty_recovery']['assessment']
    assert recorded == {'status': 'recorded', 'answer': 'supported',
                        'evidence_refs': [state.evidence.entries[0]['evidence_ref']]}


@pytest.mark.parametrize('message,expected', [
    ('openai_codex_invalid_json', 'v3_runtime_error:openai_codex_invalid_json'),
    ('synthetic-private-output', 'v3_runtime_error:ValueError'),
])
def test_tactical_provider_error_has_safe_diagnostic_and_preserves_prior_evidence(message, expected):
    task, context, calls = fixture()
    task.hierarchical_state = new_tactical_state(phase()).to_dict()
    task.metadata['v3_phase_history'] = [{'phase_id': 'prior', 'evidence': {'entries': [
        {'evidence_ref': 'verified-prior', 'status': 'success', 'result': {'value': 42}}]}}]
    class BrokenInference:
        def generate_json(self, _request):
            raise ValueError(message)
    context.inference = BrokenInference()
    processor = HierarchicalCAPDProcessor()
    processor.executor = PhaseExecutor(reveal_policy=ToolRevealPolicy())
    processor.process(task, context)
    assert task.status == 'failed'
    assert task.outcome['reason'] == expected
    assert task.metadata['v3_phase_history'][0]['evidence']['entries'][0]['evidence_ref'] == 'verified-prior'
    assert 'synthetic-private-output' not in json.dumps(task.to_dict())
    assert calls == []
