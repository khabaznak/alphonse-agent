"""Synthetic retrieval stalls reach a verified qualified answer or a parked question."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.inference import InferencePurpose
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import (
    PhasePlan, PhaseSubgoal, PhaseOutcome, PhaseStatus, SideEffectClass, new_tactical_state,
)
from alphonse.agent_v2.core.intelligence.v3.outer import V3OuterController
from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
from alphonse.agent_v2.core.intelligence.v3.retrieval_handoff import prepare_handoff, validate_handoff
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
from alphonse.agent_v2.core.tools.registry.native.respond import build_respond_tool_definition
from alphonse.agent_v2.core.tools.registry.native.ask_question import build_ask_question_tool_definition
from alphonse.agent_v2.system_one import SystemOneTacticalReview, SystemOneReviewResult, SystemOneActRecommendation


def read_stage(identity):
    return PhaseSubgoal(identity, 'Inspect the ' + identity + ' source for dated instructions or an explicit gap',
        'source_results_or_gap', allowed_capabilities=('memory_recall',))


def communication(question=False, depends_on=()):
    return PhaseSubgoal('communicate', 'Report supported facts with limitations or ask for the exact missing source',
        'question' if question else 'response', depends_on=depends_on,
        allowed_capabilities=('user_interaction' if question else 'user_response',),
        allowed_side_effects=(SideEffectClass.USER_RESPONSE,))


def lookup_phase(identity='lookup', remaining=False):
    reads = (read_stage('conversation'), read_stage('database')) if remaining else (read_stage('database'),)
    return PhasePlan(identity, 'Answer from dated records with explicit limitations',
        (*reads, communication(depends_on=(reads[-1].subgoal_id,))), criterion_ids=('ac-1',),
        authorized_capabilities=('memory_recall', 'user_response'))


def stalled(remaining=False, identity='lookup'):
    state = new_tactical_state(lookup_phase(identity, remaining))
    stage = state.phase.subgoals[0]
    state.evidence.append({'evidence_ref': identity + ':read', 'phase_id': identity,
        'subgoal_id': stage.subgoal_id, 'tool_id': 'native.search_memory', 'status': 'success',
        'result': {'source': 'fixture', 'dated_instructions': 'Tuesday: FixtureBeta', 'current_status': 'not confirmed'}})
    state.status = PhaseStatus.BLOCKED
    outcome = PhaseOutcome(identity, PhaseStatus.BLOCKED, blockers=('tactical_stage_no_new_progress',))
    return state, outcome


@pytest.mark.parametrize('remaining', [False, True])
def test_handoff_preserves_pending_criteria_and_checkpointed_budget(remaining):
    task = TaskState(user='fixture', goal='What does the recorded schedule say for today?')
    task.set_acceptance_contract_from_markdown('1.- [ ] Supported answer with limitations')
    contract = deepcopy(task.acceptance_contract)
    state, outcome = stalled(remaining)
    context = CoreLoopContext(messages=InMemoryMessageQueue())
    V3OuterController().review_and_route(task, state, outcome, context)
    receipt = task.metadata['act_directive']['retrieval_resolution']
    assert receipt['route'] == ('remaining_source' if remaining else 'answer_or_clarify')
    assert task.acceptance_contract == contract and task.status != 'completed'
    assert not state.completed_subgoal_ids
    restored = TaskState.from_dict(task.to_checkpoint_dict())
    assert prepare_handoff(restored, state, outcome) is None
    next_state, next_outcome = stalled(identity='next')
    if remaining:
        assert prepare_handoff(restored, next_state, next_outcome)['route'] == 'answer_or_clarify'
        next_state, next_outcome = stalled(identity='third')
    assert prepare_handoff(restored, next_state, next_outcome) is None


@pytest.mark.parametrize('problem', ['write_phase', 'failed_reads', 'other_blocker', 'cancelled'])
def test_handoff_cannot_override_operational_failures_or_mutations(problem):
    task = TaskState(user='fixture', goal='Read fixture')
    state, outcome = stalled()
    if problem == 'write_phase':
        from dataclasses import replace
        stage = replace(state.phase.subgoals[0], allowed_side_effects=(SideEffectClass.PROJECT_MUTATION,))
        state.phase = replace(state.phase, subgoals=(stage, communication()))
    elif problem == 'failed_reads': state.evidence.entries[0]['status'] = 'failed'
    elif problem == 'other_blocker': outcome = PhaseOutcome('lookup', PhaseStatus.BLOCKED, blockers=('access_denied',))
    else: outcome = PhaseOutcome('lookup', PhaseStatus.CANCELLED)
    assert prepare_handoff(task, state, outcome) is None


@pytest.mark.parametrize('bad', ['read_again', 'mutation', 'multiple_reads', 'missing_dependency'])
def test_resolution_plan_rejects_retrieval_loops_and_effects(bad):
    if bad == 'read_again':
        phase = lookup_phase()
        route = 'answer_or_clarify'
    else:
        route = 'remaining_source'
        stages = [read_stage('database'), communication(depends_on=('database',))]
        if bad == 'mutation':
            from dataclasses import replace
            stages[0] = replace(stages[0], allowed_side_effects=(SideEffectClass.PROJECT_MUTATION,))
        elif bad == 'multiple_reads': stages.insert(1, read_stage('other'))
        else: stages[-1] = communication()
        phase = PhasePlan('bad', 'Invalid recovery', tuple(stages), authorized_capabilities=('memory_recall', 'user_response'))
    with pytest.raises(ValueError, match='retrieval_handoff_'):
        validate_handoff(phase, {'retrieval_resolution': {'route': route}})


@pytest.mark.parametrize('remaining', [False, True])
@pytest.mark.parametrize('question,verified', [(False, True), (True, False), (False, False)])
def test_actual_processor_exits_stalled_retrieval_through_normal_answer_review(tmp_path, remaining, question, verified):
    task = TaskState(task_id='fixture-task', message_id='fixture-message', user='fixture', goal='What is scheduled today?',
        metadata={'v3_admission': {'completed': True, 'route': 'task'}, 'v3_no_progress_phase_count': 2})
    task.set_acceptance_contract_from_markdown('1.- [ ] Explain the dated schedule, preserve gaps, or ask for the specific missing source')
    reads, plans = [], []
    registry = InMemoryToolRegistry()
    def search(arguments):
        reads.append(arguments)
        return {'source_id': 'fixture-record', 'date': '2026-10-30', 'instructions': 'FixtureBeta: Tuesday',
                'limitation': 'A later revision is not established by this source.', 'page': len(reads)}
    registry.register(ToolDefinition(descriptor=ToolDescriptor(tool_id='native.search_memory', name='Search fixture',
        kind=ToolKind.NATIVE, read_only=True, metadata={'v3_capabilities': ['memory_recall']}), callable=search))
    registry.register(build_respond_tool_definition()); registry.register(build_ask_question_tool_definition())
    class Jev:
        def evaluate_tactical_progress(self, *, action, **kwargs):
            read = action['tool_id'] == 'native.search_memory'
            return SystemOneTacticalReview(not read, .99, True, answers={
                'runtime_declared_subgoal_complete': .72 if read else .99,
                'runtime_action_added_progress': .65 if read else .99,
            })
        def evaluate(self, *, evidence, **kwargs):
            assert any(row.get('tool_id') == 'native.respond' for row in evidence['entries'])
            refs = [row['evidence_ref'] for row in evidence['entries'] if row.get('status') == 'success']
            return SystemOneReviewResult(({'criterion_id': 'ac-1', 'status': 'satisfied' if verified else 'pending',
                'evidence_refs': refs},), ())
        def recommend_act(self, **kwargs):
            return SystemOneActRecommendation('complete' if verified else 'fail', .99, True, 'Fixture review',
                answers={'mission_outcome_is_fulfilled': .99 if verified else .01, 'closure_explanation_is_warranted': .01})
        def review_task_failure(self, **kwargs): return False
    class Inference:
        def generate_json(self, request):
            if request.purpose == InferencePurpose.TACTICAL_ACTION:
                return self.plan_tool_call(request)
            assert request.purpose == InferencePurpose.PHASE_PLANNING
            directive = task.metadata.get('act_directive', {}).get('retrieval_resolution')
            if not plans:
                phase = lookup_phase(remaining=remaining)
            else:
                assert 'explicit limitations' in request.prompt
                assert 'FixtureBeta' in request.prompt
                assert directive
                if directive['route'] == 'remaining_source':
                    phase = PhasePlan('remaining', 'Inspect the unattempted database then communicate',
                        (read_stage('database'), communication(depends_on=('database',))), criterion_ids=('ac-1',),
                        authorized_capabilities=('memory_recall', 'user_response'))
                else:
                    phase = PhasePlan('resolve', 'Answer or ask from the existing evidence', (communication(question),),
                        criterion_ids=('ac-1',), authorized_capabilities=('user_interaction' if question else 'user_response',))
            plans.append(phase)
            return SimpleNamespace(json_value=phase.to_dict())
        def plan_tool_call(self, request):
            if plans[-1].phase_id == 'resolve':
                if question:
                    tool, args = 'native.ask_question', {'question': 'Is there a revision later than October 30?', 'question_kind': 'open_text'}
                else:
                    tool, args = 'native.respond', {'message': 'The October 30 record schedules FixtureBeta on Tuesday. I cannot establish whether a later revision changed it.'}
            else:
                tool, args = 'native.search_memory', {'query': 'dated fixture instructions', 'offset': len(reads)}
            value = {'tool_id': tool, 'arguments': args, 'acceptance_questions': [{
                'question_id': 'fixture_observed', 'type': 'noul', 'instructions': 'Does the result supply the stated fixture output?',
                'criteria': {'true': 'Output is present and relevant.', 'false': 'Output is absent or irrelevant.'}}]}
            return SimpleNamespace(tool_call=value, json_value=value)
        def generate_markdown(self, request): return SimpleNamespace(content='Fixture completion remains unverified.')
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, inference=Inference(), system_one=Jev(),
                              question_store=SQLiteQuestionStore(db_path=tmp_path / 'questions.sqlite'))
    result = HierarchicalCAPDProcessor().process(task, context)
    assert len(reads) == (6 if remaining else 3), (task.outcome, task.metadata.get('v3_terminal_error_type'), task.updates_md)
    assert len(plans) == (3 if remaining else 2)
    assert task.status == ('waiting_user' if question else 'completed' if verified else 'failed')
    assert not task.metadata.get('v3_no_progress_stop')
    if not question:
        assert task.metadata['prepared_user_response']['message'].startswith('The October 30 record')
        assert task.acceptance_criteria_all_complete() == verified
        assert task.metadata['system_one_review']['status'] == 'used'
    else:
        assert result.status.value == 'parked'
        assert not task.acceptance_criteria_all_complete()


@pytest.mark.parametrize('recovers', [True, False])
def test_planner_corrects_an_ignored_final_handoff_once(recovers):
    from alphonse.agent_v2.core.intelligence.v3.planner import plan_phase, V3PhasePlanValidationError
    task = TaskState(user='fixture', goal='Answer from the fixture record', metadata={
        'act_directive': {'retrieval_resolution': {'route': 'answer_or_clarify'}},
    })
    task.set_acceptance_contract_from_markdown('1.- [ ] Supported answer with explicit limitations')
    original = deepcopy(task.acceptance_contract)
    calls = []
    valid = PhasePlan('resolve', 'Communicate supported findings and gaps', (communication(),),
        criterion_ids=('ac-1',), authorized_capabilities=('user_response',))
    def generate(request):
        calls.append(request)
        if len(calls) == 2:
            assert 'Plan validation feedback' in request.prompt
            assert 'No further retrieval or mutation' in request.prompt
        return SimpleNamespace(json_value=(valid if recovers and len(calls) == 2 else lookup_phase()).to_dict())
    context = CoreLoopContext(messages=InMemoryMessageQueue(), inference=SimpleNamespace(generate_json=generate))
    if recovers:
        assert plan_phase(task, context).phase_id == 'resolve'
    else:
        with pytest.raises(V3PhasePlanValidationError, match='retrieval_handoff_'):
            plan_phase(task, context)
    assert len(calls) == 2 and task.acceptance_contract == original
    assert task.metadata['v3_plan_validation_recovery']['reason'] == 'retrieval_handoff'


def test_failed_scope_verification_does_not_get_a_retrieval_handoff():
    task = TaskState(user='fixture', goal='Read authorized fixture')
    state, outcome = stalled()
    state.evidence.entries[0]['result'] = {'affected_paths': ['outside-scope.txt']}
    V3OuterController().review_and_route(task, state, outcome, CoreLoopContext(messages=InMemoryMessageQueue()))
    assert task.metadata['v3_phase_review']['status'] == 'verification_failed'
    assert 'v3_retrieval_handoffs' not in task.metadata


# Existing scripted providers delegate the shared interpretation call explicitly.
pytestmark = pytest.mark.usefixtures("task_understanding_provider")
