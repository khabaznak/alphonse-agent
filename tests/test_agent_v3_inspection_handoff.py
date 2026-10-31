"""Inspection resolves to an authorized effect or a concrete question, never success by acknowledgment."""
from copy import deepcopy
import shlex
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.contracts import (
    MutationScope, PhasePlan, PhaseSubgoal, PhaseStatus, SideEffectClass, new_tactical_state,
)
from alphonse.agent_v2.core.intelligence.v3.executor import PhaseExecutor, MAX_UPDATE_INSPECTION_ACTIONS
from alphonse.agent_v2.core.intelligence.v3.outer import V3OuterController, StrategicAction
from alphonse.agent_v2.core.intelligence.v3.planner import plan_phase
from alphonse.agent_v2.core.intelligence.v3.revealing import ToolRevealPolicy
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.ask_question import build_ask_question_tool_definition
from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
from alphonse.agent_v2.system_one import (
    JevCriterionDecisionProvider, SystemOneSettings, SystemOneToolRegistrySelection, SystemOneTacticalReview,
)


def inspection_phase(mixed_stage=False):
    first = PhaseSubgoal('inspect', 'Inspect then apply the authorized update' if mixed_stage else 'Read the existing record and identify sufficient evidence or exact missing fields', 'inspection',
        allowed_capabilities=('local_shell',),
        allowed_side_effects=(SideEffectClass.READ_ONLY, SideEffectClass.PROJECT_MUTATION) if mixed_stage else (SideEffectClass.READ_ONLY,))
    last = (PhaseSubgoal('report', 'Report the verified result', 'response', allowed_capabilities=('user_response',), allowed_side_effects=(SideEffectClass.USER_RESPONSE,))
        if mixed_stage else PhaseSubgoal('update', 'Persist the reviewed value and read it back', 'verified_record', depends_on=('inspect',),
            allowed_capabilities=('local_shell',), allowed_side_effects=(SideEffectClass.PROJECT_MUTATION,)))
    return PhasePlan('inspect', 'Inspect then save the authorized fixture value', (first, last), criterion_ids=('ac-1',),
        authorized_capabilities=('local_shell', 'user_response') if mixed_stage else ('local_shell',), mutation_scope=MutationScope(('records.sqlite',)))


def registry():
    tools = InMemoryToolRegistry()
    tools.register(build_bash_tool_definition())
    tools.register(build_ask_question_tool_definition())
    return tools


def command(script):
    return shlex.quote(sys.executable) + ' -c ' + shlex.quote(script)


@pytest.mark.parametrize('route', ['ready_for_action', 'requires_user_input'])
@pytest.mark.parametrize('repeat_inspection_plan', [False, True])
@pytest.mark.parametrize('mixed_stage', [False, True])
def test_inspection_handoff_commits_and_reads_back_or_parks_with_specific_question(tmp_path, monkeypatch, route, repeat_inspection_plan, mixed_stage):
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', '0')
    database = tmp_path / 'records.sqlite'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE records(id INTEGER PRIMARY KEY, value TEXT)')
        db.execute("INSERT INTO records VALUES(1, 'old')")
    task = TaskState(user='fixture-owner', task_id='inspection-test', goal='Save record 1 with the reviewed value', intelligence_engine='hierarchical_v3', intelligence_schema_version=3)
    task.set_acceptance_contract_from_markdown('1.- [ ] Record 1 is persisted with the reviewed value and read back')
    questions = SQLiteQuestionStore(':memory:')
    calls = []

    class Jev:
        def evaluate_tactical_progress(self, **values):
            if values['phase']['phase_id'] == 'inspect':
                # The source cannot be independently re-read, so the original
                # over-demanding stage stays incomplete despite resolving the decision.
                return SystemOneTacticalReview(False, .99, True, answers={
                    'runtime_action_is_inspection': .99,
                    'runtime_declared_subgoal_complete': .05,
                    'runtime_action_added_progress': .99,
                    'runtime_inspection_ready_for_action': .99 if route == 'ready_for_action' else .01,
                    'runtime_inspection_requires_user_input': .99 if route == 'requires_user_input' else .01,
                })
            return SystemOneTacticalReview(True, .99, True)

        def curate_request_tools(self, **values):
            # In the question branch, simulate curation omitting the needed tool.
            return SystemOneToolRegistrySelection(selected_tool_ids=('native.bash',))

        def select_plan_tools(self, **values):
            return SystemOneToolRegistrySelection(selected_tool_ids=('native.bash',))

    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry(), system_one=Jev(), question_store=questions)
    phase = inspection_phase(mixed_stage)
    state = new_tactical_state(phase)
    read = command("import sqlite3; c=sqlite3.connect('file:records.sqlite?mode=ro', uri=True); print(c.execute('SELECT * FROM records WHERE id=1').fetchone()); print('SOURCE: reviewed extraction already confirmed' if "
                   + repr(route == 'ready_for_action') + " else 'MISSING_FIELD: start_date; source unreadable')")

    def inspect(*args):
        calls.append('inspect')
        return {'tool_id': 'native.bash', 'arguments': {'command': read, 'cwd': str(tmp_path)}}

    outcome = PhaseExecutor(action_selector=inspect, reveal_policy=ToolRevealPolicy()).run(task, state, context)
    assert outcome.status == PhaseStatus.BLOCKED
    assert calls == ['inspect']
    assert not state.completed_subgoal_ids  # Handoff is not fabricated inspection or mutation completion.
    assert not task.acceptance_criteria_all_complete()
    _, decision = V3OuterController().review_and_route(task, state, outcome, context)
    assert decision.action == (StrategicAction.ASK_USER if route == 'requires_user_input' else StrategicAction.REPLAN)
    assert task.metadata['act_directive']['inspection_resolution']['route'] == route
    assert task.status == 'running'  # Only native.ask_question can actually park it.
    with sqlite3.connect(database) as db:
        assert db.execute('SELECT value FROM records').fetchone()[0] == 'old'

    task.metadata['v3_phase_history'] = [{'phase': phase.to_dict(), 'evidence': state.evidence.to_dict(), 'outcome': outcome.to_dict()}]
    next_phase = dict(schema_version=3, phase_id='resolve', objective='Resolve inspection evidence',
        acceptance_criteria=[], criterion_ids=['ac-1'], mutation_scope={'allowed_paths': ['records.sqlite'] if route == 'ready_for_action' else [], 'allow_external_effects': False},
        authorized_capabilities=['local_shell'] if route == 'ready_for_action' else ['user_interaction'],
        subgoals=[dict(subgoal_id='resolve', objective='Persist and verify the reviewed value' if route == 'ready_for_action' else 'Ask for the missing start_date only',
            required_output_type='verified_record' if route == 'ready_for_action' else 'question', depends_on=[],
            allowed_capabilities=['local_shell'] if route == 'ready_for_action' else ['user_interaction'],
            allowed_side_effects=['project_mutation'] if route == 'ready_for_action' else ['user_response'],
            completion={'kind': 'output_present'}, failure_policy='stop')])
    requests = []

    def generate(request):
        requests.append(request)
        assert 'inspection_resolution' in request.prompt
        assert 'Do not request generic permission again' in request.prompt or 'ask only for the specific missing information' in request.prompt
        if repeat_inspection_plan and len(requests) == 1:
            return SimpleNamespace(json_value=inspection_phase().to_dict())
        return SimpleNamespace(json_value=deepcopy(next_phase))

    context.inference = SimpleNamespace(generate_json=generate)
    planned = plan_phase(task, context)
    assert len(requests) == (2 if repeat_inspection_plan else 1)
    assert planned.subgoals[0].subgoal_id == 'resolve'
    assert not task.acceptance_criteria_all_complete()
    resumed = new_tactical_state(planned, cumulative_evidence=state.evidence.entries)
    if route == 'ready_for_action':
        write = command("import sqlite3; c=sqlite3.connect('records.sqlite'); c.execute('BEGIN'); c.execute(\"UPDATE records SET value='reviewed' WHERE id=1\"); c.commit(); c.close(); r=sqlite3.connect('file:records.sqlite?mode=ro',uri=True); print('PERSISTED_READ_BACK',r.execute('SELECT value FROM records WHERE id=1').fetchone()[0])")
        action = {'tool_id': 'native.bash', 'arguments': {'command': write, 'cwd': str(tmp_path)}}
    else:
        action = {'tool_id': 'native.ask_question', 'arguments': {'question': 'What start_date should be recorded? The source did not establish that field.', 'question_kind': 'open_text'}}
    final = PhaseExecutor(action_selector=lambda *_: action, reveal_policy=ToolRevealPolicy()).run(task, resumed, context)
    with sqlite3.connect(database) as db:
        assert db.execute('SELECT value FROM records').fetchone()[0] == ('reviewed' if route == 'ready_for_action' else 'old')
    if route == 'ready_for_action':
        assert final.status == PhaseStatus.PHASE_COMPLETE
        assert 'PERSISTED_READ_BACK reviewed' in resumed.evidence.entries[-1]['result']['stdout']
    else:
        assert final.status == PhaseStatus.WAITING_USER
        assert 'start_date' in resumed.evidence.entries[-1]['result']['question_interrupt']['message']
        assert not task.acceptance_criteria_all_complete()
        from alphonse.agent_v2.core.intelligence.v3.processor import _resume_inspection_from_answer
        interrupt = resumed.evidence.entries[-1]['result']['question_interrupt']
        answered = questions.route_answer(respondent_user_id='fixture-owner', question_id=interrupt['question_id'], text='2026-10-31')
        assert answered.handled and answered.resumed_task is not None
        continuation = answered.resumed_task
        _resume_inspection_from_answer(continuation, resumed)
        assert not continuation.metadata['act_directive'].get('inspection_resolution')
        assert continuation.metadata['act_directive']['action'] == 'replan'
        assert not continuation.acceptance_criteria_all_complete()


@pytest.mark.parametrize('mixed_stage', [False, True])
def test_useful_but_unfinished_inspection_has_a_task_wide_bound(tmp_path, mixed_stage):
    task = TaskState(user='fixture-owner', goal='Save the reviewed record')
    # Simulate previous phases using up all but one attempt. Renaming the phase
    # or producing different successful output must not reset the inspection budget.
    task.metadata['v3_update_inspection_actions'] = MAX_UPDATE_INSPECTION_ACTIONS - 1
    class Jev:
        def evaluate_tactical_progress(self, **kwargs):
            return SystemOneTacticalReview(False, .99, True, answers={'runtime_declared_subgoal_complete': .05, 'runtime_action_added_progress': .99, 'runtime_action_is_inspection': .99})
    state = new_tactical_state(inspection_phase(mixed_stage))
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry(), system_one=Jev())
    outcome = PhaseExecutor(action_selector=lambda *_: {'tool_id': 'native.bash', 'arguments': {'command': "printf 'Another useful inspection result'", 'cwd': str(tmp_path)}}, reveal_policy=ToolRevealPolicy()).run(task, state, context)
    assert outcome.blockers == ('inspection_decision_required',)
    assert len(state.actions) == 1
    assert task.metadata['v3_inspection_resolution']['route'] == 'decision_required'
    assert not state.completed_subgoal_ids


@pytest.mark.parametrize('ready,missing', [(True, .01), (float('nan'), .01), (.99, .99), (.5, .5)])
def test_malformed_or_conflicting_inspection_judgments_cannot_authorize_a_handoff(tmp_path, ready, missing):
    class Jev:
        def evaluate_tactical_progress(self, **kwargs):
            return SystemOneTacticalReview(False, .99, True, answers={'runtime_declared_subgoal_complete': .01,
                'runtime_action_added_progress': .01, 'runtime_inspection_ready_for_action': ready, 'runtime_inspection_requires_user_input': missing})
    state = new_tactical_state(inspection_phase())
    task = TaskState(user='fixture-owner')
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry(), system_one=Jev())
    outcome = PhaseExecutor(action_selector=lambda *_: {'tool_id': 'native.bash', 'arguments': {'command': 'true', 'cwd': str(tmp_path)}}, reveal_policy=ToolRevealPolicy()).run(task, state, context)
    assert outcome.status == PhaseStatus.BLOCKED
    assert not task.metadata.get('v3_inspection_resolution')
    assert not state.completed_subgoal_ids


@pytest.mark.parametrize('mixed_stage', [False, True])
def test_jev_separates_readiness_from_stage_success_and_never_generates_a_question(mixed_stage):
    seen = []
    def transport(url, key, payload, timeout):
        seen.append(payload)
        return {'answers': {name: {'type': 'noul', 'noul': .99 if name == 'runtime_inspection_requires_user_input' else .01}
                            for name in payload['questions']}}
    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key='fixture'), transport=transport)
    phase = inspection_phase(mixed_stage)
    review = provider.evaluate_tactical_progress(goal='Save reviewed record', phase=phase.to_dict(),
        subgoal={'allowed_side_effects': ['read_only', 'project_mutation'] if mixed_stage else ['read_only']}, action={'status': 'success'}, execution_log=[])
    assert not review.complete
    assert 'runtime_action_is_inspection' in seen[0]['questions']
    assert review.answers['runtime_inspection_requires_user_input'] == .99
    assert all(q['type'] == 'noul' for q in seen[0]['questions'].values())
    assert 'previously reviewed/confirmed extraction' in seen[0]['questions']['runtime_inspection_ready_for_action']['criteria']['true']


def test_inspection_handoff_gets_its_decision_phase_before_no_progress_stop(monkeypatch):
    from alphonse.agent_v2.core.intelligence.v3.contracts import PhaseOutcome
    from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
    task = TaskState(user='fixture-owner', goal='Save reviewed fixture', metadata={
        'v3_no_progress_phase_count': 2,
        'v3_inspection_resolution': {'route': 'requires_user_input', 'phase_id': 'inspect', 'evidence_refs': []},
    })
    first = new_tactical_state(inspection_phase())
    first.status = PhaseStatus.BLOCKED
    question = PhasePlan('question', 'Ask for the missing start_date', (PhaseSubgoal(
        'question', 'Ask for start_date', 'question', allowed_capabilities=('user_interaction',),
        allowed_side_effects=(SideEffectClass.USER_RESPONSE,),
    ),), authorized_capabilities=('user_interaction',))
    second = new_tactical_state(question)
    second.status = PhaseStatus.WAITING_USER
    states = iter([first, second])
    outcomes = iter([PhaseOutcome('inspect', PhaseStatus.BLOCKED, blockers=('inspection_requires_user_input',)),
                     PhaseOutcome('question', PhaseStatus.WAITING_USER)])
    processor = HierarchicalCAPDProcessor()
    monkeypatch.setattr(processor, '_new_state', lambda *_: next(states))
    monkeypatch.setattr(processor.executor, 'run', lambda *_: next(outcomes))
    result = processor.process(task, CoreLoopContext(messages=InMemoryMessageQueue()))
    assert result.status.value == 'parked'
    assert task.status == 'waiting_user'
    assert not task.metadata.get('v3_no_progress_stop')


@pytest.mark.parametrize('receipt_kind,receipt_phase,receipt_task', [
    ('historical_message', 'question', 'fixture-task'),
    ('authenticated_question_answer', 'old-question', 'fixture-task'),
    ('authenticated_question_answer', 'question', 'other-task'),
])
def test_historical_or_unrelated_answers_cannot_release_question_handoff(receipt_kind, receipt_phase, receipt_task):
    from alphonse.agent_v2.core.intelligence.v3.processor import _resume_inspection_from_answer
    phase = PhasePlan('question', 'Ask for missing details', (PhaseSubgoal('question', 'Ask for start_date', 'question'),))
    state = new_tactical_state(phase)
    state.status = PhaseStatus.WAITING_USER
    task = TaskState(task_id='fixture-task', metadata={
        'act_directive': {'action': 'ask_user', 'inspection_resolution': {'route': 'requires_user_input'}},
        'v3_resumption_evidence': [{'kind': receipt_kind, 'phase_id': receipt_phase, 'task_id': receipt_task}],
        'v3_update_inspection_actions': MAX_UPDATE_INSPECTION_ACTIONS,
    })
    before = deepcopy(task.metadata)
    _resume_inspection_from_answer(task, state)
    assert task.metadata == before


def test_fresh_read_can_verify_an_already_satisfied_idempotent_update_without_a_write(tmp_path):
    from alphonse.agent_v2.system_one import SystemOneActRecommendation, SystemOneReviewResult
    database = tmp_path / 'records.sqlite'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE records(id INTEGER PRIMARY KEY, value TEXT, updated_at TEXT)')
        db.execute("INSERT INTO records VALUES(1,'reviewed','original timestamp')")
    task = TaskState(user='fixture-owner', goal='Ensure record 1 has value reviewed')
    task.set_acceptance_contract_from_markdown('1.- [ ] Record 1 persists value reviewed and is verified by fresh read-back')
    phase = inspection_phase(True)
    state = new_tactical_state(PhasePlan('verify-existing', 'Verify requested persisted state', (phase.subgoals[0],),
        criterion_ids=('ac-1',), authorized_capabilities=('local_shell',), mutation_scope=MutationScope(('records.sqlite',))))

    class Jev:
        def evaluate_tactical_progress(self, **kwargs):
            assert kwargs['action']['result']['stdout'].strip() == '1|reviewed'
            return SystemOneTacticalReview(True, .99, True)

        def evaluate(self, **kwargs):
            entry = kwargs['evidence']['entries'][-1]
            assert entry['result']['stdout'].strip() == '1|reviewed'
            return SystemOneReviewResult(({'criterion_id': 'ac-1', 'status': 'satisfied',
                'evidence_refs': [entry['evidence_ref']]},), ())

        def recommend_act(self, **kwargs):
            return SystemOneActRecommendation('complete', .99, True, 'Exact requested state freshly verified', answers={
                'mission_outcome_is_fulfilled': .99, 'mission_requires_persisted_effect': .99,
            })

    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry(), system_one=Jev())
    read = command("import sqlite3; c=sqlite3.connect('file:records.sqlite?mode=ro',uri=True); row=c.execute('SELECT id,value FROM records WHERE id=1').fetchone(); print(str(row[0])+'|'+row[1])")
    outcome = PhaseExecutor(action_selector=lambda *_: {'tool_id': 'native.bash', 'arguments': {'command': read, 'cwd': str(tmp_path)}}, reveal_policy=ToolRevealPolicy()).run(task, state, context)
    task.metadata['prepared_user_response'] = {'message': 'Record 1 already has the requested value, verified by fresh read-back.',
        'work_evidence_refs': [state.evidence.entries[-1]['evidence_ref']]}
    _, decision = V3OuterController().review_and_route(task, state, outcome, context)
    assert decision.action == StrategicAction.COMPLETE
    with sqlite3.connect('file:'+str(database)+'?mode=ro', uri=True) as db:
        assert db.execute('SELECT updated_at FROM records WHERE id=1').fetchone()[0] == 'original timestamp'
    assert len(state.actions) == 1
    assert not task.metadata.get('v3_inspection_resolution')
