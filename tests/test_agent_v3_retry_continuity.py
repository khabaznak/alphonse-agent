"""Offline native memory workflows; no household data or live providers."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.contracts import PhasePlan, PhaseStatus, new_tactical_state
from alphonse.agent_v2.core.intelligence.v3.executor import PhaseExecutor
from alphonse.agent_v2.core.intelligence.v3.planner import plan_phase
from alphonse.agent_v2.core.intelligence.v3.revealing import ToolRevealPolicy
from alphonse.agent_v2.core.memory.retry_context import prepare_retry_continuity, retry_continuity_context
from alphonse.agent_v2.core.memory.workflow import MEMORY_WORKFLOW
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.ask_question import build_ask_question_tool_definition
from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
from alphonse.agent_v2.core.tools.registry.native.memory import (
    build_search_memory_tool_definition, build_read_memory_source_tool_definition,
    execute_search_memory, execute_read_memory_source,
)
from alphonse.agent_v2.core.tools.registry.native.scheduled_task import build_scheduled_task_tool_definition
from alphonse.agent_v2.system_one import SystemOneToolRegistrySelection, SystemOneTacticalReview


class RelevanceAndProgress:
    """Jev curates tools and judges observed progress, never historical episodes."""
    def judge_recall(self, **kwargs):
        return None  # Optional relevance may be unavailable; it grants no source authority.

    def select_retry_context(self, **kwargs):
        pytest.fail('No Jev source selection')

    def curate_request_tools(self, *, tools, **kwargs):
        return SystemOneToolRegistrySelection(selected_tool_ids=tuple(t.tool_id for t in tools))

    def select_plan_tools(self, *, tools, **kwargs):
        return SystemOneToolRegistrySelection(selected_tool_ids=tuple(t.tool_id for t in tools))

    def evaluate_tactical_progress(self, **kwargs):
        return SystemOneTacticalReview(True, .99, True)


def task():
    return TaskState(task_id='fixture', message_id='fixture', user='owner', goal='Retry the updated fixture file')


def phase(*subgoals, capabilities, paths=()):
    return dict(schema_version=3, phase_id='fixture-phase', objective='Complete the requested operation',
        acceptance_criteria=['Requested operation is verified'], criterion_ids=['ac-1'],
        authorized_capabilities=capabilities,
        mutation_scope={'allowed_paths': list(paths), 'allow_external_effects': False}, subgoals=list(subgoals))


def subgoal(name, capability, effect='read_only', depends=()):
    return dict(subgoal_id=name, objective=name, required_output_type=name, depends_on=list(depends),
        allowed_capabilities=[capability], allowed_side_effects=[effect], completion={'kind': 'output_present'})


def setup(tmp_path, definitions):
    tools = InMemoryToolRegistry()
    for definition in definitions:
        tools.register(definition)
    return CoreLoopContext(messages=InMemoryMessageQueue(), tools=tools, system_one=RelevanceAndProgress(),
        conversation_store=SQLiteConversationStore(tmp_path / 'history.sqlite'),
        question_store=SQLiteQuestionStore(tmp_path / 'questions.sqlite'))


def run(task, plan, context, actions):
    state = new_tactical_state(PhasePlan.from_dict(plan))
    actions = iter(actions)
    outcome = PhaseExecutor(action_selector=lambda *_: next(actions), reveal_policy=ToolRevealPolicy()).run(task, state, context)
    return state, outcome


@pytest.mark.parametrize('old_status', ['ambiguous', 'unavailable', 'over_budget', 'selected'])
@pytest.mark.parametrize('projected', [False, True])
def test_standalone_reminder_plans_schedules_and_reads_back_despite_optional_history(tmp_path, monkeypatch, old_status, projected):
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', str(int(projected)))
    from alphonse.agent_v2.core.io.identity import V2IdentityResolver
    from alphonse.agent_v2.users import V2UserStore
    users = V2UserStore(); users.set_users_root(tmp_path / 'users')
    users.create_user(display_name='Fixture Owner', user_id='owner')
    users.bind_address(user_id='owner', integration_id='telegram', provider_key='telegram', provider_user_id='fixture-address')
    context = setup(tmp_path, [build_scheduled_task_tool_definition()])
    context.identity_resolver = V2IdentityResolver(user_store=users)
    context.schedule_store = ScheduledTaskStore(tmp_path / 'schedules.sqlite')
    current = task(); current.goal = 'Remind me to water the plants on 2027-01-01 at 09:00 UTC'
    prepare_retry_continuity(current, context)
    current.metadata['v3_retry_continuity'].update(status=old_status, source_ids=['conversation:missing'])
    plan = phase(subgoal('schedule', 'scheduling', 'project_mutation'), capabilities=['scheduling'])
    def infer(request):
        assert MEMORY_WORKFLOW in request.prompt
        return SimpleNamespace(json_value=deepcopy(plan))
    context.inference = SimpleNamespace(generate_json=infer)
    planned = plan_phase(current, context)
    # Simulate a legacy ambiguous checkpoint even after planning: execution must
    # not reinterpret this optional history as a global mutation restriction.
    current.metadata['v3_retry_continuity']['status'] = old_status
    state, outcome = run(current, planned.to_dict(), context, [{'tool_id': 'native.scheduled_task', 'arguments': {
        'name': 'Water plants', 'description': 'Water the plants', 'prompt': 'Water the plants',
        'schedule_kind': 'once', 'run_at': '2027-01-01T09:00:00+00:00', 'delivery_mode': 'direct'}}])
    assert outcome.status == PhaseStatus.PHASE_COMPLETE, outcome
    records = ScheduledTaskStore(tmp_path / 'schedules.sqlite').list_tasks(owner_user_id='owner')
    assert len(records) == 1 and records[0].description == 'Water the plants'
    assert records[0].next_run_at == '2027-01-01T09:00:00+00:00'
    assert state.evidence.entries[0]['result']['verification']['source'] == 'schedule_store_readback'


def record(context, text, **kwargs):
    return context.conversation_store.record(owner_user_id=kwargs.pop('owner', 'owner'), role='user',
        content=text, source='fixture', source_message_id=text, **kwargs)


def test_updated_file_retry_searches_expands_and_freshly_inspects_current_steering(tmp_path):
    context = setup(tmp_path, [build_search_memory_tool_definition(), build_read_memory_source_tool_definition(), build_bash_tool_definition()])
    current = task()
    current.append_conversation_message('owner', 'Use the November fixture, which I have now updated.')
    old = record(context, 'October fixture: old.txt; observed OLD CONTENT', created_at='2026-10-01T12:00:00+00:00')
    intended = record(context, 'November fixture: current.txt; observed BEFORE UPDATE; uncertain title', created_at='2026-11-01T12:00:00+00:00')
    (tmp_path / 'current.txt').write_text('FRESH UPDATED CONTENT')
    source_id = 'conversation:' + intended.event_id
    plan = phase(subgoal('find', 'memory_recall'), subgoal('expand', 'memory_recall', depends=['find']),
        subgoal('fresh_read', 'local_shell', depends=['expand']), capabilities=['memory_recall', 'local_shell'])
    actions = [
        {'tool_id': 'native.search_memory', 'arguments': {'query': 'November fixture', 'source_types': ['conversation']}},
        {'tool_id': 'native.read_memory_source', 'arguments': {'source_id': source_id, 'neighbors': 0}},
        {'tool_id': 'native.bash', 'arguments': {'command': 'cat current.txt', 'cwd': str(tmp_path)}}]
    state, outcome = run(current, plan, context, actions)
    assert outcome.status == PhaseStatus.PHASE_COMPLETE, outcome
    evidence = state.evidence.entries
    assert evidence[0]['result']['results'][0]['source_id'] == source_id
    assert evidence[1]['result']['historical_only']
    assert 'uncertain title' in evidence[1]['result']['items'][0]['content']
    assert evidence[1]['result']['items'][0]['source']['observed_at'] == intended.observed_at
    assert evidence[2]['result']['stdout'].strip() == 'FRESH UPDATED CONTENT'
    assert 'BEFORE UPDATE' not in evidence[2]['result']['stdout']
    assert old.event_id != intended.event_id


def test_conflicting_episodes_ask_one_specific_persisted_question_after_independent_work(tmp_path):
    context = setup(tmp_path, [build_search_memory_tool_definition(), build_ask_question_tool_definition(), build_bash_tool_definition()])
    current = task()
    record(context, 'Fixture invoice INV-A, amount 42')
    record(context, 'Fixture invoice INV-B, amount 84')
    plan = phase(subgoal('independent', 'local_shell'), subgoal('recall', 'memory_recall'),
        subgoal('ask', 'user_interaction', 'user_response', depends=['recall']), capabilities=['local_shell', 'memory_recall', 'user_interaction'])
    question = 'Which fixture invoice should I update: INV-A (42) or INV-B (84)?'
    state, outcome = run(current, plan, context, [
        {'tool_id': 'native.bash', 'arguments': {'command': 'pwd', 'cwd': str(tmp_path)}},
        {'tool_id': 'native.search_memory', 'arguments': {'query': 'Fixture invoice', 'source_types': ['conversation']}},
        {'tool_id': 'native.ask_question', 'arguments': {'question': question, 'question_kind': 'open_text'}}])
    assert outcome.status == PhaseStatus.WAITING_USER, outcome
    assert state.completed_subgoal_ids == ['independent', 'recall']
    assert len(state.evidence.entries[1]['result']['results']) == 2
    restored = SQLiteQuestionStore(tmp_path / 'questions.sqlite').load_task_checkpoint(current.task_id)
    assert restored is not None and question in str(restored.conversation_records)
    from alphonse.agent_v2.core.intelligence.v3.contracts import TacticalState
    resumed = TacticalState.from_dict(restored.hierarchical_state)
    again = PhaseExecutor(action_selector=lambda *_: pytest.fail('Parked checkpoint must wait, not ask again')).run(restored, resumed, context)
    assert again.status == PhaseStatus.WAITING_USER
    assert len(resumed.actions) == 3


def test_private_deleted_and_unavailable_sources_stay_inaccessible_after_checkpoint(tmp_path):
    context = setup(tmp_path, [])
    current = task()
    # Native read context binds requester at every call, including after restart.
    tool_context = SimpleNamespace(task=current, conversation_store=context.conversation_store)
    own = record(context, 'Fixture current resource current.txt')
    private = record(context, 'PRIVATE fixture path', owner='someone-else')
    own_id = 'conversation:' + own.event_id
    assert execute_read_memory_source({'source_id': own_id}, context=tool_context)['status'] == 'found'
    current.metadata['memory_source_ids'] = [own_id]
    tool_context.task = TaskState.from_dict(current.to_checkpoint_dict())
    context.conversation_store.delete(event_id=own.event_id, owner_user_id='owner')
    for identity in [own_id, 'conversation:' + private.event_id, 'conversation:missing']:
        assert execute_read_memory_source({'source_id': identity}, context=tool_context)['status'] == 'not_found'
    tool_context.conversation_store = None
    assert execute_read_memory_source({'source_id': own_id}, context=tool_context)['status'] == 'unavailable'
    unavailable = execute_search_memory({'query': 'Fixture'}, context=tool_context)
    assert unavailable['status'] != 'empty' and unavailable['coverage']['conversation'] == 'unavailable'


def test_preparation_does_not_automatically_read_or_interpret_history(tmp_path):
    current = task(); context = setup(tmp_path, [])
    context.conversation_store = SimpleNamespace(list_recent=lambda **kw: pytest.fail('No automatic history scan'))
    prepare_retry_continuity(current, context)
    assert current.metadata['v3_retry_continuity']['status'] == 'not_needed'
    restored = TaskState.from_dict(current.to_checkpoint_dict())
    prepare_retry_continuity(restored, context)
    assert retry_continuity_context(restored, context) == ''

pytestmark = pytest.mark.usefixtures("task_understanding_provider")
