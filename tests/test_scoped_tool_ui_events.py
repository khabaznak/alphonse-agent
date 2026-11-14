"""Synthetic regression for user isolation of tool UI event payloads."""
from threading import RLock
from types import SimpleNamespace

from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import PhaseExecutor, PhasePlan, PhaseSubgoal, ToolRevealPolicy, new_tactical_state
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.invocation import ToolInvocationService
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
from alphonse.agent_v2.daemon import V2Daemon


def setup():
    registry = InMemoryToolRegistry()
    descriptor = ToolDescriptor(tool_id='native.fixture_secret', name='fixture_secret',
        kind=ToolKind.NATIVE, description='Synthetic private result',
        metadata={'v3_capabilities': ['fixture_read']}, read_only=True)
    registry.register(ToolDefinition(descriptor=descriptor, argument_schema={},
                                     callable=lambda arguments: {'secret': arguments['secret']}))
    events = []
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, ui_event_sink=events.append)
    return context, events


def project(events):
    daemon = V2Daemon.__new__(V2Daemon)
    daemon._ui_event_lock = RLock()
    daemon._ui_event_sequence = 0
    daemon._ui_event_journal = []
    daemon.runtime = SimpleNamespace(ui_events=events)
    daemon._ag_ui = SimpleNamespace(map_event=lambda event: [{'type': event.event_type, 'payload': event.payload}])
    return daemon


def test_tool_payloads_only_reach_originating_actor():
    context, events = setup()
    alice = TaskState(task_id='a', user='alice')
    bob = TaskState(task_id='b', user='bob')
    ToolInvocationService(context=context, task=alice).invoke('native.fixture_secret', {'secret': 'alice-private'})
    ToolInvocationService(context=context, task=bob).invoke('native.fixture_secret', {'secret': 'bob-private'})
    daemon = project(events)
    alice_events, _ = daemon.ui_events_since(user='alice')
    bob_events, _ = daemon.ui_events_since(user='bob')
    assert len(alice_events) == len(bob_events) == 2
    assert 'alice-private' in str(alice_events) and 'bob-private' not in str(alice_events)
    assert 'bob-private' in str(bob_events) and 'alice-private' not in str(bob_events)
    assert {event['event']['type'] for event in alice_events} == {'tool_call_started', 'tool_call_result'}
    assert all(event['event']['payload']['task_id'] == 'a' for event in alice_events)


def test_missing_actor_suppresses_tool_payloads():
    context, events = setup()
    result = ToolInvocationService(context=context, task=TaskState(task_id='anonymous')).invoke(
        'native.fixture_secret', {'secret': 'must-not-broadcast'})
    assert result['status'] == 'success'
    assert events == []


def test_sensitive_tool_events_are_both_scoped_and_redacted():
    registry = InMemoryToolRegistry()
    descriptor = ToolDescriptor(tool_id='native.sensitive_fixture', name='fixture',
        kind=ToolKind.NATIVE, description='Synthetic sensitive result', read_only=True,
        metadata={'sensitive_content': True})
    registry.register(ToolDefinition(descriptor=descriptor, argument_schema={},
        callable=lambda args: {'secret': args['secret']}))
    events = []
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, ui_event_sink=events.append)
    outcome = ToolInvocationService(context=context, task=TaskState(task_id='a', user='alice')).invoke(
        'native.sensitive_fixture', {'secret': 'private-fixture-value'})
    assert outcome['status'] == 'success'
    daemon = project(events)
    alice, _ = daemon.ui_events_since(user='alice')
    bob, _ = daemon.ui_events_since(user='bob')
    assert len(alice) == 2 and bob == []
    assert 'private-fixture-value' not in str(alice)
    assert all(row['event']['payload']['task_id'] == 'a' for row in alice)


def test_v3_reveal_event_is_scoped_and_missing_actor_is_not_broadcast():
    context, events = setup()
    subgoal = PhaseSubgoal('read', 'Read synthetic', 'observation', allowed_capabilities=('fixture_read',))
    phase = PhasePlan('phase', 'Read synthetic', (subgoal,), authorized_capabilities=('fixture_read',))
    executor = PhaseExecutor(reveal_policy=ToolRevealPolicy())
    executor._revealed_tools(TaskState(task_id='a', user='alice'), new_tactical_state(phase), subgoal, context)
    executor._revealed_tools(TaskState(task_id='anonymous'), new_tactical_state(phase), subgoal, context)
    daemon = project(events)
    alice_events, _ = daemon.ui_events_since(user='alice')
    bob_events, _ = daemon.ui_events_since(user='bob')
    assert len(alice_events) == 1 and alice_events[0]['event']['type'] == 'tactical_tools_revealed'
    assert bob_events == []


def test_v2_direct_tool_events_are_scoped_and_execution_still_succeeds():
    from alphonse.agent_v2.evaluation.legacy_pdca.nodes import do_node
    context, events = setup()
    for actor in ('alice', 'bob', ''):
        task = TaskState(task_id=actor or 'anonymous', user=actor)
        task.append_plan_call({'id': actor or 'none', 'tool_id': 'native.fixture_secret',
                               'tool_name': 'fixture_secret', 'arguments': {'secret': actor + '-private'}})
        do_node(task, context=context)
        assert task.get_next_planned_call() is None
    daemon = project(events)
    alice, _ = daemon.ui_events_since(user='alice')
    bob, _ = daemon.ui_events_since(user='bob')
    assert len(alice) == len(bob) == 2
    assert 'alice-private' in str(alice) and 'bob-private' not in str(alice)
    assert 'bob-private' in str(bob) and 'alice-private' not in str(bob)
    assert '-private' not in str([event for event in alice if event['user'] != 'alice'])


def test_v2_program_events_are_scoped_and_execution_still_succeeds():
    from alphonse.agent_v2.evaluation.legacy_pdca.nodes import do_node
    context, events = setup()
    context.program_runner = SimpleNamespace(run=lambda **kwargs: {
        'status': 'success', 'program_result': 'private-program-result', 'tool_calls': []})
    task = TaskState(task_id='program', user='alice')
    task.append_plan_call({'id': 'p', 'execution_mode': 'program',
        'program': {'source': 'result = 1'}, 'tool_name': 'program'})
    do_node(task, context=context)
    assert task.get_next_planned_call() is None
    daemon = project(events)
    alice, _ = daemon.ui_events_since(user='alice')
    bob, _ = daemon.ui_events_since(user='bob')
    assert len(alice) == 2 and 'private-program-result' in str(alice)
    assert bob == []
    no_actor = TaskState(task_id='anonymous')
    no_actor.append_plan_call({'id': 'q', 'execution_mode': 'program', 'program': {'source': 'result = 1'}})
    do_node(no_actor, context=context)
    assert events == []  # The previous batch was consumed by project().


def test_real_agui_routes_delegated_run_to_owner_and_question_to_respondent():
    from alphonse.agent_v2.core.core import CoreUiEvent
    from alphonse.agent_v2.core.questions import SQLiteQuestionStore
    from alphonse.agent_v2.interfaces.ag_ui import AgUiAdapter
    context, events = setup()
    store = SQLiteQuestionStore()
    task = TaskState(task_id='delegated', user='alice', goal='owner-private-goal')
    question = store.create_question(task=task, question='Please choose', kind='yes_no', respondent_user_id='bob')
    context.emit_ui_event('run_finished', {'task': task.to_dict(), 'question': question.to_dict(), 'status': 'waiting_user'})
    context.emit_ui_event('question_interrupt_opened', {'question': question.to_dict()})
    daemon = project(events)
    daemon._ag_ui = AgUiAdapter(question_store=store)
    alice, _ = daemon.ui_events_since(user='alice')
    bob, _ = daemon.ui_events_since(user='bob')
    assert 'owner-private-goal' in str(alice)
    assert 'owner-private-goal' not in str(bob)
    assert any(row['event']['type'] == 'STATE_SNAPSHOT' for row in alice)
    assert any(row['event'].get('name') == 'question_interrupt_opened' for row in bob)
    assert not any(row['event']['type'] == 'STATE_SNAPSHOT' for row in bob)


def test_real_agui_scopes_tool_results_and_drops_unowned_sensitive_events():
    from alphonse.agent_v2.core.core import CoreUiEvent
    from alphonse.agent_v2.core.questions import SQLiteQuestionStore
    from alphonse.agent_v2.interfaces.ag_ui import AgUiAdapter
    context, events = setup()
    ToolInvocationService(context=context, task=TaskState(task_id='a', user='alice')).invoke(
        'native.fixture_secret', {'secret': 'real-adapter-private'})
    # Some events enter the daemon directly, bypassing CoreLoopContext.
    events.extend([CoreUiEvent('question_interrupt_resolved', {'answer': 'secret-answer'}),
                   CoreUiEvent('question_answer_rejected', {'answer': 'secret-answer'}),
                   CoreUiEvent('run_started', {'task': {'user': '', 'goal': 'secret-goal'}}),
                   CoreUiEvent('scheduled_task_created', {'task_state': {'user': '', 'goal': 'secret-schedule'}}),
                   CoreUiEvent('scheduled_task_created', {'task_state': 'malformed-private', 'user': 'bob'}),
                   CoreUiEvent('scheduled_task_created', {'task_state': {}, 'question': {'respondent_user_id': 'bob'}, 'detail': 'empty-private'}),
                   CoreUiEvent('scheduled_task_created', {'task': {'user': 'alice'}, 'task_state': {'user': 'bob'}, 'detail': 'conflicting-private'}),
                   CoreUiEvent('state_snapshot', {'status': 'working', 'goal': 'secret-goal'}),
                   CoreUiEvent('state_snapshot', {'status': 'working'}),
                   CoreUiEvent('public_status', {'state': 'idle'})])
    daemon = project(events)
    daemon._ag_ui = AgUiAdapter(question_store=SQLiteQuestionStore())
    alice, _ = daemon.ui_events_since(user='alice')
    bob, _ = daemon.ui_events_since(user='bob')
    assert 'real-adapter-private' in str(alice) and 'real-adapter-private' not in str(bob)
    assert 'secret-answer' not in str(alice) and 'secret-answer' not in str(bob)
    assert 'secret-goal' not in str(alice) and 'secret-goal' not in str(bob)
    assert 'secret-schedule' not in str(alice) and 'secret-schedule' not in str(bob)
    assert all(value not in str(alice) and value not in str(bob) for value in
               ('malformed-private', 'empty-private', 'conflicting-private'))
    assert any(row['event'].get('snapshot', {}).get('status') == 'working' for row in bob)
    assert any(row['event'].get('name') == 'public_status' for row in bob)


def test_anonymous_v2_and_v3_processors_do_not_publish_task(monkeypatch):
    from alphonse.agent_v2.evaluation.legacy_pdca.processor import PDCAIntelligenceProcessor
    from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
    import alphonse.agent_v2.evaluation.legacy_pdca.processor as pdca_processor
    context, events = setup()
    anonymous = TaskState(task_id='anonymous', goal='private-anonymous-goal')
    # Drive the real V2 processor's parked-event and terminal-emission branches
    # with a disposable graph result; no real provider calls occur.
    def parked(task, context):
        task.status = 'waiting_user'
        return task
    monkeypatch.setattr(pdca_processor, 'run_pdca_once', parked)
    PDCAIntelligenceProcessor().process(anonymous, context)
    assert not events
    other = TaskState(task_id='anonymous-v3', goal='private-anonymous-v3')
    HierarchicalCAPDProcessor().process(other, context)
    assert not events
