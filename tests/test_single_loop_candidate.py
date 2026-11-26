"""Synthetic single-loop prototype checks; no provider or household access."""
from types import SimpleNamespace
from dataclasses import replace
import json
import subprocess
import sys

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateHost, CandidateState, run_candidate
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import BudgetLimits, SQLiteTaskBudgetStore, TaskBudget
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition


def test_simple_loop_core_import_does_not_load_v3():
    subprocess.run([sys.executable, '-c', '''
import sys
import alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate
assert not any(name == 'alphonse.agent_v2.core.intelligence.v3'
               or name.startswith('alphonse.agent_v2.core.intelligence.v3.')
               for name in sys.modules)
'''], check=True)


def vector(actions=0, model_calls=0, input_tokens=0, output_tokens=0):
    return dict(actions=actions, model_calls=model_calls, input_tokens=input_tokens, output_tokens=output_tokens)


@pytest.fixture
def fixture(tmp_path):
    registry = InMemoryToolRegistry()
    calls, snapshots, seen, sources = [], [], [], {'goal': 'Read the fixture value'}
    descriptor = ToolDescriptor('native.read', 'read', ToolKind.NATIVE, read_only=True,
        argument_schema={'type': 'object', 'properties': {'key': {'type': 'string'}}, 'required': ['key'], 'additionalProperties': False})
    registry.register(ToolDefinition(descriptor, lambda args: calls.append(args) or {'candidate_complete': True, 'value': '42'}))
    registry.register(ToolDefinition(ToolDescriptor('native.write', 'write', ToolKind.NATIVE, read_only=False),
                                     lambda args: pytest.fail('mutation dispatched')))
    store = SQLiteTaskBudgetStore(str(tmp_path / 'budget.db'))
    store.create('task', TaskBudget(BudgetLimits(vector(3, 6, 3000, 1000), vector(0, 1, 100, 50), 100000, 1000), started_at_ms=0))
    task = TaskState(task_id='task', user='fixture-user')
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry)
    script = iter([{'kind': 'tool', 'tool_id': 'native.read', 'arguments': {'key': 'x'}, 'usage': {'input_tokens': 100, 'output_tokens': 20}},
                   {'kind': 'answer', 'text': 'The value is 42.', 'usage': {'input_tokens': 100, 'output_tokens': 20}}])
    def choose(text, max_output_tokens):
        seen.append(text)
        return next(script)
    def record(actor, result):
        key = 'result-' + str(len(sources))
        sources[key] = json.dumps(result['result'])
        return key
    host = CandidateHost(task, context, store, snapshots.append, choose,
                         lambda actor, ref: actor == 'fixture-user' and ref in sources,
                         lambda actor, ref: sources[ref], lambda actor, tool, args: True,
                         record, lambda actor, response, refs: response == 'The value is 42.' and bool(refs),
                         lambda: 100)
    return SimpleNamespace(host=host, state=CandidateState('task', 'fixture-user', 'goal'), calls=calls,
                           snapshots=snapshots, sources=sources, seen=seen, registry=registry)


def test_read_then_grounded_answer_uses_projection_budget_and_real_invocation(fixture):
    result = run_candidate(fixture.state, fixture.host)
    assert result.status == 'completed'
    assert result.response == 'The value is 42.'
    assert fixture.calls == [{'key': 'x'}]
    assert '42' in fixture.seen[-1]
    assert fixture.host.budgets.load('task').usage()['charged'] == vector(1, 2, 200, 40)
    assert fixture.snapshots[-1]['status'] == 'completed'
    assert any(x['inflight'].startswith('tool-') for x in fixture.snapshots)


def test_host_pause_occurs_between_settled_actions_without_replaying(fixture):
    seen = []
    def pause(state):
        seen.append(state.snapshot())
        return bool(state.evidence_sources)
    paused = run_candidate(fixture.state, replace(fixture.host, checkpoint_pause=pause))
    assert paused.status == 'running' and not paused.inflight
    assert fixture.calls == [{'key': 'x'}]
    assert len(paused.evidence_sources) == 1
    assert fixture.host.budgets.load('task').usage()['in_flight'] == []
    assert len(seen) == 2
    result = run_candidate(paused, fixture.host)
    assert result.status == 'completed'
    assert fixture.calls == [{'key': 'x'}]


def test_pause_callback_failure_never_dispatches_or_overwrites_state(fixture):
    def fail(_state):
        raise RuntimeError('pause unavailable')
    with pytest.raises(RuntimeError, match='pause unavailable'):
        run_candidate(fixture.state, replace(fixture.host, checkpoint_pause=fail))
    assert fixture.calls == [] and fixture.seen == [] and fixture.snapshots == []


def test_current_candidate_blocks_native_bash_without_dispatch(fixture):
    from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
    bash = build_bash_tool_definition()
    fixture.registry.register(ToolDefinition(bash.descriptor,
        lambda *_args, **_kwargs: pytest.fail('unapproved Bash must not execute'),
        argument_schema=bash.argument_schema, accepts_context=True))
    host = replace(fixture.host, choose=lambda *_: {
        'kind': 'tool', 'tool_id': 'native.bash', 'arguments': {'command': 'printf dojo'},
        'usage': {'input_tokens': 100, 'output_tokens': 20}})
    result = run_candidate(fixture.state, host)
    assert (result.status, result.blocker) == ('blocked', 'candidate_requires_read_only_native_tool')


def test_disposable_bash_uses_explicit_host_callback_and_observes_failure(fixture):
    from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
    bash = build_bash_tool_definition()
    fixture.registry.register(ToolDefinition(bash.descriptor,
        lambda *_args, **_kwargs: pytest.fail('host Bash must not execute'), argument_schema=bash.argument_schema))
    decisions = iter([
        {'kind': 'tool', 'tool_id': 'native.bash', 'arguments': {'command': 'false'},
         'usage': {'input_tokens': 100, 'output_tokens': 20}},
        {'kind': 'answer', 'text': 'The command failed.', 'usage': {'input_tokens': 100, 'output_tokens': 20}},
    ])
    calls = []
    def execute(actor, tool, args):
        calls.append((actor, tool, args))
        return {'status': 'success', 'result': {'candidate_complete': True, 'exit_code': 1, 'stdout': '', 'stderr': 'failed'}}
    host = replace(fixture.host, choose=lambda *_: next(decisions),
        authorize_disposable_tool=lambda actor, tool, args: actor == 'fixture-user' and tool == 'native.bash',
        execute_disposable_tool=execute,
        verify_response=lambda actor, text, refs: text == 'The command failed.' and len(refs) == 1)
    result = run_candidate(fixture.state, host)
    assert result.status == 'completed' and result.response == 'The command failed.'
    assert calls == [('fixture-user', 'native.bash', {'command': 'false'})]
    assert '"exit_code": 1' in fixture.sources[result.evidence_sources[0]]
    assert host.budgets.load('task').usage()['charged']['actions'] == 1


def test_disposable_bash_exception_preserves_unknown_inflight_and_never_replays(fixture):
    from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
    fixture.registry.register(build_bash_tool_definition())
    calls = []
    def execute(*_):
        calls.append('executed')
        raise RuntimeError('fixture execution unavailable')
    host = replace(fixture.host, choose=lambda *_: {
        'kind': 'tool', 'tool_id': 'native.bash', 'arguments': {'command': 'printf dojo'},
        'usage': {'input_tokens': 100, 'output_tokens': 20}},
        authorize_disposable_tool=lambda *_: True, execute_disposable_tool=execute)
    result = run_candidate(fixture.state, host)
    assert result.blocker == 'candidate_disposable_effect_unconfirmed'
    assert result.inflight and host.budgets.load('task').usage()['in_flight'] == [result.inflight]
    restored = CandidateState.restore({**result.snapshot(), 'status': 'running', 'blocker': ''})
    assert run_candidate(restored, host).blocker == 'candidate_unresolved_inflight'
    assert calls == ['executed']


def test_disposable_bash_denied_before_dispatch(fixture):
    from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
    fixture.registry.register(build_bash_tool_definition())
    host = replace(fixture.host, choose=lambda *_: {
        'kind': 'tool', 'tool_id': 'native.bash', 'arguments': {'command': 'printf dojo'},
        'usage': {'input_tokens': 100, 'output_tokens': 20}},
        authorize_disposable_tool=lambda *_: False,
        execute_disposable_tool=lambda *_: pytest.fail('denied action executed'))
    assert run_candidate(fixture.state, host).blocker == 'candidate_read_denied'


def test_partial_result_cannot_complete(fixture):
    fixture.registry.register(ToolDefinition(ToolDescriptor('native.read', 'read', ToolKind.NATIVE, read_only=True),
                                             lambda args: {'candidate_complete': False, 'value': '42'}))
    result = run_candidate(fixture.state, fixture.host)
    assert result.status == 'blocked'
    assert result.blocker == 'candidate_observation_incomplete'
    assert result.evidence_sources == []


@pytest.mark.parametrize('owner,content,expected', [
    ('fixture-user', 'The synthetic meeting starts at ten.', 'completed'),
    ('another-user', 'A private synthetic note.', 'candidate_observation_incomplete'),
    ('fixture-user', 'Long text ' * 500, 'candidate_observation_incomplete'),
])
def test_real_conversation_read_requires_visible_complete_owner_observation(tmp_path, fixture, owner, content, expected):
    from alphonse.agent_v2.conversations import SQLiteConversationStore
    from alphonse.agent_v2.core.tools.registry.native.memory import (
        build_read_memory_source_tool_definition, complete_conversation_read,
    )
    conversations = SQLiteConversationStore(tmp_path / 'conversations.sqlite')
    event = conversations.record(owner_user_id=owner, role='user', content=content, source='test',
                                 source_message_id='synthetic-note')
    source_id = 'conversation:' + event.event_id
    fixture.host.context.conversation_store = conversations
    fixture.registry.register(build_read_memory_source_tool_definition())
    decisions = iter([
        {'kind': 'tool', 'tool_id': 'native.read_memory_source', 'arguments': {'source_id': source_id},
         'usage': {'input_tokens': 100, 'output_tokens': 20}},
        {'kind': 'answer', 'text': 'The synthetic meeting starts at ten.',
         'usage': {'input_tokens': 100, 'output_tokens': 20}},
    ])
    host = replace(fixture.host, choose=lambda *_: next(decisions),
                   authorize_read=lambda actor, tool, args: actor == 'fixture-user' and
                       tool == 'native.read_memory_source' and args == {'source_id': source_id},
                   verify_response=lambda actor, response, refs: bool(refs) and response == content,
                   validate_observation=complete_conversation_read, input_reservation=2048)
    result = run_candidate(fixture.state, host)
    if expected == 'completed':
        assert result.status == 'completed' and result.response == content, result.blocker
        assert source_id in fixture.sources[result.evidence_sources[0]]
    else:
        assert (result.status, result.blocker) == ('blocked', expected)
        assert result.evidence_sources == []


def test_observation_validator_error_fails_closed(fixture):
    def invalid(*_):
        raise RuntimeError('private diagnostic')
    result = run_candidate(fixture.state, replace(fixture.host, validate_observation=invalid))
    assert (result.status, result.blocker) == ('blocked', 'candidate_observation_incomplete')
    assert not result.evidence_sources


def test_action_bound_recording_failure_does_not_publish_private_error(fixture):
    def record(*_args):
        raise RuntimeError('synthetic private detail')
    result = run_candidate(fixture.state, replace(fixture.host, record_bound_observation=record))
    assert (result.status, result.blocker) == ('blocked', 'candidate_evidence_recording_unavailable')
    assert result.evidence_sources == []
    assert 'synthetic private detail' not in repr(result)


def test_candidate_discovers_then_reads_owner_conversation(tmp_path, fixture):
    import re
    from alphonse.agent_v2.conversations import SQLiteConversationStore
    from alphonse.agent_v2.core.tools.registry.native.memory import (
        build_read_memory_source_tool_definition, build_search_memory_tool_definition,
        revalidate_conversation_observation,
    )
    conversations = SQLiteConversationStore(tmp_path / 'conversations.sqlite')
    source = conversations.record(owner_user_id='fixture-user', role='user',
        content='Synthetic pharmacy Cedar', source='test', source_message_id='one')
    conversations.record(owner_user_id='someone-else', role='user',
        content='Synthetic pharmacy private', source='test', source_message_id='two')
    fixture.host.context.conversation_store = conversations
    fixture.registry.register(build_search_memory_tool_definition())
    fixture.registry.register(build_read_memory_source_tool_definition())
    seen = []
    def choose(projection, _):
        seen.append(projection)
        if len(seen) == 1:
            return {'kind': 'tool', 'tool_id': 'native.search_memory',
                    'arguments': {'query': 'pharmacy', 'source_types': ['conversation']},
                    'usage': {'input_tokens': 100, 'output_tokens': 20}}
        if len(seen) == 2:
            found = re.search(r'conversation:[0-9a-f-]{36}', projection)
            assert found and 'Synthetic pharmacy private' not in projection
            return {'kind': 'tool', 'tool_id': 'native.read_memory_source',
                    'arguments': {'source_id': found.group(), 'neighbors': 0},
                    'usage': {'input_tokens': 100, 'output_tokens': 20}}
        return {'kind': 'answer', 'text': 'Synthetic pharmacy Cedar',
                'usage': {'input_tokens': 100, 'output_tokens': 20}}
    def validate(actor, tool_id, args, payload):
        return revalidate_conversation_observation(actor, tool_id, args, payload,
            context=fixture.host.context.tool_execution_context(fixture.host.task))
    budgets = SQLiteTaskBudgetStore(str(tmp_path / 'discovery-budget.db'))
    budgets.create('task', TaskBudget(BudgetLimits(vector(3, 6, 24000, 1000),
        vector(0, 1, 100, 50), 100000, 1000), started_at_ms=0))
    host = replace(fixture.host, choose=choose, validate_observation=validate, budgets=budgets,
        input_reservation=6000, projection_budget=8000,
        authorize_read=lambda actor, tool, args: actor == 'fixture-user' and tool in {
            'native.search_memory', 'native.read_memory_source'},
        verify_response=lambda actor, text, refs: len(refs) == 2 and text == 'Synthetic pharmacy Cedar')
    result = run_candidate(fixture.state, host)
    assert result.status == 'completed', result.blocker
    assert 'conversation:' + source.event_id in seen[1]
    assert 'Synthetic pharmacy Cedar' in seen[2]


def test_candidate_rejects_conversation_deleted_after_read(tmp_path, fixture):
    from alphonse.agent_v2.conversations import SQLiteConversationStore
    from alphonse.agent_v2.core.tools.registry.native.memory import (
        build_read_memory_source_tool_definition, revalidate_conversation_observation,
    )
    conversations = SQLiteConversationStore(tmp_path / 'conversations.sqlite')
    event = conversations.record(owner_user_id='fixture-user', role='user', content='Synthetic private note',
                                 source='test', source_message_id='synthetic-note')
    source_id = 'conversation:' + event.event_id
    fixture.host.context.conversation_store = conversations
    fixture.registry.register(build_read_memory_source_tool_definition())
    def validate(actor, tool_id, args, payload):
        assert conversations.delete(event_id=event.event_id, owner_user_id=actor)
        return revalidate_conversation_observation(actor, tool_id, args, payload,
            context=fixture.host.context.tool_execution_context(fixture.host.task))
    host = replace(fixture.host, validate_observation=validate,
        choose=lambda *_: {'kind': 'tool', 'tool_id': 'native.read_memory_source',
                           'arguments': {'source_id': source_id, 'neighbors': 0},
                           'usage': {'input_tokens': 100, 'output_tokens': 20}})
    result = run_candidate(fixture.state, host)
    assert (result.status, result.blocker) == ('blocked', 'candidate_observation_incomplete')
    assert result.evidence_sources == []


@pytest.mark.parametrize('change,expected_status,expected_reason', [
    ('cancel', 'cancelled', 'cancelled'),
    ('revoke', 'blocked', 'candidate_read_revoked'),
    ('deadline', 'blocked', 'candidate_deadline_exceeded'),
])
def test_real_memory_validation_rechecks_before_recording(tmp_path, fixture, change, expected_status, expected_reason):
    from alphonse.agent_v2.conversations import SQLiteConversationStore
    from alphonse.agent_v2.core.tools.registry.native.memory import (
        build_read_memory_source_tool_definition, revalidate_conversation_observation,
    )
    conversations = SQLiteConversationStore(tmp_path / 'conversations.sqlite')
    event = conversations.record(owner_user_id='fixture-user', role='user', content='Synthetic note',
                                 source='test', source_message_id='synthetic-note')
    source_id = 'conversation:' + event.event_id
    fixture.host.context.conversation_store = conversations
    fixture.registry.register(build_read_memory_source_tool_definition())
    conditions = {'cancel': False, 'revoke': False, 'deadline': False}
    recorded = []
    context = fixture.host.context
    context.cancellation_checker = lambda: conditions['cancel']
    def validate(actor, tool_id, args, payload):
        assert revalidate_conversation_observation(actor, tool_id, args, payload,
            context=context.tool_execution_context(fixture.host.task))
        conditions[change] = True
        return True
    host = replace(fixture.host, validate_observation=validate,
        authorize_read=lambda *_: not conditions['revoke'],
        deadline_checker=lambda: conditions['deadline'],
        record_bound_observation=lambda *args: recorded.append(args) or 'unexpected',
        choose=lambda *_: {'kind': 'tool', 'tool_id': 'native.read_memory_source',
                           'arguments': {'source_id': source_id, 'neighbors': 0},
                           'usage': {'input_tokens': 100, 'output_tokens': 20}})
    result = run_candidate(fixture.state, host)
    assert (result.status, result.blocker) == (expected_status, expected_reason)
    assert not recorded and not result.evidence_sources


def test_candidate_revalidates_action_bound_search_before_answer(tmp_path, fixture):
    from alphonse.agent_v2.conversations import SQLiteConversationStore
    from alphonse.agent_v2.core.tools.registry.native.memory import (
        build_search_memory_tool_definition, revalidate_conversation_observation,
    )
    conversations = SQLiteConversationStore(tmp_path / 'conversations.sqlite')
    event = conversations.record(owner_user_id='fixture-user', role='user',
        content='Synthetic current pharmacy Cedar', source='test', source_message_id='one')
    fixture.host.context.conversation_store = conversations
    fixture.registry.register(build_search_memory_tool_definition())
    observations = {}
    def record(actor, tool_id, arguments, result):
        ref = 'bound-' + result['call_id']
        observations[ref] = (actor, tool_id, arguments, result['result'])
        return ref
    def allowed(actor, ref):
        if ref == 'goal':
            return actor == 'fixture-user'
        owner, tool_id, args, payload = observations[ref]
        return owner == actor and revalidate_conversation_observation(actor, tool_id, args, payload,
            context=fixture.host.context.tool_execution_context(fixture.host.task))
    def load(actor, ref):
        return fixture.sources['goal'] if ref == 'goal' else json.dumps(observations[ref][3])
    calls = []
    def choose(_projection, _cap):
        calls.append(True)
        if len(calls) == 1:
            return {'kind': 'tool', 'tool_id': 'native.search_memory',
                    'arguments': {'query': 'pharmacy', 'source_types': ['conversation']},
                    'usage': {'input_tokens': 100, 'output_tokens': 20}}
        assert conversations.delete(event_id=event.event_id, owner_user_id='fixture-user')
        return {'kind': 'answer', 'text': 'Synthetic current pharmacy Cedar',
                'usage': {'input_tokens': 100, 'output_tokens': 20}}
    budgets = SQLiteTaskBudgetStore(str(tmp_path / 'bound-budget.db'))
    budgets.create('task', TaskBudget(BudgetLimits(vector(3, 6, 24000, 1000),
        vector(0, 1, 100, 50), 100000, 1000), started_at_ms=0))
    host = replace(fixture.host, choose=choose, authorize_source=allowed, load_source=load,
        budgets=budgets, input_reservation=6000, projection_budget=8000,
        validate_observation=lambda actor, tool, args, payload: revalidate_conversation_observation(
            actor, tool, args, payload, context=fixture.host.context.tool_execution_context(fixture.host.task)),
        record_bound_observation=record)
    result = run_candidate(fixture.state, host)
    assert (result.status, result.blocker) == ('blocked', 'candidate_evidence_changed_or_revoked')
    assert len(observations) == 1
    assert next(iter(observations.values()))[2] == {'query': 'pharmacy', 'source_types': ['conversation']}


def test_revalidation_rejects_changed_search_and_wrong_actor(tmp_path, fixture):
    from alphonse.agent_v2.conversations import SQLiteConversationStore
    from alphonse.agent_v2.core.tools.registry.native.memory import (
        execute_search_memory, revalidate_conversation_observation,
    )
    conversations = SQLiteConversationStore(tmp_path / 'conversations.sqlite')
    conversations.record(owner_user_id='fixture-user', role='user', content='Synthetic pharmacy A',
                         source='test', source_message_id='one')
    fixture.host.context.conversation_store = conversations
    context = fixture.host.context.tool_execution_context(fixture.host.task)
    arguments = {'query': 'pharmacy', 'source_types': ['conversation']}
    payload = execute_search_memory(arguments, context=context)
    assert revalidate_conversation_observation('fixture-user', 'native.search_memory', arguments, payload,
                                               context=context)
    assert not revalidate_conversation_observation('other', 'native.search_memory', arguments, payload,
                                                   context=context)
    checks = iter([False, True])
    assert not revalidate_conversation_observation('fixture-user', 'native.search_memory', arguments, payload,
                                                   context=replace(context, cancellation_checker=lambda: next(checks)))
    conversations.record(owner_user_id='fixture-user', role='user', content='Synthetic pharmacy B',
                         source='test', source_message_id='two')
    assert not revalidate_conversation_observation('fixture-user', 'native.search_memory', arguments, payload,
                                                   context=context)


def test_conversation_search_rejects_partial_or_other_owner():
    from alphonse.agent_v2.core.tools.registry.native.memory import complete_conversation_search
    args = {'query': 'pharmacy', 'source_types': ['conversation']}
    base = {'schema_version': 2, 'query': 'pharmacy', 'status': 'empty', 'coverage': {'conversation': 'complete'},
            'backend': 'sqlite_fts5', 'next_offset': None, 'truncated': False, 'total_matches': 0, 'results': []}
    assert complete_conversation_search('owner', 'native.search_memory', args, base)
    assert not complete_conversation_search('owner', 'native.search_memory', args, {**base, 'status': 'partial'})
    assert not complete_conversation_search('owner', 'native.search_memory', args, {**base, 'next_offset': 1})
    other = {'source_id': 'conversation:other', 'source_type': 'conversation', 'truncated': False,
             'source': {'source_id': 'conversation:other', 'kind': 'conversation',
                        'access': {'owner_user_id': 'other'}}}
    assert not complete_conversation_search('owner', 'native.search_memory', args,
        {**base, 'status': 'found', 'total_matches': 1, 'results': [other]})


def test_no_fabricated_completion_without_grounding(fixture):
    host = replace(fixture.host, choose=lambda _, cap: {'kind': 'answer', 'text': '42',
                   'usage': {'input_tokens': 10, 'output_tokens': 10}}, verify_response=lambda *args: False)
    result = run_candidate(fixture.state, host)
    assert (result.status, result.blocker) == ('blocked', 'candidate_answer_unsupported')


def test_invalid_action_format_gets_one_bounded_repair_without_execution(fixture):
    decisions = iter([
        {'status': 'answer in the wrong envelope', 'usage': {'input_tokens': 10, 'output_tokens': 10}},
        {'kind': 'answer', 'text': 'supported', 'usage': {'input_tokens': 10, 'output_tokens': 10}},
    ])
    prompts = []
    def choose(text, _):
        prompts.append(text)
        return next(decisions)
    host = replace(fixture.host, choose=choose, verify_response=lambda *_: True, input_reservation=1024)
    result = run_candidate(fixture.state, host)
    assert (result.status, result.response, result.invalid_decision_retries) == ('completed', 'supported', 1)
    assert fixture.calls == [] and len(prompts) == 2
    assert 'invalid format and was not executed' in prompts[1]
    assert 'answer in the wrong envelope' not in prompts[1]
    assert CandidateState.restore(fixture.snapshots[-1]).invalid_decision_retries == 1


def test_repeated_invalid_action_is_bounded_and_cannot_bypass_authorization(fixture):
    decisions = iter([
        {'kind': [], 'usage': {'input_tokens': 10, 'output_tokens': 10}},
        {'unexpected': 'still invalid', 'usage': {'input_tokens': 10, 'output_tokens': 10}},
    ])
    host = replace(fixture.host, choose=lambda *_: next(decisions), input_reservation=1024)
    result = run_candidate(fixture.state, host)
    assert (result.status, result.blocker, result.invalid_decision_retries) == ('blocked', 'candidate_decision_invalid', 1)
    assert fixture.calls == []


def test_repair_does_not_authorize_an_effectful_tool(fixture):
    decisions = iter([
        {'kind': 'invalid', 'usage': {'input_tokens': 10, 'output_tokens': 10}},
        {'kind': 'tool', 'tool_id': 'native.write', 'arguments': {}, 'usage': {'input_tokens': 10, 'output_tokens': 10}},
    ])
    result = run_candidate(fixture.state, replace(fixture.host, choose=lambda *_: next(decisions), input_reservation=1024))
    assert (result.status, result.blocker) == ('blocked', 'candidate_requires_read_only_native_tool')
    assert fixture.calls == []


def test_format_retry_is_not_reset_by_restart(fixture):
    def checkpoint(snapshot):
        fixture.snapshots.append(snapshot)
        if snapshot['decision_format_pending']:
            raise RuntimeError('simulated restart')
    invalid = lambda *_: {'kind': 'invalid', 'usage': {'input_tokens': 10, 'output_tokens': 10}}
    host = replace(fixture.host, choose=invalid, checkpoint=checkpoint, input_reservation=1024)
    with pytest.raises(RuntimeError, match='simulated restart'):
        run_candidate(fixture.state, host)
    restored = CandidateState.restore(fixture.snapshots[-1])
    result = run_candidate(restored, replace(host, checkpoint=fixture.snapshots.append))
    assert (result.status, result.blocker, result.invalid_decision_retries) == ('blocked', 'candidate_decision_invalid', 1)
    assert fixture.calls == []


def test_cancellation_prevents_format_repair_call(fixture):
    cancelled = {'value': False}
    calls = []
    def choose(*_):
        calls.append(True)
        return {'kind': 'invalid', 'usage': {'input_tokens': 10, 'output_tokens': 10}}
    def checkpoint(snapshot):
        if snapshot['decision_format_pending']:
            cancelled['value'] = True
    fixture.host.context.cancellation_checker = lambda: cancelled['value']
    result = run_candidate(fixture.state, replace(fixture.host, choose=choose, checkpoint=checkpoint,
                                                  input_reservation=1024))
    assert result.status == 'cancelled' and len(calls) == 1
    assert fixture.calls == []


def test_older_candidate_snapshots_restore_without_format_retries():
    snapshot = CandidateState('task', 'actor', 'goal').snapshot()
    for version in (1, 2, 3, 6, 7, 8):
        old = {**snapshot, 'schema_version': version}
        old.pop('reused_web_fetch_ref')
        if version < 8:
            old.pop('progress_reviewed_count')
            old.pop('progress_assessment')
        if version < 7:
            old.pop('parse_repair_pending')
        if version < 4:
            for key in ('answer_review_state', 'prerequisite_feedback', 'prerequisite_retries', 'retry_feedback_pending'):
                old.pop(key)
        if version < 3:
            old.pop('invalid_decision_retries')
            old.pop('decision_format_pending')
        if version == 1:
            old.pop('unsupported_retries')
            old.pop('recovery_pending')
        restored = CandidateState.restore(old)
        assert restored.invalid_decision_retries == 0 and not restored.decision_format_pending
        assert not restored.parse_repair_pending
        assert restored.answer_review_state == 'not_started'
        assert restored.progress_reviewed_count == 0 and restored.progress_assessment == 'unknown'
        assert restored.reused_web_fetch_ref == ''


def test_reused_web_fetch_retains_middle_of_recorded_page_when_compacting(fixture):
    from alphonse.agent_v2.core.intelligence.simple_loop.evidence_projection import observation_excerpt
    from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import _project

    fixture.sources['web-page'] = 'x' * 4600 + 'important middle' + 'x' * 4600
    fixture.sources['other-page'] = 'y' * 9200
    state = CandidateState.restore(CandidateState('task', 'fixture-user', 'goal',
        evidence_sources=['web-page', 'other-page'], reused_web_fetch_ref='web-page').snapshot())
    host = replace(fixture.host, projection_budget=16000,
                   compact_source=lambda _ref, text, limit: observation_excerpt(text, limit))
    projection = _project(host, state, 4096)
    assert 'important middle' in projection.text
    assert set(projection.manifest['compacted_source_digests']) == {'other-page'}


def test_deferred_retry_survives_restore_and_stops_repeated_selection(fixture):
    fixture.sources['prior'] = '{"status":"failed"}'
    state = CandidateState('task', 'fixture-user', 'goal', evidence_sources=['prior'],
                           retry_feedback_pending=True)
    restored = CandidateState.restore(json.loads(json.dumps(state.snapshot(), sort_keys=True)))
    assert restored.retry_feedback_pending is True
    prompts = []
    def choose(prompt, _cap):
        prompts.append(prompt)
        return {'kind': 'tool', 'tool_id': 'native.read', 'arguments': {'key': 'x'},
                'usage': {'input_tokens': 10, 'output_tokens': 10}}
    host = replace(fixture.host, choose=choose, input_reservation=1024,
                   defer_identical_failed_read=lambda _actor, _tool, _args, refs: refs == ('prior',))
    result = run_candidate(restored, host)
    assert (result.status, result.blocker) == ('blocked', 'candidate_retry_deferred')
    assert 'immediate repeat was not dispatched' in prompts[0]
    assert fixture.calls == []


def test_deferred_retry_allows_different_action_after_restore(fixture):
    fixture.sources['prior'] = '{"status":"failed"}'
    state = CandidateState.restore(CandidateState('task', 'fixture-user', 'goal',
        evidence_sources=['prior'], retry_feedback_pending=True).snapshot())
    decisions = iter([
        {'kind': 'tool', 'tool_id': 'native.read', 'arguments': {'key': 'different'},
         'usage': {'input_tokens': 10, 'output_tokens': 10}},
        {'kind': 'answer', 'text': 'The value is 42.', 'usage': {'input_tokens': 10, 'output_tokens': 10}},
    ])
    host = replace(fixture.host, choose=lambda *_: next(decisions), input_reservation=1024,
                   defer_identical_failed_read=lambda _actor, _tool, args, _refs: args == {'key': 'x'})
    result = run_candidate(state, host)
    assert result.status == 'completed'
    assert fixture.calls == [{'key': 'different'}]
    assert not result.retry_feedback_pending


def test_answer_review_allows_relevant_read_without_forcing_distractors(fixture):
    decisions = iter([
        {'kind': 'answer', 'text': 'provisional private draft', 'usage': {'input_tokens': 10, 'output_tokens': 10}},
        {'kind': 'tool', 'tool_id': 'native.read', 'arguments': {'key': 'relevant'},
         'usage': {'input_tokens': 10, 'output_tokens': 10}},
        {'kind': 'answer', 'text': 'supported', 'usage': {'input_tokens': 10, 'output_tokens': 10}},
    ])
    prompts = []
    def choose(text, _):
        prompts.append(text)
        return next(decisions)
    host = replace(fixture.host, choose=choose, verify_response=lambda *_: True,
                   input_reservation=1024, synthetic_preanswer_review=True)
    result = run_candidate(fixture.state, host)
    assert (result.status, result.response, result.answer_review_state) == ('completed', 'supported', 'used')
    assert fixture.calls == [{'key': 'relevant'}]
    assert len(prompts) == 3 and 'unexamined authorized source' in prompts[1]
    assert 'provisional private draft' not in str(fixture.snapshots) + str(prompts)


def test_answer_review_never_grants_read_authorization(fixture):
    decisions = iter([
        {'kind': 'answer', 'text': 'provisional', 'usage': {'input_tokens': 10, 'output_tokens': 10}},
        {'kind': 'tool', 'tool_id': 'native.read', 'arguments': {'key': 'denied'},
         'usage': {'input_tokens': 10, 'output_tokens': 10}},
    ])
    host = replace(fixture.host, choose=lambda *_: next(decisions), authorize_read=lambda *_: False,
                   input_reservation=1024, synthetic_preanswer_review=True)
    result = run_candidate(fixture.state, host)
    assert (result.status, result.blocker) == ('blocked', 'candidate_read_denied')
    assert fixture.calls == []


def test_answer_review_restart_does_not_repeat_completed_review(fixture):
    decisions = iter([
        {'kind': 'answer', 'text': 'provisional', 'usage': {'input_tokens': 10, 'output_tokens': 10}},
        {'kind': 'tool', 'tool_id': 'native.read', 'arguments': {'key': 'relevant'},
         'usage': {'input_tokens': 10, 'output_tokens': 10}},
        {'kind': 'answer', 'text': 'supported', 'usage': {'input_tokens': 10, 'output_tokens': 10}},
    ])
    def checkpoint(snapshot):
        fixture.snapshots.append(snapshot)
        if snapshot['answer_review_state'] == 'used' and snapshot['evidence_sources']:
            raise RuntimeError('synthetic restart')
    host = replace(fixture.host, choose=lambda *_: next(decisions), checkpoint=checkpoint,
                   verify_response=lambda *_: True, input_reservation=1024, synthetic_preanswer_review=True)
    with pytest.raises(RuntimeError, match='synthetic restart'):
        run_candidate(fixture.state, host)
    restored = CandidateState.restore(fixture.snapshots[-1])
    assert restored.answer_review_state == 'used' and not restored.inflight
    result = run_candidate(restored, replace(host, checkpoint=fixture.snapshots.append))
    assert (result.status, result.response) == ('completed', 'supported')
    assert fixture.calls == [{'key': 'relevant'}]


def test_cancellation_prevents_answer_review_call(fixture):
    cancelled = {'value': False}
    calls = []
    source_loads_after_cancellation = []
    def choose(*_):
        calls.append(True)
        return {'kind': 'answer', 'text': 'provisional', 'usage': {'input_tokens': 10, 'output_tokens': 10}}
    def checkpoint(snapshot):
        if snapshot['answer_review_state'] == 'pending':
            cancelled['value'] = True
    def load_source(actor, ref):
        if cancelled['value']:
            source_loads_after_cancellation.append(ref)
        return fixture.host.load_source(actor, ref)
    fixture.host.context.cancellation_checker = lambda: cancelled['value']
    host = replace(fixture.host, choose=choose, checkpoint=checkpoint, input_reservation=1024,
                   synthetic_preanswer_review=True, load_source=load_source)
    result = run_candidate(fixture.state, host)
    assert result.status == 'cancelled' and len(calls) == 1
    assert source_loads_after_cancellation == []
    budget = host.budgets.load('task')
    assert not budget.usage()['in_flight']
    assert len(budget.to_dict()['reservations']) == 1


@pytest.mark.parametrize('checker', [lambda: None, lambda: (_ for _ in ()).throw(RuntimeError('clock failed'))])
def test_ambiguous_or_failed_deadline_checker_stops_before_model_dispatch(fixture, checker):
    def no_dispatch(*_args):
        pytest.fail('deadline control did not fail closed')
    result = run_candidate(fixture.state, replace(fixture.host, choose=no_dispatch, deadline_checker=checker))
    assert (result.status, result.blocker) == ('blocked', 'candidate_deadline_exceeded')
    assert fixture.calls == []


def test_effectful_action_is_never_dispatched(fixture):
    host = replace(fixture.host, choose=lambda _, cap: {'kind': 'tool', 'tool_id': 'native.write', 'arguments': {},
                   'usage': {'input_tokens': 10, 'output_tokens': 10}})
    result = run_candidate(fixture.state, host)
    assert result.blocker == 'candidate_requires_read_only_native_tool'


def test_revocation_between_inference_and_tool_prevents_dispatch(fixture):
    host = replace(fixture.host, authorize_read=lambda *args: False)
    result = run_candidate(fixture.state, host)
    assert result.blocker == 'candidate_read_denied'
    assert fixture.calls == []


def test_cancellation_after_external_tool_io_withholds_completion(fixture):
    cancelled = {'value': False}
    def read(args):
        cancelled['value'] = True
        return {'candidate_complete': True, 'value': '42'}
    fixture.registry.register(ToolDefinition(ToolDescriptor('native.read', 'read', ToolKind.NATIVE, read_only=True), read))
    fixture.host.context.cancellation_checker = lambda: cancelled['value']
    result = run_candidate(fixture.state, fixture.host)
    assert result.status == 'cancelled'
    assert result.evidence_sources == []


def test_restart_from_inflight_checkpoint_does_not_repeat(fixture):
    restored = CandidateState.restore({**fixture.state.snapshot(), 'inflight': 'tool-unconfirmed'})
    result = run_candidate(restored, fixture.host)
    assert result.blocker == 'candidate_unresolved_inflight'
    assert fixture.calls == []


def test_checkpoint_failure_before_action_stops_dispatch(fixture):
    def checkpoint(row):
        if row['inflight'].startswith('tool-'):
            raise OSError('simulated persistence failure')
    with pytest.raises(OSError):
        run_candidate(fixture.state, replace(fixture.host, checkpoint=checkpoint))
    assert fixture.calls == []
    assert fixture.host.budgets.load('task').usage()['in_flight']


def test_budget_exhaustion_returns_honest_blocker(fixture):
    def no_action(_, cap):
        return {'kind': 'tool', 'tool_id': 'native.read', 'arguments': {'key': 'x'},
                'usage': {'input_tokens': 10, 'output_tokens': 10}}
    result = run_candidate(fixture.state, replace(fixture.host, choose=no_action, input_reservation=1024))
    assert result.status == 'blocked'
    assert 'budget_exhausted:actions' in result.blocker
    assert len(fixture.calls) == 3
    assert result.response == ''


def test_question_waits_without_claiming_delivery(fixture):
    host = replace(fixture.host, choose=lambda _, cap: {'kind': 'question', 'text': 'Which source?',
                   'usage': {'input_tokens': 10, 'output_tokens': 10}})
    result = run_candidate(fixture.state, host)
    assert result.status == 'waiting_user'
    assert result.blocker == 'question_delivery_not_integrated'


def test_source_revocation_during_model_call_blocks_action(fixture):
    def choose(_, cap):
        fixture.sources.pop('goal')
        return {'kind': 'tool', 'tool_id': 'native.read', 'arguments': {'key': 'x'},
                'usage': {'input_tokens': 10, 'output_tokens': 10}}
    result = run_candidate(fixture.state, replace(fixture.host, choose=choose))
    assert result.blocker == 'candidate_evidence_changed_or_revoked'
    assert fixture.calls == []


def test_cross_task_budget_confusion_fails_closed(fixture):
    fixture.host.task.task_id = 'other'
    result = run_candidate(fixture.state, fixture.host)
    assert result.blocker == 'candidate_actor_mismatch'
    assert fixture.calls == []


def test_incomplete_usage_prevents_next_model_call_and_never_claims_success(fixture):
    host = replace(fixture.host, choose=lambda _, cap: {'kind': 'tool', 'tool_id': 'native.read',
                   'arguments': {'key': 'x'}, 'usage': {'input_tokens': 100}})
    result = run_candidate(fixture.state, host)
    assert result.status == 'blocked'
    assert result.blocker == 'candidate_model_usage_unknown_or_overrun'
    assert len(fixture.calls) == 0
    assert host.budgets.load('task').usage()['unknown_usage']


def test_changed_authorized_goal_during_model_call_blocks_decision(fixture):
    def choose(_, cap):
        fixture.sources['goal'] = 'A changed user request'
        return {'kind': 'tool', 'tool_id': 'native.read', 'arguments': {'key': 'x'},
                'usage': {'input_tokens': 10, 'output_tokens': 10}}
    result = run_candidate(fixture.state, replace(fixture.host, choose=choose))
    assert result.blocker == 'candidate_evidence_changed_or_revoked'
    assert fixture.calls == []


def test_revocation_during_answer_verification_prevents_completion(fixture):
    def verify(actor, text, refs):
        fixture.sources.pop('goal')
        return True
    host = replace(fixture.host, verify_response=verify)
    result = run_candidate(fixture.state, host)
    assert result.status == 'blocked'
    assert result.blocker == 'candidate_evidence_changed_or_revoked'
    assert result.response == ''


def test_changed_evidence_during_answer_verification_prevents_completion(fixture):
    def verify(actor, text, refs):
        fixture.sources[refs[0]] = 'changed source'
        return True
    result = run_candidate(fixture.state, replace(fixture.host, verify_response=verify))
    assert result.blocker == 'candidate_evidence_changed_or_revoked'


def test_model_output_over_forecast_can_complete_within_task_budget(fixture):
    host = replace(fixture.host, choose=lambda _, cap: {'kind': 'answer', 'text': 'unsupported',
        'usage': {'input_tokens': 10, 'output_tokens': cap + 1}}, verify_response=lambda *args: True)
    result = run_candidate(fixture.state, host)
    assert result.status == 'completed'
    assert host.budgets.load('task').usage()['charged']['output_tokens'] == host.output_reservation + 1


def test_model_input_over_forecast_is_accounted_and_tool_can_run_within_budget(fixture):
    decisions = iter([
        {'kind': 'tool', 'tool_id': 'native.read', 'arguments': {'key': 'x'},
         'usage': {'input_tokens': fixture.host.input_reservation + 1, 'output_tokens': 10}},
        {'kind': 'answer', 'text': 'The value is 42.', 'usage': {'input_tokens': 10, 'output_tokens': 10}},
    ])
    host = replace(fixture.host, choose=lambda *_: next(decisions))
    result = run_candidate(fixture.state, host)
    usage = host.budgets.load('task').usage()
    assert result.status == 'completed'
    assert usage['charged']['input_tokens'] == host.input_reservation + 11
    assert usage['unknown_usage'] == [] and fixture.calls == [{'key': 'x'}]


def test_model_actual_usage_over_work_budget_never_dispatches_tool(fixture):
    host = replace(fixture.host, choose=lambda *_: {'kind': 'tool', 'tool_id': 'native.read',
        'arguments': {'key': 'x'}, 'usage': {'input_tokens': 2901, 'output_tokens': 10}})
    result = run_candidate(fixture.state, host)
    assert result.status == 'blocked' and result.blocker == 'candidate_model_usage_overrun'
    assert host.budgets.load('task').usage()['charged']['input_tokens'] == 2901
    assert fixture.calls == []


def test_projection_larger_than_reservation_never_calls_model(fixture):
    fixture.sources['goal'] = 'x' * 1000
    called = []
    host = replace(fixture.host, choose=lambda *_: called.append(True))
    result = run_candidate(fixture.state, host)
    assert result.blocker == 'candidate_input_reservation_too_small'
    assert not called
    assert host.budgets.load('task').usage()['charged'] == vector()


def test_compaction_does_not_hide_evidence_changes_from_the_freshness_check(fixture):
    from alphonse.agent_v2.core.intelligence.simple_loop.evidence_projection import observation_excerpt

    fixture.state.evidence_sources = ['result']
    fixture.sources['result'] = 'a' * 1000 + 'old' + 'b' * 1000

    def prepare(prompt, _):
        assert 'observation excerpted' in prompt and 'old' not in prompt

        def choose():
            fixture.sources['result'] = fixture.sources['result'].replace('old', 'new')
            return {'kind': 'answer', 'text': 'The value is 42.',
                    'usage': {'input_tokens': 100, 'output_tokens': 20}}

        return choose

    host = replace(fixture.host, input_reservation=1024, prepare_choice=prepare,
                   compact_source=lambda ref, text, limit: observation_excerpt(text, limit))
    result = run_candidate(fixture.state, host)
    assert result.blocker == 'candidate_evidence_changed_or_revoked'
    assert fixture.calls == []


def test_crash_after_settlement_before_evidence_checkpoint_does_not_repeat_read(fixture):
    snapshots = []
    def checkpoint(row):
        if row['evidence_sources']:
            raise OSError('simulated crash after evidence storage')
        snapshots.append(row)
    with pytest.raises(OSError):
        run_candidate(fixture.state, replace(fixture.host, checkpoint=checkpoint))
    assert len(fixture.calls) == 1
    persisted = CandidateState.restore(snapshots[-1])
    assert persisted.inflight.startswith('tool-')
    recovered = run_candidate(persisted, replace(fixture.host, checkpoint=lambda _: None))
    assert recovered.blocker == 'candidate_unresolved_inflight'
    assert len(fixture.calls) == 1


def test_changed_source_at_final_dispatch_fence_blocks_tool(fixture):
    calls = []
    def authorize(actor, tool, args):
        calls.append(True)
        if len(calls) == 2:
            fixture.sources['goal'] = 'changed at dispatch'
        return True
    result = run_candidate(fixture.state, replace(fixture.host, authorize_read=authorize))
    assert result.blocker == 'candidate_evidence_changed_or_revoked'
    assert fixture.calls == []


def test_missing_evidence_recovery_reads_before_answer(fixture):
    decisions = iter([
        {'kind': 'answer', 'text': '42', 'usage': {'input_tokens': 10, 'output_tokens': 10}},
        {'kind': 'tool', 'tool_id': 'native.read', 'arguments': {'key': 'x'}, 'usage': {'input_tokens': 10, 'output_tokens': 10}},
        {'kind': 'answer', 'text': 'The value is 42.', 'usage': {'input_tokens': 10, 'output_tokens': 10}},
    ])
    def choose(text, cap):
        fixture.seen.append(text)
        return next(decisions)
    host = replace(fixture.host, choose=choose, input_reservation=1024,
                   remaining_advertised_evidence=lambda actor, refs: not refs)
    result = run_candidate(fixture.state, host)
    assert result.status == 'completed'
    assert result.unsupported_retries == 1
    assert fixture.calls == [{'key': 'x'}]
    assert 'A previous answer was not verified' in fixture.seen[1]
    assert 'A previous answer was not verified' in fixture.seen[2]
    assert CandidateState.restore(fixture.snapshots[-1]).unsupported_retries == 1


def test_recovery_feedback_persists_across_multiple_reads_without_extra_retries(fixture):
    decisions = iter([
        {'kind': 'answer', 'text': 'guess'},
        {'kind': 'tool', 'tool_id': 'native.read', 'arguments': {'key': 'x'}},
        {'kind': 'tool', 'tool_id': 'native.read', 'arguments': {'key': 'y'}},
        {'kind': 'answer', 'text': 'The value is 42.'},
    ])
    def choose(text, cap):
        fixture.seen.append(text)
        return {**next(decisions), 'usage': {'input_tokens': 10, 'output_tokens': 10}}
    host = replace(fixture.host, choose=choose, input_reservation=1024,
                   verify_response=lambda actor, response, refs: len(refs) == 2 and response == 'The value is 42.',
                   remaining_advertised_evidence=lambda actor, refs: True)
    result = run_candidate(fixture.state, host)
    assert result.status == 'completed' and result.unsupported_retries == 1
    assert [call['key'] for call in fixture.calls] == ['x', 'y']
    assert 'A previous answer was not verified' not in fixture.seen[0]
    assert all('A previous answer was not verified' in text for text in fixture.seen[1:])


def test_owner_unverified_answer_gets_one_budgeted_reconsideration(fixture):
    decisions = iter([
        {'kind': 'answer', 'text': 'unsupported'},
        {'kind': 'tool', 'tool_id': 'native.read', 'arguments': {'key': 'x'}},
        {'kind': 'answer', 'text': 'The value is 42.'},
    ])
    def choose(text, cap):
        fixture.seen.append(text)
        return {**next(decisions), 'usage': {'input_tokens': 10, 'output_tokens': 10}}
    host = replace(fixture.host, choose=choose, input_reservation=1024, retry_unverified_answer=True)
    result = run_candidate(fixture.state, host)
    assert result.status == 'completed'
    assert result.unsupported_retries == 1
    assert fixture.calls == [{'key': 'x'}]
    assert 'reason may be unknown' in fixture.seen[1]


def test_owner_reconsideration_does_not_release_rejected_answer(fixture):
    host = replace(fixture.host, retry_unverified_answer=True, input_reservation=1024,
        choose=lambda *_: {'kind': 'answer', 'text': 'unverified', 'usage': {'input_tokens': 10, 'output_tokens': 10}},
        verify_response=lambda *_: False)
    result = run_candidate(fixture.state, host)
    assert (result.status, result.blocker, result.response) == ('blocked', 'candidate_answer_unsupported', '')
    assert result.unsupported_retries == 1
    assert fixture.host.budgets.load('task').usage()['charged']['model_calls'] == 2


def test_unsupported_recovery_is_bounded_and_no_answer_delivered(fixture):
    host = replace(fixture.host,
        choose=lambda *_: {'kind': 'answer', 'text': 'guessed', 'usage': {'input_tokens': 10, 'output_tokens': 10}},
        remaining_advertised_evidence=lambda actor, refs: True, input_reservation=1024)
    result = run_candidate(fixture.state, host)
    assert (result.status, result.blocker, result.response) == ('blocked', 'candidate_answer_unsupported', '')
    assert result.unsupported_retries == 2
    assert fixture.host.budgets.load('task').usage()['charged']['model_calls'] == 3
    assert fixture.calls == []


def test_recovery_does_not_override_verification_failure_or_denied_reads(fixture):
    def fail_verification(*_):
        raise RuntimeError('unavailable')
    host = replace(fixture.host,
        choose=lambda *_: {'kind': 'answer', 'text': 'guessed', 'usage': {'input_tokens': 10, 'output_tokens': 10}},
        verify_response=fail_verification, remaining_advertised_evidence=lambda *_: True)
    result = run_candidate(fixture.state, host)
    assert (result.status, result.blocker) == ('blocked', 'candidate_verification_unavailable')
    assert result.unsupported_retries == 0


def test_recovery_does_not_bypass_read_authorization(fixture):
    script = iter([
        {'kind': 'answer', 'text': 'guessed', 'usage': {'input_tokens': 10, 'output_tokens': 10}},
        {'kind': 'tool', 'tool_id': 'native.read', 'arguments': {'key': 'x'}, 'usage': {'input_tokens': 10, 'output_tokens': 10}},
    ])
    host = replace(fixture.host, choose=lambda *_: next(script), input_reservation=1024,
                   remaining_advertised_evidence=lambda *_: True, authorize_read=lambda *_: False)
    result = run_candidate(fixture.state, host)
    assert (result.status, result.blocker) == ('blocked', 'candidate_read_denied')
    assert fixture.calls == []


def test_recovery_limit_persists_after_restart(fixture):
    snapshots = []
    def checkpoint(snapshot):
        snapshots.append(snapshot)
        if snapshot['recovery_pending'] and snapshot['unsupported_retries'] == 1:
            raise OSError('simulated process interruption')
    host = replace(fixture.host, checkpoint=checkpoint, input_reservation=1024,
        choose=lambda *_: {'kind': 'answer', 'text': 'guess', 'usage': {'input_tokens': 10, 'output_tokens': 10}},
        remaining_advertised_evidence=lambda *_: True)
    with pytest.raises(OSError):
        run_candidate(fixture.state, host)
    restored = CandidateState.restore(snapshots[-1])
    result = run_candidate(restored, replace(host, checkpoint=fixture.snapshots.append))
    assert (result.status, result.blocker, result.unsupported_retries) == ('blocked', 'candidate_answer_unsupported', 2)
    assert fixture.host.budgets.load('task').usage()['charged']['model_calls'] == 3


def test_recovery_eligibility_exception_fails_closed(fixture):
    def unavailable(*_):
        raise RuntimeError('unavailable')
    host = replace(fixture.host, remaining_advertised_evidence=unavailable,
        choose=lambda *_: {'kind': 'answer', 'text': 'guess', 'usage': {'input_tokens': 10, 'output_tokens': 10}})
    result = run_candidate(fixture.state, host)
    assert (result.status, result.blocker, result.unsupported_retries) == ('blocked', 'candidate_answer_unsupported', 0)
    assert fixture.host.budgets.load('task').usage()['charged']['model_calls'] == 1


def test_cancellation_during_recovery_prevents_further_model_calls(fixture):
    cancelled = {'value': False}
    calls = []
    fixture.host.context.cancellation_checker = lambda: cancelled['value']
    def missing(*_):
        cancelled['value'] = True
        return True
    host = replace(fixture.host, input_reservation=1024, remaining_advertised_evidence=missing,
        choose=lambda *_: calls.append(True) or {'kind': 'answer', 'text': 'guess', 'usage': {'input_tokens': 10, 'output_tokens': 10}})
    result = run_candidate(fixture.state, host)
    assert result.status == 'cancelled' and result.response == ''
    assert len(calls) == 1


def test_recovery_policy_exception_fails_closed(fixture):
    def policy(*_):
        raise RuntimeError('unavailable')
    host = replace(fixture.host,
        choose=lambda *_: {'kind': 'answer', 'text': 'guess', 'usage': {'input_tokens': 10, 'output_tokens': 10}},
        remaining_advertised_evidence=policy)
    result = run_candidate(fixture.state, host)
    assert (result.status, result.blocker, result.unsupported_retries) == ('blocked', 'candidate_answer_unsupported', 0)


def test_chooser_failure_categories_do_not_include_private_exception_text():
    from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import _chooser_failure_category
    assert _chooser_failure_category(ValueError('candidate_router_input_exceeds_reservation')) == 'input_exceeds_reservation'
    assert _chooser_failure_category(ValueError('candidate_router_input_exceeds_reservation: private')) == 'chooser_error'
    assert _chooser_failure_category(RuntimeError('private')) == 'chooser_error'
