"""Synthetic single-loop prototype checks; no provider or household access."""
from types import SimpleNamespace
from dataclasses import replace
import json

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateHost, CandidateState, run_candidate
from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetLimits, SQLiteTaskBudgetStore, TaskBudget
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition


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
    for version in (1, 2, 3):
        old = {key: value for key, value in snapshot.items() if key != 'answer_review_state'}
        if version < 3:
            old.pop('invalid_decision_retries')
            old.pop('decision_format_pending')
        old['schema_version'] = version
        if version == 1:
            old.pop('unsupported_retries')
            old.pop('recovery_pending')
        restored = CandidateState.restore(old)
        assert restored.invalid_decision_retries == 0 and not restored.decision_format_pending
        assert restored.answer_review_state == 'not_started'


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


def test_model_output_overrun_cannot_complete(fixture):
    host = replace(fixture.host, choose=lambda _, cap: {'kind': 'answer', 'text': 'unsupported',
        'usage': {'input_tokens': 10, 'output_tokens': cap + 1}}, verify_response=lambda *args: True)
    result = run_candidate(fixture.state, host)
    assert result.status == 'blocked'
    assert result.blocker == 'candidate_model_usage_unknown_or_overrun'
    assert host.budgets.load('task').usage()['charged']['output_tokens'] == host.output_reservation + 1


def test_projection_larger_than_reservation_never_calls_model(fixture):
    fixture.sources['goal'] = 'x' * 1000
    called = []
    host = replace(fixture.host, choose=lambda *_: called.append(True))
    result = run_candidate(fixture.state, host)
    assert result.blocker == 'candidate_input_reservation_too_small'
    assert not called
    assert host.budgets.load('task').usage()['charged'] == vector()


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
