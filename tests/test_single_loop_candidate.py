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


def test_no_fabricated_completion_without_grounding(fixture):
    host = replace(fixture.host, choose=lambda _, cap: {'kind': 'answer', 'text': '42',
                   'usage': {'input_tokens': 10, 'output_tokens': 10}}, verify_response=lambda *args: False)
    result = run_candidate(fixture.state, host)
    assert (result.status, result.blocker) == ('blocked', 'candidate_answer_unsupported')


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
