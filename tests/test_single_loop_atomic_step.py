"""Synthetic only: one candidate decision persisted through the runtime task row."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.single_loop_atomic_step import run_atomic_step
from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateHost, CandidateState
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import ENGINE, ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetLimits, TaskBudget
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore


def v(actions=0, model_calls=0, input_tokens=0, output_tokens=0):
    return dict(actions=actions, model_calls=model_calls, input_tokens=input_tokens, output_tokens=output_tokens)


@pytest.fixture
def f(tmp_path):
    questions = SQLiteQuestionStore(tmp_path / 'runtime.sqlite')
    task = TaskState(task_id='task', user='alice', goal='Find synthetic note', intelligence_engine=ENGINE,
                     intelligence_schema_version=1, metadata={'intelligence_engine': ENGINE})
    initial = questions.save_task_checkpoint(task, status='running')
    state = CandidateState('task', 'alice', 'goal')
    budget = TaskBudget(BudgetLimits(v(2, 3, 3000, 500), v(0, 1, 100, 50), 10000, 1000), started_at_ms=1000)
    store = ExperimentalCheckpointStore(questions)
    checkpoint = store.initialize('task', 'alice', expected_version=initial, state=state, budget=budget)
    data = {'goal': 'Find synthetic note'}
    choices = []
    def choose(prompt, cap):
        choices.append((prompt, cap))
        return {'kind': 'tool', 'tool_id': 'native.read', 'arguments': {},
                'usage': {'input_tokens': 100, 'output_tokens': 20}}
    def no_dispatch(*args):
        pytest.fail('tool, checkpoint callback or response verification must not be invoked')
    host = CandidateHost(task=task, context=CoreLoopContext(messages=InMemoryMessageQueue()), budgets=None,
        checkpoint=no_dispatch, choose=choose, authorize_source=lambda actor, ref: actor == 'alice' and ref in data,
        load_source=lambda actor, ref: data[ref], authorize_read=no_dispatch, record_evidence=no_dispatch,
        verify_response=no_dispatch, now_ms=lambda: 1100, input_reservation=1024, output_reservation=80)
    return SimpleNamespace(questions=questions, task=task, state=state, store=store, version=checkpoint.version,
                           host=host, data=data, choices=choices)


def test_model_reservation_and_settlement_use_runtime_row_without_tool_dispatch(f):
    result = run_atomic_step(f.state, f.host, f.store, expected_version=f.version)
    assert (result.status, result.reason) == ('blocked', 'tool_dispatch_and_evidence_not_integrated')
    assert result.checkpoint_version == f.version + 2
    checkpoint = f.store.load('task', 'alice', expected_version=result.checkpoint_version)
    assert checkpoint.state.inflight == ''
    assert checkpoint.budget.usage()['charged'] == v(model_calls=1, input_tokens=100, output_tokens=20)
    assert len(f.choices) == 1 and f.choices[0][1] == 80


def test_process_interruption_preserves_atomic_model_reservation(f):
    def crash(*args):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run_atomic_step(f.state, replace(f.host, choose=crash), f.store, expected_version=f.version)
    checkpoint = f.store.load('task', 'alice', expected_version=f.version + 1)
    assert checkpoint.state.inflight.startswith('model-')
    assert checkpoint.budget.usage()['in_flight'] == [checkpoint.state.inflight]
    result = run_atomic_step(checkpoint.state, f.host, f.store, expected_version=checkpoint.version)
    assert result.reason == 'inflight_requires_reconciliation'
    assert f.choices == []


def test_existing_inflight_read_cannot_be_dispatched_again(f):
    reserved = f.store.reserve('task', 'alice', expected_version=f.version, reservation_id='read',
                               kind='tool', amounts=v(actions=1), now_ms=1100)
    result = run_atomic_step(reserved.state, f.host, f.store, expected_version=reserved.version)
    assert result.reason == 'inflight_requires_reconciliation'
    assert f.choices == []


def test_cancel_during_model_retains_inflight(f):
    def cancel(prompt, cap):
        f.questions.mark_task_checkpoint_terminal('task', status='cancelled')
        f.host.context.cancellation_checker = lambda: True
        return {'kind': 'answer', 'text': '42', 'usage': {'input_tokens': 10, 'output_tokens': 1}}
    result = run_atomic_step(f.state, replace(f.host, choose=cancel), f.store, expected_version=f.version)
    assert result.status == 'cancelled'
    info = f.store.inspect_cancelled('task', 'alice', expected_version=f.version + 1)
    assert info['kind'] == 'model' and info['inflight'].startswith('model-')


def test_durable_cancel_before_dispatch_blocks_stale_cancellation_checker(f):
    loaded = []
    def load(actor, ref):
        loaded.append(ref)
        if len(loaded) == 2:
            assert f.questions.mark_task_checkpoint_terminal('task', status='cancelled')
        return f.data[ref]
    result = run_atomic_step(f.state, replace(f.host, load_source=load), f.store, expected_version=f.version)
    assert result.reason == 'checkpoint_changed_before_model'
    assert not f.choices
    info = f.store.inspect_cancelled('task', 'alice', expected_version=f.version + 1)
    assert info['kind'] == 'model' and info['inflight'].startswith('model-')


def test_stale_checkpoint_does_not_call_model(f):
    result = run_atomic_step(f.state, f.host, f.store, expected_version=f.version - 1)
    assert result.status == 'blocked' and not f.choices


@pytest.mark.parametrize('before_dispatch', [10000, 1099, True])
def test_expired_or_regressed_clock_after_reservation_does_not_dispatch(f, before_dispatch):
    # Work window ends at started_at + elapsed - final reserve = 10000.
    times = iter((1100, before_dispatch))
    result = run_atomic_step(f.state, replace(f.host, now_ms=lambda: next(times)), f.store,
                             expected_version=f.version)
    assert result.reason == 'deadline_or_clock_invalid_before_model'
    assert not f.choices
    # A fresh store instance represents restart: no automatic refund or retry.
    restored = ExperimentalCheckpointStore(SQLiteQuestionStore(f.questions.db_path))
    checkpoint = restored.load('task', 'alice', expected_version=result.checkpoint_version)
    assert checkpoint.state.inflight.startswith('model-')
    assert checkpoint.budget.usage()['in_flight'] == [checkpoint.state.inflight]
    retry = run_atomic_step(checkpoint.state, f.host, restored, expected_version=checkpoint.version)
    assert retry.reason == 'inflight_requires_reconciliation'
    assert not f.choices


def test_clock_failure_after_reservation_blocks_and_retains_inflight(f):
    calls = iter((1100, None))
    def clock():
        value = next(calls)
        if value is None:
            raise RuntimeError('clock unavailable')
        return value
    result = run_atomic_step(f.state, replace(f.host, now_ms=clock), f.store, expected_version=f.version)
    assert result.reason == 'deadline_or_clock_invalid_before_model'
    assert not f.choices
    checkpoint = f.store.load('task', 'alice', expected_version=result.checkpoint_version)
    assert checkpoint.state.inflight.startswith('model-')
    assert checkpoint.budget.usage()['in_flight'] == [checkpoint.state.inflight]


def test_changed_goal_after_model_cannot_turn_into_action(f):
    def choose(prompt, cap):
        f.data['goal'] = 'A different request'
        return {'kind': 'tool', 'tool_id': 'native.read', 'arguments': {},
                'usage': {'input_tokens': 10, 'output_tokens': 1}}
    result = run_atomic_step(f.state, replace(f.host, choose=choose), f.store, expected_version=f.version)
    assert result.reason == 'evidence_changed_or_revoked'


def test_unknown_usage_blocks_model_and_later_tool(f):
    result = run_atomic_step(f.state, replace(f.host, choose=lambda *_: {'kind': 'tool', 'usage': {}}),
                             f.store, expected_version=f.version)
    assert result.reason == 'model_usage_unknown_or_overrun'
    checkpoint = f.store.load('task', 'alice', expected_version=result.checkpoint_version)
    assert checkpoint.budget.usage()['unknown_usage']
    with pytest.raises(ValueError, match='model_usage_unresolved'):
        f.store.reserve('task', 'alice', expected_version=checkpoint.version, reservation_id='tool',
                        kind='tool', amounts=v(actions=1), now_ms=1200)


def test_answer_and_question_cannot_claim_terminal_or_waiting_state(f):
    result = run_atomic_step(f.state, replace(f.host, choose=lambda *_: {
        'kind': 'answer', 'text': '42', 'usage': {'input_tokens': 10, 'output_tokens': 1}}),
        f.store, expected_version=f.version)
    assert result.reason == 'communication_and_completion_not_integrated'
    assert f.store.load('task', 'alice', expected_version=result.checkpoint_version).state.status == 'running'
