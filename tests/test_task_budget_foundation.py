"""Disposable tests for the opt-in budget accounting contract; no providers."""
import json

import pytest

from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetExhausted, BudgetLimits, TaskBudget


def vector(actions=0, model_calls=0, input_tokens=0, output_tokens=0):
    return dict(actions=actions, model_calls=model_calls, input_tokens=input_tokens, output_tokens=output_tokens)


def budget():
    return TaskBudget(BudgetLimits(vector(3, 4, 1000, 300), vector(0, 1, 150, 50), 10000, 1000), started_at_ms=1000)


def model(input_tokens=200, output_tokens=60):
    return vector(model_calls=1, input_tokens=input_tokens, output_tokens=output_tokens)


def roundtrip(value):
    return TaskBudget.from_dict(json.loads(json.dumps(value.to_dict())))


def test_restart_with_reserved_call_does_not_refund_or_allow_same_id():
    state = budget()
    assert state.reserve('attempt-1', kind='model', amounts=model(), now_ms=1100) == 1
    restored = roundtrip(state)
    assert restored.usage()['in_flight'] == ['attempt-1']
    assert restored.usage()['charged'] == model()
    with pytest.raises(ValueError, match='duplicate'):
        restored.reserve('attempt-1', kind='model', amounts=model(), now_ms=1200)
    with pytest.raises(BudgetExhausted, match='unresolved'):
        restored.reserve('attempt-2', kind='model', amounts=model(), now_ms=1200)
    restored.settle('attempt-1', outcome='failure', input_tokens=120, output_tokens=10)
    assert restored.reserve('attempt-2', kind='model', amounts=model(), now_ms=1201) == 4
    assert roundtrip(restored).usage()['charged'] == model(320, 70) | {'model_calls': 2}


def test_failure_timeout_and_retries_are_individual_attempts():
    state = budget()
    for n in range(3):
        state.reserve(str(n), kind='model', amounts=model(100, 20), now_ms=1001 + n)
        state.settle(str(n), outcome='timeout' if n == 0 else 'failure', input_tokens=90, output_tokens=10)
    with pytest.raises(BudgetExhausted, match='model_calls'):
        state.reserve('fourth', kind='model', amounts=model(100, 20), now_ms=1004)
    state.reserve('last-reply', kind='model', amounts=model(100, 30), now_ms=1005, lane='final')
    with pytest.raises(BudgetExhausted, match='final_attempt'):
        state.reserve('another-reply', kind='model', amounts=model(10, 10), now_ms=1006, lane='final')
    with pytest.raises(BudgetExhausted, match='final_already_started'):
        state.reserve('extra', kind='tool', amounts=vector(actions=1), now_ms=1006)


def test_unknown_usage_charges_reservation_and_blocks_all_models_including_final():
    state = budget()
    state.reserve('failed', kind='model', amounts=model(), now_ms=1001)
    state.settle('failed', outcome='failure', input_tokens=123)
    snapshot = roundtrip(state)
    assert snapshot.usage()['charged'] == model(123, 60)
    assert snapshot.usage()['unknown_usage'] == ['failed']
    assert snapshot.to_dict()['reservations']['failed']['unknown_tokens'] == ['output_tokens']
    with pytest.raises(BudgetExhausted, match='usage_unresolved'):
        snapshot.reserve('retry', kind='model', amounts=model(), now_ms=1002)
    with pytest.raises(BudgetExhausted, match='usage_unresolved'):
        snapshot.reserve('reply', kind='model', amounts=model(100, 30), now_ms=1002, lane='final')


def test_overrun_is_accounted_and_cannot_be_hidden_by_reservation():
    state = budget()
    state.reserve('overrun', kind='model', amounts=model(), now_ms=1001)
    state.settle('overrun', outcome='success', input_tokens=1100, output_tokens=100)
    assert state.usage()['charged']['input_tokens'] == 1100
    with pytest.raises(BudgetExhausted, match='input_tokens'):
        state.reserve('reply', kind='model', amounts=model(10, 10), now_ms=1002, lane='final')
    # Caller must use a deterministic no-model final response if even reserve is exhausted.


def test_actions_count_on_mixed_success_failure_and_cancel_and_reservation_survives_restart():
    state = budget()
    for n, outcome in enumerate(('success', 'failure', 'cancelled')):
        state.reserve(str(n), kind='tool', amounts=vector(actions=1), now_ms=1001 + n)
        state.settle(str(n), outcome=outcome)
        state = roundtrip(state)
    with pytest.raises(BudgetExhausted, match='actions'):
        state.reserve('other', kind='tool', amounts=vector(actions=1), now_ms=1010)
    with pytest.raises(BudgetExhausted, match='cancelled'):
        state.reserve('cancelled', kind='model', amounts=model(), now_ms=1010, cancelled=True)


def test_elapsed_includes_wait_and_downtime_and_protects_final_window():
    state = budget()
    state.reserve('work', kind='tool', amounts=vector(actions=1), now_ms=1001)
    state.settle('work', outcome='success')
    state = roundtrip(state)
    with pytest.raises(BudgetExhausted, match='elapsed'):
        state.reserve('late', kind='tool', amounts=vector(actions=1), now_ms=10000)
    state.reserve('final', kind='model', amounts=model(100, 30), now_ms=10000, lane='final')
    state.settle('final', outcome='success', input_tokens=100, output_tokens=30)
    with pytest.raises(BudgetExhausted, match='elapsed'):
        state.reserve('too-late', kind='model', amounts=model(10, 10), now_ms=11000, lane='final')
    with pytest.raises(ValueError, match='clock'):
        state.reserve('backwards', kind='tool', amounts=vector(actions=1), now_ms=9999)


def test_only_known_undispatched_reservation_may_be_released():
    state = budget()
    state.reserve('not-dispatched', kind='model', amounts=model(), now_ms=1001)
    state.release_undispatched('not-dispatched')
    assert roundtrip(state).usage()['charged'] == vector()
    with pytest.raises(ValueError, match='not_pending'):
        state.settle('not-dispatched', outcome='success', input_tokens=0, output_tokens=0)
    state.reserve('new-id', kind='model', amounts=model(), now_ms=1002)
    assert state.usage()['in_flight'] == ['new-id']


def test_invalid_and_future_snapshots_are_rejected():
    state = budget().to_dict()
    state['schema_version'] = 2
    with pytest.raises(ValueError, match='schema'):
        TaskBudget.from_dict(state)
    with pytest.raises(ValueError):
        BudgetLimits(vector(1, 1, 1, 1), vector(0, 2, 0, 0), 100, 20)
    with pytest.raises(ValueError):
        budget().reserve('bad', kind='model', amounts=model(0, 1), now_ms=1001)


def test_file_store_compare_and_swap_prevents_two_workers_spending_same_budget(tmp_path):
    from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetRevisionConflict, SQLiteTaskBudgetStore
    path = str(tmp_path / 'budgets.sqlite')
    store = SQLiteTaskBudgetStore(path)
    store.create('task', budget())
    worker_one = store.load('task')
    worker_two = SQLiteTaskBudgetStore(path).load('task')
    worker_one.reserve('one', kind='model', amounts=model(), now_ms=1001)
    worker_two.reserve('two', kind='model', amounts=model(), now_ms=1001)
    store.save('task', worker_one, expected_revision=0)
    with pytest.raises(BudgetRevisionConflict):
        store.save('task', worker_two, expected_revision=0)
    recovered = SQLiteTaskBudgetStore(path).load('task')
    assert recovered.usage()['in_flight'] == ['one']
    recovered.settle('one', outcome='timeout')
    store.save('task', recovered, expected_revision=1)
    assert store.load('task').usage()['unknown_usage'] == ['one']


def test_store_requires_atomic_step_before_next_transition(tmp_path):
    from alphonse.agent_v2.core.intelligence.v3.task_budget import SQLiteTaskBudgetStore
    store = SQLiteTaskBudgetStore(str(tmp_path / 'budgets.sqlite'))
    state = budget()
    store.create('task', state)
    state.reserve('one', kind='tool', amounts=vector(actions=1), now_ms=1001)
    state.settle('one', outcome='success')
    with pytest.raises(ValueError, match='single_transition'):
        store.save('task', state, expected_revision=0)
    assert store.load('task').usage()['charged'] == vector()


def test_unknown_usage_reconciles_only_with_explicit_later_observation():
    state = budget()
    state.reserve('call', kind='model', amounts=model(), now_ms=1001)
    state.settle('call', outcome='timeout', input_tokens=100)
    with pytest.raises(ValueError, match='reconciliation'):
        state.reconcile_unknown('call', input_tokens=1)
    state.reconcile_unknown('call', output_tokens=5)
    restored = roundtrip(state)
    assert restored.usage()['unknown_usage'] == []
    assert restored.usage()['charged'] == model(100, 5)
    assert restored.to_dict()['reservations']['call']['outcome'] == 'timeout'
    restored.reserve('next', kind='model', amounts=model(), now_ms=1002)


def test_inflight_prior_model_blocks_final_until_usage_is_reconciled():
    state = budget()
    state.reserve('inflight', kind='model', amounts=model(), now_ms=1001)
    restored = roundtrip(state)
    with pytest.raises(BudgetExhausted, match='usage_unresolved'):
        restored.reserve('reply', kind='model', amounts=model(100, 30), now_ms=1002, lane='final')
    restored.settle('inflight', outcome='timeout')
    with pytest.raises(BudgetExhausted, match='usage_unresolved'):
        restored.reserve('reply', kind='model', amounts=model(100, 30), now_ms=1003, lane='final')
    restored.reconcile_unknown('inflight', input_tokens=180, output_tokens=50)
    restored.reserve('reply', kind='model', amounts=model(100, 30), now_ms=1004, lane='final')


def test_deadline_rejection_is_persisted_and_clock_rollback_cannot_reopen_work(tmp_path):
    from alphonse.agent_v2.core.intelligence.v3.task_budget import SQLiteTaskBudgetStore
    store = SQLiteTaskBudgetStore(str(tmp_path / 'budget.sqlite'))
    store.create('task', budget())
    with pytest.raises(BudgetExhausted, match='elapsed'):
        store.reserve('task', expected_revision=0, reservation_id='late', kind='tool',
                      amounts=vector(actions=1), now_ms=10000)
    restored = SQLiteTaskBudgetStore(str(tmp_path / 'budget.sqlite')).load('task')
    assert restored.revision == 1
    assert restored.last_observed_at_ms == 10000
    with pytest.raises(ValueError, match='clock'):
        store.reserve('task', expected_revision=1, reservation_id='rollback', kind='tool',
                      amounts=vector(actions=1), now_ms=2000)
    # The final reserve is still available until the total deadline.
    store.reserve('task', expected_revision=1, reservation_id='final', kind='model',
                  amounts=model(100, 30), now_ms=10000, lane='final')


def test_total_deadline_latch_blocks_final_after_restart(tmp_path):
    from alphonse.agent_v2.core.intelligence.v3.task_budget import SQLiteTaskBudgetStore
    store = SQLiteTaskBudgetStore(str(tmp_path / 'budget.sqlite'))
    store.create('task', budget())
    with pytest.raises(BudgetExhausted, match='elapsed'):
        store.reserve('task', expected_revision=0, reservation_id='final', kind='model',
                      amounts=model(100, 30), now_ms=11000, lane='final')
    with pytest.raises(ValueError, match='clock'):
        store.reserve('task', expected_revision=1, reservation_id='final2', kind='model',
                      amounts=model(100, 30), now_ms=10000, lane='final')


def test_rejection_for_non_time_limit_persists_clock_before_restart(tmp_path):
    from alphonse.agent_v2.core.intelligence.v3.task_budget import SQLiteTaskBudgetStore
    store = SQLiteTaskBudgetStore(str(tmp_path / 'budget.sqlite'))
    state = budget()
    store.create('task', state)
    state = store.reserve('task', expected_revision=0, reservation_id='first', kind='model',
                          amounts=model(), now_ms=1100)
    # A second model is rejected for unresolved usage while still inside the
    # work window; the timestamp must be fenced before returning that error.
    with pytest.raises(BudgetExhausted, match='usage_unresolved'):
        store.reserve('task', expected_revision=state.revision, reservation_id='second', kind='model',
                      amounts=model(), now_ms=9500)
    restored = SQLiteTaskBudgetStore(str(tmp_path / 'budget.sqlite')).load('task')
    assert restored.last_observed_at_ms == 9500
    restored.settle('first', outcome='success', input_tokens=100, output_tokens=10)
    store.save('task', restored, expected_revision=2)
    with pytest.raises(ValueError, match='clock'):
        store.reserve('task', expected_revision=3, reservation_id='reply', kind='model',
                      amounts=model(100, 30), now_ms=2000, lane='final')
