"""Claim-bound candidate reservations use disposable shared SQLite only."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateState
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import ENGINE, CheckpointConflict, ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetLimits, BudgetExhausted, TaskBudget
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimSuperseded
from alphonse.agent_v2.core.messages.experimental_claim_checkpoint import reserve_model_under_claim, settle_model_under_claim
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.users import V2UserStore
from unsafe_reclaim_fixture import simulate_old_writer_requeue


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / 'runtime.sqlite'
    users = V2UserStore(path)
    users.set_users_root(tmp_path / 'profiles')
    users.create_user(user_id='alice', display_name='Alice')
    queue = SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    questions = SQLiteQuestionStore(path)
    now = datetime.now(timezone.utc)
    queue.enqueue(CoreMessage(now, 'synthetic', 'alice'), message_id='message')
    claim = queue.claim_next(lease_seconds=60).claim_receipt
    task = TaskState(task_id='task', message_id='message', user='alice', goal='synthetic',
                     intelligence_engine=ENGINE, intelligence_schema_version=1)
    version = questions.save_task_checkpoint(task, status='running')
    store = ExperimentalCheckpointStore(questions)
    vector = dict(actions=2, model_calls=2, input_tokens=500, output_tokens=200)
    reserve = dict(actions=0, model_calls=0, input_tokens=0, output_tokens=0)
    checkpoint = store.initialize('task', 'alice', expected_version=version,
        state=CandidateState('task', 'alice', 'goal'),
        budget=TaskBudget(BudgetLimits(vector, reserve, 100000, 0), started_at_ms=int(now.timestamp() * 1000)))
    return queue, users, questions, store, claim, checkpoint.version, now


def reserve(f, **kwargs):
    _, _, _, store, claim, version, now = f
    return reserve_model_under_claim(store, kwargs.pop('claim', claim), task_id='task',
        actor=kwargs.pop('actor', 'alice'), expected_version=kwargs.pop('version', version),
        reservation_id=kwargs.pop('reservation_id', 'model-1'), input_tokens=100, output_tokens=50,
        clock=kwargs.pop('clock', lambda: now), **kwargs)


def test_current_claim_reserves_atomically_and_restart_preserves_marker(setup):
    queue, _, questions, store, claim, _, _ = setup
    result = reserve(setup)
    restarted = ExperimentalCheckpointStore(SQLiteQuestionStore(questions.db_path)).load(
        'task', 'alice', expected_version=result.version)
    assert restarted.state.inflight == 'model-1'
    assert restarted.budget.usage()['in_flight'] == ['model-1']
    assert claim.claim_generation not in str(questions.load_task_checkpoint('task').metadata)
    with pytest.raises(CheckpointConflict, match='reconciliation'):
        reserve(setup, version=result.version, reservation_id='model-2')
    assert queue.status_for('message') == 'processing'


def test_lost_claim_or_expiry_rolls_back_entire_reservation(setup):
    queue, _, questions, store, claim, version, now = setup
    samples = iter((now, now, now + timedelta(days=1)))
    with pytest.raises(ClaimSuperseded):
        reserve(setup, clock=lambda: next(samples))
    assert not store.load('task', 'alice', expected_version=version).state.inflight
    simulate_old_writer_requeue(queue, 'message')
    assert queue.claim_next().claim_receipt.claim_generation != claim.claim_generation
    with pytest.raises(ClaimSuperseded):
        reserve(setup)
    assert not store.load('task', 'alice', expected_version=version).state.inflight


def test_wrong_actor_cancelled_or_stale_checkpoint_does_not_reserve(setup):
    _, users, questions, store, _, version, _ = setup
    with pytest.raises(ClaimFenceUnavailable):
        reserve(setup, actor='bob')
    with pytest.raises(CheckpointConflict):
        reserve(setup, version=version + 1)
    users.update_user('alice', is_active=False)
    with pytest.raises(ClaimFenceUnavailable):
        reserve(setup)
    users.update_user('alice', is_active=True)
    assert questions.mark_task_checkpoint_terminal('task', status='cancelled')
    with pytest.raises(CheckpointConflict):
        reserve(setup)
    assert store.inspect_cancelled('task', 'alice', expected_version=version)['inflight'] is None


def test_lease_check_serializes_concurrent_reservations(setup):
    def run(identity):
        try:
            return reserve(setup, reservation_id=identity).state.inflight
        except CheckpointConflict:
            return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(run, ['first', 'second']))
    assert values.count('conflict') == 1
    assert len(set(values) - {'conflict'}) == 1


def test_trigger_corruption_and_clock_failure_roll_back(setup):
    _, _, questions, store, _, version, now = setup
    with questions._connect() as conn:
        conn.execute("""CREATE TRIGGER corrupt_candidate AFTER UPDATE ON v2_task_checkpoints
            BEGIN UPDATE v2_task_checkpoints SET task_state_json=json_set(NEW.task_state_json, '$.goal', 'changed')
            WHERE task_id=NEW.task_id; END""")
    with pytest.raises(ClaimFenceUnavailable, match='changed'):
        reserve(setup)
    assert not store.load('task', 'alice', expected_version=version).state.inflight
    with questions._connect() as conn:
        conn.execute('DROP TRIGGER corrupt_candidate')
    def failed_clock():
        raise RuntimeError('unavailable')
    with pytest.raises(ClaimFenceUnavailable, match='clock_unavailable'):
        reserve(setup, clock=failed_clock)
    assert not store.load('task', 'alice', expected_version=version).state.inflight


def test_rejected_budget_clock_observation_is_fenced_and_persisted(setup):
    _, _, _, store, _, version, now = setup
    # Move only the task deadline into the past; the queue lease is still live.
    with store.store._connect() as conn:
        raw = conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0]
        import json
        data = json.loads(raw)
        data['metadata']['single_loop_checkpoint']['budget']['limits']['elapsed_ms'] = 1
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(data),))
    with pytest.raises(BudgetExhausted):
        reserve(setup, clock=lambda: now + timedelta(milliseconds=2))
    restored = store.load('task', 'alice', expected_version=version + 1)
    assert not restored.state.inflight
    assert restored.budget.last_observed_at_ms == int((now + timedelta(milliseconds=2)).timestamp() * 1000)


def settle(f, version, **kwargs):
    _, _, _, store, claim, _, now = f
    return settle_model_under_claim(store, kwargs.pop('claim', claim), task_id='task',
        actor=kwargs.pop('actor', 'alice'), expected_version=version, reservation_id='model-1',
        outcome=kwargs.pop('outcome', 'success'), input_tokens=kwargs.pop('input_tokens', 25),
        output_tokens=kwargs.pop('output_tokens', 10), clock=kwargs.pop('clock', lambda: now), **kwargs)


def test_exact_acquisition_settles_and_generic_writer_cannot_bypass(setup):
    _, _, questions, store, claim, _, _ = setup
    reserved = reserve(setup)
    with pytest.raises(CheckpointConflict, match='fenced_settlement_required'):
        store.settle('task', 'alice', expected_version=reserved.version, reservation_id='model-1', outcome='success')
    metadata = questions.load_task_checkpoint('task').metadata
    assert claim.claim_generation not in str(metadata)
    settled = settle(setup, reserved.version)
    assert not settled.state.inflight
    assert settled.budget.to_dict()['reservations']['model-1']['outcome'] == 'success'
    assert 'single_loop_model_claim' not in questions.load_task_checkpoint('task').metadata


def test_same_owner_new_generation_cannot_settle_old_reservation(setup):
    queue, _, _, store, old, _, _ = setup
    reserved = reserve(setup)
    simulate_old_writer_requeue(queue, 'message')
    newer = queue.claim_next().claim_receipt
    assert newer.lease_owner == old.lease_owner and newer.claim_generation != old.claim_generation
    with pytest.raises(ClaimSuperseded):
        settle(setup, reserved.version)
    with pytest.raises(CheckpointConflict, match='binding_unavailable'):
        settle(setup, reserved.version, claim=newer)
    assert store.load('task', 'alice', expected_version=reserved.version).state.inflight == 'model-1'


def test_settlement_preserves_marker_on_expiry_cancellation_and_invalid_usage(setup):
    _, _, questions, store, _, _, now = setup
    reserved = reserve(setup)
    moments = iter((now, now + timedelta(days=1)))
    with pytest.raises(ClaimSuperseded):
        settle(setup, reserved.version, clock=lambda: next(moments))
    with pytest.raises(ValueError):
        settle(setup, reserved.version, input_tokens=True)
    assert questions.mark_task_checkpoint_terminal('task', status='cancelled')
    with pytest.raises(CheckpointConflict):
        settle(setup, reserved.version)
    assert store.inspect_cancelled('task', 'alice', expected_version=reserved.version)['inflight'] == 'model-1'


def test_settlement_keeps_unknown_usage_and_blocks_future_model_reservation(setup):
    _, _, _, store, _, _, _ = setup
    reserved = reserve(setup)
    settled = settle(setup, reserved.version, input_tokens=None)
    assert settled.budget.usage()['unknown_usage'] == ['model-1']
    with pytest.raises(CheckpointConflict, match='usage_unresolved'):
        reserve(setup, version=settled.version, reservation_id='model-2')


def test_settlement_rolls_back_if_trigger_changes_checkpoint(setup):
    _, _, questions, store, _, _, _ = setup
    reserved = reserve(setup)
    with questions._connect() as conn:
        conn.execute("""CREATE TRIGGER corrupt_model_settlement AFTER UPDATE ON v2_task_checkpoints
            BEGIN UPDATE v2_task_checkpoints SET task_state_json=json_set(NEW.task_state_json, '$.goal', 'changed')
            WHERE task_id=NEW.task_id; END""")
    with pytest.raises(ClaimFenceUnavailable, match='changed'):
        settle(setup, reserved.version)
    restored = store.load('task', 'alice', expected_version=reserved.version)
    assert restored.state.inflight == 'model-1'
    assert restored.budget.usage()['in_flight'] == ['model-1']


def test_orphaned_claim_binding_blocks_legacy_initialization_and_reservation(setup):
    _, _, questions, store, _, version, now = setup
    with questions._connect() as conn:
        raw = conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0]
        import json
        data = json.loads(raw)
        data['metadata']['single_loop_model_claim'] = {'corrupt': True}
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(data),))
    amounts = {'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}
    with pytest.raises(CheckpointConflict, match='binding_unresolved'):
        store.reserve('task', 'alice', expected_version=version, reservation_id='new', kind='model',
                      amounts=amounts, now_ms=int(now.timestamp() * 1000))
    with pytest.raises(CheckpointConflict, match='fenced_settlement_required'):
        store.settle('task', 'alice', expected_version=version, reservation_id='new', outcome='success')
    # Even a missing envelope must not let initialization adopt the orphan.
    with questions._connect() as conn:
        del data['metadata']['single_loop_checkpoint']
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(data),))
    total = dict(actions=2, model_calls=2, input_tokens=500, output_tokens=200)
    final = dict(actions=0, model_calls=0, input_tokens=0, output_tokens=0)
    with pytest.raises(CheckpointConflict, match='binding_unresolved'):
        store.initialize('task', 'alice', expected_version=version,
            state=CandidateState('task', 'alice', 'goal'),
            budget=TaskBudget(BudgetLimits(total, final, 10000, 0), started_at_ms=int(now.timestamp() * 1000)))
