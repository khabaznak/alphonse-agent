"""Quiescent recovery may never revive uncertain or still-running work."""
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest

from test_owner_atomic_checkpoint import setup
from test_owner_continuation_activation import prepare
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_activation import activate_held_same_claim
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_reclaim import reclaim_quiescent_activated
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_reclaim import reclaim_approved_pause
from alphonse.agent_v2.core.intelligence.simple_loop.owner_atomic_checkpoint import OwnerAtomicCheckpoint
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_detached import persist_continuation_cancellation
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_checkpoint import CheckpointConflict


def activated(setup, monkeypatch):
    queue, task, atomic, version, now = prepare(setup)
    monkeypatch.setattr('alphonse.agent_v2.core.intelligence.simple_loop.owner_read_recovery.restore_owner_read_observations',
                        lambda *args, **kwargs: None)
    context = CoreLoopContext(messages=queue, claim_receipt=atomic.receipt, cancellation_checker=lambda: False)
    atomic.clock = lambda: now + timedelta(milliseconds=30)
    saved = activate_held_same_claim(atomic, expected_version=version, context=context, cancellation_checker=lambda: False)
    return queue, task, atomic, saved, now


def recover(queue, store, task, now, cancellation=lambda: False):
    return reclaim_quiescent_activated(queue, store, task.message_id, lease_owner='new-worker',
                                       clock=lambda: now, cancellation_checker=cancellation)


def test_live_lease_cannot_be_taken_and_expired_quiescent_lease_can(setup, monkeypatch):
    queue, task, atomic, saved, now = activated(setup, monkeypatch)
    with pytest.raises(CheckpointConflict):
        recover(queue, atomic.store, task, now)
    recovered, version = recover(queue, atomic.store, task, now + timedelta(minutes=10))
    assert recovered.claim_receipt.claim_generation != atomic.receipt.claim_generation
    assert version > saved.version


@pytest.mark.parametrize('marker', ['owner_read_pending_action', 'owner_read_pending_decision',
    'owner_read_selected_draft', 'owner_bash_dispatch_started', 'owner_bash_return_unreconciled'])
def test_uncertain_markers_never_reclaim(setup, monkeypatch, marker):
    queue, task, atomic, _, now = activated(setup, monkeypatch)
    with queue._connect() as conn:
        row = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', (task.task_id,)).fetchone()
        raw = json.loads(row['task_state_json'])
        raw['metadata'][marker] = {'status': 'unconfirmed'}
        conn.execute('UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id=?', (json.dumps(raw), task.task_id))
    with pytest.raises(CheckpointConflict):
        recover(queue, atomic.store, task, now + timedelta(minutes=10))


@pytest.mark.parametrize('cancelled', [True, None])
def test_cancelled_or_unknown_recovery_rejected(setup, monkeypatch, cancelled):
    queue, task, atomic, _, now = activated(setup, monkeypatch)
    with pytest.raises(CheckpointConflict):
        recover(queue, atomic.store, task, now + timedelta(minutes=10), cancellation=lambda: cancelled)


def test_inflight_reservation_is_never_recovered(setup, monkeypatch):
    queue, task, atomic, saved, now = activated(setup, monkeypatch)
    atomic.reserve(expected_version=saved.version, reservation_id='inflight', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50},
        now_ms=int((now + timedelta(milliseconds=31)).timestamp() * 1000))
    with pytest.raises(CheckpointConflict):
        recover(queue, atomic.store, task, now + timedelta(minutes=10))


def test_repeated_crashes_preserve_claim_lineage_and_prior_usage(setup, monkeypatch):
    queue, task, atomic, saved, now = activated(setup, monkeypatch)
    first, _ = recover(queue, atomic.store, task, now + timedelta(minutes=10))
    second, version = recover(queue, atomic.store, task, now + timedelta(minutes=20))
    assert first.claim_receipt.claim_generation != second.claim_receipt.claim_generation
    with queue._connect() as conn:
        row = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', (task.task_id,)).fetchone()
    raw = json.loads(row['task_state_json'])
    assert len(raw['metadata']['owner_continuation_crash_lineage']) == 2
    state, budget = atomic.store._decode(raw, task.task_id, task.user)
    assert state.evidence_sources == saved.state.evidence_sources
    assert budget.usage()['charged'] == saved.budget.usage()['charged']


def test_competing_recovery_workers_only_one_gets_live_claim(setup, monkeypatch):
    queue, task, atomic, _, now = activated(setup, monkeypatch)
    barrier = threading.Barrier(2)
    def attempt(name):
        barrier.wait(timeout=5)
        try:
            return reclaim_quiescent_activated(queue, atomic.store, task.message_id, lease_owner=name,
                clock=lambda: now + timedelta(minutes=10), cancellation_checker=lambda: False)
        except CheckpointConflict:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, ('worker-a', 'worker-b')))
    successful = [item for item in results if item is not None]
    assert len(successful) == 1


def test_persisted_cancel_after_reclaim_blocks_activation_even_if_process_state_is_lost(setup, monkeypatch):
    queue, task, original, version, now = prepare(setup)
    new, version = reclaim_approved_pause(queue, original.store, task.message_id, lease_owner='new-worker',
        clock=lambda: now, cancellation_checker=lambda: False)
    persist_continuation_cancellation(queue, task.message_id, task.user)
    fresh = OwnerAtomicCheckpoint(queue, original.store, task, new.claim_receipt, clock=lambda: now)
    fresh.version = version
    context = CoreLoopContext(messages=queue, claim_receipt=new.claim_receipt, cancellation_checker=lambda: False)
    monkeypatch.setattr('alphonse.agent_v2.core.intelligence.simple_loop.owner_read_recovery.restore_owner_read_observations',
                        lambda *args, **kwargs: None)
    with pytest.raises(CheckpointConflict):
        activate_held_same_claim(fresh, expected_version=version, context=context, cancellation_checker=lambda: False)


def test_persisted_cancel_after_activation_blocks_further_reservations(setup, monkeypatch):
    queue, task, atomic, saved, now = activated(setup, monkeypatch)
    persist_continuation_cancellation(queue, task.message_id, task.user)
    with pytest.raises(CheckpointConflict):
        atomic.reserve(expected_version=saved.version, reservation_id='after-cancel', kind='model',
            amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50},
            now_ms=int((now + timedelta(milliseconds=31)).timestamp() * 1000))
