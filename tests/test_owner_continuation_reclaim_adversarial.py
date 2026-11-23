"""Adversarial fresh-claim cases; all data and owners are disposable."""
from datetime import datetime, timedelta, timezone
import os
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from test_owner_atomic_checkpoint import setup
from test_owner_continuation_question import paused_checkpoint
from alphonse.agent_v2.core.intelligence.simple_loop.owner_atomic_checkpoint import OwnerAtomicCheckpoint
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_question import (
    start_question, acknowledge_question, record_choice,
)
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_grant import issue_grant, prepare_handoff
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_reclaim import reclaim_approved_pause
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_detached import open_detached_paused_checkpoint
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_checkpoint import CheckpointConflict
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimSuperseded


def approved_fixture(setup):
    queue, acquired, task, store, atomic, version = paused_checkpoint(setup)
    version, token = start_question(atomic, expected_version=version, question='Continue?', cancellation_checker=lambda: False)
    version = acknowledge_question(atomic, expected_version=version, token=token, cancellation_checker=lambda: False)
    version = record_choice(atomic, expected_version=version, peer_uid=os.getuid(), approve=True, cancellation_checker=lambda: False)
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    version = issue_grant(atomic, expected_version=version, now_ms=now, lifetime_ms=300000, cancellation_checker=lambda: False)
    version = prepare_handoff(atomic, expected_version=version, now_ms=now, cancellation_checker=lambda: False)
    return queue, acquired, task, store, atomic, version


def reserve(checkpoint, version):
    return checkpoint.reserve(expected_version=version, reservation_id='another-model', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=102)


def test_new_claim_fences_live_original_worker_and_remains_paused(setup):
    queue, old_claim, task, store, atomic, version = approved_fixture(setup)
    new_claim, new_version = reclaim_approved_pause(queue, store, task.message_id, lease_owner='next-worker',
                                                     cancellation_checker=lambda: False)
    assert new_claim.claim_receipt.claim_generation != old_claim.claim_receipt.claim_generation
    with pytest.raises((ClaimSuperseded, ClaimFenceUnavailable, CheckpointConflict)):
        reserve(atomic, version)
    fresh = OwnerAtomicCheckpoint(queue, store, task, new_claim.claim_receipt, clock=lambda: datetime.now(timezone.utc))
    with pytest.raises(CheckpointConflict, match='owner_continuation_paused'):
        reserve(fresh, new_version)


def test_crash_after_reclaim_can_be_reclaimed_again_without_dispatch(setup):
    queue, _, task, store, _, _ = approved_fixture(setup)
    first, _ = reclaim_approved_pause(queue, store, task.message_id, lease_owner='worker-before-crash',
                                      cancellation_checker=lambda: False)
    second, version = reclaim_approved_pause(queue, store, task.message_id, lease_owner='worker-after-crash',
                                             cancellation_checker=lambda: False)
    assert first.claim_receipt.claim_generation != second.claim_receipt.claim_generation
    fresh = OwnerAtomicCheckpoint(queue, store, task, second.claim_receipt, clock=lambda: datetime.now(timezone.utc))
    with pytest.raises(CheckpointConflict, match='owner_continuation_paused'):
        reserve(fresh, version)


@pytest.mark.parametrize('cancelled', [True, None])
def test_cancelled_or_unknown_request_cannot_be_reclaimed(setup, cancelled):
    queue, acquired, task, store, _, _ = approved_fixture(setup)
    with pytest.raises(CheckpointConflict):
        reclaim_approved_pause(queue, store, task.message_id, lease_owner='worker', cancellation_checker=lambda: cancelled)
    with queue._connect() as conn:
        row = conn.execute('SELECT claim_generation FROM v2_inbound_messages WHERE message_id=?', (task.message_id,)).fetchone()
    assert row['claim_generation'] == acquired.claim_receipt.claim_generation


def test_competing_reclaims_leave_at_most_one_current_claim(setup):
    queue, _, task, store, _, _ = approved_fixture(setup)
    barrier = threading.Barrier(2)
    def reclaim(worker):
        barrier.wait(timeout=5)
        try:
            return reclaim_approved_pause(queue, store, task.message_id, lease_owner=worker,
                                          cancellation_checker=lambda: False)
        except (ClaimSuperseded, ClaimFenceUnavailable, CheckpointConflict):
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reclaim, ('worker-a', 'worker-b')))
    claims = [row[0] for row in results if row is not None]
    assert claims
    with queue._connect() as conn:
        row = conn.execute('SELECT claim_generation FROM v2_inbound_messages WHERE message_id=?', (task.message_id,)).fetchone()
    assert sum(claim.claim_receipt.claim_generation == row['claim_generation'] for claim in claims) == 1


def test_expired_grant_cannot_be_reclaimed(setup):
    queue, acquired, task, store, _, _ = approved_fixture(setup)
    future = datetime.now(timezone.utc) + timedelta(minutes=6)
    with pytest.raises(CheckpointConflict):
        reclaim_approved_pause(queue, store, task.message_id, lease_owner='late-worker', clock=lambda: future,
                               cancellation_checker=lambda: False)
    with queue._connect() as conn:
        row = conn.execute('SELECT claim_generation FROM v2_inbound_messages WHERE message_id=?', (task.message_id,)).fetchone()
    assert row['claim_generation'] == acquired.claim_receipt.claim_generation


def test_cancellation_during_reclaim_rolls_back_claim_and_checkpoint(setup):
    queue, acquired, task, store, _, version = approved_fixture(setup)
    checks = iter([False, True])
    with pytest.raises(CheckpointConflict):
        reclaim_approved_pause(queue, store, task.message_id, lease_owner='worker', cancellation_checker=lambda: next(checks))
    with queue._connect() as conn:
        queue_row = conn.execute('SELECT claim_generation FROM v2_inbound_messages WHERE message_id=?', (task.message_id,)).fetchone()
        checkpoint_row = conn.execute('SELECT version FROM v2_task_checkpoints WHERE task_id=?', (task.task_id,)).fetchone()
    assert queue_row['claim_generation'] == acquired.claim_receipt.claim_generation
    assert checkpoint_row['version'] == version


def test_expired_claim_adapter_rolls_back_pause_fence_removal(setup):
    queue, _, task, store, _, version = approved_fixture(setup)
    detached = open_detached_paused_checkpoint(queue, store, task.message_id)
    with pytest.raises(CheckpointConflict, match='owner_detached_pause_unavailable'):
        with detached._transaction() as conn:
            row, raw, state, budget = detached._load(conn, version, allow_paused=True)
            del raw['metadata']['owner_continuation_pause']
            store._write(conn, row, raw, state, budget)
    with queue._connect() as conn:
        checkpoint_row = conn.execute('SELECT version FROM v2_task_checkpoints WHERE task_id=?', (task.task_id,)).fetchone()
    assert checkpoint_row['version'] == version


def test_deactivated_owner_cannot_reclaim(setup):
    queue, acquired, task, store, _, _ = approved_fixture(setup)
    with queue._connect() as conn:
        conn.execute('UPDATE v2_users SET is_active=0 WHERE user_id=?', (task.user,))
    with pytest.raises(CheckpointConflict):
        reclaim_approved_pause(queue, store, task.message_id, lease_owner='worker', cancellation_checker=lambda: False)
    with queue._connect() as conn:
        row = conn.execute('SELECT claim_generation FROM v2_inbound_messages WHERE message_id=?', (task.message_id,)).fetchone()
    assert row['claim_generation'] == acquired.claim_receipt.claim_generation
