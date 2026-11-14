"""Opt-in queue generation tests against disposable SQLite only."""
from datetime import datetime, timedelta, timezone
from dataclasses import asdict, replace
from copy import deepcopy
from unsafe_reclaim_fixture import simulate_old_writer_requeue

import pytest

from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue


def message(user='alice'):
    return CoreMessage(timestamp=datetime.now(timezone.utc), prompt='synthetic', user=user)


def test_default_has_no_migration_and_opt_in_claim_receipt_stays_internal(tmp_path):
    path = tmp_path / 'queue.sqlite'
    legacy = SQLiteMessageQueue(path, lease_owner='worker')
    with legacy._connect() as conn:
        assert 'claim_generation' not in {row['name'] for row in conn.execute('PRAGMA table_info(v2_inbound_messages)')}
    legacy.enqueue(message(), message_id='m')
    queue = SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    claimed = queue.claim_next()
    assert claimed.claim_receipt.message_id == 'm'
    assert claimed.claim_receipt.user_id == 'alice'
    assert len(claimed.claim_receipt.claim_generation) == 32
    assert claimed.claim_receipt.claim_generation not in repr(claimed)
    assert 'claim_receipt' not in asdict(claimed)
    assert claimed.claim_receipt.claim_generation not in str(asdict(claimed))
    assert 'claim_generation' not in asdict(claimed.claim_receipt)
    assert replace(claimed).claim_receipt is claimed.claim_receipt
    assert deepcopy(claimed).claim_receipt.claim_generation == claimed.claim_receipt.claim_generation
    assert replace(claimed.claim_receipt, claim_generation='0' * 32) != claimed.claim_receipt
    assert 'claim_generation' not in claimed.message.metadata
    with pytest.raises(ValueError, match='requires_receipt'):
        queue.claim_next(include_owned=True)


def test_migration_never_adopts_processing_legacy_claim(tmp_path):
    path = tmp_path / 'queue.sqlite'
    queue = SQLiteMessageQueue(path, lease_owner='old')
    queue.enqueue(message(), message_id='m')
    assert queue.claim_next()
    with pytest.raises(ValueError, match='legacy_inflight'):
        SQLiteMessageQueue(path, lease_owner='new', experimental_claim_fencing=True)
    with queue._connect() as conn:
        assert conn.execute("SELECT status FROM v2_inbound_messages WHERE message_id='m'").fetchone()[0] == 'processing'
        assert 'claim_generation' not in {r['name'] for r in conn.execute('PRAGMA table_info(v2_inbound_messages)')}


def test_same_owner_reclaim_rotates_generation_and_legacy_ack_retry_cannot_steal(tmp_path):
    path = tmp_path / 'queue.sqlite'
    queue = SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    queue.enqueue(message(), message_id='m')
    old = queue.claim_next(lease_seconds=1).claim_receipt
    assert queue.reclaim_expired(now=datetime.now(timezone.utc) + timedelta(seconds=2)) == 0
    simulate_old_writer_requeue(queue, 'm')
    newer = queue.claim_next().claim_receipt
    assert old.claim_generation != newer.claim_generation
    legacy = SQLiteMessageQueue(path, lease_owner='worker')
    assert not legacy.ack('m') and not legacy.retry('m')
    assert not queue.ack_fenced(old)
    assert not queue.ack_fenced(replace(newer, user_id='bob'))
    # A live generation alone cannot establish task completion or effects.
    assert not queue.ack_fenced(newer)
    with queue._connect() as conn:
        row = conn.execute("SELECT status,claim_generation FROM v2_inbound_messages WHERE message_id='m'").fetchone()
        assert tuple(row) == ('processing', newer.claim_generation)


@pytest.mark.parametrize('wait_outbox_id', ['', 'unconfirmed-delivery'])
def test_fenced_handoff_release_refuses_unverified_legacy_handoff_even_with_live_claim(tmp_path, wait_outbox_id):
    path = tmp_path / 'queue.sqlite'
    queue = SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    queue.enqueue(message(), message_id='m')
    receipt = queue.claim_next().claim_receipt
    with queue._connect() as conn:
        conn.execute('INSERT INTO v2_scheduled_handoffs(message_id,task_id,owner_id,wait_outbox_id) VALUES (?,?,?,?)',
                     ('m', 'task', 'alice', wait_outbox_id))
    assert not queue.release_scheduled_handoff('m', lease_owner='worker')
    assert not queue.release_scheduled_handoff_fenced(replace(receipt, claim_generation='a' * 32))
    assert not queue.release_scheduled_handoff_fenced(receipt, clock=lambda: pytest.fail('unverified handoff must not enter release'))
    assert not queue.ack_fenced(receipt)
    with queue._connect() as conn:
        row = conn.execute("SELECT status,claim_generation FROM v2_inbound_messages WHERE message_id='m'").fetchone()
        assert tuple(row) == ('processing', receipt.claim_generation)
        assert conn.execute("SELECT 1 FROM v2_scheduled_handoffs WHERE message_id='m'").fetchone()
    legacy = SQLiteMessageQueue(path, lease_owner='worker')
    assert legacy.claim_next() is None


def test_expired_or_wrong_actor_receipt_cannot_ack_or_release(tmp_path):
    path = tmp_path / 'queue.sqlite'
    queue = SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    queue.enqueue(message(), message_id='m')
    receipt = queue.claim_next().claim_receipt
    with queue._connect() as conn:
        conn.execute('INSERT INTO v2_scheduled_handoffs(message_id,task_id,owner_id) VALUES (?,?,?)', ('m', 'task', 'bob'))
    assert not queue.release_scheduled_handoff_fenced(receipt)
    with queue._connect() as conn:
        conn.execute("UPDATE v2_inbound_messages SET lease_expires_at=? WHERE message_id='m'",
                     ((datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat(),))
    assert not queue.ack_fenced(receipt)
    assert not queue.release_scheduled_handoff_fenced(receipt)


@pytest.mark.parametrize('outcome', ['completed', 'cancelled', 'processing'])
def test_fenced_ack_refuses_all_unverified_outcomes(tmp_path, outcome):
    path = tmp_path / 'queue.sqlite'
    queue = SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    queue.enqueue(message(), message_id='m')
    receipt = queue.claim_next().claim_receipt
    result = queue.ack_fenced(receipt, outcome=outcome, clock=lambda: pytest.fail('must not enter ack'))
    assert not result
    with queue._connect() as conn:
        row = conn.execute("SELECT status,claim_generation FROM v2_inbound_messages WHERE message_id='m'").fetchone()
        assert tuple(row) == ('processing', receipt.claim_generation)


def test_reclaim_does_not_replay_expired_fenced_row(tmp_path):
    path = tmp_path / 'queue.sqlite'
    queue = SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    queue.enqueue(message(), message_id='m')
    receipt = queue.claim_next(lease_seconds=1).claim_receipt
    legacy = SQLiteMessageQueue(path, lease_owner='worker')
    future = datetime.now(timezone.utc) + timedelta(seconds=2)
    assert legacy.reclaim_expired(now=future) == 0
    assert queue.reclaim_expired(now=future) == 0
    assert queue.status_for('m') == 'processing'
    assert not queue.ack_fenced(receipt)


def test_experimental_staging_and_legacy_mutation_are_unavailable(tmp_path):
    path = tmp_path / 'queue.sqlite'
    queue = SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    queue.enqueue(message(), message_id='m')
    queue.claim_next()
    with pytest.raises(RuntimeError, match='scheduled_handoff_unavailable'):
        queue.stage_scheduled_handoff(None, None, None)
    with pytest.raises(RuntimeError, match='delivery_wait_unavailable'):
        queue.stage_delivery_wait(None, None, '', '')
    assert not queue.ack('m') and not queue.retry('m')


def test_reused_generation_fails_closed_before_reacquisition(tmp_path, monkeypatch):
    path = tmp_path / 'queue.sqlite'
    queue = SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    queue.enqueue(message(), message_id='m')
    first = queue.claim_next(lease_seconds=1).claim_receipt
    assert queue.reclaim_expired(now=datetime.now(timezone.utc) + timedelta(seconds=2)) == 0
    simulate_old_writer_requeue(queue, 'm')
    monkeypatch.setattr('alphonse.agent_v2.core.messages.claim_fence.new_claim_generation', lambda: first.claim_generation)
    with pytest.raises(ValueError, match='generation_reused'):
        queue.claim_next()
    assert queue.status_for('m') == 'pending'


def test_blank_actor_is_not_claimed_in_fenced_mode(tmp_path):
    queue = SQLiteMessageQueue(tmp_path / 'queue.sqlite', lease_owner='worker', experimental_claim_fencing=True)
    queue.enqueue(message(' '), message_id='m')
    assert queue.claim_next() is None
    assert queue.status_for('m') == 'pending'


def test_fenced_event_claims_remain_unavailable_without_receipt_generation(tmp_path):
    queue = SQLiteMessageQueue(tmp_path / 'queue.sqlite', lease_owner='worker', experimental_claim_fencing=True)
    queue.enqueue(message(), message_id='event:fixture')
    assert queue.claim_next() is None
    assert queue.status_for('event:fixture') == 'pending'


def test_renewal_refuses_live_lease_without_checkpoint_clock_protocol(tmp_path):
    queue = SQLiteMessageQueue(tmp_path / 'queue.sqlite', lease_owner='worker', experimental_claim_fencing=True)
    queue.enqueue(message(), message_id='m')
    claim = queue.claim_next(lease_seconds=10).claim_receipt
    with queue._connect() as conn:
        before = datetime.fromisoformat(conn.execute("SELECT lease_expires_at FROM v2_inbound_messages WHERE message_id='m'").fetchone()[0])
    now = before - timedelta(seconds=5)
    assert not queue.renew_claim_fenced(claim, lease_seconds=60, is_cancelled=lambda: False, clock=lambda: now)
    with queue._connect() as conn:
        row = conn.execute("SELECT lease_expires_at,claim_generation FROM v2_inbound_messages WHERE message_id='m'").fetchone()
    assert datetime.fromisoformat(row['lease_expires_at']) == before
    assert row['claim_generation'] == claim.claim_generation


def test_renewal_rejects_wrong_identity_and_invalid_contract(tmp_path):
    path = tmp_path / 'queue.sqlite'
    queue = SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    queue.enqueue(message(), message_id='m')
    claim = queue.claim_next().claim_receipt
    assert not queue.renew_claim_fenced(replace(claim, user_id='bob'), lease_seconds=60, is_cancelled=lambda: False)
    assert not queue.renew_claim_fenced(replace(claim, lease_owner='other'), lease_seconds=60, is_cancelled=lambda: False)
    assert not queue.renew_claim_fenced(replace(claim, claim_generation='a' * 32), lease_seconds=60, is_cancelled=lambda: False)
    with pytest.raises(ValueError, match='arguments_invalid'):
        queue.renew_claim_fenced(claim, lease_seconds=0, is_cancelled=lambda: False)
    with pytest.raises(ValueError, match='arguments_invalid'):
        queue.renew_claim_fenced(claim, lease_seconds=60, is_cancelled=None)
    legacy = SQLiteMessageQueue(path, lease_owner='worker')
    assert not legacy.renew_claim_fenced(claim, lease_seconds=60, is_cancelled=lambda: False)


def test_unavailable_renewal_never_invokes_callbacks_or_mutates_claim(tmp_path):
    queue = SQLiteMessageQueue(tmp_path / 'queue.sqlite', lease_owner='worker', experimental_claim_fencing=True)
    queue.enqueue(message(), message_id='m')
    claim = queue.claim_next(lease_seconds=10).claim_receipt
    with queue._connect() as conn:
        expiry = datetime.fromisoformat(conn.execute("SELECT lease_expires_at FROM v2_inbound_messages WHERE message_id='m'").fetchone()[0])
    def forbidden():
        pytest.fail('unavailable renewal must not inspect callbacks')
    assert not queue.renew_claim_fenced(claim, lease_seconds=60, is_cancelled=forbidden, clock=forbidden)
    with queue._connect() as conn:
        row = conn.execute("SELECT status,lease_expires_at,claim_generation FROM v2_inbound_messages WHERE message_id='m'").fetchone()
    assert row['status'] == 'processing' and datetime.fromisoformat(row['lease_expires_at']) == expiry
    assert row['claim_generation'] == claim.claim_generation


def test_reclaim_before_renewal_never_revives_old_claim(tmp_path):
    queue = SQLiteMessageQueue(tmp_path / 'queue.sqlite', lease_owner='worker', experimental_claim_fencing=True)
    queue.enqueue(message(), message_id='m')
    claim = queue.claim_next(lease_seconds=10).claim_receipt
    with queue._connect() as conn:
        expiry = datetime.fromisoformat(conn.execute("SELECT lease_expires_at FROM v2_inbound_messages WHERE message_id='m'").fetchone()[0])
    assert queue.reclaim_expired(now=expiry + timedelta(seconds=1)) == 0
    simulate_old_writer_requeue(queue, 'm')
    assert not queue.renew_claim_fenced(claim, lease_seconds=60, is_cancelled=lambda: False,
                                        clock=lambda: expiry - timedelta(seconds=1))
    assert queue.status_for('m') == 'pending'


def test_renewal_and_auto_reclaim_leave_claim_unchanged(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    queue = SQLiteMessageQueue(tmp_path / 'queue.sqlite', lease_owner='worker', experimental_claim_fencing=True)
    queue.enqueue(message(), message_id='m')
    claim = queue.claim_next(lease_seconds=10).claim_receipt
    with queue._connect() as conn:
        expiry = datetime.fromisoformat(conn.execute("SELECT lease_expires_at FROM v2_inbound_messages WHERE message_id='m'").fetchone()[0])
    with ThreadPoolExecutor(max_workers=2) as pool:
        renewing = pool.submit(queue.renew_claim_fenced, claim, lease_seconds=120,
                               is_cancelled=lambda: False, clock=lambda: expiry - timedelta(seconds=5))
        reclaiming = pool.submit(queue.reclaim_expired, now=expiry + timedelta(seconds=1))
        assert not renewing.result(timeout=10)
        assert reclaiming.result(timeout=10) == 0
    assert queue.status_for('m') == 'processing'
    with queue._connect() as conn:
        assert datetime.fromisoformat(conn.execute("SELECT lease_expires_at FROM v2_inbound_messages WHERE message_id='m'").fetchone()[0]) == expiry


def test_renewal_cannot_touch_held_message(tmp_path):
    from alphonse.agent_v2.core.messages.supersession_hold import install_hold_schema
    queue = SQLiteMessageQueue(tmp_path / 'queue.sqlite', lease_owner='worker', experimental_claim_fencing=True)
    queue.enqueue(message(), message_id='m')
    claim = queue.claim_next().claim_receipt
    with queue._connect() as conn:
        install_hold_schema(conn)
        conn.execute('INSERT INTO v2_supersession_holds VALUES(?,?,?,?,?,?,?,?,?)',
                     ('m', 'alice', claim.claim_generation, 'task', 'hierarchical_v3', 1, 2, 'running',
                      datetime.now(timezone.utc).isoformat()))
    assert not queue.renew_claim_fenced(claim, lease_seconds=60, is_cancelled=lambda: False)


def test_opt_in_rejects_nullable_lookalike_generation_column(tmp_path):
    path = tmp_path / 'queue.sqlite'
    legacy = SQLiteMessageQueue(path, lease_owner='worker')
    legacy.enqueue(message(), message_id='m')
    legacy.claim_next()
    with legacy._connect() as conn:
        conn.execute('ALTER TABLE v2_inbound_messages ADD COLUMN claim_generation TEXT')
    with pytest.raises(ValueError, match='schema_invalid'):
        SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    assert legacy.status_for('m') == 'processing'


def test_opt_in_rejects_malformed_existing_generation(tmp_path):
    path = tmp_path / 'queue.sqlite'
    queue = SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    queue.enqueue(message(), message_id='m')
    with queue._connect() as conn:
        conn.execute("UPDATE v2_inbound_messages SET claim_generation='not-a-generation' WHERE message_id='m'")
    with pytest.raises(ValueError, match='generation_invalid'):
        SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    assert queue.status_for('m') == 'pending'


@pytest.mark.parametrize('first_mode', ['legacy', 'fenced'])
def test_claim_modes_do_not_run_concurrently_in_this_binary(tmp_path, first_mode):
    """A pending fenced handoff still belongs to its fenced lifecycle."""
    path = tmp_path / 'queue.sqlite'
    legacy = SQLiteMessageQueue(path, lease_owner='legacy-worker')
    fenced = SQLiteMessageQueue(path, lease_owner='fenced-worker', experimental_claim_fencing=True)
    legacy.enqueue(message(), message_id='first')
    legacy.enqueue(message(), message_id='second')
    first, other = (legacy, fenced) if first_mode == 'legacy' else (fenced, legacy)
    claim = first.claim_next()
    assert claim is not None
    assert other.claim_next() is None
    if first_mode == 'fenced':
        with fenced._connect() as conn:
            conn.execute('INSERT INTO v2_scheduled_handoffs(message_id,task_id,owner_id) VALUES(?,?,?)',
                         (claim.message_id, 'task', 'alice'))
        assert not fenced.release_scheduled_handoff_fenced(claim.claim_receipt)
        # An unverified handoff remains held by its current fenced claim.
        assert legacy.claim_next() is None
        assert not fenced.ack_fenced(claim.claim_receipt)
        assert other.claim_next() is None
        return
    else:
        assert legacy.ack(claim.message_id)
    assert other.claim_next() is not None


def test_claim_mode_race_allows_only_one_protocol_to_acquire(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    path = tmp_path / 'queue.sqlite'
    legacy = SQLiteMessageQueue(path, lease_owner='legacy-worker')
    fenced = SQLiteMessageQueue(path, lease_owner='fenced-worker', experimental_claim_fencing=True)
    legacy.enqueue(message(), message_id='first')
    legacy.enqueue(message(), message_id='second')
    barrier = Barrier(2)
    def acquire(queue):
        barrier.wait(timeout=5)
        return queue.claim_next()
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(acquire, (legacy, fenced)))
    assert sum(claim is not None for claim in claims) == 1
    with legacy._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM v2_inbound_messages WHERE status='processing'").fetchone()[0] == 1


def test_claim_mode_diagnostic_exposes_no_task_data_and_fails_closed_without_disposition(tmp_path):
    path = tmp_path / 'queue.sqlite'
    legacy = SQLiteMessageQueue(path, lease_owner='legacy-worker')
    fenced = SQLiteMessageQueue(path, lease_owner='fenced-worker', experimental_claim_fencing=True)
    assert legacy.claim_mode_blocked() is False
    legacy.enqueue(message(), message_id='secret-id')
    claim = fenced.claim_next()
    assert legacy.claim_mode_blocked() is True
    assert fenced.claim_mode_blocked() is False
    assert not fenced.ack_fenced(claim.claim_receipt)
    assert legacy.claim_mode_blocked() is True


def test_claim_mode_diagnostic_does_not_hide_storage_error(tmp_path):
    import sqlite3
    queue = SQLiteMessageQueue(tmp_path / 'queue.sqlite')
    with queue._connect() as conn:
        conn.execute('DROP TABLE v2_inbound_messages')
    with pytest.raises(sqlite3.Error):
        queue.claim_mode_blocked()
    with queue._connect() as conn:
        conn.execute('CREATE TABLE v2_inbound_messages(message_id TEXT)')
    with pytest.raises(sqlite3.Error):
        queue.claim_mode_blocked()


def test_claim_mode_rejects_generation_corrupted_after_migration(tmp_path):
    path = tmp_path / 'shared.sqlite'
    legacy = SQLiteMessageQueue(path, lease_owner='legacy')
    fenced = SQLiteMessageQueue(path, lease_owner='fenced', experimental_claim_fencing=True)
    legacy.enqueue(message(), message_id='first')
    with legacy._connect() as conn:
        conn.execute("UPDATE v2_inbound_messages SET claim_generation='not-a-generation' WHERE message_id='first'")
    assert fenced.claim_mode_blocked() is True
    assert legacy.claim_mode_blocked() is True
    assert fenced.claim_next() is None
    assert legacy.claim_next() is None
