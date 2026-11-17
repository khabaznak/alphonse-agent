"""Owner ingress authorization and same-transaction fencing, using disposable data."""
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimSuperseded
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def owner_claim(tmp_path):
    path = tmp_path / 'fixture.db'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Fixture', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    SQLiteOutboundStore(path)
    from alphonse.agent_v2.conversations import SQLiteConversationStore
    SQLiteConversationStore(path)
    message_id = queue.enqueue_trusted_local_owner(prompt='fixture', peer_uid=os.getuid())
    acquired = queue.claim_next()
    task = TaskState.from_message(acquired.message, message_id=message_id)
    task.task_id = message_id
    with queue._connect() as conn:
        conn.execute('CREATE TABLE fixture_writes (value TEXT NOT NULL)')
    return queue, acquired.claim_receipt, task, owner


def test_owner_authorized_transaction_commits_and_verifier_uses_it(owner_claim):
    queue, claim, task, _ = owner_claim
    with queue.trusted_local_owner_claim_transaction(claim, task) as conn:
        conn.execute("INSERT INTO fixture_writes VALUES ('accepted')")
    assert queue.verify_trusted_local_owner_claim(claim, task)
    with queue._connect() as conn:
        assert [row[0] for row in conn.execute('SELECT value FROM fixture_writes')] == ['accepted']


def test_invalid_and_expired_claims_cannot_write(owner_claim):
    queue, claim, task, _ = owner_claim
    for stale, clock in ((replace(claim, claim_generation='0' * 32), None),
                         (claim, lambda: datetime.now(timezone.utc) + timedelta(days=1))):
        with pytest.raises(ClaimSuperseded):
            with queue.trusted_local_owner_claim_transaction(stale, task, clock=clock) as conn:
                conn.execute("INSERT INTO fixture_writes VALUES ('bad')")
    with queue._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM fixture_writes').fetchone()[0] == 0


def test_revocation_during_transaction_rolls_back_caller_writes(owner_claim):
    queue, claim, task, owner = owner_claim
    with pytest.raises(ClaimFenceUnavailable):
        with queue.trusted_local_owner_claim_transaction(claim, task) as conn:
            conn.execute("INSERT INTO fixture_writes VALUES ('bad')")
            conn.execute('UPDATE v2_users SET is_active=0 WHERE user_id=?', (owner.user_id,))
    with queue._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM fixture_writes').fetchone()[0] == 0
        assert conn.execute('SELECT is_active FROM v2_users WHERE user_id=?', (owner.user_id,)).fetchone()[0] == 1


def test_origin_mutation_or_task_substitution_rejected(owner_claim):
    queue, claim, task, _ = owner_claim
    with pytest.raises(ClaimFenceUnavailable):
        with queue.trusted_local_owner_claim_transaction(claim, replace(task, goal='other')):
            pass
    with pytest.raises(ClaimFenceUnavailable):
        with queue.trusted_local_owner_claim_transaction(claim, task) as conn:
            conn.execute("INSERT INTO fixture_writes VALUES ('bad')")
            conn.execute('DELETE FROM v2_trusted_owner_ingress WHERE message_id=?', (claim.message_id,))
    assert queue.verify_trusted_local_owner_claim(claim, task)
    with queue._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM fixture_writes').fetchone()[0] == 0


def test_expiry_before_commit_rolls_back_caller_writes(owner_claim):
    queue, claim, task, _ = owner_claim
    now = datetime.now(timezone.utc)
    samples = iter((now, now + timedelta(days=1)))
    with pytest.raises(ClaimSuperseded):
        with queue.trusted_local_owner_claim_transaction(claim, task, clock=lambda: next(samples)) as conn:
            conn.execute("INSERT INTO fixture_writes VALUES ('bad')")
    with queue._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM fixture_writes').fetchone()[0] == 0


@pytest.mark.parametrize('final_delta, error', [(timedelta(days=1), ClaimSuperseded),
                                                (timedelta(seconds=-1), ClaimFenceUnavailable)])
def test_finalizer_clock_failure_rolls_back_claim_and_writes(owner_claim, final_delta, error):
    queue, claim, task, _ = owner_claim
    now = datetime.now(timezone.utc)
    samples = iter((now, now, now + final_delta))
    def delete_claim(conn):
        conn.execute('DELETE FROM v2_inbound_messages WHERE message_id=?', (claim.message_id,))
    with pytest.raises(error):
        with queue.trusted_local_owner_claim_transaction(claim, task, clock=lambda: next(samples),
                                                         _finalize=delete_claim) as conn:
            conn.execute("INSERT INTO fixture_writes VALUES ('bad')")
    with queue._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM fixture_writes').fetchone()[0] == 0
        assert conn.execute('SELECT 1 FROM v2_inbound_messages WHERE message_id=?', (claim.message_id,)).fetchone()


def test_abandon_rejects_requeued_processing_message(owner_claim):
    queue, claim, task, _ = owner_claim
    SQLiteQuestionStore(queue.db_path)
    with queue._connect() as conn:
        # Unsafe old-writer fixture: pending status must not erase evidence of
        # a previous processing attempt.
        conn.execute("UPDATE v2_inbound_messages SET status='pending' WHERE message_id=?", (claim.message_id,))
    with pytest.raises(ValueError, match='owner_abandon_unavailable'):
        queue.abandon_trusted_owner_message(claim.message_id)
    with queue._connect() as conn:
        assert conn.execute('SELECT 1 FROM v2_inbound_messages WHERE message_id=?', (claim.message_id,)).fetchone()


def test_abandon_pristine_pending_message_is_explicit_and_idempotent(owner_claim):
    queue, _, _, _ = owner_claim
    SQLiteQuestionStore(queue.db_path)
    message_id = queue.enqueue_trusted_local_owner(prompt='not started', peer_uid=os.getuid())
    expected = {'status': 'abandoned', 'delivery_status': 'no_pickup_recorded'}
    assert queue.abandon_trusted_owner_message(message_id) == expected
    assert queue.abandon_trusted_owner_message(message_id) == expected


def test_abandon_eligibility_is_read_only_and_action_rechecks_after_state_changes(owner_claim):
    queue, _, _, _ = owner_claim
    SQLiteQuestionStore(queue.db_path)
    message_id = queue.enqueue_trusted_local_owner(prompt='not started', peer_uid=os.getuid())
    with queue._connect() as conn:
        before = list(conn.iterdump())
    assert queue.can_abandon_trusted_owner_message(message_id) is True
    with queue._connect() as conn:
        assert list(conn.iterdump()) == before
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='v2_owner_abandoned'").fetchone() is None
        conn.execute('''INSERT INTO v2_outbox(outbox_message_id,integration_id,provider_key,channel_target,
            message,kind,task_id,created_at,updated_at)
            VALUES ('later','desktop','desktop','fixture','synthetic','response',?,'now','now')''', (message_id,))
    assert queue.can_abandon_trusted_owner_message(message_id) is False
    with pytest.raises(ValueError, match='owner_abandon_unavailable'):
        queue.abandon_trusted_owner_message(message_id)
    with queue._connect() as conn:
        assert conn.execute('SELECT 1 FROM v2_inbound_messages WHERE message_id=?', (message_id,)).fetchone()


def test_abandon_batch_uses_one_read_only_connection_without_store_initialization(owner_claim, monkeypatch):
    from alphonse.agent_v2.core.messages import sqlite_queue as module
    queue, _, _, _ = owner_claim
    SQLiteQuestionStore(queue.db_path)
    first = queue.enqueue_trusted_local_owner(prompt='first', peer_uid=os.getuid())
    second = queue.enqueue_trusted_local_owner(prompt='second', peer_uid=os.getuid())
    original_connect = module.sqlite3.connect
    calls = []
    def connect(*args, **kwargs):
        calls.append((args, kwargs))
        return original_connect(*args, **kwargs)
    monkeypatch.setattr(module.sqlite3, 'connect', connect)
    monkeypatch.setattr(queue, '_connect', lambda: pytest.fail('preview must not initialize the writable store'))
    result = queue.can_abandon_trusted_owner_messages([first, second])
    assert result == {first: True, second: True}
    assert len(calls) == 1 and calls[0][0][0].endswith('?mode=ro')
    assert calls[0][1]['uri'] is True and calls[0][1]['timeout'] <= .1


def test_abandon_batch_rejects_oversize_and_fails_closed_on_budget_interruption(owner_claim, monkeypatch):
    from alphonse.agent_v2.core.messages import sqlite_queue as module
    queue, _, _, _ = owner_claim
    SQLiteQuestionStore(queue.db_path)
    message_id = queue.enqueue_trusted_local_owner(prompt='not started', peer_uid=os.getuid())
    assert queue.can_abandon_trusted_owner_messages([message_id] * 51) == {}
    assert queue.can_abandon_trusted_owner_message(message_id) is True
    samples = iter([0.0] + [10.0] * 10000)
    with monkeypatch.context() as patch:
        patch.setattr(module.time, 'monotonic', lambda: next(samples, 10.0))
        assert queue.can_abandon_trusted_owner_messages([message_id]) == {message_id: False}
    assert queue.can_abandon_trusted_owner_message(message_id) is True


def test_abandon_rejects_tampered_origin_and_revoked_owner(owner_claim):
    queue, _, _, owner = owner_claim
    SQLiteQuestionStore(queue.db_path)
    message_id = queue.enqueue_trusted_local_owner(prompt='not started', peer_uid=os.getuid())
    with queue._connect() as conn:
        conn.execute("UPDATE v2_inbound_messages SET prompt='changed' WHERE message_id=?", (message_id,))
    with pytest.raises(ValueError, match='owner_abandon_unavailable'):
        queue.abandon_trusted_owner_message(message_id)
    with queue._connect() as conn:
        conn.execute("UPDATE v2_inbound_messages SET prompt='not started' WHERE message_id=?", (message_id,))
        conn.execute('UPDATE v2_users SET is_active=0 WHERE user_id=?', (owner.user_id,))
    with pytest.raises(ValueError, match='owner_abandon_unavailable'):
        queue.abandon_trusted_owner_message(message_id)
    with queue._connect() as conn:
        assert conn.execute('SELECT 1 FROM v2_inbound_messages WHERE message_id=?', (message_id,)).fetchone()
