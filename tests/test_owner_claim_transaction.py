"""Owner ingress authorization and same-transaction fencing, using disposable data."""
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimSuperseded
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def owner_claim(tmp_path):
    path = tmp_path / 'fixture.db'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Fixture', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
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
