"""Experimental boundary only; no user publication or delivery is tested."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.io.checkpoint_claim_fence import experimental_checkpoint_claim_transaction
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimSuperseded
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore


@pytest.fixture
def fixture(tmp_path):
    path = tmp_path / 'runtime.sqlite'
    queue = SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    questions = SQLiteQuestionStore(path)
    now = datetime.now(timezone.utc)
    queue.enqueue(CoreMessage(timestamp=now, prompt='synthetic', user='alice'), message_id='message')
    claim = queue.claim_next(lease_seconds=60).claim_receipt
    task = TaskState(task_id='task', message_id='message', user='alice', goal='synthetic',
                     intelligence_engine='single_loop_experimental', intelligence_schema_version=1)
    version = questions.save_task_checkpoint(task, status='running')
    with queue._connect() as conn:
        conn.execute('CREATE TABLE fixture_writes (value TEXT)')
    return queue, questions, claim, version, now


def boundary(conn, fixture, *, actor='alice', version=None, clock=None):
    _, _, claim, expected, now = fixture
    return experimental_checkpoint_claim_transaction(conn, claim, task_id='task', actor=actor,
        expected_version=expected if version is None else version, clock=clock or (lambda: now))


def test_current_claim_and_checkpoint_allow_same_store_work(fixture):
    queue, _, _, _, _ = fixture
    with queue._connect() as conn:
        with boundary(conn, fixture):
            conn.execute("INSERT INTO fixture_writes VALUES ('synthetic')")
        assert conn.execute('SELECT count(*) FROM fixture_writes').fetchone()[0] == 1


def test_advanced_checkpoint_rejects_live_claim_before_work(fixture):
    queue, questions, _, _, _ = fixture
    with questions._connect() as conn:
        conn.execute("UPDATE v2_task_checkpoints SET version=version+1 WHERE task_id='task'")
    with queue._connect() as conn:
        with pytest.raises(ClaimFenceUnavailable, match='checkpoint_claim_unavailable'):
            with boundary(conn, fixture):
                conn.execute("INSERT INTO fixture_writes VALUES ('stale')")
        assert conn.execute('SELECT count(*) FROM fixture_writes').fetchone()[0] == 0


@pytest.mark.parametrize('change', ['cancelled', 'message', 'owner', 'schema', 'malformed'])
def test_invalid_checkpoint_never_admits_work(fixture, change):
    queue, _, _, _, _ = fixture
    with queue._connect() as conn:
        raw = conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0]
        task = json.loads(raw)
        if change == 'cancelled':
            conn.execute("UPDATE v2_task_checkpoints SET status='cancelled' WHERE task_id='task'")
        else:
            if change == 'message': task['message_id'] = 'other'
            if change == 'owner': task['user'] = 'bob'
            if change == 'schema': task['intelligence_schema_version'] = True
            encoded = '{invalid' if change == 'malformed' else json.dumps(task)
            conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (encoded,))
        conn.commit()
        with pytest.raises(ClaimFenceUnavailable, match='checkpoint_claim_unavailable'):
            with boundary(conn, fixture):
                pytest.fail('invalid checkpoint reached callback')


def test_mutating_checkpoint_or_expiring_claim_rolls_back_work(fixture):
    queue, _, _, _, now = fixture
    with queue._connect() as conn:
        # Both mutations must roll back, but the independent expiry scenario
        # need only run once rather than once for each SQL mutation.
        for sql in ("UPDATE v2_task_checkpoints SET status='cancelled' WHERE task_id='task'",
                    "UPDATE v2_task_checkpoints SET project_id='other' WHERE task_id='task'"):
            with pytest.raises(ClaimFenceUnavailable, match='checkpoint_changed_by_caller'):
                with boundary(conn, fixture):
                    conn.execute("INSERT INTO fixture_writes VALUES ('must rollback')")
                    conn.execute(sql)
            assert conn.execute('SELECT count(*) FROM fixture_writes').fetchone()[0] == 0
            assert conn.execute("SELECT status FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0] == 'running'
        moments = iter((now, now + timedelta(days=1)))
        with pytest.raises(ClaimSuperseded):
            with boundary(conn, fixture, clock=lambda: next(moments)):
                conn.execute("INSERT INTO fixture_writes VALUES ('must rollback')")
        assert conn.execute('SELECT count(*) FROM fixture_writes').fetchone()[0] == 0


def test_other_queue_actor_is_not_interpreted_as_delegated_owner(fixture):
    queue, _, _, _, _ = fixture
    with queue._connect() as conn:
        with pytest.raises(ClaimFenceUnavailable, match='checkpoint_claim_unavailable'):
            with boundary(conn, fixture, actor='bob'):
                pytest.fail('unsupported delegated task reached callback')
