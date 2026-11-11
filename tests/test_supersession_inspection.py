"""Read-only, unregistered supersession classification against disposable SQLite."""
from datetime import datetime, timedelta, timezone
import json
import sqlite3

import pytest

from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages.claim_fence import (
    ClaimFenceUnavailable, ClaimReceipt, ClaimSuperseded, new_claim_generation,
)
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.messages.supersession_inspection import inspect_supersession
from alphonse.agent_v2.core.questions import SQLiteQuestionStore


@pytest.fixture
def f(tmp_path):
    path = tmp_path / 'runtime.sqlite3'
    queue = SQLiteMessageQueue(path, lease_owner='worker')
    questions = SQLiteQuestionStore(path)
    now = datetime.now(timezone.utc)
    queue.enqueue(CoreMessage(timestamp=now, prompt='synthetic', user='alice'), message_id='message')
    queue.claim_next(lease_owner='worker', lease_seconds=60)
    task = TaskState(task_id='task', message_id='message', user='alice', goal='synthetic',
                     intelligence_engine='hierarchical_v3', intelligence_schema_version=3)
    version = questions.save_task_checkpoint(task, status='running')
    generation = new_claim_generation()
    with queue._connect() as conn:
        # Simulates the future migration; production currently lacks this column.
        conn.execute("ALTER TABLE v2_inbound_messages ADD COLUMN claim_generation TEXT NOT NULL DEFAULT ''")
        conn.execute('UPDATE v2_inbound_messages SET claim_generation=? WHERE message_id=?', (generation, 'message'))
    receipt = ClaimReceipt('message', 'alice', 'worker', generation)
    return path, queue, questions, receipt, version, now


def inspect(f, *, actor='alice', task_id='task', engine='hierarchical_v3', version=None, clock=None, claim=None):
    path, _, _, receipt, expected, now = f
    with sqlite3.connect(path) as conn:
        return inspect_supersession(conn, claim or receipt, task_id=task_id, actor=actor, engine=engine,
                                    observed_version=expected if version is None else version,
                                    clock=clock or (lambda: now))


def test_current_and_newer_active_do_not_modify_queue_or_checkpoint(f):
    _, queue, questions, _, version, _ = f
    assert inspect(f).classification == 'current'
    task = questions.load_task_checkpoint('task')
    task.metadata['new_steering'] = 'retained synthetic input'
    questions.save_task_checkpoint(task, status='running')
    observed = inspect(f)
    assert observed.classification == 'newer_active' and observed.checkpoint_version == version + 1
    with queue._connect() as conn:
        row = conn.execute("SELECT status,lease_owner,claim_generation FROM v2_inbound_messages WHERE message_id='message'").fetchone()
    assert tuple(row) == ('processing', 'worker', f[3].claim_generation)
    assert questions.load_task_checkpoint('task').metadata['new_steering'] == 'retained synthetic input'


@pytest.mark.parametrize(('status', 'classification'), [
    ('cancelled', 'cancelled'), ('waiting_user', 'waiting_user'), ('done', 'terminal'),
    ('failed', 'terminal'), ('queued', 'queued'),
])
def test_authoritative_status_is_observed_but_not_acknowledged(f, status, classification):
    _, queue, _, _, _, _ = f
    with queue._connect() as conn:
        conn.execute("UPDATE v2_task_checkpoints SET status=? WHERE task_id='task'", (status,))
    assert inspect(f).classification == classification
    with queue._connect() as conn:
        assert conn.execute("SELECT status FROM v2_inbound_messages WHERE message_id='message'").fetchone()[0] == 'processing'
        assert conn.execute("SELECT status FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0] == status


def test_missing_wrong_actor_engine_and_older_state_are_opaque(f):
    assert inspect(f, actor='bob').classification == 'unavailable'
    assert inspect(f, task_id='missing').classification == 'unavailable'
    assert inspect(f, engine='tactical_v2').classification == 'unavailable'
    assert inspect(f, version=f[4] + 1).classification == 'unavailable'


@pytest.mark.parametrize('change', ['message', 'owner', 'schema', 'metadata_schema', 'malformed'])
def test_corrupt_or_unbound_checkpoint_never_produces_a_disposition(f, change):
    _, queue, _, _, _, _ = f
    with queue._connect() as conn:
        raw = conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0]
        task = json.loads(raw)
        if change == 'message':
            task['message_id'] = 'other-message'
        elif change == 'owner':
            task['user'] = 'bob'
        elif change == 'schema':
            task['intelligence_schema_version'] = 999
        elif change == 'metadata_schema':
            task['metadata']['intelligence_schema_version'] = True
        raw = '{invalid' if change == 'malformed' else json.dumps(task)
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (raw,))
    assert inspect(f).classification == 'unavailable'
    with queue._connect() as conn:
        assert conn.execute("SELECT status FROM v2_inbound_messages WHERE message_id='message'").fetchone()[0] == 'processing'


def test_stale_or_expired_claim_cannot_inspect(f):
    stale = ClaimReceipt('message', 'alice', 'worker', new_claim_generation())
    with pytest.raises(ClaimSuperseded):
        inspect(f, claim=stale)
    with pytest.raises(ClaimSuperseded):
        inspect(f, clock=lambda: f[5] + timedelta(days=1))


def test_expiry_during_transaction_prevents_return(f):
    moments = iter((f[5], f[5] + timedelta(days=1)))
    with pytest.raises(ClaimSuperseded):
        inspect(f, clock=lambda: next(moments))


def test_missing_schema_and_in_memory_db_fail_closed(f):
    conn = sqlite3.connect(':memory:')
    try:
        with pytest.raises(ClaimFenceUnavailable, match='shared_file'):
            inspect_supersession(conn, f[3], task_id='task', actor='alice', engine='hierarchical_v3',
                                 observed_version=1, clock=lambda: f[5])
    finally:
        conn.close()
    path = f[0].parent / 'empty.sqlite'
    with sqlite3.connect(path) as conn:
        with pytest.raises(ClaimFenceUnavailable):
            inspect_supersession(conn, f[3], task_id='task', actor='alice', engine='hierarchical_v3',
                                 observed_version=1, clock=lambda: f[5])
