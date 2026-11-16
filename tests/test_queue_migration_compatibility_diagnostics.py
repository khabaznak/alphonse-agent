"""Disposable reproductions of mixed-version risks; not rollout acceptance tests."""
from datetime import datetime, timezone
import hashlib
import json
import os
import sqlite3

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.daemon import _claim_single_instance_lock
from alphonse.agent_v2.users import V2UserStore


def test_old_sql_can_claim_after_the_migration_drain_check(tmp_path):
    path = tmp_path / 'queue.sqlite'
    queue = SQLiteMessageQueue(path, lease_owner='new', experimental_claim_fencing=True)
    queue.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc), prompt='synthetic', user='alice'), message_id='m')
    # Representative old-worker SQL predates the generation column. It can run
    # after the one-time migration check and create a new unfenced processing row.
    with sqlite3.connect(path) as old:
        old.execute("UPDATE v2_inbound_messages SET status='processing',lease_owner='old' WHERE message_id='m'")
    with sqlite3.connect(path) as check:
        assert check.execute("SELECT status,claim_generation FROM v2_inbound_messages WHERE message_id='m'").fetchone() == ('processing', '')


def test_old_sql_can_ack_new_generation_with_a_reused_owner(tmp_path):
    path = tmp_path / 'queue.sqlite'
    queue = SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    queue.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc), prompt='synthetic', user='alice'), message_id='m')
    receipt = queue.claim_next().claim_receipt
    # The owner matches, but this old-style delete never checked the generation.
    with sqlite3.connect(path) as old:
        removed = old.execute("DELETE FROM v2_inbound_messages WHERE message_id=? AND status='processing' AND lease_owner=?", ('m', 'worker')).rowcount
    assert removed == 1
    assert not queue.ack_fenced(receipt)


def test_socket_instance_lock_does_not_fence_database_writers(tmp_path, monkeypatch):
    """The daemon lock is advisory, socket-path scoped, and not a DB fence."""
    path = tmp_path / 'shared-queue.sqlite'
    queue = SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    queue.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc), prompt='synthetic', user='alice'), message_id='m')
    receipt = queue.claim_next().claim_receipt

    # Both processes can target the same configured DB while choosing distinct
    # IPC socket paths, and each acquires a different advisory flock.
    monkeypatch.setenv('ALPHONSE_V2_DB_PATH', str(path))
    monkeypatch.setenv('ALPHONSE_V2_SOCKET_PATH', str(tmp_path / 'first.sock'))
    first = _claim_single_instance_lock('first')
    try:
        monkeypatch.setenv('ALPHONSE_V2_SOCKET_PATH', str(tmp_path / 'second.sock'))
        second = _claim_single_instance_lock('second')
        try:
            # A pre-generation queue writer never participates in either lock.
            with sqlite3.connect(path) as old:
                removed = old.execute(
                    "DELETE FROM v2_inbound_messages WHERE message_id=? AND status='processing' AND lease_owner=?",
                    ('m', 'worker'),
                ).rowcount
            assert removed == 1
            assert not queue.ack_fenced(receipt)
        finally:
            second.close()
    finally:
        first.close()


def test_five_column_owner_origin_migrates_without_rewriting_legacy_claim(tmp_path):
    """Actual old origin rows remain claimable; old positional inserts fail closed."""
    path = tmp_path / 'legacy-owner.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    SQLiteConversationStore(path)
    queue = SQLiteMessageQueue(path, lease_owner='old', experimental_claim_fencing=True)
    message_id = 'a' * 32
    metadata = {'intelligence_engine': 'single_loop_experimental',
                'intelligence_schema_version': 1, 'routing_disposition': 'pdca_task'}
    queue.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc), prompt='Synthetic prior request',
                              user=owner.user_id, metadata=metadata), message_id=message_id)
    with queue._connect() as conn:
        row = conn.execute('SELECT * FROM v2_inbound_messages WHERE message_id=?', (message_id,)).fetchone()
        payload = [row[key] for key in ('message_id', 'timestamp', 'prompt', 'user_id', 'project_id',
                                        'tag', 'correlation_id', 'memory_session_id', 'metadata_json')]
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        conn.execute('''CREATE TABLE v2_trusted_owner_ingress (
            message_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, peer_uid INTEGER NOT NULL,
            payload_sha256 TEXT NOT NULL, created_at TEXT NOT NULL) STRICT''')
        conn.execute('INSERT INTO v2_trusted_owner_ingress VALUES (?,?,?,?,?)',
                     (message_id, owner.user_id, os.getuid(), digest, row['timestamp']))
    migrated = SQLiteMessageQueue(path, lease_owner='new', experimental_claim_fencing=True)
    with migrated._connect() as conn:
        origin = conn.execute('SELECT * FROM v2_trusted_owner_ingress WHERE message_id=?', (message_id,)).fetchone()
        assert (origin['payload_sha256'], origin['history_version'], origin['history_event_id']) == (digest, 0, '')
        with pytest.raises(sqlite3.OperationalError):
            conn.execute('INSERT INTO v2_trusted_owner_ingress VALUES (?,?,?,?,?)',
                         ('b' * 32, owner.user_id, os.getuid(), digest, row['timestamp']))
    acquired = migrated.claim_next()
    task = TaskState.from_message(acquired.message, message_id=message_id)
    task.task_id = message_id
    assert migrated.verify_trusted_local_owner_claim(acquired.claim_receipt, task)
