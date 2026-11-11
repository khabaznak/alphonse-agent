"""Disposable reproductions of mixed-version risks; not rollout acceptance tests."""
from datetime import datetime, timezone
import sqlite3

from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.daemon import _claim_single_instance_lock


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
