from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.messages import MessageSelector
from alphonse.agent_v2.core.messages import SQLiteMessageQueue


def test_sqlite_queue_persists_messages_and_acknowledges_processing(tmp_path: Path) -> None:
    path = tmp_path / "messages.sqlite3"
    queue = SQLiteMessageQueue(path)
    queued = queue.enqueue(
        CoreMessage(
            timestamp=datetime.now(timezone.utc),
            prompt="hello",
            user="alex",
            correlation_id="corr-1",
            metadata={"channel": {"integration_id": "telegram-home"}},
        )
    )

    restarted = SQLiteMessageQueue(path)
    claimed = restarted.dequeue(MessageSelector(user="alex"))

    assert claimed is not None
    assert claimed.message.prompt == "hello"
    assert claimed.message.metadata["channel"]["integration_id"] == "telegram-home"
    assert restarted.ack(claimed.message_id) is True
    assert restarted.size() == 0
    assert queued.message_id == claimed.message_id


def test_sqlite_queue_reclaims_expired_processing_messages(tmp_path: Path) -> None:
    path = tmp_path / "messages.sqlite3"
    queue = SQLiteMessageQueue(path, lease_owner="worker-1")
    queued = queue.enqueue(
        CoreMessage(timestamp=datetime.now(timezone.utc), prompt="recover", user="alex")
    )
    assert queue.claim_next(lease_seconds=1) is not None

    restarted = SQLiteMessageQueue(path, lease_owner="worker-2")
    assert restarted.reclaim_expired(now=datetime.now(timezone.utc) + timedelta(seconds=2)) == 1
    recovered = restarted.dequeue()

    assert recovered is not None
    assert recovered.message_id == queued.message_id


def test_sqlite_queue_does_not_reclaim_active_processing_lease(tmp_path: Path) -> None:
    path = tmp_path / "messages.sqlite3"
    queue = SQLiteMessageQueue(path, lease_owner="worker-1")
    queue.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc), prompt="active", user="alex"))
    assert queue.claim_next(lease_seconds=60) is not None

    restarted = SQLiteMessageQueue(path, lease_owner="worker-2")
    assert restarted.reclaim_expired(now=datetime.now(timezone.utc)) == 0
    assert restarted.dequeue() is None


def test_sqlite_queue_does_not_return_the_callers_owned_processing_message(tmp_path: Path) -> None:
    queue = SQLiteMessageQueue(tmp_path / "messages.sqlite3", lease_owner="worker-1")
    queue.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc), prompt="active", user="alex"))
    assert queue.claim_next(lease_seconds=60) is not None

    assert queue.dequeue() is None
    assert queue.claim_next(include_owned=True) is not None


def test_sqlite_queue_reclaims_legacy_processing_rows_without_a_lease(tmp_path: Path) -> None:
    queue = SQLiteMessageQueue(tmp_path / "messages.sqlite3")
    queued = queue.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc), prompt="legacy", user="alex"))
    with queue._connect() as conn:
        conn.execute(
            "UPDATE v2_inbound_messages SET status = 'processing' WHERE message_id = ?",
            (queued.message_id,),
        )

    assert queue.reclaim_expired() == 1
    assert queue.peek() is not None


def test_sqlite_queue_marks_inbound_message_failed_after_retry_limit(tmp_path: Path) -> None:
    queue = SQLiteMessageQueue(tmp_path / "messages.sqlite3", lease_owner="worker-1")
    queued = queue.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc), prompt="fails", user="alex"))
    assert queue.claim_next() is not None

    assert queue.retry(queued.message_id, error="provider_failed", max_attempts=1) is True
    assert queue.status_counts()["failed"] == 1
    assert queue.peek() is None
