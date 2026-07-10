from __future__ import annotations

from datetime import datetime, timezone
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


def test_sqlite_queue_requeues_abandoned_processing_messages(tmp_path: Path) -> None:
    path = tmp_path / "messages.sqlite3"
    queue = SQLiteMessageQueue(path)
    queued = queue.enqueue(
        CoreMessage(timestamp=datetime.now(timezone.utc), prompt="recover", user="alex")
    )
    assert queue.dequeue() is not None

    restarted = SQLiteMessageQueue(path)
    recovered = restarted.dequeue()

    assert recovered is not None
    assert recovered.message_id == queued.message_id
