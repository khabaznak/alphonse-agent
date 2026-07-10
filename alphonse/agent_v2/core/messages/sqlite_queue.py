"""Durable inbound message queue for the v2 daemon."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.messages.queue import MessageSelector
from alphonse.agent_v2.core.messages.queue import QueuedMessage


class SQLiteMessageQueue:
    """SQLite-backed queue with processing recovery and explicit acknowledgement."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory_connection: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._memory_connection = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_connection.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls) -> "SQLiteMessageQueue":
        return cls(
            os.getenv("ALPHONSE_V2_MESSAGES_DB_PATH")
            or os.getenv("ALPHONSE_V2_DB_PATH")
            or str(Path.home() / ".alphonse" / "v2-messages.sqlite3")
        )

    def enqueue(self, message: CoreMessage, *, message_id: str | None = None) -> QueuedMessage:
        queued = QueuedMessage(message=message, message_id=str(message_id or QueuedMessage(message).message_id))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_inbound_messages (
                  message_id, sequence, timestamp, prompt, user_id, project_id, tag,
                  correlation_id, metadata_json, queued_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                ON CONFLICT(message_id) DO NOTHING
                """,
                (
                    queued.message_id,
                    time.time_ns(),
                    queued.message.timestamp.isoformat(),
                    queued.message.prompt,
                    queued.message.user,
                    queued.message.project_id,
                    queued.message.tag,
                    queued.message.correlation_id,
                    json.dumps(queued.message.metadata, sort_keys=True),
                    queued.queued_at.isoformat(),
                ),
            )
        return queued

    def peek(self, selector: MessageSelector | None = None) -> QueuedMessage | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM v2_inbound_messages WHERE status = 'pending' {_where(selector)} ORDER BY sequence LIMIT 1",
                _values(selector),
            ).fetchone()
        return _row_to_message(row) if row is not None else None

    def dequeue(self, selector: MessageSelector | None = None) -> QueuedMessage | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM v2_inbound_messages WHERE status = 'pending' {_where(selector)} ORDER BY sequence LIMIT 1",
                _values(selector),
            ).fetchone()
            if row is None:
                return None
            cursor = conn.execute(
                "UPDATE v2_inbound_messages SET status = 'processing', processing_at = ? WHERE message_id = ? AND status = 'pending'",
                (_now_iso(), str(row["message_id"])),
            )
            if cursor.rowcount != 1:
                return None
        return _row_to_message(row)

    def ack(self, message_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM v2_inbound_messages WHERE message_id = ? AND status = 'processing'",
                (str(message_id or "").strip(),),
            )
            return cursor.rowcount == 1

    def requeue(self, message_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE v2_inbound_messages SET status = 'pending', processing_at = '' WHERE message_id = ? AND status = 'processing'",
                (str(message_id or "").strip(),),
            )
            return cursor.rowcount == 1

    def size(self, selector: MessageSelector | None = None) -> int:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS count FROM v2_inbound_messages WHERE status = 'pending' {_where(selector)}",
                _values(selector),
            ).fetchone()
        return int(row["count"] if row is not None else 0)

    def _connect(self) -> sqlite3.Connection:
        if self._memory_connection is not None:
            return _ConnectionProxy(self._memory_connection)
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS v2_inbound_messages (
                  message_id TEXT PRIMARY KEY,
                  sequence INTEGER NOT NULL,
                  timestamp TEXT NOT NULL,
                  prompt TEXT NOT NULL,
                  user_id TEXT NOT NULL,
                  project_id TEXT NOT NULL DEFAULT '',
                  tag TEXT NOT NULL DEFAULT '',
                  correlation_id TEXT NOT NULL DEFAULT '',
                  metadata_json TEXT NOT NULL DEFAULT '{}',
                  queued_at TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'pending',
                  processing_at TEXT NOT NULL DEFAULT ''
                ) STRICT;
                """
            )
            conn.execute("UPDATE v2_inbound_messages SET status = 'pending', processing_at = '' WHERE status = 'processing'")


class _ConnectionProxy:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def __enter__(self) -> sqlite3.Connection:
        return self.connection

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        return False


def _where(selector: MessageSelector | None) -> str:
    filters: list[str] = []
    if selector is not None:
        if selector.user is not None:
            filters.append("user_id = ?")
        if selector.project_id is not None:
            filters.append("project_id = ?")
        if selector.tag is not None:
            filters.append("tag = ?")
        if selector.correlation_id is not None:
            filters.append("correlation_id = ?")
    return f"AND {' AND '.join(filters)}" if filters else ""


def _values(selector: MessageSelector | None) -> tuple[str, ...]:
    if selector is None:
        return ()
    values: list[str] = []
    for value in (selector.user, selector.project_id, selector.tag, selector.correlation_id):
        if value is not None:
            values.append(str(value))
    return tuple(values)


def _row_to_message(row: sqlite3.Row) -> QueuedMessage:
    timestamp = datetime.fromisoformat(str(row["timestamp"]))
    queued_at = datetime.fromisoformat(str(row["queued_at"]))
    metadata = json.loads(str(row["metadata_json"] or "{}"))
    return QueuedMessage(
        message=CoreMessage(
            timestamp=timestamp,
            prompt=str(row["prompt"]),
            user=str(row["user_id"]),
            project_id=str(row["project_id"] or ""),
            tag=str(row["tag"] or ""),
            correlation_id=str(row["correlation_id"] or ""),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        ),
        message_id=str(row["message_id"]),
        queued_at=queued_at,
        sequence=int(row["sequence"]),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
