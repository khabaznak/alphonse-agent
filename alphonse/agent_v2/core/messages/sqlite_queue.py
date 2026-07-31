"""Durable inbound message queue for the v2 daemon."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.messages.queue import MessageSelector
from alphonse.agent_v2.core.messages.queue import QueuedMessage
from alphonse.agent_v2.database import connect_database, default_database_path


class SQLiteMessageQueue:
    """SQLite-backed queue with processing recovery and explicit acknowledgement."""

    def __init__(self, db_path: str | Path = ":memory:", *, lease_owner: str = "default") -> None:
        self.db_path = str(db_path)
        self.lease_owner = str(lease_owner or "default").strip() or "default"
        self._memory_connection: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._memory_connection = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_connection.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls, *, lease_owner: str = "default") -> "SQLiteMessageQueue":
        return cls(default_database_path(), lease_owner=lease_owner)

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
                f"""
                SELECT * FROM v2_inbound_messages
                WHERE (status = 'pending' OR (status = 'retry_wait' AND next_attempt_at <= ?))
                  {_where(selector)}
                ORDER BY sequence LIMIT 1
                """,
                (_now_iso(), *_values(selector)),
            ).fetchone()
        return _row_to_message(row) if row is not None else None

    def dequeue(self, selector: MessageSelector | None = None) -> QueuedMessage | None:
        return self.claim_next(selector=selector)

    def claim_next(
        self,
        selector: MessageSelector | None = None,
        *,
        lease_owner: str | None = None,
        lease_seconds: int = 60,
        include_owned: bool = False,
    ) -> QueuedMessage | None:
        owner = str(lease_owner or self.lease_owner or "default").strip() or "default"
        now = _now()
        lease_expires_at = (now + timedelta(seconds=max(1, int(lease_seconds)))).isoformat()
        with self._connect() as conn:
            if include_owned:
                owned = conn.execute(
                    f"""
                    SELECT * FROM v2_inbound_messages
                    WHERE status = 'processing' AND lease_owner = ?
                      {_where(selector)}
                    ORDER BY sequence LIMIT 1
                    """,
                    (owner, *_values(selector)),
                ).fetchone()
                if owned is not None:
                    return _row_to_message(owned)
            row = conn.execute(
                f"""
                SELECT * FROM v2_inbound_messages
                WHERE (status = 'pending' OR (status = 'retry_wait' AND next_attempt_at <= ?))
                  {_where(selector)}
                ORDER BY sequence LIMIT 1
                """,
                (now.isoformat(), *_values(selector)),
            ).fetchone()
            if row is None:
                return None
            cursor = conn.execute(
                """
                UPDATE v2_inbound_messages
                SET status = 'processing',
                    processing_at = ?,
                    lease_owner = ?,
                    lease_expires_at = ?,
                    attempt_count = attempt_count + 1
                WHERE message_id = ?
                  AND (status = 'pending' OR (status = 'retry_wait' AND next_attempt_at <= ?))
                """,
                (now.isoformat(), owner, lease_expires_at, str(row["message_id"]), now.isoformat()),
            )
            if cursor.rowcount != 1:
                return None
            updated = conn.execute(
                "SELECT * FROM v2_inbound_messages WHERE message_id = ?",
                (str(row["message_id"]),),
            ).fetchone()
        return _row_to_message(updated) if updated is not None else None

    def ack(self, message_id: str, lease_owner: str | None = None) -> bool:
        owner = str(lease_owner or self.lease_owner or "default").strip() or "default"
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM v2_inbound_messages
                WHERE message_id = ? AND status = 'processing' AND lease_owner = ?
                """,
                (str(message_id or "").strip(), owner),
            )
            return cursor.rowcount == 1

    def retry(
        self,
        message_id: str,
        *,
        error: str = "",
        next_attempt_at: datetime | str | None = None,
        lease_owner: str | None = None,
        max_attempts: int | None = None,
    ) -> bool:
        owner = str(lease_owner or self.lease_owner or "default").strip() or "default"
        retry_at = _coerce_iso(next_attempt_at or _now())
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT attempt_count FROM v2_inbound_messages
                WHERE message_id = ? AND status = 'processing' AND lease_owner = ?
                """,
                (str(message_id or "").strip(), owner),
            ).fetchone()
            if row is None:
                return False
            attempts = int(row["attempt_count"] or 0)
            terminal = max_attempts is not None and attempts >= max(1, int(max_attempts))
            cursor = conn.execute(
                """
                UPDATE v2_inbound_messages
                SET status = ?,
                    processing_at = '',
                    lease_owner = '',
                    lease_expires_at = '',
                    next_attempt_at = ?,
                    last_error = ?
                WHERE message_id = ? AND status = 'processing' AND lease_owner = ?
                """,
                (
                    "failed" if terminal else "retry_wait",
                    "" if terminal else retry_at,
                    str(error or "").strip(),
                    str(message_id or "").strip(),
                    owner,
                ),
            )
            return cursor.rowcount == 1

    def requeue(self, message_id: str) -> bool:
        return self.retry(message_id, next_attempt_at=_now())

    def reclaim_expired(self, *, now: datetime | None = None) -> int:
        timestamp = (now or _now()).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE v2_inbound_messages
                SET status = 'pending',
                    processing_at = '',
                    lease_owner = '',
                    lease_expires_at = ''
                WHERE status = 'processing'
                  AND (
                    (lease_expires_at != '' AND lease_expires_at <= ?)
                    OR (lease_owner = '' AND lease_expires_at = '')
                  )
                """,
                (timestamp,),
            )
            return int(cursor.rowcount)

    def size(self, selector: MessageSelector | None = None) -> int:
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*) AS count FROM v2_inbound_messages
                WHERE (status = 'pending' OR (status = 'retry_wait' AND next_attempt_at <= ?))
                  {_where(selector)}
                """,
                (_now_iso(), *_values(selector)),
            ).fetchone()
        return int(row["count"] if row is not None else 0)

    def status_counts(self) -> dict[str, int]:
        counts = {"pending": 0, "processing": 0, "retry_wait": 0, "failed": 0}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM v2_inbound_messages GROUP BY status"
            ).fetchall()
        for row in rows:
            status = str(row["status"] or "")
            if status in counts:
                counts[status] = int(row["count"] or 0)
        return counts

    def _connect(self) -> sqlite3.Connection:
        if self._memory_connection is not None:
            return _ConnectionProxy(self._memory_connection)
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return connect_database(path)

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
                  processing_at TEXT NOT NULL DEFAULT '',
                  attempt_count INTEGER NOT NULL DEFAULT 0,
                  lease_owner TEXT NOT NULL DEFAULT '',
                  lease_expires_at TEXT NOT NULL DEFAULT '',
                  next_attempt_at TEXT NOT NULL DEFAULT '',
                  last_error TEXT NOT NULL DEFAULT '',
                  CHECK (status IN ('pending', 'processing', 'retry_wait', 'failed'))
                ) STRICT;
                """
            )
            _ensure_columns(
                conn,
                "v2_inbound_messages",
                {
                    "attempt_count": "INTEGER NOT NULL DEFAULT 0",
                    "lease_owner": "TEXT NOT NULL DEFAULT ''",
                    "lease_expires_at": "TEXT NOT NULL DEFAULT ''",
                    "next_attempt_at": "TEXT NOT NULL DEFAULT ''",
                    "last_error": "TEXT NOT NULL DEFAULT ''",
                },
            )


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


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _coerce_iso(value: datetime | str) -> str:
    if isinstance(value, datetime):
        timestamp = value
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc).isoformat()
    return str(value or "").strip() or _now_iso()
