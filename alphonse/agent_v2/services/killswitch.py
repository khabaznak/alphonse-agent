"""Thread-safe active-task cancellation and durable kill-switch audit records."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from alphonse.agent_v2.database import connect_database


@dataclass(frozen=True)
class ActiveTask:
    message_id: str
    task_id: str
    user_id: str
    project_id: str
    metadata: dict[str, Any]


class KillSwitchCoordinator:
    """Owns the single active v2 task and its cooperative cancellation flag."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active: ActiveTask | None = None
        self._cancelled_message_ids: set[str] = set()

    def activate(self, *, message_id: str, task_id: str, user_id: str, project_id: str, metadata: dict[str, Any]) -> None:
        with self._lock:
            self._active = ActiveTask(message_id, task_id, user_id, project_id, dict(metadata))

    def request_cancel(self) -> ActiveTask | None:
        with self._lock:
            if self._active is None:
                return None
            self._cancelled_message_ids.add(self._active.message_id)
            return self._active

    def is_cancelled(self, message_id: str) -> bool:
        with self._lock:
            return str(message_id or "") in self._cancelled_message_ids

    def clear(self, message_id: str = "") -> None:
        with self._lock:
            target = str(message_id or "")
            if self._active is not None and (not target or self._active.message_id == target):
                self._cancelled_message_ids.discard(self._active.message_id)
                self._active = None

    def active(self) -> ActiveTask | None:
        with self._lock:
            return self._active


class KillSwitchAuditStore:
    """Small append-only audit store deliberately independent of inbound work."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory_connection: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._memory_connection = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def record(self, *, actor_user_id: str, source: dict[str, Any], active: ActiveTask | None, status: str, notification_outbox_id: str = "", notification_error: str = "") -> str:
        audit_id = f"killswitch-{uuid4().hex}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_killswitch_audit (
                  audit_id, actor_user_id, source_json, target_message_id, target_task_id,
                  target_user_id, status, notification_outbox_id, notification_error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_id, str(actor_user_id or ""), json.dumps(dict(source or {}), sort_keys=True),
                    active.message_id if active else "", active.task_id if active else "",
                    active.user_id if active else "", str(status or ""), str(notification_outbox_id or ""),
                    str(notification_error or ""), datetime.now(timezone.utc).isoformat(),
                ),
            )
        return audit_id

    def _connect(self) -> Any:
        if self._memory_connection is not None:
            return _ConnectionProxy(self._memory_connection)
        return connect_database(self.db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS v2_killswitch_audit (
                  audit_id TEXT PRIMARY KEY, actor_user_id TEXT NOT NULL, source_json TEXT NOT NULL,
                  target_message_id TEXT NOT NULL DEFAULT '', target_task_id TEXT NOT NULL DEFAULT '',
                  target_user_id TEXT NOT NULL DEFAULT '', status TEXT NOT NULL,
                  notification_outbox_id TEXT NOT NULL DEFAULT '', notification_error TEXT NOT NULL DEFAULT '',
                  created_at TEXT NOT NULL
                ) STRICT
                """
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
