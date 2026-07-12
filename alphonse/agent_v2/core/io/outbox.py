"""Durable outbound queue and deterministic projection for v2 I/O."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from alphonse.agent_v2.core.core import StateSnapshot
from alphonse.agent_v2.core.io.channels import ChannelAddress
from alphonse.agent_v2.core.io.channels import channel_address_from_metadata
from alphonse.agent_v2.core.io.identity import V2IdentityResolver


@dataclass(frozen=True)
class OutboundMessage:
    """Normalized outbound message for a concrete integration instance."""

    outbox_message_id: str
    integration_id: str
    provider_key: str
    channel_target: str
    message: str
    kind: str
    audience_user_id: str
    correlation_id: str
    status: str
    provider_message_id: str = ""
    reply_to_provider_message_id: str = ""
    task_id: str = ""
    question_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    claimed_at: str = ""
    delivered_at: str = ""
    created_at: str = ""
    updated_at: str = ""
    attempt_count: int = 0
    lease_owner: str = ""
    lease_expires_at: str = ""
    next_attempt_at: str = ""
    last_error: str = ""
    max_attempts: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "outbox_message_id": self.outbox_message_id,
            "integration_id": self.integration_id,
            "provider_key": self.provider_key,
            "channel_target": self.channel_target,
            "message": self.message,
            "kind": self.kind,
            "audience_user_id": self.audience_user_id,
            "correlation_id": self.correlation_id,
            "status": self.status,
            "provider_message_id": self.provider_message_id,
            "reply_to_provider_message_id": self.reply_to_provider_message_id,
            "task_id": self.task_id,
            "question_id": self.question_id,
            "metadata": dict(self.metadata),
            "claimed_at": self.claimed_at,
            "delivered_at": self.delivered_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "attempt_count": self.attempt_count,
            "lease_owner": self.lease_owner,
            "lease_expires_at": self.lease_expires_at,
            "next_attempt_at": self.next_attempt_at,
            "last_error": self.last_error,
            "max_attempts": self.max_attempts,
        }


@dataclass(frozen=True)
class OutboundSelector:
    """Selector for consumers draining the outbound queue."""

    integration_id: str | None = None
    channel_target: str | None = None
    status: str | None = "pending"
    correlation_id: str | None = None
    audience_user_id: str | None = None


class SQLiteOutboundStore:
    """SQLite-backed outbound queue for v2 integration delivery."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory_connection: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._memory_connection = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_connection.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls) -> "SQLiteOutboundStore":
        return cls(_default_outbox_db_path())

    def enqueue(
        self,
        *,
        address: ChannelAddress,
        message: str,
        kind: str = "response",
        audience_user_id: str = "",
        correlation_id: str = "",
        task_id: str = "",
        question_id: str = "",
        reply_to_provider_message_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> OutboundMessage:
        text = str(message or "").strip()
        if not text:
            raise ValueError("outbound_message_required")
        now = _now_iso()
        record = OutboundMessage(
            outbox_message_id=str(uuid4()),
            integration_id=str(address.integration_id or "").strip(),
            provider_key=str(address.provider_key or "").strip().lower(),
            channel_target=str(address.channel_target or "").strip(),
            message=text,
            kind=str(kind or "response").strip() or "response",
            audience_user_id=str(audience_user_id or address.alphonse_user_id or "").strip(),
            correlation_id=str(correlation_id or "").strip(),
            status="pending",
            reply_to_provider_message_id=str(reply_to_provider_message_id or address.provider_message_id or "").strip(),
            task_id=str(task_id or "").strip(),
            question_id=str(question_id or "").strip(),
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        if not record.integration_id:
            raise ValueError("outbound_integration_id_required")
        if not record.provider_key:
            raise ValueError("outbound_provider_key_required")
        if not record.channel_target:
            raise ValueError("outbound_channel_target_required")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_outbox (
                  outbox_message_id, integration_id, provider_key, channel_target, message,
                  kind, audience_user_id, correlation_id, status, provider_message_id,
                  reply_to_provider_message_id, task_id, question_id, metadata_json,
                  claimed_at, delivered_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', '', ?, ?, ?, ?, '', '', ?, ?)
                """,
                (
                    record.outbox_message_id,
                    record.integration_id,
                    record.provider_key,
                    record.channel_target,
                    record.message,
                    record.kind,
                    record.audience_user_id,
                    record.correlation_id,
                    record.reply_to_provider_message_id,
                    record.task_id,
                    record.question_id,
                    json.dumps(record.metadata, sort_keys=True),
                    record.created_at,
                    record.updated_at,
                ),
            )
        return record

    def list_pending(self, selector: OutboundSelector | None = None, *, limit: int = 100) -> list[OutboundMessage]:
        return self.list(selector or OutboundSelector(), limit=limit)

    def status_counts(self) -> dict[str, int]:
        counts = {"pending": 0, "claimed": 0, "retry_wait": 0, "delivered": 0, "failed": 0}
        with self._connect() as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS count FROM v2_outbox GROUP BY status").fetchall()
        for row in rows:
            status = str(row["status"] or "")
            if status in counts:
                counts[status] = int(row["count"] or 0)
        return counts

    def get(self, outbox_message_id: str) -> OutboundMessage | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM v2_outbox WHERE outbox_message_id = ?",
                (str(outbox_message_id or "").strip(),),
            ).fetchone()
        return _message_from_row(row) if row is not None else None

    def list(self, selector: OutboundSelector | None = None, *, limit: int = 100) -> list[OutboundMessage]:
        where, values = _selector_where(selector)
        values.append(max(1, min(int(limit), 1000)))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM v2_outbox
                {where}
                ORDER BY created_at ASC
                LIMIT ?
                """,
                tuple(values),
            ).fetchall()
        return [_message_from_row(row) for row in rows]

    def claim_next(
        self,
        selector: OutboundSelector | None = None,
        *,
        lease_owner: str = "outbox",
        lease_seconds: int = 60,
    ) -> OutboundMessage | None:
        selector = selector or OutboundSelector()
        pending_selector = OutboundSelector(
            integration_id=selector.integration_id,
            channel_target=selector.channel_target,
            status=None if (selector.status or "pending") == "pending" else selector.status,
            correlation_id=selector.correlation_id,
            audience_user_id=selector.audience_user_id,
        )
        where, values = _selector_where(pending_selector)
        now = _now()
        owner = str(lease_owner or "outbox").strip() or "outbox"
        lease_expires_at = (now + timedelta(seconds=max(1, int(lease_seconds)))).isoformat()
        eligibility = "status = 'pending' OR (status = 'retry_wait' AND next_attempt_at <= ?)"
        values = [now.isoformat(), *values]
        if where:
            where = where.replace("WHERE ", f"WHERE ({eligibility}) AND ", 1)
        else:
            where = f"WHERE ({eligibility})"
        with self._connect() as conn:
            self.reclaim_expired(conn=conn, now=now)
            row = conn.execute(
                f"""
                SELECT * FROM v2_outbox
                {where}
                ORDER BY created_at ASC
                LIMIT 1
                """,
                tuple(values),
            ).fetchone()
            if row is None:
                return None
            message_id = str(row["outbox_message_id"])
            cursor = conn.execute(
                """
                UPDATE v2_outbox
                SET status = 'claimed',
                    claimed_at = ?,
                    updated_at = ?,
                    attempt_count = attempt_count + 1,
                    lease_owner = ?,
                    lease_expires_at = ?,
                    next_attempt_at = ''
                WHERE outbox_message_id = ?
                  AND (status = 'pending' OR (status = 'retry_wait' AND next_attempt_at <= ?))
                """,
                (now.isoformat(), now.isoformat(), owner, lease_expires_at, message_id, now.isoformat()),
            )
            if cursor.rowcount != 1:
                return None
            updated = conn.execute("SELECT * FROM v2_outbox WHERE outbox_message_id = ?", (message_id,)).fetchone()
        return _message_from_row(updated) if updated is not None else None

    def mark_delivered(self, outbox_message_id: str, *, provider_message_id: str = "") -> bool:
        now = _now_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE v2_outbox
                SET status = 'delivered',
                    provider_message_id = ?,
                    delivered_at = ?,
                    updated_at = ?,
                    lease_owner = '',
                    lease_expires_at = '',
                    next_attempt_at = ''
                WHERE outbox_message_id = ?
                """,
                (
                    str(provider_message_id or "").strip(),
                    now,
                    now,
                    str(outbox_message_id or "").strip(),
                ),
            )
            return cursor.rowcount == 1

    def mark_failed(
        self,
        outbox_message_id: str,
        *,
        error: str,
        retry_after_seconds: float | None = None,
    ) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT metadata_json, attempt_count, max_attempts FROM v2_outbox WHERE outbox_message_id = ?",
                (str(outbox_message_id or "").strip(),),
            ).fetchone()
            if row is None:
                return False
            metadata = _json_object(row["metadata_json"])
            normalized_error = str(error or "").strip()
            metadata["last_error"] = normalized_error
            attempts = int(row["attempt_count"] or 0)
            max_attempts = int(row["max_attempts"] or 5)
            retry_delay = 2.0 ** max(0, min(attempts - 1, 5))
            if retry_after_seconds is not None:
                retry_delay = max(0.0, float(retry_after_seconds))
            next_attempt_at = (_now() + timedelta(seconds=retry_delay)).isoformat()
            terminal = attempts >= max_attempts
            cursor = conn.execute(
                """
                UPDATE v2_outbox
                SET status = ?,
                    metadata_json = ?,
                    updated_at = ?,
                    lease_owner = '',
                    lease_expires_at = '',
                    next_attempt_at = ?,
                    last_error = ?
                WHERE outbox_message_id = ?
                """,
                (
                    "failed" if terminal else "retry_wait",
                    json.dumps(metadata, sort_keys=True),
                    _now_iso(),
                    "" if terminal else next_attempt_at,
                    normalized_error,
                    str(outbox_message_id or "").strip(),
                ),
            )
            return cursor.rowcount == 1

    def reclaim_expired(
        self,
        *,
        conn: sqlite3.Connection | None = None,
        now: datetime | None = None,
    ) -> int:
        timestamp = (now or _now()).isoformat()

        def _reclaim(connection: sqlite3.Connection) -> int:
            cursor = connection.execute(
                """
                UPDATE v2_outbox
                SET status = 'pending',
                    lease_owner = '',
                    lease_expires_at = '',
                    claimed_at = '',
                    updated_at = ?
                WHERE status = 'claimed'
                  AND lease_expires_at != ''
                  AND lease_expires_at <= ?
                """,
                (timestamp, timestamp),
            )
            return int(cursor.rowcount)

        if conn is not None:
            return _reclaim(conn)
        with self._connect() as opened:
            return _reclaim(opened)

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
                CREATE TABLE IF NOT EXISTS v2_outbox (
                  outbox_message_id TEXT PRIMARY KEY,
                  integration_id TEXT NOT NULL,
                  provider_key TEXT NOT NULL,
                  channel_target TEXT NOT NULL,
                  message TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  audience_user_id TEXT NOT NULL DEFAULT '',
                  correlation_id TEXT NOT NULL DEFAULT '',
                  status TEXT NOT NULL DEFAULT 'pending',
                  provider_message_id TEXT NOT NULL DEFAULT '',
                  reply_to_provider_message_id TEXT NOT NULL DEFAULT '',
                  task_id TEXT NOT NULL DEFAULT '',
                  question_id TEXT NOT NULL DEFAULT '',
                  metadata_json TEXT NOT NULL DEFAULT '{}',
                  claimed_at TEXT NOT NULL DEFAULT '',
                  delivered_at TEXT NOT NULL DEFAULT '',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  attempt_count INTEGER NOT NULL DEFAULT 0,
                  lease_owner TEXT NOT NULL DEFAULT '',
                  lease_expires_at TEXT NOT NULL DEFAULT '',
                  next_attempt_at TEXT NOT NULL DEFAULT '',
                  last_error TEXT NOT NULL DEFAULT '',
                  max_attempts INTEGER NOT NULL DEFAULT 5,
                  CHECK (status IN ('pending', 'claimed', 'retry_wait', 'delivered', 'failed'))
                ) STRICT;

                CREATE INDEX IF NOT EXISTS idx_v2_outbox_consumer
                  ON v2_outbox (integration_id, channel_target, status, created_at);

                CREATE INDEX IF NOT EXISTS idx_v2_outbox_question
                  ON v2_outbox (question_id, status);
                """
            )
            _ensure_columns(
                conn,
                {
                    "attempt_count": "INTEGER NOT NULL DEFAULT 0",
                    "lease_owner": "TEXT NOT NULL DEFAULT ''",
                    "lease_expires_at": "TEXT NOT NULL DEFAULT ''",
                    "next_attempt_at": "TEXT NOT NULL DEFAULT ''",
                    "last_error": "TEXT NOT NULL DEFAULT ''",
                    "max_attempts": "INTEGER NOT NULL DEFAULT 5",
                },
            )
            _migrate_status_check(conn)


def build_outbox_delivery_sink(
    *,
    outbox: SQLiteOutboundStore,
    identity_resolver: V2IdentityResolver | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a delivery sink for tools that already emit delivery events."""
    resolver = identity_resolver or V2IdentityResolver()

    def _sink(event: dict[str, Any]) -> dict[str, Any]:
        event_type = str(event.get("event_type") or "").strip()
        if event_type != "question.deliver":
            return {"ignored": True}
        question = event.get("question") if isinstance(event.get("question"), dict) else {}
        task = event.get("task") if isinstance(event.get("task"), dict) else {}
        task_metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        origin = channel_address_from_metadata(task_metadata)
        respondent = str(question.get("respondent_user_id") or task.get("user") or "").strip()
        resolved = resolver.resolve_outbound_address(alphonse_user_id=respondent, fallback_address=origin)
        if not resolved.resolved or resolved.address is None:
            if origin is not None:
                outbound = outbox.enqueue(
                    address=origin,
                    message=(
                        "I could not resolve where to deliver this question. "
                        f"Please map user {respondent or 'unknown'} to a messaging integration."
                    ),
                    kind="identity_resolution",
                    audience_user_id=str(task.get("user") or origin.alphonse_user_id or "").strip(),
                    correlation_id=str(task.get("correlation_id") or "").strip(),
                    task_id=str(question.get("task_id") or task.get("task_id") or "").strip(),
                    question_id=str(question.get("question_id") or "").strip(),
                    metadata={
                        "reason": resolved.reason,
                        "respondent_user_id": respondent,
                        "question": dict(question),
                    },
                )
                return {
                    "outbox_message_id": outbound.outbox_message_id,
                    "integration_id": outbound.integration_id,
                    "provider_key": outbound.provider_key,
                    "channel_target": outbound.channel_target,
                    "unresolved": True,
                    "reason": resolved.reason,
                }
            return {
                "exception": "question_delivery_unresolved",
                "reason": resolved.reason,
                "respondent_user_id": respondent,
            }
        outbound = outbox.enqueue(
            address=resolved.address,
            message=str(question.get("message") or "").strip(),
            kind="question",
            audience_user_id=respondent,
            correlation_id=str(task.get("correlation_id") or "").strip(),
            task_id=str(question.get("task_id") or task.get("task_id") or "").strip(),
            question_id=str(question.get("question_id") or "").strip(),
            metadata={
                "question": dict(question),
                "origin_channel": origin.to_dict() if origin is not None else {},
                "respondent_channel": resolved.address.to_dict(),
            },
        )
        return {
            "outbox_message_id": outbound.outbox_message_id,
            "integration_id": outbound.integration_id,
            "provider_key": outbound.provider_key,
            "channel_target": outbound.channel_target,
        }

    return _sink


def project_snapshot_to_outbox(
    *,
    snapshot: StateSnapshot,
    outbox: SQLiteOutboundStore,
) -> OutboundMessage | None:
    """Project a completed task snapshot response into the outbox."""
    metadata = snapshot.metadata or {}
    task_state = metadata.get("task_state")
    if not isinstance(task_state, dict):
        return None
    task_metadata = task_state.get("metadata") if isinstance(task_state.get("metadata"), dict) else {}
    origin = channel_address_from_metadata(task_metadata)
    if origin is None:
        return None
    response = _latest_tool_result_response(task_state)
    if not response:
        return None
    projected = task_metadata.get("outbox_projected")
    if isinstance(projected, dict) and str(projected.get("response")) == response:
        return None
    return outbox.enqueue(
        address=origin,
        message=response,
        kind="response",
        audience_user_id=str(task_state.get("user") or origin.alphonse_user_id or "").strip(),
        correlation_id=str(task_state.get("correlation_id") or "").strip(),
        task_id=str(task_state.get("task_id") or "").strip(),
        metadata={
            "source": "task_result",
            "tool": "respond",
            "project_id": str(task_state.get("project_id") or "").strip(),
            "scheduled_task_id": str(task_metadata.get("scheduled_task_id") or "").strip(),
            "scheduled_run_id": str(task_metadata.get("scheduled_run_id") or "").strip(),
            "occurrence_key": str(task_metadata.get("occurrence_key") or "").strip(),
        },
    )


def _latest_tool_result_response(task_state: dict[str, Any]) -> str:
    calls = _json_list(task_state.get("plan_json"))
    for call in reversed(calls):
        if not isinstance(call, dict):
            continue
        execution = call.get("execution")
        if not isinstance(execution, dict) or str(execution.get("status") or "") != "success":
            continue
        tool_id = str(call.get("tool_id") or "").strip()
        result = execution.get("result")
        if not isinstance(result, dict):
            continue
        if str(result.get("message") or "").strip():
            return str(result.get("message") or "").strip()
        if tool_id == "native.bash":
            stdout = str(result.get("stdout") or "").strip()
            stderr = str(result.get("stderr") or "").strip()
            if stdout:
                return stdout
            if stderr:
                return stderr
        if tool_id == "native.scheduled_task":
            name = str(result.get("name") or "Scheduled task").strip()
            next_run_at = str(result.get("next_run_at") or "").strip()
            if next_run_at:
                return f'Scheduled "{name}" for {next_run_at}.'
            return f'Scheduled "{name}".'
    return ""


def _selector_where(selector: OutboundSelector | None) -> tuple[str, list[Any]]:
    if selector is None:
        return "", []
    filters: list[str] = []
    values: list[Any] = []
    if selector.integration_id is not None:
        filters.append("integration_id = ?")
        values.append(str(selector.integration_id or "").strip())
    if selector.channel_target is not None:
        filters.append("channel_target = ?")
        values.append(str(selector.channel_target or "").strip())
    if selector.status is not None:
        filters.append("status = ?")
        values.append(str(selector.status or "").strip())
    if selector.correlation_id is not None:
        filters.append("correlation_id = ?")
        values.append(str(selector.correlation_id or "").strip())
    if selector.audience_user_id is not None:
        filters.append("audience_user_id = ?")
        values.append(str(selector.audience_user_id or "").strip())
    return (f"WHERE {' AND '.join(filters)}" if filters else "", values)


def _message_from_row(row: sqlite3.Row) -> OutboundMessage:
    return OutboundMessage(
        outbox_message_id=str(row["outbox_message_id"]),
        integration_id=str(row["integration_id"]),
        provider_key=str(row["provider_key"]),
        channel_target=str(row["channel_target"]),
        message=str(row["message"]),
        kind=str(row["kind"]),
        audience_user_id=str(row["audience_user_id"]),
        correlation_id=str(row["correlation_id"]),
        status=str(row["status"]),
        provider_message_id=str(row["provider_message_id"]),
        reply_to_provider_message_id=str(row["reply_to_provider_message_id"]),
        task_id=str(row["task_id"]),
        question_id=str(row["question_id"]),
        metadata=_json_object(row["metadata_json"]),
        claimed_at=str(row["claimed_at"]),
        delivered_at=str(row["delivered_at"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        attempt_count=int(row["attempt_count"] or 0),
        lease_owner=str(row["lease_owner"] or ""),
        lease_expires_at=str(row["lease_expires_at"] or ""),
        next_attempt_at=str(row["next_attempt_at"] or ""),
        last_error=str(row["last_error"] or ""),
        max_attempts=int(row["max_attempts"] or 5),
    )


class _ConnectionProxy:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def __enter__(self) -> sqlite3.Connection:
        return self.conn

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        return False


def _json_object(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except ValueError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    try:
        parsed = json.loads(str(value or "[]"))
    except ValueError:
        return []
    return list(parsed) if isinstance(parsed, list) else []


def _now_iso() -> str:
    return _now().isoformat()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_columns(conn: sqlite3.Connection, columns: dict[str, str]) -> None:
    existing = {str(row["name"]) for row in conn.execute("PRAGMA table_info(v2_outbox)").fetchall()}
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE v2_outbox ADD COLUMN {name} {definition}")


def _migrate_status_check(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'v2_outbox'"
    ).fetchone()
    sql = str(row["sql"] or "") if row is not None else ""
    if "retry_wait" in sql:
        return
    conn.execute("ALTER TABLE v2_outbox RENAME TO v2_outbox_legacy")
    conn.executescript(
        """
        CREATE TABLE v2_outbox (
          outbox_message_id TEXT PRIMARY KEY,
          integration_id TEXT NOT NULL,
          provider_key TEXT NOT NULL,
          channel_target TEXT NOT NULL,
          message TEXT NOT NULL,
          kind TEXT NOT NULL,
          audience_user_id TEXT NOT NULL DEFAULT '',
          correlation_id TEXT NOT NULL DEFAULT '',
          status TEXT NOT NULL DEFAULT 'pending',
          provider_message_id TEXT NOT NULL DEFAULT '',
          reply_to_provider_message_id TEXT NOT NULL DEFAULT '',
          task_id TEXT NOT NULL DEFAULT '',
          question_id TEXT NOT NULL DEFAULT '',
          metadata_json TEXT NOT NULL DEFAULT '{}',
          claimed_at TEXT NOT NULL DEFAULT '',
          delivered_at TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          attempt_count INTEGER NOT NULL DEFAULT 0,
          lease_owner TEXT NOT NULL DEFAULT '',
          lease_expires_at TEXT NOT NULL DEFAULT '',
          next_attempt_at TEXT NOT NULL DEFAULT '',
          last_error TEXT NOT NULL DEFAULT '',
          max_attempts INTEGER NOT NULL DEFAULT 5,
          CHECK (status IN ('pending', 'claimed', 'retry_wait', 'delivered', 'failed'))
        ) STRICT;
        """
    )
    conn.execute(
        """
        INSERT INTO v2_outbox (
          outbox_message_id, integration_id, provider_key, channel_target, message,
          kind, audience_user_id, correlation_id, status, provider_message_id,
          reply_to_provider_message_id, task_id, question_id, metadata_json,
          claimed_at, delivered_at, created_at, updated_at, attempt_count,
          lease_owner, lease_expires_at, next_attempt_at, last_error, max_attempts
        )
        SELECT
          outbox_message_id, integration_id, provider_key, channel_target, message,
          kind, audience_user_id, correlation_id,
          CASE WHEN status IN ('pending', 'claimed', 'delivered', 'failed') THEN status ELSE 'pending' END,
          provider_message_id, reply_to_provider_message_id, task_id, question_id,
          metadata_json, claimed_at, delivered_at, created_at, updated_at,
          attempt_count, lease_owner, lease_expires_at, next_attempt_at, last_error, max_attempts
        FROM v2_outbox_legacy
        """
    )
    conn.execute("DROP TABLE v2_outbox_legacy")
    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_v2_outbox_consumer
          ON v2_outbox (integration_id, channel_target, status, created_at);

        CREATE INDEX IF NOT EXISTS idx_v2_outbox_question
          ON v2_outbox (question_id, status);
        """
    )


def _default_outbox_db_path() -> str:
    configured = os.getenv("ALPHONSE_V2_OUTBOX_DB_PATH") or os.getenv("ALPHONSE_V2_DB_PATH")
    if configured:
        return configured
    return str(Path.home() / ".alphonse" / "v2-outbox.sqlite3")
