"""Durable outbound queue and deterministic projection for v2 I/O."""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import NAMESPACE_URL, uuid4, uuid5

from alphonse.agent_v2.core.core import StateSnapshot
from alphonse.agent_v2.core.io.channels import ChannelAddress
from alphonse.agent_v2.core.io.channels import channel_address_from_metadata
from alphonse.agent_v2.core.io.identity import V2IdentityResolver
from alphonse.agent_v2.database import (
    AlphonseSQLiteConnection, DEFAULT_BUSY_TIMEOUT_MS, connect_database,
    default_database_path, ensure_schema_migrations,
)


logger = logging.getLogger(__name__)


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
    project_id: str = ""
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
            "project_id": self.project_id,
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
    project_id: str | None = None


class SQLiteOutboundStore:
    """SQLite-backed outbound queue for v2 integration delivery."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self.finding_delivery_guard: Callable[[OutboundMessage], bool] | None = None
        self.scheduled_delivery_guard: Callable[[OutboundMessage], bool] | None = None
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
        project_id: str = "",
        reply_to_provider_message_id: str = "",
        metadata: dict[str, Any] | None = None,
        idempotency_key: str = "",
        connection: sqlite3.Connection | None = None,
    ) -> OutboundMessage:
        text = str(message or "").strip()
        if not text:
            raise ValueError("outbound_message_required")
        now = _now_iso()
        payload = dict(metadata or {})
        resolved_project_id = str(project_id or payload.get("project_id") or "").strip()
        dedupe_key = str(idempotency_key or "").strip()
        record = OutboundMessage(
            outbox_message_id=str(uuid5(NAMESPACE_URL, f"alphonse-outbox:{dedupe_key}")) if dedupe_key else str(uuid4()),
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
            project_id=resolved_project_id,
            metadata=payload,
            created_at=now,
            updated_at=now,
        )
        if not record.integration_id:
            raise ValueError("outbound_integration_id_required")
        if not record.provider_key:
            raise ValueError("outbound_provider_key_required")
        if not record.channel_target:
            raise ValueError("outbound_channel_target_required")
        if connection is not None:
            self._validate_connection(connection)
            self._insert(connection, record)
            if dedupe_key:
                cursor = connection.cursor()
                cursor.row_factory = sqlite3.Row
                existing = cursor.execute(
                    "SELECT * FROM main.v2_outbox WHERE outbox_message_id = ?",
                    (record.outbox_message_id,),
                ).fetchone()
                if existing is not None:
                    return _message_from_row(existing)
        else:
            with self._connect() as conn:
                self._insert(conn, record)
                if dedupe_key:
                    existing = conn.execute(
                        "SELECT * FROM v2_outbox WHERE outbox_message_id = ?",
                        (record.outbox_message_id,),
                    ).fetchone()
                    if existing is not None:
                        return _message_from_row(existing)
        return record

    def _validate_connection(self, connection: sqlite3.Connection) -> None:
        """Require supplied transactions to target this store's main database."""
        if self._memory_connection is not None:
            if connection is not self._memory_connection:
                raise ValueError("outbox_connection_database_mismatch")
            return
        try:
            databases = connection.execute("PRAGMA database_list").fetchall()
            main_path = next((row[2] for row in databases if row[1] == "main"), "")
            if not main_path or Path(main_path).resolve() != Path(self.db_path).expanduser().resolve():
                raise ValueError("outbox_connection_database_mismatch")
        except (sqlite3.Error, OSError, RuntimeError):
            raise ValueError("outbox_connection_database_mismatch") from None

    @staticmethod
    def _insert(conn: sqlite3.Connection, record: OutboundMessage) -> None:
        conn.execute(
            """
            INSERT OR IGNORE INTO main.v2_outbox (
              outbox_message_id, integration_id, provider_key, channel_target, message,
              kind, audience_user_id, correlation_id, status, provider_message_id,
              reply_to_provider_message_id, task_id, question_id, project_id, metadata_json,
              claimed_at, delivered_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', '', ?, ?, ?, ?, ?, '', '', ?, ?)
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
                record.project_id,
                json.dumps(record.metadata, sort_keys=True),
                record.created_at,
                record.updated_at,
            ),
        )

    def list_pending(self, selector: OutboundSelector | None = None, *, limit: int = 100) -> list[OutboundMessage]:
        return self.list(selector or OutboundSelector(), limit=limit)

    def list_unconfirmed_scheduled(self) -> list[OutboundMessage]:
        """Return all ambiguous scheduled sends for restart reconciliation."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM v2_outbox WHERE status='unconfirmed'
                   AND json_valid(metadata_json)
                   AND COALESCE(json_extract(metadata_json, '$.occurrence_key'), '') != ''"""
            ).fetchall()
        return [_message_from_row(row) for row in rows]

    def list_confirmed_scheduled_evidence(self) -> list[OutboundMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM v2_outbox WHERE status='delivered' AND json_valid(metadata_json)
                   AND json_extract(metadata_json, '$.scheduled_delivery_evidence')=1"""
            ).fetchall()
        return [_message_from_row(row) for row in rows]

    def list_for_occurrence(self, occurrence_key: str) -> list[OutboundMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM v2_outbox WHERE json_valid(metadata_json)
                   AND json_extract(metadata_json, '$.occurrence_key') = ?
                   AND COALESCE(json_extract(metadata_json, '$.scheduled_question'), 0) != 1""",
                (occurrence_key,),
            ).fetchall()
        return [_message_from_row(row) for row in rows]

    def status_counts(self) -> dict[str, int]:
        counts = {"pending": 0, "claimed": 0, "retry_wait": 0, "delivered": 0, "failed": 0, "unconfirmed": 0}
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

    def mark_scheduled_unconfirmed(self, outbox_message_id: str, *, task_id: str, occurrence_key: str, recipient_user_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute(
                """UPDATE v2_outbox SET status='unconfirmed', next_attempt_at='', lease_owner='', lease_expires_at='',
                   metadata_json=json_set(metadata_json, '$.delivery_state', 'unconfirmed', '$.unconfirmed_lease_owner', lease_owner),
                   last_error='scheduled_delivery_wait_expired', updated_at=?
                   WHERE outbox_message_id=? AND task_id=? AND audience_user_id=?
                     AND status IN ('pending','claimed','retry_wait') AND json_valid(metadata_json)
                     AND json_extract(metadata_json, '$.occurrence_key')=?""",
                (_now_iso(), outbox_message_id, task_id, recipient_user_id, occurrence_key),
            ).rowcount == 1

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
        # Every consumer uses this gate, including restarted stores. A store
        # without the authoritative finding service fails closed.
        for _ in range(100):
            message = self._claim_next_unchecked(selector, lease_owner=lease_owner, lease_seconds=lease_seconds)
            if message is None:
                return message
            if message.metadata.get("scheduled_task_id") and message.metadata.get("occurrence_key"):
                try:
                    allowed_schedule = self.scheduled_delivery_guard is not None and self.scheduled_delivery_guard(message)
                except Exception:
                    allowed_schedule = False
                if not allowed_schedule:
                    with self._connect() as conn:
                        conn.execute("""UPDATE v2_outbox SET status='failed',
                            metadata_json=json_set(metadata_json, '$.delivery_state', 'cancelled_or_unavailable'),
                            last_error='scheduled_delivery_not_authorized', lease_owner='', lease_expires_at=''
                            WHERE outbox_message_id=? AND status='claimed'""", (message.outbox_message_id,))
                    continue
            if message.kind != 'finding':
                return message
            try:
                allowed = self.finding_delivery_guard is not None and self.finding_delivery_guard(message)
            except Exception:
                allowed = False
            if allowed:
                return message
            with self._connect() as conn:
                conn.execute("""UPDATE v2_outbox SET status='failed', message='[Finding unavailable]',
                    metadata_json=json_set(metadata_json, '$.finding_delivery_state', 'unavailable'),
                    lease_owner='', lease_expires_at='' WHERE outbox_message_id=?""", (message.outbox_message_id,))
        return None

    def redact_finding(self, finding_id: str) -> None:
        with self._connect() as conn:
            conn.execute("""UPDATE v2_outbox SET message='[Finding unavailable]',
                status=CASE WHEN status='delivered' THEN status ELSE 'failed' END,
                metadata_json=json_set(metadata_json, '$.finding_delivery_state', 'revoked'),
                last_error='', lease_owner='', lease_expires_at=''
                WHERE kind='finding' AND json_extract(metadata_json, '$.finding_id')=?""", (finding_id,))

    def _claim_next_unchecked(
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
            project_id=selector.project_id,
        )
        where, values = _selector_where(pending_selector)
        now = _now()
        owner = str(lease_owner or "outbox").strip() or "outbox"
        lease_expires_at = (now + timedelta(seconds=max(1, int(lease_seconds)))).isoformat()
        eligibility = "status = 'pending' OR (status = 'retry_wait' AND next_attempt_at <= ?)"
        if self.finding_delivery_guard is None:
            eligibility = f"({eligibility}) AND kind != 'finding'"
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
                    metadata_json = CASE WHEN kind='finding'
                        THEN json_set(metadata_json, '$.finding_delivery_state', 'delivered')
                        WHEN json_valid(metadata_json) AND COALESCE(json_extract(metadata_json, '$.occurrence_key'), '') != ''
                        THEN json_set(metadata_json, '$.delivery_state', 'confirmed') ELSE metadata_json END,
                    provider_message_id = ?,
                    delivered_at = ?,
                    updated_at = ?,
                    lease_owner = '',
                    lease_expires_at = '',
                    next_attempt_at = '',
                    last_error = ''
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
            # A failed send may have reached the provider. Findings must never
            # automatically resend after ambiguity, even after process restart.
            is_finding = bool(metadata.get("finding_id"))
            if is_finding:
                metadata["finding_delivery_state"] = "ambiguous"
                metadata["last_error"] = "finding_send_outcome_unknown"
                normalized_error = "finding_send_outcome_unknown"
            # Once dispatch began, a provider exception can follow a successful
            # send. A scheduled message needs explicit reconciliation, not an
            # automatic resend that may duplicate the reminder.
            scheduled = bool(metadata.get("occurrence_key"))
            if scheduled:
                metadata["delivery_state"] = "unconfirmed"
            terminal = is_finding or scheduled or attempts >= max_attempts
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
                    "unconfirmed" if scheduled else "failed" if terminal else "retry_wait",
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
                SET status = CASE WHEN kind='finding' THEN 'failed'
                    WHEN json_valid(metadata_json) AND COALESCE(json_extract(metadata_json, '$.occurrence_key'), '') != ''
                    THEN 'unconfirmed' ELSE 'pending' END,
                    metadata_json = CASE WHEN kind='finding'
                      THEN json_set(metadata_json, '$.finding_delivery_state', 'ambiguous')
                      WHEN json_valid(metadata_json) AND COALESCE(json_extract(metadata_json, '$.occurrence_key'), '') != ''
                      THEN json_set(metadata_json, '$.delivery_state', 'unconfirmed', '$.unconfirmed_lease_owner', lease_owner) ELSE metadata_json END,
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
        return connect_database(path)

    @contextmanager
    def _schema_connect(self):
        if self._memory_connection is not None:
            with _ConnectionProxy(self._memory_connection) as conn:
                yield conn
            return
        # connect_database creates migration bookkeeping before returning. Use
        # the same connection settings without DDL so an unsupported store is
        # not modified before its outbox schema is checked under the lock.
        path = Path(self.db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=DEFAULT_BUSY_TIMEOUT_MS / 1000, factory=AlphonseSQLiteConnection)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute('PRAGMA foreign_keys = ON')
            conn.execute(f'PRAGMA busy_timeout = {DEFAULT_BUSY_TIMEOUT_MS}')
            with conn:
                yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._schema_connect() as conn:
            # Hold the write lock across inspection and DDL. executescript
            # commits implicitly, so use individual statements below.
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT 1 FROM main.sqlite_master WHERE type='table' AND name='v2_outbox'"
            ).fetchone()
            if existing is not None:
                _reject_unsupported_statuses(conn)
                if _outbox_status_schema(conn) != _OUTBOX_STATUSES:
                    _validate_outbox_rebuild(conn)
            if self._memory_connection is None:
                ensure_schema_migrations(conn)
            conn.execute(
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
                  project_id TEXT NOT NULL DEFAULT '',
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
                  CHECK (status IN ('pending', 'claimed', 'retry_wait', 'delivered', 'failed', 'unconfirmed'))
                ) STRICT
                """
            )
            conn.execute("""CREATE INDEX IF NOT EXISTS idx_v2_outbox_consumer
                         ON v2_outbox (integration_id, channel_target, status, created_at)""")
            conn.execute("""CREATE INDEX IF NOT EXISTS idx_v2_outbox_question
                         ON v2_outbox (question_id, status)""")
            _ensure_columns(
                conn,
                {
                    "attempt_count": "INTEGER NOT NULL DEFAULT 0",
                    "lease_owner": "TEXT NOT NULL DEFAULT ''",
                    "lease_expires_at": "TEXT NOT NULL DEFAULT ''",
                    "next_attempt_at": "TEXT NOT NULL DEFAULT ''",
                    "last_error": "TEXT NOT NULL DEFAULT ''",
                    "max_attempts": "INTEGER NOT NULL DEFAULT 5",
                    "project_id": "TEXT NOT NULL DEFAULT ''",
                },
            )
            conn.execute(
                """
                UPDATE v2_outbox
                SET project_id = CASE
                  WHEN json_valid(metadata_json) THEN COALESCE(json_extract(metadata_json, '$.project_id'), '')
                  ELSE ''
                END
                WHERE project_id = ''
                """
            )
            _migrate_status_check(conn)
            # Older versions automatically retried expired/failed sends. Once
            # an attempt has begun, migration cannot prove the provider did
            # not accept it, so do not resume those automatic retries.
            conn.execute(
                """UPDATE v2_outbox SET status='unconfirmed', next_attempt_at='',
                   metadata_json=json_set(metadata_json, '$.delivery_state', 'unconfirmed'),
                   last_error='scheduled_delivery_outcome_unknown'
                   WHERE status IN ('pending','retry_wait') AND attempt_count>0
                     AND json_valid(metadata_json)
                     AND COALESCE(json_extract(metadata_json, '$.occurrence_key'), '') != ''"""
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_v2_outbox_project
                  ON v2_outbox (audience_user_id, project_id, status, created_at)
                """
            )
        if self._memory_connection is None:
            # WAL cannot be enabled inside a transaction. It is intentionally
            # configured only after a successful, committed schema preflight.
            conn = connect_database(Path(self.db_path).expanduser())
            conn.close()


def build_outbox_delivery_sink(
    *,
    outbox: SQLiteOutboundStore,
    identity_resolver: V2IdentityResolver | None = None,
    communication_router: Any | None = None,
    conversation_store: Any | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a delivery sink for task lifecycle and tool delivery events."""
    resolver = identity_resolver or V2IdentityResolver()

    def _sink(event: dict[str, Any]) -> dict[str, Any]:
        event_type = str(event.get("event_type") or "").strip()
        if event_type == "communication.resolve_recipient":
            if communication_router is None:
                return {"status": "unavailable"}
            recipient = communication_router._resolve_user(str(event.get("recipient") or "").strip())
            if not recipient or recipient == "ambiguous":
                return {"status": "unavailable"}
            return {"status": "resolved", "recipient_user_id": recipient}
        if event_type in {"communication.delivery_status", "communication.delivery_timeout"}:
            task = event.get("task") if isinstance(event.get("task"), dict) else {}
            metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
            outbound = outbox.get(str(event.get("outbox_message_id") or ""))
            if (outbound is None or not task.get("task_id") or outbound.task_id != task.get("task_id")
                    or not metadata.get("occurrence_key")
                    or outbound.metadata.get("occurrence_key") != metadata.get("occurrence_key")
                    or outbound.audience_user_id != event.get("recipient_user_id")):
                return {"status": "unavailable"}
            if event_type == "communication.delivery_timeout":
                outbox.mark_scheduled_unconfirmed(
                    outbound.outbox_message_id, task_id=str(task["task_id"]),
                    occurrence_key=str(metadata["occurrence_key"]), recipient_user_id=outbound.audience_user_id,
                )
                outbound = outbox.get(outbound.outbox_message_id)
            return {"status": outbound.status, "outbox_message_id": outbound.outbox_message_id,
                    "provider_message_id": outbound.provider_message_id, "recipient_read": "unknown"}
        if event_type == "task.acknowledge":
            task = event.get("task") if isinstance(event.get("task"), dict) else {}
            task_metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
            origin = channel_address_from_metadata(task_metadata)
            if origin is None:
                return {"status": "origin_unavailable"}
            message = str(event.get("message") or "").strip()
            if not message:
                return {"status": "message_unavailable"}
            outbound = outbox.enqueue(
                address=origin,
                message=message,
                kind="task_acknowledgement",
                audience_user_id=str(task.get("user") or origin.alphonse_user_id or "").strip(),
                correlation_id=str(task.get("correlation_id") or "").strip(),
                task_id=str(task.get("task_id") or "").strip(),
                project_id=str(task.get("project_id") or "").strip(),
                metadata={"source": "v3_admission", "project_id": str(task.get("project_id") or "").strip()},
                idempotency_key=str(event.get("idempotency_key") or "").strip(),
            )
            if conversation_store is not None:
                conversation_store.record(
                    owner_user_id=str(task.get("user") or origin.alphonse_user_id or "").strip(),
                    project_id=str(task.get("project_id") or "").strip(),
                    memory_session_id=str(task.get("memory_session_id") or "").strip(),
                    role="assistant",
                    content=outbound.message,
                    source=outbound.integration_id,
                    source_message_id=f"outbound:{outbound.outbox_message_id}",
                    created_at=outbound.created_at,
                )
            return {
                "status": "queued",
                "outbox_message_id": outbound.outbox_message_id,
                "integration_id": outbound.integration_id,
                "provider_key": outbound.provider_key,
                "channel_target": outbound.channel_target,
            }
        if event_type == "communication.deliver":
            if communication_router is None:
                return {"status": "delivery_unavailable"}
            task = event.get("task") if isinstance(event.get("task"), dict) else {}
            task_metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
            tactical = task.get("hierarchical_state") if isinstance(task.get("hierarchical_state"), dict) else {}
            phase = tactical.get("phase") if isinstance(tactical.get("phase"), dict) else {}
            origin_raw = event.get("origin") if isinstance(event.get("origin"), dict) else {}
            origin = ChannelAddress(**origin_raw) if origin_raw else channel_address_from_metadata(task.get("metadata") if isinstance(task.get("metadata"), dict) else {})
            if origin is None:
                return {"status": "origin_unavailable"}
            return communication_router.deliver(
                sender_user_id=str(event.get("sender_user_id") or task.get("user") or "").strip(),
                origin=origin,
                recipient_reference=str(event.get("recipient") or "").strip(),
                message=str(event.get("message") or "").strip(),
                expects_reply=bool(event.get("expects_reply")),
                asset_ids=list(event.get("asset_ids") or []),
                task_id=str(task.get("task_id") or ""),
                occurrence_key=str(task_metadata.get("occurrence_key") or ""),
                scheduled_task_id=str(task_metadata.get("scheduled_task_id") or ""),
                phase_id=str(phase.get("phase_id") or ""),
                subgoal_id=str(tactical.get("active_subgoal_id") or ""),
            )
        if event_type != "question.deliver":
            return {"ignored": True}
        question = event.get("question") if isinstance(event.get("question"), dict) else {}
        task = event.get("task") if isinstance(event.get("task"), dict) else {}
        task_metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        scheduled_question = ({"scheduled_task_id": str(task_metadata.get("scheduled_task_id") or ""),
                               "occurrence_key": str(task_metadata.get("occurrence_key") or ""),
                               "scheduled_question": True}
                              if task_metadata.get("source") == "scheduled_task" else {})
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
                    project_id=str(question.get("project_id") or task.get("project_id") or "").strip(),
                    metadata={
                        **scheduled_question,
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
            project_id=str(question.get("project_id") or task.get("project_id") or "").strip(),
            metadata={
                **scheduled_question,
                "question": dict(question),
                "origin_channel": origin.to_dict() if origin is not None else {},
                "respondent_channel": resolved.address.to_dict(),
            },
        )
        if conversation_store is not None:
            conversation_store.record(
                owner_user_id=respondent,
                project_id=str(question.get("project_id") or task.get("project_id") or "").strip(),
                memory_session_id=str(task.get("memory_session_id") or "").strip(),
                role="assistant",
                content=str(question.get("message") or "").strip(),
                source=outbound.integration_id,
                source_message_id=f"question:{str(question.get('question_id') or '').strip()}",
                created_at=outbound.created_at,
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
    identity_resolver: V2IdentityResolver | None = None,
    mirror_automation_messages_to_preferred_channel: bool = False,
    connection: sqlite3.Connection | None = None,
) -> OutboundMessage | None:
    """Project a completed task snapshot response into the outbox."""
    metadata = snapshot.metadata or {}
    task_state = metadata.get("task_state")
    if not isinstance(task_state, dict):
        return None
    # A prepared response is intermediate evidence, not permission to announce
    # completion. Parked tasks can retain one while awaiting a user answer.
    if task_state.get("status") != "completed":
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
    projected = outbox.enqueue(
        address=origin,
        message=response,
        kind="response",
        audience_user_id=str(task_state.get("user") or origin.alphonse_user_id or "").strip(),
        correlation_id=str(task_state.get("correlation_id") or "").strip(),
        task_id=str(task_state.get("task_id") or "").strip(),
        project_id=str(task_state.get("project_id") or "").strip(),
        metadata={
            "source": "task_result",
            "tool": "respond",
            "project_id": str(task_state.get("project_id") or "").strip(),
            "scheduled_task_id": str(task_metadata.get("scheduled_task_id") or "").strip(),
            "scheduled_run_id": str(task_metadata.get("scheduled_run_id") or "").strip(),
            "occurrence_key": str(task_metadata.get("occurrence_key") or "").strip(),
        },
        idempotency_key=(
            f"scheduled-result:{task_metadata['occurrence_key']}"
            if str(task_metadata.get("occurrence_key") or "").strip() else ""
        ),
        connection=connection,
    )
    if mirror_automation_messages_to_preferred_channel and str(task_metadata.get("scheduled_task_id") or "").strip():
        _enqueue_preferred_automation_copy(
            outbox=outbox,
            original=projected,
            recipient_user_id=str(task_state.get("user") or origin.alphonse_user_id or "").strip(),
            resolver=identity_resolver,
            connection=connection,
        )
    return projected


def _enqueue_preferred_automation_copy(
    *,
    outbox: SQLiteOutboundStore,
    original: OutboundMessage,
    recipient_user_id: str,
    resolver: V2IdentityResolver | None,
    connection: sqlite3.Connection | None = None,
) -> None:
    if resolver is None or not recipient_user_id:
        logger.info(
            "automation preferred-channel copy skipped correlation_id=%s reason=%s",
            original.correlation_id,
            "identity_resolver_unavailable" if resolver is None else "recipient_missing",
        )
        return
    resolved = resolver.resolve_outbound_address(alphonse_user_id=recipient_user_id)
    preferred = resolved.address if resolved.resolved else None
    if preferred is None:
        logger.info(
            "automation preferred-channel copy skipped correlation_id=%s reason=%s",
            original.correlation_id,
            resolved.reason or "preferred_delivery_not_found",
        )
        return
    if (
        preferred.integration_id == original.integration_id
        and preferred.provider_key == original.provider_key
        and preferred.channel_target == original.channel_target
    ):
        logger.info("automation preferred-channel copy skipped correlation_id=%s reason=destination_matches_origin", original.correlation_id)
        return
    try:
        metadata = dict(original.metadata)
        metadata.pop("occurrence_key", None)
        metadata["automation_preferred_channel_copy"] = True
        metadata["automation_copy_origin"] = {
            "integration_id": original.integration_id,
            "provider_key": original.provider_key,
            "channel_target": original.channel_target,
        }
        outbox.enqueue(
            address=preferred,
            message=original.message,
            kind=original.kind,
            audience_user_id=recipient_user_id,
            correlation_id=original.correlation_id,
            task_id=original.task_id,
            project_id=original.project_id,
            metadata=metadata,
            idempotency_key=f"automation-copy:{original.outbox_message_id}:{preferred.integration_id}:{preferred.provider_key}:{preferred.channel_target}",
            connection=connection,
        )
        logger.info("automation preferred-channel copy queued correlation_id=%s", original.correlation_id)
    except Exception:
        logger.exception("automation preferred-channel copy failed correlation_id=%s", original.correlation_id)


def _latest_tool_result_response(task_state: dict[str, Any]) -> str:
    metadata = task_state.get("metadata") if isinstance(task_state.get("metadata"), dict) else {}
    prepared = metadata.get("prepared_user_response") if isinstance(metadata, dict) else None
    if isinstance(prepared, dict) and str(prepared.get("message") or "").strip():
        return str(prepared.get("message") or "").strip()
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
        if tool_id == "native.respond" and str(result.get("message") or "").strip():
            return str(result.get("message") or "").strip()
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
    if selector.project_id is not None:
        filters.append("project_id = ?")
        values.append(str(selector.project_id or "").strip())
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
        project_id=str(row["project_id"]),
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


def _reject_unsupported_statuses(conn: sqlite3.Connection) -> None:
    # A newer or corrupt store may contain a state this binary does not
    # understand. Treating that row as pending could dispatch a message that
    # was deliberately held or already has an uncertain delivery outcome.
    # Reject before renaming the table or copying any message contents.
    try:
        unsupported = conn.execute(
            """SELECT 1 FROM main.v2_outbox
               WHERE status IS NULL OR status NOT IN
                 ('pending', 'claimed', 'retry_wait', 'delivered', 'failed', 'unconfirmed')
               LIMIT 1"""
        ).fetchone()
    except sqlite3.Error:
        raise ValueError("outbox_status_schema_invalid") from None
    if unsupported is not None:
        raise ValueError("outbox_status_unsupported")


_OUTBOX_STATUSES = frozenset({'pending', 'claimed', 'retry_wait', 'delivered', 'failed', 'unconfirmed'})


def _outbox_status_schema(conn: sqlite3.Connection) -> frozenset[str]:
    """Accept only known status CHECK grammars, including older subsets.

    Unknown future states must not be accepted just because the schema also
    mentions `unconfirmed`. This parser deliberately fails closed for a CHECK
    expression it cannot recognize; it is not a mixed-version worker fence.
    """
    row = conn.execute(
        "SELECT sql FROM main.sqlite_master WHERE type='table' AND name='v2_outbox'"
    ).fetchone()
    sql = str(row[0] or '') if row is not None else ''
    # SQLite retains comments in sqlite_master. A commented-out CHECK must not
    # be mistaken for a real constraint. Known product schemas use none.
    if any(marker in sql for marker in ('--', '/*', '*/')):
        raise ValueError('outbox_status_schema_unsupported')
    # Mask string literals before locating the CHECK; its wording may occur in
    # a default string and is not itself a constraint. Preserve offsets to
    # extract only the actual IN-list from the original schema.
    masked = re.sub(r"'(?:''|[^'])*'", lambda match: ' ' * len(match.group()), sql)
    # Known schemas use unquoted identifiers. Refuse other quoting rather than
    # treating an identifier containing CHECK text as SQL syntax.
    if any(char in masked for char in "'\"`[]"):
        raise ValueError('outbox_status_schema_unsupported')
    matches = list(re.finditer(r"\bCHECK\s*\(\s*status\s+IN\s*\(([^()]*)\)\s*\)", masked, re.IGNORECASE))
    if len(matches) != 1:
        raise ValueError('outbox_status_schema_unsupported')
    raw = sql[matches[0].start(1):matches[0].end(1)]
    parts = re.findall(r"'([a-z_]+)'", raw, re.IGNORECASE)
    if (not parts or re.sub(r"'([a-z_]+)'|[\s,]", '', raw, flags=re.IGNORECASE)
            or len(parts) != len(set(parts)) or not set(parts) <= _OUTBOX_STATUSES):
        raise ValueError('outbox_status_schema_unsupported')
    return frozenset(parts)


_OUTBOX_REBUILD_COLUMNS = frozenset({
    'outbox_message_id', 'integration_id', 'provider_key', 'channel_target', 'message', 'kind',
    'audience_user_id', 'correlation_id', 'status', 'provider_message_id', 'reply_to_provider_message_id',
    'task_id', 'question_id', 'project_id', 'metadata_json', 'claimed_at', 'delivered_at', 'created_at',
    'updated_at', 'attempt_count', 'lease_owner', 'lease_expires_at', 'next_attempt_at', 'last_error', 'max_attempts',
})
_OUTBOX_ADDED_COLUMNS = frozenset({'project_id', 'attempt_count', 'lease_owner', 'lease_expires_at',
                                   'next_attempt_at', 'last_error', 'max_attempts'})
_OUTBOX_INDEX_COLUMNS = {
    'idx_v2_outbox_consumer': ('integration_id', 'channel_target', 'status', 'created_at'),
    'idx_v2_outbox_question': ('question_id', 'status'),
    'idx_v2_outbox_project': ('audience_user_id', 'project_id', 'status', 'created_at'),
}
_OUTBOX_NO_DEFAULT = frozenset({'outbox_message_id', 'integration_id', 'provider_key', 'channel_target',
                                 'message', 'kind', 'created_at', 'updated_at'})
_OUTBOX_INTEGER_DEFAULTS = {'attempt_count': '0', 'max_attempts': '5'}


def _validate_outbox_rebuild(conn: sqlite3.Connection) -> None:
    """Refuse a rebuild that would discard schema objects or external links."""
    info = list(conn.execute('PRAGMA main.table_xinfo(v2_outbox)'))
    columns = {str(row[1]) for row in info}
    if columns - _OUTBOX_REBUILD_COLUMNS or _OUTBOX_REBUILD_COLUMNS - columns - _OUTBOX_ADDED_COLUMNS:
        raise ValueError('outbox_rebuild_schema_unsupported')
    for row in info:
        name = str(row[1])
        expected_type = 'INTEGER' if name in _OUTBOX_INTEGER_DEFAULTS else 'TEXT'
        expected_default = (_OUTBOX_INTEGER_DEFAULTS[name] if name in _OUTBOX_INTEGER_DEFAULTS else
                            None if name in _OUTBOX_NO_DEFAULT else "'pending'" if name == 'status' else
                            "'{}'" if name == 'metadata_json' else "''")
        if (str(row[2]).upper() != expected_type or row[3] != 1 or row[4] != expected_default
                or row[5] != int(name == 'outbox_message_id') or row[6] != 0):
            raise ValueError('outbox_rebuild_schema_unsupported')
    sql_row = conn.execute("SELECT sql FROM main.sqlite_master WHERE type='table' AND name='v2_outbox'").fetchone()
    sql = str(sql_row[0] or '') if sql_row else ''
    if (len(re.findall(r'\bCHECK\s*\(', sql, re.IGNORECASE)) != 1
            or re.search(r'\b(?:COLLATE|GENERATED|UNIQUE)\b|\bWITHOUT\s+ROWID\b', sql, re.IGNORECASE)):
        raise ValueError('outbox_rebuild_schema_unsupported')
    # ALTER TABLE RENAME rewrites dependent views and triggers as well. Dropping
    # the renamed table would leave those references broken; refuse rather than
    # attempting to infer or rewrite household schema objects.
    for dependency in conn.execute("SELECT tbl_name,sql FROM main.sqlite_master WHERE type IN ('trigger','view')"):
        if (str(dependency[0]).casefold() == 'v2_outbox'
                or re.search(r'\bv2_outbox\b', str(dependency[1] or ''), re.IGNORECASE)):
            raise ValueError('outbox_rebuild_schema_unsupported')
    for index in conn.execute('PRAGMA main.index_list(v2_outbox)'):
        name, unique, origin, partial = str(index[1]), index[2], index[3], index[4]
        if origin == 'pk':
            continue
        if name not in _OUTBOX_INDEX_COLUMNS or unique or partial:
            raise ValueError('outbox_rebuild_schema_unsupported')
        # Only known constant names are interpolated into PRAGMA.
        key_columns = [row for row in conn.execute(f'PRAGMA main.index_xinfo({name})') if row[5] == 1]
        actual = tuple(str(row[2]) for row in key_columns)
        if (actual != _OUTBOX_INDEX_COLUMNS[name]
                or any(row[3] != 0 or str(row[4]).upper() != 'BINARY' for row in key_columns)):
            raise ValueError('outbox_rebuild_schema_unsupported')
    tables = conn.execute("SELECT name FROM main.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
    for row in tables:
        name = str(row[0])
        quoted = name.replace('"', '""')
        for foreign_key in conn.execute(f'PRAGMA main.foreign_key_list("{quoted}")'):
            if name == 'v2_outbox' or str(foreign_key[2]).casefold() == 'v2_outbox':
                raise ValueError('outbox_rebuild_schema_unsupported')


def _migrate_status_check(conn: sqlite3.Connection) -> None:
    _reject_unsupported_statuses(conn)
    if _outbox_status_schema(conn) == _OUTBOX_STATUSES:
        return
    if not conn.in_transaction:
        raise ValueError('outbox_migration_requires_transaction')
    _validate_outbox_rebuild(conn)
    conn.execute("ALTER TABLE v2_outbox RENAME TO v2_outbox_legacy")
    conn.execute(
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
          project_id TEXT NOT NULL DEFAULT '',
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
          CHECK (status IN ('pending', 'claimed', 'retry_wait', 'delivered', 'failed', 'unconfirmed'))
        ) STRICT;
        """
    )
    conn.execute(
        """
        INSERT INTO v2_outbox (
          outbox_message_id, integration_id, provider_key, channel_target, message,
          kind, audience_user_id, correlation_id, status, provider_message_id,
          reply_to_provider_message_id, task_id, question_id, project_id, metadata_json,
          claimed_at, delivered_at, created_at, updated_at, attempt_count,
          lease_owner, lease_expires_at, next_attempt_at, last_error, max_attempts
        )
        SELECT
          outbox_message_id, integration_id, provider_key, channel_target, message,
          kind, audience_user_id, correlation_id,
          status,
          provider_message_id, reply_to_provider_message_id, task_id, question_id,
          project_id, metadata_json, claimed_at, delivered_at, created_at, updated_at,
          attempt_count, lease_owner, lease_expires_at, next_attempt_at, last_error, max_attempts
        FROM v2_outbox_legacy
        """
    )
    conn.execute("DROP TABLE v2_outbox_legacy")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_v2_outbox_consumer ON v2_outbox (integration_id, channel_target, status, created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_v2_outbox_question ON v2_outbox (question_id, status)")


def _default_outbox_db_path() -> str:
    return str(default_database_path())
