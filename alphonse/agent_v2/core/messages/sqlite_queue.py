"""Durable inbound message queue for the v2 daemon."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from threading import RLock
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.messages.queue import MessageSelector
from alphonse.agent_v2.core.messages.queue import QueuedMessage
from alphonse.agent_v2.database import connect_database, default_database_path


class SQLiteMessageQueue:
    """SQLite-backed queue with processing recovery and explicit acknowledgement."""

    def __init__(self, db_path: str | Path = ":memory:", *, lease_owner: str = "default",
                 experimental_claim_fencing: bool = False) -> None:
        self.db_path = str(db_path)
        self._lock = RLock()
        self.lease_owner = str(lease_owner or "default").strip() or "default"
        if type(experimental_claim_fencing) is not bool:
            raise ValueError("claim_fencing_invalid")
        if experimental_claim_fencing and self.db_path == ":memory:":
            raise ValueError("claim_fencing_file_database_required")
        self.experimental_claim_fencing = experimental_claim_fencing
        self._memory_connection: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._memory_connection = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_connection.row_factory = sqlite3.Row
        self._ensure_schema()
        if self.experimental_claim_fencing:
            self._enable_experimental_claim_fencing()

    def _enable_experimental_claim_fencing(self) -> None:
        """Test-only entry point; old workers must be drained before any use."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _ensure_columns(conn, "v2_inbound_messages", {"claim_generation": "TEXT NOT NULL DEFAULT ''"})
            column = next(row for row in conn.execute("PRAGMA table_info(v2_inbound_messages)")
                          if row["name"] == "claim_generation")
            if str(column["type"]).upper() != "TEXT" or column["notnull"] != 1 or column["dflt_value"] != "''":
                raise ValueError("claim_fencing_schema_invalid")
            if conn.execute("""SELECT 1 FROM v2_inbound_messages WHERE claim_generation IS NULL
                OR (claim_generation != '' AND (length(claim_generation) != 32
                    OR claim_generation GLOB '*[^0-9a-f]*')) LIMIT 1""").fetchone():
                raise ValueError("claim_fencing_generation_invalid")
            if conn.execute("SELECT 1 FROM v2_inbound_messages WHERE status='processing' AND claim_generation='' LIMIT 1").fetchone():
                raise ValueError("claim_fencing_legacy_inflight")

    @classmethod
    def default(cls, *, lease_owner: str = "default") -> "SQLiteMessageQueue":
        return cls(default_database_path(), lease_owner=lease_owner)

    def enqueue(self, message: CoreMessage, *, message_id: str | None = None) -> QueuedMessage:
        queued = QueuedMessage(message=message, message_id=str(message_id or QueuedMessage(message).message_id))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if queued.message_id.startswith("event:"):
                receipt = conn.execute("SELECT status FROM v2_inbound_receipts WHERE message_id=?", (queued.message_id,)).fetchone()
                if receipt is not None:
                    return queued
                conn.execute("INSERT INTO v2_inbound_receipts VALUES(?,?,?)", (queued.message_id, "pending", _now_iso()))
            conn.execute(
                """
                INSERT INTO v2_inbound_messages (
                  message_id, sequence, timestamp, prompt, user_id, project_id, tag,
                  correlation_id, memory_session_id, metadata_json, queued_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
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
                    queued.message.memory_session_id,
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

    def list_pending(self, selector: MessageSelector | None = None, *, limit: int = 1000) -> list[QueuedMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM v2_inbound_messages
                WHERE (status = 'pending' OR (status = 'retry_wait' AND next_attempt_at <= ?))
                  {_where(selector)}
                ORDER BY sequence LIMIT ?
                """,
                (_now_iso(), *_values(selector), max(0, int(limit))),
            ).fetchall()
        return [_row_to_message(row) for row in rows]

    def dequeue(self, selector: MessageSelector | None = None) -> QueuedMessage | None:
        return self.claim_next(selector=selector)

    def claim_mode_blocked(self) -> bool:
        """Report incompatible same-binary claims without exposing task data.

        This is a snapshot for diagnostics, not an ownership fence. The actual
        claim repeats the check under its write transaction. Database errors must
        propagate so a caller never treats unavailable storage as a clear queue.
        Old binaries and raw SQL are not fenced by this helper.
        """
        with self._connect() as conn:
            return _claim_mode_conflict(conn, fenced=self.experimental_claim_fencing)

    def stage_scheduled_handoff(self, task: Any, question_store: Any, schedule_store: Any) -> bool:
        """Atomically pin a checkpoint before yielding to a verified occurrence."""
        if self.experimental_claim_fencing:
            raise RuntimeError("claim_fencing_scheduled_handoff_unavailable")
        if (self.db_path == ":memory:" or self.db_path != str(getattr(question_store, "db_path", ""))
                or self.db_path != str(getattr(schedule_store, "db_path", ""))):
            return False
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        if not task.message_id or not task.task_id:
            return False
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            guard = _legacy_unfenced_guard(conn, "q")
            active = conn.execute(
                f"""SELECT q.user_id FROM v2_inbound_messages q WHERE q.message_id=? AND q.status='processing' AND q.lease_owner=? {guard}
                   AND (q.user_id=? OR (json_valid(q.metadata_json) AND EXISTS (
                     SELECT 1 FROM v2_questions question WHERE question.question_id=json_extract(q.metadata_json, '$.answered_question_id')
                       AND question.task_id=? AND question.respondent_user_id=q.user_id AND question.status='answered')))""",
                (task.message_id, self.lease_owner, task.user, task.task_id),
            ).fetchone()
            due = conn.execute(
                """SELECT 1 FROM v2_inbound_messages q
                   JOIN v2_scheduled_task_executions e ON e.queued_message_id=q.message_id
                   JOIN v2_scheduled_tasks s ON s.scheduled_task_id=e.scheduled_task_id
                   WHERE ((q.status='pending' AND e.status='enqueued') OR
                          (q.status='retry_wait' AND q.next_attempt_at<=? AND e.status='processing'))
                     AND s.status='active' AND s.owner_user_id=q.user_id
                     AND json_valid(q.metadata_json)
                     AND json_extract(q.metadata_json, '$.source')='scheduled_task'
                     AND json_extract(q.metadata_json, '$.occurrence_key')=e.occurrence_key
                   LIMIT 1""", (_now_iso(),),
            ).fetchone()
            if active is None or due is None:
                return False
            if "v3_handoff_history_source_ids" not in metadata:
                sources = metadata.get("conversation_sources") or []
                metadata["v3_handoff_history_source_ids"] = [
                    str(item.get("source_id")) for item in sources
                    if isinstance(item, dict) and item.get("source_id")
                ]
            question_store.save_task_checkpoint(task, status="queued", connection=conn)
            conn.execute(
                "INSERT OR REPLACE INTO v2_scheduled_handoffs(message_id, task_id, owner_id) VALUES(?,?,?)",
                (task.message_id, task.task_id, str(active["user_id"])),
            )
            return True

    def handoff_task_id(self, message_id: str, owner_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT task_id FROM v2_scheduled_handoffs WHERE message_id=? AND owner_id=?",
                               (message_id, owner_id)).fetchone()
        return str(row[0]) if row else ""

    def stage_delivery_wait(self, task: Any, question_store: Any, outbox_message_id: str, recipient_user_id: str) -> bool:
        if self.experimental_claim_fencing:
            raise RuntimeError("claim_fencing_delivery_wait_unavailable")
        if self.db_path == ":memory:" or self.db_path != str(getattr(question_store, "db_path", "")):
            return False
        occurrence = str(task.metadata.get("occurrence_key") or "")
        if not occurrence or not task.task_id or not task.message_id:
            return False
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            guard = _legacy_unfenced_guard(conn, "q")
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_outbox'").fetchone() is None:
                return False
            row = conn.execute(
                f"""SELECT q.user_id FROM v2_inbound_messages q JOIN v2_outbox o ON o.outbox_message_id=?
                   WHERE q.message_id=? AND q.status='processing' AND q.lease_owner=? {guard}
                     AND (q.user_id=? OR (json_valid(q.metadata_json) AND EXISTS (
                       SELECT 1 FROM v2_questions question WHERE question.question_id=json_extract(q.metadata_json, '$.answered_question_id')
                         AND question.task_id=? AND question.respondent_user_id=q.user_id AND question.status='answered')))
                     AND o.task_id=? AND o.audience_user_id=? AND o.status IN ('pending','claimed','retry_wait')
                     AND json_valid(o.metadata_json) AND json_extract(o.metadata_json, '$.occurrence_key')=?""",
                (outbox_message_id, task.message_id, self.lease_owner, task.user, task.task_id,
                 task.task_id, recipient_user_id, occurrence),
            ).fetchone()
            if row is None:
                return False
            question_store.save_task_checkpoint(task, status="queued", connection=conn)
            conn.execute(
                "INSERT OR REPLACE INTO v2_scheduled_handoffs(message_id, task_id, owner_id, wait_outbox_id) VALUES(?,?,?,?)",
                (task.message_id, task.task_id, str(row["user_id"]), outbox_message_id),
            )
            return True

    def release_scheduled_handoff(self, message_id: str, *, lease_owner: str) -> bool:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            guard = _legacy_unfenced_guard(conn)
            handoff = conn.execute("SELECT wait_outbox_id FROM v2_scheduled_handoffs WHERE message_id=?", (message_id,)).fetchone()
            wait = bool(handoff and handoff["wait_outbox_id"])
            retry_at = (_now() + timedelta(seconds=5)).isoformat() if wait else ""
            cursor = conn.execute(
                f"""UPDATE v2_inbound_messages SET status=?, processing_at='', lease_owner='',
                   lease_expires_at='', next_attempt_at=?,
                   attempt_count=MAX(attempt_count-1, 0),
                   sequence=(SELECT MAX(sequence)+1 FROM v2_inbound_messages)
                   WHERE message_id=? AND status='processing' AND lease_owner=? {guard}
                     AND EXISTS (SELECT 1 FROM v2_scheduled_handoffs WHERE message_id=?)""",
                ("retry_wait" if wait else "pending", retry_at, message_id, lease_owner, message_id),
            )
            return cursor.rowcount == 1

    def cancel_scheduled_handoff(self, task: Any, question_store: Any, *, lease_owner: str) -> bool:
        """Cancel the checkpoint and queued task in the same transaction."""
        if self.db_path != str(getattr(question_store, "db_path", "")):
            return False
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            guard = _legacy_unfenced_guard(conn, "q")
            row = conn.execute(
                f"""SELECT 1 FROM v2_inbound_messages q JOIN v2_scheduled_handoffs h ON h.message_id=q.message_id
                   WHERE q.message_id=? AND q.status='processing' AND q.lease_owner=? AND h.task_id=? AND h.owner_id=q.user_id {guard}""",
                (task.message_id, lease_owner, task.task_id),
            ).fetchone()
            if row is None:
                return False
            task.status = "cancelled"
            task.outcome = {"status": "cancelled", "reason": "Execution was cancelled by the kill switch."}
            question_store.save_task_checkpoint(task, status="cancelled", connection=conn)
            conn.execute("DELETE FROM v2_inbound_messages WHERE message_id=?", (task.message_id,))
            conn.execute("DELETE FROM v2_scheduled_handoffs WHERE message_id=?", (task.message_id,))
            conn.execute("UPDATE v2_inbound_receipts SET status='cancelled', updated_at=? WHERE message_id=?", (_now_iso(), task.message_id))
            return True

    def resume_confirmed_scheduled_delivery(self, outbox_message_id: str, question_store: Any) -> bool:
        """Resume only a previously unconfirmed task after matching provider evidence."""
        if self.db_path == ":memory:" or self.db_path != str(getattr(question_store, "db_path", "")):
            return False
        from alphonse.agent_v2.core.intelligence.task_state import TaskState
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            guard = _legacy_unfenced_guard(conn, "q")
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_outbox'").fetchone() is None:
                return False
            row = conn.execute(
                """SELECT c.task_state_json, c.task_id, c.owner_id, e.occurrence_key
                   FROM v2_outbox o
                   JOIN v2_task_checkpoints c ON c.task_id=o.task_id AND c.status='failed'
                   JOIN v2_scheduled_task_executions e ON e.queued_message_id=c.task_id
                   JOIN v2_scheduled_tasks s ON s.scheduled_task_id=e.scheduled_task_id AND s.owner_user_id=c.owner_id
                   WHERE o.outbox_message_id=? AND o.status='delivered' AND s.status='active' AND e.status='unconfirmed'
                     AND json_valid(o.metadata_json) AND json_extract(o.metadata_json, '$.occurrence_key')=e.occurrence_key
                     AND json_valid(c.task_state_json)
                     AND json_extract(c.task_state_json, '$.outcome.reason')='scheduled_delivery_outcome_unavailable'
                     AND json_extract(c.task_state_json, '$.metadata.v3_unconfirmed_delivery_outbox_id')=o.outbox_message_id""",
                (outbox_message_id,),
            ).fetchone()
            if row is None:
                return False
            task = TaskState.from_dict(json.loads(row["task_state_json"]))
            queued = conn.execute(
                f"""SELECT q.user_id FROM v2_inbound_messages q WHERE q.message_id=? AND q.status='failed' {guard}
                   AND (q.user_id=? OR (json_valid(q.metadata_json) AND EXISTS (
                       SELECT 1 FROM v2_questions question
                       WHERE question.question_id=json_extract(q.metadata_json, '$.answered_question_id')
                         AND question.task_id=? AND question.respondent_user_id=q.user_id AND question.status='answered'
                   )))""", (task.message_id, task.user, task.task_id),
            ).fetchone()
            if queued is None or task.user != row["owner_id"]:
                return False
            retry = task.metadata.get("v3_tactical_delivery_retry")
            if isinstance(retry, dict) and task.hierarchical_state.get("status") == "blocked":
                task.hierarchical_state["status"] = "running"
            task.status = "running"
            task.outcome = None
            for key in ("v3_failure_response", "v3_terminal_error_type", "v3_unconfirmed_delivery_outbox_id", "v3_route"):
                task.metadata.pop(key, None)
            question_store.save_task_checkpoint(task, status="queued", connection=conn)
            conn.execute("INSERT OR REPLACE INTO v2_scheduled_handoffs(message_id,task_id,owner_id,wait_outbox_id) VALUES(?,?,?,'')",
                         (task.message_id, task.task_id, str(queued["user_id"])))
            conn.execute("""UPDATE v2_inbound_messages SET status='pending', attempt_count=0, last_error='',
                next_attempt_at='', sequence=(SELECT MAX(sequence)+1 FROM v2_inbound_messages) WHERE message_id=? AND status='failed'""",
                (task.message_id,))
            conn.execute("""UPDATE v2_scheduled_task_executions SET status='processing', error='', last_error='', updated_at=?
                WHERE occurrence_key=? AND status='unconfirmed'""", (_now_iso(), row["occurrence_key"]))
            return True

    def claim_next(
        self,
        selector: MessageSelector | None = None,
        *,
        lease_owner: str | None = None,
        lease_seconds: int = 60,
        include_owned: bool = False,
    ) -> QueuedMessage | None:
        owner = str(lease_owner or self.lease_owner or "default").strip() or "default"
        if self.experimental_claim_fencing and include_owned:
            raise ValueError("claim_fencing_owned_claim_requires_receipt")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            # Do not start work under both ownership protocols at once in this
            # binary. The lock makes the check atomic with acquisition; it does
            # not fence older binaries or writers using raw SQL.
            if _claim_mode_conflict(conn, fenced=self.experimental_claim_fencing):
                return None
            # Sample after obtaining the writer lock, not before a potentially
            # long lock wait that could otherwise create an expired claim.
            now = _now()
            lease_expires_at = (now + timedelta(seconds=max(1, int(lease_seconds)))).isoformat()
            legacy_guard = "" if self.experimental_claim_fencing else _legacy_unfenced_guard(conn)
            holds = _exclude_supersession_holds(conn)
            # Event receipts currently lack a generation after queue-row removal;
            # fenced event claims are unsupported until that lifecycle is fenced.
            identity_guard = (" AND trim(user_id) != '' AND trim(message_id) != '' AND message_id NOT LIKE 'event:%'"
                              if self.experimental_claim_fencing else "")
            if include_owned:
                owned = conn.execute(
                    f"""
                    SELECT * FROM v2_inbound_messages
                    WHERE status = 'processing' AND lease_owner = ?
                      {legacy_guard} {holds}
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
                  {legacy_guard} {holds} {identity_guard}
                  {_where(selector)}
                ORDER BY sequence LIMIT 1
                """,
                (now.isoformat(), *_values(selector)),
            ).fetchone()
            if row is None:
                return None
            generation_assignment = ", claim_generation = ?" if self.experimental_claim_fencing else ""
            generation_values = ()
            if self.experimental_claim_fencing:
                from .claim_fence import new_claim_generation
                generation = new_claim_generation()
                if generation == str(row["claim_generation"] or ""):
                    raise ValueError("claim_generation_reused")
                generation_values = (generation,)
            cursor = conn.execute(
                f"""
                UPDATE v2_inbound_messages
                SET status = 'processing',
                    processing_at = ?,
                    lease_owner = ?,
                    lease_expires_at = ?,
                    attempt_count = attempt_count + 1 {generation_assignment}
                WHERE message_id = ?
                  AND (status = 'pending' OR (status = 'retry_wait' AND next_attempt_at <= ?))
                  {legacy_guard} {holds} {identity_guard}
                """,
                (now.isoformat(), owner, lease_expires_at, *generation_values, str(row["message_id"]), now.isoformat()),
            )
            if cursor.rowcount != 1:
                return None
            updated = conn.execute(
                "SELECT * FROM v2_inbound_messages WHERE message_id = ?",
                (str(row["message_id"]),),
            ).fetchone()
        return _row_to_message(updated, include_claim=self.experimental_claim_fencing) if updated is not None else None

    def ack(self, message_id: str, lease_owner: str | None = None, *, outcome: str = "completed") -> bool:
        if outcome not in {"completed", "cancelled", "processing"}:
            raise ValueError("invalid_queue_outcome")
        owner = str(lease_owner or self.lease_owner or "default").strip() or "default"
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            guard = _legacy_unfenced_guard(conn)
            cursor = conn.execute(f"""DELETE FROM v2_inbound_messages
                WHERE message_id=? AND status='processing' AND lease_owner=? {guard}""", (message_id, owner))
            if cursor.rowcount:
                conn.execute("DELETE FROM v2_scheduled_handoffs WHERE message_id=?", (message_id,))
                conn.execute("UPDATE v2_inbound_receipts SET status=?,updated_at=? WHERE message_id=?",
                             (outcome, _now_iso(), message_id))
            return cursor.rowcount == 1

    def ack_fenced(self, receipt: Any, *, outcome: str = "completed", clock: Any = None) -> bool:
        """Refuse acknowledgement until terminal effects are durably fenced.

        A current receipt proves queue ownership only. It does not prove that
        the actor-bound task reached the requested terminal state, that its
        effects were reconciled, or that the response was delivered. Preserve
        the claim for a future receipt-bound disposition instead of deleting it.
        """
        if outcome not in {"completed", "cancelled", "processing"}:
            raise ValueError("invalid_queue_outcome")
        return False

    def release_scheduled_handoff_fenced(self, receipt: Any, *, clock: Any = None) -> bool:
        """Refuse release until receipt-bound handoff staging is implemented.

        The legacy handoff row contains neither the claim generation nor a
        verified checkpoint/effect boundary. Its presence cannot prove that a
        continuation is safe, even while the presented claim is current.
        Preserve both claim and row for explicit reconciliation.
        """
        return False

    def renew_claim_fenced(self, receipt: Any, *, lease_seconds: int,
                           is_cancelled: Any, clock: Any = None) -> bool:
        """Refuse renewal until checkpoint and durable clock fencing exist.

        A queue receipt and caller-supplied cancellation callback alone cannot
        establish that the task is still current. Nor can a pair of wall-clock
        samples detect a regression across separate renewals or process restart.
        Leave the claim unchanged for explicit recovery instead of reviving a
        stale worker. This does not affect legacy queue behavior.
        """
        from .claim_fence import ClaimReceipt
        if not self.experimental_claim_fencing or not isinstance(receipt, ClaimReceipt):
            return False
        if type(lease_seconds) is not int or not 1 <= lease_seconds <= 3600 or not callable(is_cancelled):
            raise ValueError("claim_renewal_arguments_invalid")
        if clock is not None and not callable(clock):
            raise ValueError("claim_renewal_arguments_invalid")
        return False

    def finish_event(self, message_id: str, outcome: str) -> None:
        if outcome not in {"completed", "cancelled", "failed"}:
            raise ValueError("invalid_queue_outcome")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            guard = _legacy_unfenced_guard(conn, "q")
            conn.execute(f"""UPDATE v2_inbound_receipts SET status=?,updated_at=?
                WHERE message_id=? AND status='processing' AND NOT EXISTS (
                  SELECT 1 FROM v2_inbound_messages q WHERE q.message_id=v2_inbound_receipts.message_id
                  {('AND NOT (' + guard.removeprefix(' AND ') + ')') if guard else 'AND 0'}
                )""", (outcome, _now_iso(), message_id))

    def cancel_event(self, message_id: str) -> bool:
        """Create a tombstone even if cancellation races the first insertion."""
        if not message_id.startswith("event:"):
            raise ValueError("event_message_required")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            guard = _legacy_unfenced_guard(conn)
            receipt = conn.execute("SELECT status FROM v2_inbound_receipts WHERE message_id=?", (message_id,)).fetchone()
            if receipt is not None and receipt[0] in {"completed", "failed", "processing"}:
                return False
            row = conn.execute(f"SELECT status FROM v2_inbound_messages WHERE message_id=? {guard}", (message_id,)).fetchone()
            if row is None and conn.execute("SELECT 1 FROM v2_inbound_messages WHERE message_id=?", (message_id,)).fetchone():
                return False  # A generation-bearing row belongs to its fenced lifecycle.
            if row is not None and row[0] == "processing":
                return False  # Active CAPD cancellation uses the daemon kill switch.
            conn.execute("DELETE FROM v2_inbound_messages WHERE message_id=?", (message_id,))
            conn.execute("""INSERT INTO v2_inbound_receipts VALUES(?,?,?)
                ON CONFLICT(message_id) DO UPDATE SET status=CASE
                WHEN status='completed' THEN status ELSE 'cancelled' END,updated_at=excluded.updated_at""",
                (message_id, "cancelled", _now_iso()))
            return True

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
            conn.execute("BEGIN IMMEDIATE")
            guard = _legacy_unfenced_guard(conn)
            row = conn.execute(
                f"""
                SELECT attempt_count FROM v2_inbound_messages
                WHERE message_id = ? AND status = 'processing' AND lease_owner = ? {guard}
                """,
                (str(message_id or "").strip(), owner),
            ).fetchone()
            if row is None:
                return False
            attempts = int(row["attempt_count"] or 0)
            terminal = max_attempts is not None and attempts >= max(1, int(max_attempts))
            cursor = conn.execute(
                f"""
                UPDATE v2_inbound_messages
                SET status = ?,
                    processing_at = '',
                    lease_owner = '',
                    lease_expires_at = '',
                    next_attempt_at = ?,
                    last_error = ?
                WHERE message_id = ? AND status = 'processing' AND lease_owner = ? {guard}
                """,
                (
                    "failed" if terminal else "retry_wait",
                    "" if terminal else retry_at,
                    str(error or "").strip(),
                    str(message_id or "").strip(),
                    owner,
                ),
            )
            if cursor.rowcount and terminal:
                conn.execute("UPDATE v2_inbound_receipts SET status='failed',updated_at=? WHERE message_id=?", (_now_iso(), message_id))
            return cursor.rowcount == 1

    def requeue(self, message_id: str) -> bool:
        return self.retry(message_id, next_attempt_at=_now())

    def reclaim_expired(self, *, now: datetime | None = None, exclude_owner: str = "") -> int:
        # Expiry proves lost ownership, not that an in-flight effect did not
        # happen. Until receipt-bound reconciliation exists, requeueing a
        # fenced claim can silently repeat a mutation or delivery. Keep it in
        # processing for explicit recovery. Legacy queue behavior is unchanged.
        if self.experimental_claim_fencing:
            return 0
        timestamp = (now or _now()).isoformat()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            guard = " AND COALESCE(claim_generation, '') != ''" if self.experimental_claim_fencing else _legacy_unfenced_guard(conn)
            holds = _exclude_supersession_holds(conn)
            cursor = conn.execute(
                f"""
                UPDATE v2_inbound_messages
                SET status = 'pending',
                    processing_at = '',
                    lease_owner = '',
                    lease_expires_at = ''
                WHERE status = 'processing'
                  {guard} {holds}
                  AND (? = '' OR lease_owner != ?)
                  AND (
                    (lease_expires_at != '' AND lease_expires_at <= ?)
                    OR (lease_owner = '' AND lease_expires_at = '')
                  )
                """,
                (exclude_owner, exclude_owner, timestamp),
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

    def status_for(self, message_id: str) -> str:
        """Return the durable state of one message without exposing its content."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM v2_inbound_messages WHERE message_id=?",
                (str(message_id or "").strip(),),
            ).fetchone()
            if row is None:
                row = conn.execute("SELECT status FROM v2_inbound_receipts WHERE message_id=?", (str(message_id),)).fetchone()
        return str(row["status"] or "") if row is not None else ""

    def turns_ahead(self, message_id: str) -> int:
        """Count independently runnable tasks ahead of a queued message."""
        with self._connect() as conn:
            target = conn.execute(
                "SELECT sequence FROM v2_inbound_messages WHERE message_id=?",
                (str(message_id or "").strip(),),
            ).fetchone()
            if target is None:
                return 0
            row = conn.execute(
                """
                SELECT COUNT(*) AS count FROM v2_inbound_messages
                WHERE sequence < ?
                  AND status IN ('pending', 'processing', 'retry_wait')
                  AND json_extract(metadata_json, '$.routing_disposition') = 'pdca_task'
                """,
                (int(target["sequence"]),),
            ).fetchone()
        return int(row["count"] or 0) if row is not None else 0

    def has_pending_memory_session(self, session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM v2_inbound_messages WHERE memory_session_id=? AND status IN ('pending','processing','retry_wait') LIMIT 1",
                (str(session_id or "").strip(),),
            ).fetchone()
        return row is not None

    def _connect(self) -> sqlite3.Connection:
        if self._memory_connection is not None:
            return _ConnectionProxy(self._memory_connection, self._lock)
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return connect_database(path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS v2_inbound_receipts (
                  message_id TEXT PRIMARY KEY,status TEXT NOT NULL,updated_at TEXT NOT NULL
                ) STRICT;
                CREATE TABLE IF NOT EXISTS v2_scheduled_handoffs (
                  message_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, owner_id TEXT NOT NULL,
                  wait_outbox_id TEXT NOT NULL DEFAULT ''
                ) STRICT;
                CREATE TABLE IF NOT EXISTS v2_inbound_messages (
                  message_id TEXT PRIMARY KEY,
                  sequence INTEGER NOT NULL,
                  timestamp TEXT NOT NULL,
                  prompt TEXT NOT NULL,
                  user_id TEXT NOT NULL,
                  project_id TEXT NOT NULL DEFAULT '',
                  tag TEXT NOT NULL DEFAULT '',
                  correlation_id TEXT NOT NULL DEFAULT '',
                  memory_session_id TEXT NOT NULL DEFAULT '',
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
            conn.execute("""INSERT OR IGNORE INTO v2_inbound_receipts
                SELECT message_id,status,queued_at FROM v2_inbound_messages WHERE message_id LIKE 'event:%'""")
            _ensure_columns(conn, "v2_scheduled_handoffs", {"wait_outbox_id": "TEXT NOT NULL DEFAULT ''"})
            _ensure_columns(
                conn,
                "v2_inbound_messages",
                {
                    "attempt_count": "INTEGER NOT NULL DEFAULT 0",
                    "lease_owner": "TEXT NOT NULL DEFAULT ''",
                    "lease_expires_at": "TEXT NOT NULL DEFAULT ''",
                    "next_attempt_at": "TEXT NOT NULL DEFAULT ''",
                    "last_error": "TEXT NOT NULL DEFAULT ''",
                    "memory_session_id": "TEXT NOT NULL DEFAULT ''",
                },
            )


class _ConnectionProxy:
    def __init__(self, connection: sqlite3.Connection, lock: RLock) -> None:
        self.connection = connection
        self.lock = lock

    def __enter__(self) -> sqlite3.Connection:
        self.lock.acquire()
        return self.connection

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        try:
            if exc_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()
        finally:
            self.lock.release()
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
        if selector.message_id is not None:
            filters.append("message_id = ?")
    return f"AND {' AND '.join(filters)}" if filters else ""


def _values(selector: MessageSelector | None) -> tuple[str, ...]:
    if selector is None:
        return ()
    values: list[str] = []
    for value in (selector.user, selector.project_id, selector.tag, selector.correlation_id, selector.message_id):
        if value is not None:
            values.append(str(value))
    return tuple(values)


def _row_to_message(row: sqlite3.Row, *, include_claim: bool = False) -> QueuedMessage:
    timestamp = datetime.fromisoformat(str(row["timestamp"]))
    queued_at = datetime.fromisoformat(str(row["queued_at"]))
    metadata = json.loads(str(row["metadata_json"] or "{}"))
    claim = None
    if include_claim:
        from .claim_fence import ClaimReceipt
        claim = ClaimReceipt(str(row["message_id"]), str(row["user_id"]), str(row["lease_owner"]),
                             str(row["claim_generation"]))
    return QueuedMessage(
        message=CoreMessage(
            timestamp=timestamp,
            prompt=str(row["prompt"]),
            user=str(row["user_id"]),
            project_id=str(row["project_id"] or ""),
            memory_session_id=str(row["memory_session_id"] or ""),
            tag=str(row["tag"] or ""),
            correlation_id=str(row["correlation_id"] or ""),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        ),
        message_id=str(row["message_id"]),
        queued_at=queued_at,
        sequence=int(row["sequence"]),
        claim_receipt=claim,
    )


def _legacy_unfenced_guard(conn: sqlite3.Connection, alias: str = "") -> str:
    """Do not let this binary's legacy operations consume fenced rows."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(v2_inbound_messages)")}
    if "claim_generation" not in columns:
        return ""
    if alias not in {"", "q"}:
        raise ValueError("claim_guard_alias_invalid")
    prefix = f"{alias}." if alias else ""
    return f" AND COALESCE({prefix}claim_generation, '') = ''"


def _claim_mode_conflict(conn: sqlite3.Connection, *, fenced: bool) -> bool:
    """Prevent concurrent claim modes among queue clients in this binary.

    A dispatch caller must hold the SQLite write lock; a diagnostic caller gets
    only a snapshot. Generation-bearing rows remain owned by the fenced lifecycle
    even when pending or held. This does not protect against legacy binaries,
    raw SQL, or external effects, and is not a mixed-version or production
    activation barrier.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(v2_inbound_messages)")}
    if not {"message_id", "status"} <= columns:
        raise sqlite3.OperationalError("claim_mode_schema_unavailable")
    if "claim_generation" not in columns:
        return fenced
    if fenced:
        condition = ("claim_generation IS NULL OR (claim_generation != '' AND "
                     "(length(claim_generation) != 32 OR claim_generation GLOB '*[^0-9a-f]*')) OR "
                     "(status = 'processing' AND claim_generation = '')")
    else:
        condition = "claim_generation IS NULL OR claim_generation != ''"
    return conn.execute(f"SELECT 1 FROM v2_inbound_messages WHERE {condition} LIMIT 1").fetchone() is not None


def _exclude_supersession_holds(conn: sqlite3.Connection) -> str:
    if conn.execute("SELECT 1 FROM main.sqlite_master WHERE type='table' AND name='v2_supersession_holds'").fetchone():
        return " AND NOT EXISTS (SELECT 1 FROM main.v2_supersession_holds h WHERE h.message_id=v2_inbound_messages.message_id)"
    return ""


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
