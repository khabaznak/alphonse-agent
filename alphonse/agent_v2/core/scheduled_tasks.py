"""Scheduled task storage and runner for Alphonse agent v2."""

from __future__ import annotations

import json
import os
import sqlite3
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

from dateutil.rrule import rrulestr

from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.database import connect_database, default_database_path

ScheduleKind = Literal["once", "rrule"]
ScheduledTaskStatus = Literal["active", "paused", "completed", "cancelled", "failed"]
ExecutionStatus = Literal[
    "pending",
    "claimed",
    "enqueued",
    "processing",
    "response_pending",
    "retry_wait",
    "delivered",
    "failed",
    "delivery_failed",
]


@dataclass(frozen=True)
class ScheduledTaskRecord:
    scheduled_task_id: str
    owner_user_id: str
    project_id: str
    name: str
    description: str
    prompt: str
    origin_channel: dict[str, Any]
    schedule: dict[str, Any]
    timezone: str
    status: ScheduledTaskStatus
    next_run_at: str | None
    last_run_at: str | None
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduled_task_id": self.scheduled_task_id,
            "owner_user_id": self.owner_user_id,
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "prompt": self.prompt,
            "origin_channel": dict(self.origin_channel),
            "schedule": dict(self.schedule),
            "timezone": self.timezone,
            "status": self.status,
            "next_run_at": self.next_run_at,
            "last_run_at": self.last_run_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ScheduledTaskExecutionRecord:
    scheduled_task_id: str
    project_id: str
    run_id: str
    status: ExecutionStatus
    queued_message_id: str | None
    started_at: str
    finished_at: str | None
    error: str
    occurrence_key: str = ""
    attempt_count: int = 0
    lease_owner: str = ""
    lease_expires_at: str = ""
    next_attempt_at: str = ""
    response_outbox_id: str = ""
    last_error: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduled_task_id": self.scheduled_task_id,
            "project_id": self.project_id,
            "run_id": self.run_id,
            "status": self.status,
            "queued_message_id": self.queued_message_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "occurrence_key": self.occurrence_key,
            "attempt_count": self.attempt_count,
            "lease_owner": self.lease_owner,
            "lease_expires_at": self.lease_expires_at,
            "next_attempt_at": self.next_attempt_at,
            "response_outbox_id": self.response_outbox_id,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ScheduledOccurrence:
    task: ScheduledTaskRecord
    occurrence_key: str
    run_id: str
    scheduled_run_at: str
    attempt_count: int
    lease_owner: str
    lease_expires_at: str


class ScheduledTaskStore:
    """SQLite-backed v2 scheduled task store."""

    def __init__(self, db_path: str | None = ":memory:") -> None:
        self.db_path = str(db_path or ":memory:")
        self._memory_connection: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._memory_connection = sqlite3.connect(":memory:")
            self._memory_connection.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls) -> "ScheduledTaskStore":
        return cls(_default_schedule_db_path())

    def create_task(
        self,
        *,
        owner_user_id: str,
        project_id: str = "",
        name: str,
        description: str = "",
        prompt: str,
        schedule_kind: ScheduleKind,
        run_at: str | None = None,
        rrule: str | None = None,
        dtstart: str | None = None,
        origin_channel: dict[str, Any] | None = None,
        timezone_name: str = "UTC",
        enabled: bool = True,
        now: datetime | None = None,
    ) -> ScheduledTaskRecord:
        owner = str(owner_user_id or "").strip()
        name_value = str(name or "").strip()
        prompt_value = str(prompt or "").strip()
        if not owner:
            raise ValueError("scheduled_task_owner_required")
        if not name_value:
            raise ValueError("scheduled_task_name_required")
        if not prompt_value:
            raise ValueError("scheduled_task_prompt_required")
        project_value = str(project_id or "").strip()
        if not project_value:
            raise ValueError("scheduled_task_project_required")
        timezone_value = _normalize_timezone(timezone_name)
        schedule = _build_schedule(
            schedule_kind=schedule_kind,
            run_at=run_at,
            rrule=rrule,
            dtstart=dtstart,
            timezone_name=timezone_value,
        )
        current = now or _now_utc()
        next_run_at = compute_next_run_at(schedule=schedule, timezone_name=timezone_value, after=current)
        now_text = _now_iso()
        record = ScheduledTaskRecord(
            scheduled_task_id=f"scheduled_task_{uuid4().hex[:16]}",
            owner_user_id=owner,
            project_id=project_value,
            name=name_value,
            description=str(description or "").strip(),
            prompt=prompt_value,
            origin_channel=dict(origin_channel or {}),
            schedule=schedule,
            timezone=timezone_value,
            status="active" if enabled else "paused",
            next_run_at=next_run_at if enabled else None,
            last_run_at=None,
            created_at=now_text,
            updated_at=now_text,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_scheduled_tasks (
                  scheduled_task_id, owner_user_id, project_id, name, description, prompt,
                  schedule_json, origin_channel_json, timezone, status, next_run_at, last_run_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _task_values(record),
            )
        return record

    def get_task(self, scheduled_task_id: str) -> ScheduledTaskRecord | None:
        task_id = str(scheduled_task_id or "").strip()
        if not task_id:
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM v2_scheduled_tasks WHERE scheduled_task_id = ?", (task_id,)).fetchone()
        return _task_from_row(row)

    def get_task_for_owner(self, scheduled_task_id: str, *, owner_user_id: str) -> ScheduledTaskRecord | None:
        task = self.get_task(scheduled_task_id)
        return task if task is not None and task.owner_user_id == str(owner_user_id or "").strip() else None

    def list_tasks(
        self,
        *,
        owner_user_id: str | None = None,
        project_id: str | None = None,
        status: ScheduledTaskStatus | None = None,
        limit: int = 100,
    ) -> list[ScheduledTaskRecord]:
        filters: list[str] = []
        values: list[Any] = []
        if owner_user_id is not None:
            filters.append("owner_user_id = ?")
            values.append(str(owner_user_id or "").strip())
        if project_id is not None:
            filters.append("project_id = ?")
            values.append(str(project_id or "").strip())
        if status is not None:
            filters.append("status = ?")
            values.append(_normalize_status(status))
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        values.append(max(1, min(int(limit), 1000)))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM v2_scheduled_tasks
                {where}
                ORDER BY next_run_at IS NULL, next_run_at, updated_at
                LIMIT ?
                """,
                tuple(values),
            ).fetchall()
        return [_task_from_row(row) for row in rows if _task_from_row(row) is not None]

    def due_tasks(self, *, now: datetime | None = None) -> list[ScheduledTaskRecord]:
        current = _as_utc(now or _now_utc()).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM v2_scheduled_tasks
                WHERE status = 'active'
                  AND next_run_at IS NOT NULL
                  AND next_run_at <= ?
                ORDER BY next_run_at, created_at
                """,
                (current,),
            ).fetchall()
        return [_task_from_row(row) for row in rows if _task_from_row(row) is not None]

    def pause_task(self, scheduled_task_id: str) -> ScheduledTaskRecord:
        task = self._require_task(scheduled_task_id)
        if task.status != "active":
            raise ValueError("scheduled_task_not_active")
        return self._set_status(scheduled_task_id, "paused", clear_next=True)

    def resume_task(self, scheduled_task_id: str, *, now: datetime | None = None) -> ScheduledTaskRecord:
        task = self._require_task(scheduled_task_id)
        if task.status != "paused":
            raise ValueError("scheduled_task_not_paused")
        next_run_at = compute_next_run_at(schedule=task.schedule, timezone_name=task.timezone, after=now or _now_utc())
        updated = _replace_task(task, status="active", next_run_at=next_run_at, updated_at=_now_iso())
        self._save_task(updated)
        return updated

    def cancel_task(self, scheduled_task_id: str) -> ScheduledTaskRecord:
        return self._set_status(scheduled_task_id, "cancelled", clear_next=True)

    def update_task(self, scheduled_task_id: str, *, name: str, prompt: str) -> ScheduledTaskRecord:
        task = self._require_task(scheduled_task_id)
        if task.status not in {"active", "paused"}:
            raise ValueError("scheduled_task_not_editable")
        name_value = str(name or "").strip()
        prompt_value = str(prompt or "").strip()
        if not name_value:
            raise ValueError("scheduled_task_name_required")
        if not prompt_value:
            raise ValueError("scheduled_task_prompt_required")
        updated = _replace_task(task, name=name_value, prompt=prompt_value, updated_at=_now_iso())
        self._save_task(updated)
        return updated

    def delete_task(self, scheduled_task_id: str) -> bool:
        task = self._require_task(scheduled_task_id)
        with self._connect() as conn:
            conn.execute("DELETE FROM v2_scheduled_task_executions WHERE scheduled_task_id = ?", (task.scheduled_task_id,))
            return conn.execute("DELETE FROM v2_scheduled_tasks WHERE scheduled_task_id = ?", (task.scheduled_task_id,)).rowcount == 1

    def complete_task(self, scheduled_task_id: str, *, last_run_at: str | None = None) -> ScheduledTaskRecord:
        task = self._require_task(scheduled_task_id)
        updated = _replace_task(
            task,
            status="completed",
            next_run_at=None,
            last_run_at=last_run_at or _now_iso(),
            updated_at=_now_iso(),
        )
        self._save_task(updated)
        return updated

    def mark_failed(self, scheduled_task_id: str, *, error: str) -> ScheduledTaskRecord:
        _ = error
        return self._set_status(scheduled_task_id, "failed", clear_next=False)

    def update_after_run(
        self,
        task: ScheduledTaskRecord,
        *,
        now: datetime,
    ) -> ScheduledTaskRecord:
        run_at = _as_utc(now).isoformat()
        if str(task.schedule.get("kind") or "") == "once":
            return self.complete_task(task.scheduled_task_id, last_run_at=run_at)
        next_run_at = compute_next_run_at(schedule=task.schedule, timezone_name=task.timezone, after=now)
        updated = _replace_task(
            task,
            last_run_at=run_at,
            next_run_at=next_run_at,
            status="active" if next_run_at else "completed",
            updated_at=_now_iso(),
        )
        self._save_task(updated)
        return updated

    def record_execution(
        self,
        *,
        scheduled_task_id: str,
        run_id: str,
        status: ExecutionStatus,
        queued_message_id: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        error: str = "",
    ) -> ScheduledTaskExecutionRecord:
        task = self._require_task(scheduled_task_id)
        record = ScheduledTaskExecutionRecord(
            scheduled_task_id=str(scheduled_task_id or "").strip(),
            project_id=task.project_id,
            run_id=str(run_id or "").strip(),
            status=_normalize_execution_status(status),
            queued_message_id=str(queued_message_id or "").strip() or None,
            started_at=started_at or _now_iso(),
            finished_at=finished_at,
            error=str(error or "").strip(),
            occurrence_key=f"{str(scheduled_task_id or '').strip()}:{str(run_id or '').strip()}",
            attempt_count=1,
            last_error=str(error or "").strip(),
            created_at=started_at or _now_iso(),
            updated_at=finished_at or _now_iso(),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_scheduled_task_executions (
                  scheduled_task_id, project_id, occurrence_key, run_id, status, queued_message_id,
                  started_at, finished_at, error, attempt_count, last_error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.scheduled_task_id,
                    record.project_id,
                    record.occurrence_key,
                    record.run_id,
                    record.status,
                    record.queued_message_id,
                    record.started_at,
                    record.finished_at,
                    record.error,
                    record.attempt_count,
                    record.last_error,
                    record.created_at,
                    record.updated_at,
                ),
            )
        return record

    def list_executions(self, *, scheduled_task_id: str, limit: int = 100) -> list[ScheduledTaskExecutionRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM v2_scheduled_task_executions
                WHERE scheduled_task_id = ?
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (str(scheduled_task_id or "").strip(), max(1, min(int(limit), 1000))),
            ).fetchall()
        return [_execution_from_row(row) for row in rows if _execution_from_row(row) is not None]

    def claim_due_occurrences(
        self,
        *,
        worker_id: str,
        now: datetime | None = None,
        lease_seconds: float = 30.0,
        limit: int = 20,
    ) -> list[ScheduledOccurrence]:
        current = _as_utc(now or _now_utc())
        now_text = current.isoformat()
        lease_until = (current + timedelta(seconds=max(1.0, float(lease_seconds)))).isoformat()
        owner = str(worker_id or "").strip()
        if not owner:
            raise ValueError("scheduled_worker_id_required")
        claimed: list[ScheduledOccurrence] = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT * FROM v2_scheduled_tasks
                WHERE status = 'active' AND next_run_at IS NOT NULL AND next_run_at <= ?
                ORDER BY next_run_at, created_at LIMIT ?
                """,
                (now_text, max(1, min(int(limit), 100))),
            ).fetchall()
            for row in rows:
                task = _task_from_row(row)
                if task is None or not task.next_run_at:
                    continue
                scheduled_run_at = str(task.next_run_at)
                occurrence_key = f"{task.scheduled_task_id}:{scheduled_run_at}"
                run_id = f"scheduled_run_{hashlib.sha256(occurrence_key.encode()).hexdigest()[:16]}"
                existing = conn.execute(
                    "SELECT * FROM v2_scheduled_task_executions WHERE occurrence_key = ?",
                    (occurrence_key,),
                ).fetchone()
                if existing is None:
                    conn.execute(
                        """
                        INSERT INTO v2_scheduled_task_executions (
                          scheduled_task_id, project_id, occurrence_key, run_id, status, queued_message_id,
                          started_at, finished_at, error, attempt_count, lease_owner,
                          lease_expires_at, next_attempt_at, response_outbox_id, last_error,
                          created_at, updated_at
                        ) VALUES (?, ?, ?, ?, 'pending', '', ?, '', '', 0, '', '', '', '', '', ?, ?)
                        """,
                        (task.scheduled_task_id, task.project_id, occurrence_key, run_id, now_text, now_text, now_text),
                    )
                    existing = conn.execute(
                        "SELECT * FROM v2_scheduled_task_executions WHERE occurrence_key = ?",
                        (occurrence_key,),
                    ).fetchone()
                status = str(existing["status"] or "")
                expired = str(existing["lease_expires_at"] or "") <= now_text
                if status in {"delivered", "failed", "delivery_failed", "enqueued", "response_pending"}:
                    continue
                if status == "claimed" and not expired and str(existing["lease_owner"] or "") != owner:
                    continue
                attempt = int(existing["attempt_count"] or 0) + 1
                conn.execute(
                    """
                    UPDATE v2_scheduled_task_executions
                    SET status = 'claimed', attempt_count = ?, lease_owner = ?, lease_expires_at = ?,
                        next_attempt_at = '', updated_at = ?, last_error = ''
                    WHERE occurrence_key = ?
                    """,
                    (attempt, owner, lease_until, now_text, occurrence_key),
                )
                updated_task = task
                if str(task.schedule.get("kind") or "") == "once":
                    updated_task = _replace_task(task, next_run_at=None)
                else:
                    updated_task = _replace_task(
                        task,
                        next_run_at=compute_next_run_at(
                            schedule=task.schedule,
                            timezone_name=task.timezone,
                            after=current,
                        ),
                    )
                self._save_task_connection(conn, updated_task)
                claimed.append(
                    ScheduledOccurrence(
                        task=task,
                        occurrence_key=occurrence_key,
                        run_id=run_id,
                        scheduled_run_at=scheduled_run_at,
                        attempt_count=attempt,
                        lease_owner=owner,
                        lease_expires_at=lease_until,
                    )
                )
            conn.commit()
        return claimed

    def claim_retry_occurrences(
        self,
        *,
        worker_id: str,
        now: datetime | None = None,
        lease_seconds: float = 30.0,
        limit: int = 20,
    ) -> list[ScheduledOccurrence]:
        current = _as_utc(now or _now_utc())
        now_text = current.isoformat()
        lease_until = (current + timedelta(seconds=max(1.0, float(lease_seconds)))).isoformat()
        owner = str(worker_id or "").strip()
        claimed: list[ScheduledOccurrence] = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT e.occurrence_key, e.run_id, e.attempt_count, e.lease_expires_at,
                       e.status AS execution_status, t.*
                FROM v2_scheduled_task_executions e
                JOIN v2_scheduled_tasks t ON t.scheduled_task_id = e.scheduled_task_id
                WHERE e.status = 'retry_wait' AND e.next_attempt_at != '' AND e.next_attempt_at <= ?
                ORDER BY e.next_attempt_at LIMIT ?
                """,
                (now_text, max(1, min(int(limit), 100))),
            ).fetchall()
            for row in rows:
                if str(row["lease_expires_at"] or "") > now_text:
                    continue
                task = _task_from_row(row)
                if task is None:
                    continue
                attempt = int(row["attempt_count"] or 0) + 1
                conn.execute(
                    """
                    UPDATE v2_scheduled_task_executions
                    SET status = 'claimed', attempt_count = ?, lease_owner = ?, lease_expires_at = ?,
                        next_attempt_at = '', updated_at = ?
                    WHERE occurrence_key = ? AND status = 'retry_wait'
                    """,
                    (attempt, owner, lease_until, now_text, str(row["occurrence_key"])),
                )
                claimed.append(
                    ScheduledOccurrence(
                        task=task,
                        occurrence_key=str(row["occurrence_key"]),
                        run_id=str(row["run_id"]),
                        scheduled_run_at=str(row["occurrence_key"]).rsplit(":", 1)[-1],
                        attempt_count=attempt,
                        lease_owner=owner,
                        lease_expires_at=lease_until,
                    )
                )
            conn.commit()
        return claimed

    def claim_expired_occurrences(
        self,
        *,
        worker_id: str,
        now: datetime | None = None,
        lease_seconds: float = 30.0,
        limit: int = 20,
    ) -> list[ScheduledOccurrence]:
        current = _as_utc(now or _now_utc())
        now_text = current.isoformat()
        lease_until = (current + timedelta(seconds=max(1.0, float(lease_seconds)))).isoformat()
        owner = str(worker_id or "").strip()
        if not owner:
            raise ValueError("scheduled_worker_id_required")
        claimed: list[ScheduledOccurrence] = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT e.occurrence_key, e.run_id, e.attempt_count, e.lease_expires_at,
                       e.status AS execution_status, t.*
                FROM v2_scheduled_task_executions e
                JOIN v2_scheduled_tasks t ON t.scheduled_task_id = e.scheduled_task_id
                WHERE e.status = 'claimed'
                  AND e.lease_expires_at != ''
                  AND e.lease_expires_at <= ?
                ORDER BY e.lease_expires_at LIMIT ?
                """,
                (now_text, max(1, min(int(limit), 100))),
            ).fetchall()
            for row in rows:
                task = _task_from_row(row)
                if task is None:
                    continue
                attempt = int(row["attempt_count"] or 0) + 1
                cursor = conn.execute(
                    """
                    UPDATE v2_scheduled_task_executions
                    SET status = 'claimed', attempt_count = ?, lease_owner = ?, lease_expires_at = ?,
                        next_attempt_at = '', updated_at = ?, last_error = ''
                    WHERE occurrence_key = ?
                      AND status = 'claimed'
                      AND lease_expires_at != ''
                      AND lease_expires_at <= ?
                    """,
                    (attempt, owner, lease_until, now_text, str(row["occurrence_key"]), now_text),
                )
                if cursor.rowcount != 1:
                    continue
                claimed.append(
                    ScheduledOccurrence(
                        task=task,
                        occurrence_key=str(row["occurrence_key"]),
                        run_id=str(row["run_id"]),
                        scheduled_run_at=str(row["occurrence_key"]).rsplit(":", 1)[-1],
                        attempt_count=attempt,
                        lease_owner=owner,
                        lease_expires_at=lease_until,
                    )
                )
            conn.commit()
        return claimed

    def mark_occurrence_enqueued(self, occurrence_key: str, *, worker_id: str, message_id: str) -> bool:
        return self._update_occurrence(
            occurrence_key,
            worker_id=worker_id,
            status="enqueued",
            queued_message_id=message_id,
            lease_expires_at="",
        )

    def mark_occurrence_retry(
        self,
        occurrence_key: str,
        *,
        worker_id: str,
        error: str,
        next_attempt_at: str,
    ) -> bool:
        return self._update_occurrence(
            occurrence_key,
            worker_id=worker_id,
            status="retry_wait",
            last_error=error,
            error=error,
            next_attempt_at=next_attempt_at,
            lease_expires_at="",
        )

    def mark_occurrence_delivered(self, occurrence_key: str, *, response_outbox_id: str = "") -> bool:
        return self._update_occurrence(
            occurrence_key,
            status="delivered",
            response_outbox_id=response_outbox_id,
            lease_owner="",
            lease_expires_at="",
        )

    def mark_occurrence_response_pending(
        self,
        occurrence_key: str,
        *,
        response_outbox_id: str = "",
        connection: sqlite3.Connection | None = None,
    ) -> bool:
        return self._update_occurrence(
            occurrence_key,
            status="response_pending",
            response_outbox_id=str(response_outbox_id or "").strip(),
            lease_owner="",
            lease_expires_at="",
            connection=connection,
        )

    def mark_occurrence_failed(self, occurrence_key: str, *, error: str) -> bool:
        return self._update_occurrence(
            occurrence_key,
            status="delivery_failed",
            last_error=error,
            error=error,
            lease_owner="",
            lease_expires_at="",
        )

    def _update_occurrence(
        self,
        occurrence_key: str,
        *,
        worker_id: str = "",
        connection: sqlite3.Connection | None = None,
        **values: Any,
    ) -> bool:
        allowed = {
            "status", "queued_message_id", "lease_expires_at", "lease_owner", "last_error",
            "error", "next_attempt_at", "response_outbox_id",
        }
        updates = {key: value for key, value in values.items() if key in allowed}
        updates["updated_at"] = _now_iso()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        params = list(updates.values()) + [str(occurrence_key or "").strip()]
        where = "occurrence_key = ?"
        if worker_id:
            where += " AND lease_owner = ?"
            params.append(str(worker_id).strip())
        if connection is not None:
            cursor = connection.execute(
                f"UPDATE v2_scheduled_task_executions SET {assignments} WHERE {where}",
                tuple(params),
            )
            return cursor.rowcount == 1
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE v2_scheduled_task_executions SET {assignments} WHERE {where}",
                tuple(params),
            )
        return cursor.rowcount == 1

    def _set_status(self, scheduled_task_id: str, status: ScheduledTaskStatus, *, clear_next: bool) -> ScheduledTaskRecord:
        task = self._require_task(scheduled_task_id)
        updated = _replace_task(
            task,
            status=_normalize_status(status),
            next_run_at=None if clear_next else task.next_run_at,
            updated_at=_now_iso(),
        )
        self._save_task(updated)
        return updated

    def _require_task(self, scheduled_task_id: str) -> ScheduledTaskRecord:
        task = self.get_task(scheduled_task_id)
        if task is None:
            raise KeyError(f"scheduled_task_not_found: {scheduled_task_id}")
        return task

    def _save_task(self, task: ScheduledTaskRecord) -> None:
        with self._connect() as conn:
            self._save_task_connection(conn, task)

    def _save_task_connection(self, conn: sqlite3.Connection, task: ScheduledTaskRecord) -> None:
        conn.execute(
                """
                UPDATE v2_scheduled_tasks SET
                  owner_user_id = ?,
                  project_id = ?,
                  name = ?,
                  description = ?,
                  prompt = ?,
                  schedule_json = ?,
                  origin_channel_json = ?,
                  timezone = ?,
                  status = ?,
                  next_run_at = ?,
                  last_run_at = ?,
                  created_at = ?,
                  updated_at = ?
                WHERE scheduled_task_id = ?
                """,
                (
                    task.owner_user_id,
                    task.project_id,
                    task.name,
                    task.description,
                    task.prompt,
                    json.dumps(task.schedule, sort_keys=True),
                    json.dumps(task.origin_channel, sort_keys=True),
                    task.timezone,
                    task.status,
                    task.next_run_at,
                    task.last_run_at,
                    task.created_at,
                    task.updated_at,
                    task.scheduled_task_id,
                ),
            )

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
                CREATE TABLE IF NOT EXISTS v2_scheduled_tasks (
                  scheduled_task_id TEXT PRIMARY KEY,
                  owner_user_id TEXT NOT NULL,
                  project_id TEXT NOT NULL DEFAULT '',
                  name TEXT NOT NULL,
                  description TEXT NOT NULL DEFAULT '',
                  prompt TEXT NOT NULL,
                  schedule_json TEXT NOT NULL,
                  origin_channel_json TEXT NOT NULL DEFAULT '{}',
                  timezone TEXT NOT NULL,
                  status TEXT NOT NULL,
                  next_run_at TEXT,
                  last_run_at TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  CHECK (status IN ('active', 'paused', 'completed', 'cancelled', 'failed'))
                ) STRICT;

                CREATE INDEX IF NOT EXISTS idx_v2_scheduled_tasks_due
                  ON v2_scheduled_tasks (status, next_run_at);

                CREATE INDEX IF NOT EXISTS idx_v2_scheduled_tasks_owner_project
                  ON v2_scheduled_tasks (owner_user_id, project_id, status);

                CREATE TABLE IF NOT EXISTS v2_scheduled_task_executions (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  scheduled_task_id TEXT NOT NULL,
                  project_id TEXT NOT NULL DEFAULT '',
                  occurrence_key TEXT NOT NULL DEFAULT '',
                  run_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  queued_message_id TEXT,
                  started_at TEXT NOT NULL,
                  finished_at TEXT,
                  error TEXT NOT NULL DEFAULT '',
                  attempt_count INTEGER NOT NULL DEFAULT 0,
                  lease_owner TEXT NOT NULL DEFAULT '',
                  lease_expires_at TEXT NOT NULL DEFAULT '',
                  next_attempt_at TEXT NOT NULL DEFAULT '',
                  response_outbox_id TEXT NOT NULL DEFAULT '',
                  last_error TEXT NOT NULL DEFAULT '',
                  created_at TEXT NOT NULL DEFAULT '',
                  updated_at TEXT NOT NULL DEFAULT '',
                  CHECK (status IN ('pending', 'claimed', 'enqueued', 'processing', 'response_pending', 'retry_wait', 'delivered', 'failed', 'delivery_failed'))
                ) STRICT;

                CREATE INDEX IF NOT EXISTS idx_v2_scheduled_task_executions_task
                  ON v2_scheduled_task_executions (scheduled_task_id, started_at);
                """
            )
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(v2_scheduled_tasks)").fetchall()}
            if "origin_channel_json" not in columns:
                conn.execute(
                    "ALTER TABLE v2_scheduled_tasks ADD COLUMN origin_channel_json TEXT NOT NULL DEFAULT '{}'"
                )
            execution_columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(v2_scheduled_task_executions)").fetchall()
            }
            if "occurrence_key" not in execution_columns:
                conn.execute("ALTER TABLE v2_scheduled_task_executions RENAME TO v2_scheduled_task_executions_legacy")
                conn.executescript(
                    """
                    CREATE TABLE v2_scheduled_task_executions (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      scheduled_task_id TEXT NOT NULL,
                      project_id TEXT NOT NULL DEFAULT '',
                      occurrence_key TEXT NOT NULL DEFAULT '',
                      run_id TEXT NOT NULL,
                      status TEXT NOT NULL,
                      queued_message_id TEXT,
                      started_at TEXT NOT NULL,
                      finished_at TEXT,
                      error TEXT NOT NULL DEFAULT '',
                      attempt_count INTEGER NOT NULL DEFAULT 0,
                      lease_owner TEXT NOT NULL DEFAULT '',
                      lease_expires_at TEXT NOT NULL DEFAULT '',
                      next_attempt_at TEXT NOT NULL DEFAULT '',
                      response_outbox_id TEXT NOT NULL DEFAULT '',
                      last_error TEXT NOT NULL DEFAULT '',
                      created_at TEXT NOT NULL DEFAULT '',
                      updated_at TEXT NOT NULL DEFAULT '',
                      CHECK (status IN ('pending', 'claimed', 'enqueued', 'processing', 'response_pending', 'retry_wait', 'delivered', 'failed', 'delivery_failed'))
                    ) STRICT;
                    """
                )
                conn.execute(
                    """
                    INSERT INTO v2_scheduled_task_executions (
                      scheduled_task_id, project_id, occurrence_key, run_id, status, queued_message_id,
                      started_at, finished_at, error, attempt_count, created_at, updated_at
                    )
                    SELECT scheduled_task_id, '', scheduled_task_id || ':' || run_id, run_id,
                           CASE status WHEN 'queued' THEN 'enqueued' ELSE 'failed' END,
                           queued_message_id, started_at, finished_at, error, 1, started_at,
                           COALESCE(finished_at, started_at)
                    FROM v2_scheduled_task_executions_legacy
                    """
                )
                conn.execute("DROP TABLE v2_scheduled_task_executions_legacy")
                execution_columns.add("project_id")
            if "project_id" not in execution_columns:
                conn.execute("ALTER TABLE v2_scheduled_task_executions ADD COLUMN project_id TEXT NOT NULL DEFAULT ''")
            conn.execute(
                """
                UPDATE v2_scheduled_task_executions
                SET project_id=COALESCE(
                  (SELECT project_id FROM v2_scheduled_tasks t
                   WHERE t.scheduled_task_id=v2_scheduled_task_executions.scheduled_task_id),
                  ''
                )
                WHERE project_id=''
                """
            )
            conn.execute(
                "DELETE FROM v2_scheduled_task_executions WHERE occurrence_key!='' AND id NOT IN (SELECT MAX(id) FROM v2_scheduled_task_executions WHERE occurrence_key!='' GROUP BY occurrence_key)"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_v2_scheduled_task_executions_occurrence ON v2_scheduled_task_executions(occurrence_key) WHERE occurrence_key!=''"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_v2_scheduled_task_executions_project ON v2_scheduled_task_executions(project_id,started_at)"
            )


class ScheduledTaskRunner:
    """Queues due v2 scheduled tasks back into the normal message queue."""

    def __init__(self, *, store: ScheduledTaskStore, messages: Any) -> None:
        self.store = store
        self.channel = CommunicationChannel(messages)

    def run_due_once(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
        current = now or _now_utc()
        outcomes: list[dict[str, Any]] = []
        for task in self.store.due_tasks(now=current):
            outcomes.append(self._queue_task(task, now=current))
        return outcomes

    def _queue_task(self, task: ScheduledTaskRecord, *, now: datetime) -> dict[str, Any]:
        run_id = f"scheduled_run_{uuid4().hex[:16]}"
        started_at = _now_iso()
        try:
            queued = self.channel.queue_message(
                prompt=task.prompt,
                user=task.owner_user_id,
                project_id=task.project_id,
                metadata={
                    "source": "scheduled_task",
                    "scheduled_task_id": task.scheduled_task_id,
                    "project_id": task.project_id,
                    "scheduled_run_id": run_id,
                    "channel": dict(task.origin_channel),
                },
            )
            self.store.record_execution(
                scheduled_task_id=task.scheduled_task_id,
                run_id=run_id,
                status="queued",
                queued_message_id=queued.message_id,
                started_at=started_at,
                finished_at=_now_iso(),
            )
            updated = self.store.update_after_run(task, now=now)
            return {
                "scheduled_task_id": task.scheduled_task_id,
                "project_id": task.project_id,
                "run_id": run_id,
                "status": "queued",
                "queued_message_id": queued.message_id,
                "next_run_at": updated.next_run_at,
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.store.record_execution(
                scheduled_task_id=task.scheduled_task_id,
                run_id=run_id,
                status="error",
                started_at=started_at,
                finished_at=_now_iso(),
                error=error,
            )
            return {
                "scheduled_task_id": task.scheduled_task_id,
                "project_id": task.project_id,
                "run_id": run_id,
                "status": "error",
                "error": error,
            }


def compute_next_run_at(*, schedule: dict[str, Any], timezone_name: str, after: datetime) -> str | None:
    kind = str(schedule.get("kind") or "").strip()
    if kind == "once":
        run_at = _parse_datetime(str(schedule.get("run_at") or ""), timezone_name=timezone_name)
        if run_at is None:
            return None
        return _as_utc(run_at).isoformat()
    if kind != "rrule":
        return None
    rrule_value = str(schedule.get("rrule") or "").strip()
    if not rrule_value:
        return None
    tz = ZoneInfo(_normalize_timezone(timezone_name))
    dtstart = _parse_datetime(str(schedule.get("dtstart") or ""), timezone_name=timezone_name)
    if dtstart is None:
        dtstart = _as_utc(after).astimezone(tz)
    else:
        dtstart = dtstart.astimezone(tz)
    rule = rrulestr(rrule_value, dtstart=dtstart)
    next_local = rule.after(_as_utc(after).astimezone(tz), inc=False)
    if next_local is None:
        return None
    return _as_utc(next_local).isoformat()


def schedule_summary(schedule: dict[str, Any]) -> str:
    kind = str(schedule.get("kind") or "").strip()
    if kind == "once":
        return f"once at {schedule.get('run_at')}"
    if kind == "rrule":
        return f"rrule {schedule.get('rrule')}"
    return kind or "unknown"


class _ConnectionProxy:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def __enter__(self) -> sqlite3.Connection:
        return self._conn

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()


def _build_schedule(
    *,
    schedule_kind: ScheduleKind,
    run_at: str | None,
    rrule: str | None,
    dtstart: str | None,
    timezone_name: str,
) -> dict[str, Any]:
    kind = str(schedule_kind or "").strip()
    if kind not in {"once", "rrule"}:
        raise ValueError(f"invalid_schedule_kind: {schedule_kind}")
    if kind == "once":
        if not str(run_at or "").strip():
            raise ValueError("scheduled_task_run_at_required")
        parsed = _parse_datetime(str(run_at), timezone_name=timezone_name)
        if parsed is None:
            raise ValueError("scheduled_task_run_at_invalid")
        return {"kind": "once", "run_at": _as_utc(parsed).isoformat()}
    if not str(rrule or "").strip():
        raise ValueError("scheduled_task_rrule_required")
    schedule = {"kind": "rrule", "rrule": str(rrule or "").strip()}
    if str(dtstart or "").strip():
        parsed_dtstart = _parse_datetime(str(dtstart), timezone_name=timezone_name)
        if parsed_dtstart is None:
            raise ValueError("scheduled_task_dtstart_invalid")
        schedule["dtstart"] = parsed_dtstart.isoformat()
    return schedule


def _parse_datetime(value: str, *, timezone_name: str) -> datetime | None:
    rendered = str(value or "").strip()
    if not rendered:
        return None
    try:
        parsed = datetime.fromisoformat(rendered.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(_normalize_timezone(timezone_name)))
    return parsed


def _task_from_row(row: sqlite3.Row | None) -> ScheduledTaskRecord | None:
    if row is None:
        return None
    schedule = json.loads(str(row["schedule_json"] or "{}"))
    origin_channel = json.loads(str(row["origin_channel_json"] or "{}"))
    return ScheduledTaskRecord(
        scheduled_task_id=str(row["scheduled_task_id"]),
        owner_user_id=str(row["owner_user_id"]),
        project_id=str(row["project_id"] or ""),
        name=str(row["name"]),
        description=str(row["description"] or ""),
        prompt=str(row["prompt"]),
        origin_channel=dict(origin_channel) if isinstance(origin_channel, dict) else {},
        schedule=dict(schedule) if isinstance(schedule, dict) else {},
        timezone=str(row["timezone"] or "UTC"),
        status=_normalize_status(row["status"]),
        next_run_at=_optional_text(row["next_run_at"]),
        last_run_at=_optional_text(row["last_run_at"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _execution_from_row(row: sqlite3.Row | None) -> ScheduledTaskExecutionRecord | None:
    if row is None:
        return None
    return ScheduledTaskExecutionRecord(
        scheduled_task_id=str(row["scheduled_task_id"]),
        project_id=str(row["project_id"] or ""),
        run_id=str(row["run_id"]),
        status=_normalize_execution_status(row["status"]),
        queued_message_id=_optional_text(row["queued_message_id"]),
        started_at=str(row["started_at"]),
        finished_at=_optional_text(row["finished_at"]),
        error=str(row["error"] or ""),
        occurrence_key=str(row["occurrence_key"] or ""),
        attempt_count=int(row["attempt_count"] or 0),
        lease_owner=str(row["lease_owner"] or ""),
        lease_expires_at=str(row["lease_expires_at"] or ""),
        next_attempt_at=str(row["next_attempt_at"] or ""),
        response_outbox_id=str(row["response_outbox_id"] or ""),
        last_error=str(row["last_error"] or ""),
        created_at=str(row["created_at"] or ""),
        updated_at=str(row["updated_at"] or ""),
    )


def _task_values(record: ScheduledTaskRecord) -> tuple[Any, ...]:
    return (
        record.scheduled_task_id,
        record.owner_user_id,
        record.project_id,
        record.name,
        record.description,
        record.prompt,
        json.dumps(record.schedule, sort_keys=True),
        json.dumps(record.origin_channel, sort_keys=True),
        record.timezone,
        record.status,
        record.next_run_at,
        record.last_run_at,
        record.created_at,
        record.updated_at,
    )


def _replace_task(task: ScheduledTaskRecord, **values: Any) -> ScheduledTaskRecord:
    data = task.to_dict()
    data.update(values)
    return ScheduledTaskRecord(
        scheduled_task_id=str(data["scheduled_task_id"]),
        owner_user_id=str(data["owner_user_id"]),
        project_id=str(data["project_id"] or ""),
        name=str(data["name"]),
        description=str(data["description"] or ""),
        prompt=str(data["prompt"]),
        origin_channel=dict(data.get("origin_channel") or {}),
        schedule=dict(data["schedule"]),
        timezone=str(data["timezone"] or "UTC"),
        status=_normalize_status(data["status"]),
        next_run_at=_optional_text(data.get("next_run_at")),
        last_run_at=_optional_text(data.get("last_run_at")),
        created_at=str(data["created_at"]),
        updated_at=str(data["updated_at"]),
    )


def _normalize_timezone(value: object) -> str:
    rendered = str(value or "").strip() or "UTC"
    try:
        ZoneInfo(rendered)
    except Exception as exc:
        raise ValueError(f"invalid_timezone: {rendered}") from exc
    return rendered


def _normalize_status(value: object) -> ScheduledTaskStatus:
    rendered = str(value or "").strip()
    if rendered not in {"active", "paused", "completed", "cancelled", "failed"}:
        raise ValueError(f"invalid_scheduled_task_status: {value}")
    return rendered  # type: ignore[return-value]


def _normalize_execution_status(value: object) -> ExecutionStatus:
    rendered = str(value or "").strip()
    if rendered == "queued":
        rendered = "enqueued"
    if rendered == "error":
        rendered = "failed"
    if rendered not in {
        "pending", "claimed", "enqueued", "processing", "response_pending",
        "retry_wait", "delivered", "failed", "delivery_failed",
    }:
        raise ValueError(f"invalid_scheduled_task_execution_status: {value}")
    return rendered  # type: ignore[return-value]


def _as_utc(value: datetime) -> datetime:
    rendered = value
    if rendered.tzinfo is None:
        rendered = rendered.replace(tzinfo=timezone.utc)
    return rendered.astimezone(timezone.utc)


def _optional_text(value: object) -> str | None:
    rendered = str(value or "").strip()
    return rendered or None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _default_schedule_db_path() -> str:
    return str(default_database_path())
