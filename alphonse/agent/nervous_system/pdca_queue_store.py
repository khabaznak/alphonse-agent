from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path

_ACTIVE_TASK_STATUSES = {"queued", "running", "waiting_user", "paused"}
_ALL_TASK_STATUSES = _ACTIVE_TASK_STATUSES | {"done", "failed"}


def upsert_pdca_task(record: dict[str, Any]) -> str:
    task_id = str(record.get("task_id") or uuid.uuid4())
    now = _now_iso()
    status = _normalize_status(str(record.get("status") or "queued"))
    priority = _as_int(record.get("priority"), default=100, minimum=0)
    slice_cycles = _as_int(record.get("slice_cycles"), default=5, minimum=1)
    failure_streak = _as_int(record.get("failure_streak"), default=0, minimum=0)
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO pdca_tasks (
              task_id,
              owner_id,
              conversation_key,
              session_id,
              status,
              priority,
              next_run_at,
              lease_until,
              worker_id,
              slice_cycles,
              max_cycles,
              max_runtime_seconds,
              token_budget_remaining,
              failure_streak,
              last_error,
              metadata_json,
              created_at,
              updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
              owner_id = excluded.owner_id,
              conversation_key = excluded.conversation_key,
              session_id = excluded.session_id,
              status = excluded.status,
              priority = excluded.priority,
              next_run_at = excluded.next_run_at,
              lease_until = excluded.lease_until,
              worker_id = excluded.worker_id,
              slice_cycles = excluded.slice_cycles,
              max_cycles = excluded.max_cycles,
              max_runtime_seconds = excluded.max_runtime_seconds,
              token_budget_remaining = excluded.token_budget_remaining,
              failure_streak = excluded.failure_streak,
              last_error = excluded.last_error,
              metadata_json = excluded.metadata_json,
              updated_at = excluded.updated_at
            """,
            (
                task_id,
                str(record.get("owner_id") or "").strip(),
                str(record.get("conversation_key") or "").strip(),
                _none_if_blank(record.get("session_id")),
                status,
                priority,
                _none_if_blank(record.get("next_run_at")),
                _none_if_blank(record.get("lease_until")),
                _none_if_blank(record.get("worker_id")),
                slice_cycles,
                _as_optional_int(record.get("max_cycles")),
                _as_optional_int(record.get("max_runtime_seconds")),
                _as_optional_int(record.get("token_budget_remaining")),
                failure_streak,
                _none_if_blank(record.get("last_error")),
                _to_json(record.get("metadata")),
                str(record.get("created_at") or now),
                now,
            ),
        )
        conn.commit()
    return task_id


def get_pdca_task(task_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT
              task_id,
              owner_id,
              conversation_key,
              session_id,
              status,
              priority,
              next_run_at,
              lease_until,
              worker_id,
              slice_cycles,
              max_cycles,
              max_runtime_seconds,
              token_budget_remaining,
              failure_streak,
              last_error,
              metadata_json,
              created_at,
              updated_at
            FROM pdca_tasks
            WHERE task_id = ?
            """,
            (str(task_id or "").strip(),),
        ).fetchone()
    return _row_to_task(row) if row else None


def get_latest_pdca_task_for_conversation(
    *,
    conversation_key: str,
    statuses: list[str] | None = None,
) -> dict[str, Any] | None:
    key = str(conversation_key or "").strip()
    if not key:
        return None
    status_values = [str(item or "").strip().lower() for item in (statuses or list(_ACTIVE_TASK_STATUSES))]
    status_values = [item for item in status_values if item in _ALL_TASK_STATUSES]
    if not status_values:
        status_values = list(_ACTIVE_TASK_STATUSES)
    placeholders = ",".join("?" for _ in status_values)
    query = (
        """
            SELECT
              task_id,
              owner_id,
              conversation_key,
              session_id,
              status,
              priority,
              next_run_at,
              lease_until,
              worker_id,
              slice_cycles,
              max_cycles,
              max_runtime_seconds,
              token_budget_remaining,
              failure_streak,
              last_error,
              metadata_json,
              created_at,
              updated_at
            FROM pdca_tasks
            WHERE conversation_key = ?
              AND status IN ("""
        + placeholders
        + """)
            ORDER BY updated_at DESC
            LIMIT 1
        """
    )
    params: list[Any] = [key, *status_values]
    with _connect() as conn:
        row = conn.execute(query, tuple(params)).fetchone()
    return _row_to_task(row) if row else None


def list_runnable_pdca_tasks(*, now: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    now_text = str(now or _now_iso()).strip()
    safe_limit = _as_int(limit, default=20, minimum=1)
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
              task_id,
              owner_id,
              conversation_key,
              session_id,
              status,
              priority,
              next_run_at,
              lease_until,
              worker_id,
              slice_cycles,
              max_cycles,
              max_runtime_seconds,
              token_budget_remaining,
              failure_streak,
              last_error,
              metadata_json,
              created_at,
              updated_at
            FROM pdca_tasks
            WHERE status IN ('queued', 'running')
              AND (next_run_at IS NULL OR next_run_at <= ?)
              AND (lease_until IS NULL OR lease_until <= ?)
            ORDER BY priority DESC, COALESCE(next_run_at, created_at) ASC, updated_at ASC
            LIMIT ?
            """,
            (now_text, now_text, safe_limit),
        ).fetchall()
    return [_row_to_task(row) for row in rows]


def acquire_pdca_task_lease(*, task_id: str, worker_id: str, lease_seconds: int = 30, now: str | None = None) -> bool:
    task_key = str(task_id or "").strip()
    worker = str(worker_id or "").strip()
    if not task_key or not worker:
        return False
    now_dt = _parse_or_now(now)
    lease_until = now_dt + timedelta(seconds=max(int(lease_seconds or 30), 1))
    now_text = now_dt.isoformat()
    lease_text = lease_until.isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """
            UPDATE pdca_tasks
            SET
              lease_until = ?,
              worker_id = ?,
              status = CASE WHEN status = 'queued' THEN 'running' ELSE status END,
              updated_at = ?
            WHERE task_id = ?
              AND status IN ('queued', 'running')
              AND (lease_until IS NULL OR lease_until <= ?)
            """,
            (lease_text, worker, now_text, task_key, now_text),
        )
        conn.commit()
    return cur.rowcount > 0


def release_pdca_task_lease(*, task_id: str, worker_id: str) -> bool:
    task_key = str(task_id or "").strip()
    worker = str(worker_id or "").strip()
    if not task_key or not worker:
        return False
    with _connect() as conn:
        cur = conn.execute(
            """
            UPDATE pdca_tasks
            SET lease_until = NULL, worker_id = NULL, updated_at = ?
            WHERE task_id = ? AND worker_id = ?
            """,
            (_now_iso(), task_key, worker),
        )
        conn.commit()
    return cur.rowcount > 0


def update_pdca_task_status(*, task_id: str, status: str, last_error: str | None = None) -> bool:
    task_key = str(task_id or "").strip()
    if not task_key:
        return False
    normalized = _normalize_status(status)
    with _connect() as conn:
        cur = conn.execute(
            """
            UPDATE pdca_tasks
            SET status = ?, last_error = ?, updated_at = ?
            WHERE task_id = ?
            """,
            (normalized, _none_if_blank(last_error), _now_iso(), task_key),
        )
        conn.commit()
    return cur.rowcount > 0


def update_pdca_task_metadata(*, task_id: str, metadata: dict[str, Any]) -> bool:
    task_key = str(task_id or "").strip()
    if not task_key:
        return False
    payload = metadata if isinstance(metadata, dict) else {}
    with _connect() as conn:
        cur = conn.execute(
            """
            UPDATE pdca_tasks
            SET metadata_json = ?, updated_at = ?
            WHERE task_id = ?
            """,
            (json.dumps(payload, ensure_ascii=False), _now_iso(), task_key),
        )
        conn.commit()
    return cur.rowcount > 0


def save_pdca_checkpoint(
    *,
    task_id: str,
    state: dict[str, Any],
    task_state: dict[str, Any],
    expected_version: int | None = None,
) -> int | None:
    task_key = str(task_id or "").strip()
    if not task_key:
        return None
    now = _now_iso()
    state_json = json.dumps(state or {}, ensure_ascii=False)
    task_state_json = json.dumps(task_state or {}, ensure_ascii=False)
    with _connect() as conn:
        if expected_version is None:
            conn.execute(
                """
                INSERT INTO pdca_checkpoints (task_id, state_json, task_state_json, version, created_at, updated_at)
                VALUES (?, ?, ?, 1, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                  state_json = excluded.state_json,
                  task_state_json = excluded.task_state_json,
                  version = pdca_checkpoints.version + 1,
                  updated_at = excluded.updated_at
                """,
                (task_key, state_json, task_state_json, now, now),
            )
            row = conn.execute(
                "SELECT version FROM pdca_checkpoints WHERE task_id = ?",
                (task_key,),
            ).fetchone()
            conn.commit()
            return int(row[0]) if row else None

        if int(expected_version) == 0:
            try:
                conn.execute(
                    """
                    INSERT INTO pdca_checkpoints (task_id, state_json, task_state_json, version, created_at, updated_at)
                    VALUES (?, ?, ?, 1, ?, ?)
                    """,
                    (task_key, state_json, task_state_json, now, now),
                )
                conn.commit()
                return 1
            except sqlite3.IntegrityError:
                conn.rollback()
                return None

        cur = conn.execute(
            """
            UPDATE pdca_checkpoints
            SET
              state_json = ?,
              task_state_json = ?,
              version = version + 1,
              updated_at = ?
            WHERE task_id = ? AND version = ?
            """,
            (state_json, task_state_json, now, task_key, int(expected_version)),
        )
        if cur.rowcount <= 0:
            conn.rollback()
            return None
        row = conn.execute(
            "SELECT version FROM pdca_checkpoints WHERE task_id = ?",
            (task_key,),
        ).fetchone()
        conn.commit()
        return int(row[0]) if row else None


def load_pdca_checkpoint(task_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT task_id, state_json, task_state_json, version, created_at, updated_at
            FROM pdca_checkpoints
            WHERE task_id = ?
            """,
            (str(task_id or "").strip(),),
        ).fetchone()
    if not row:
        return None
    return {
        "task_id": str(row[0]),
        "state": _parse_json(str(row[1])) or {},
        "task_state": _parse_json(str(row[2])) or {},
        "version": int(row[3] or 0),
        "created_at": str(row[4] or ""),
        "updated_at": str(row[5] or ""),
    }


def append_pdca_event(*, task_id: str, event_type: str, payload: dict[str, Any] | None = None, correlation_id: str | None = None) -> str:
    event_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO pdca_events (event_id, task_id, event_type, payload_json, correlation_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                str(task_id or "").strip(),
                str(event_type or "").strip(),
                _to_json(payload or {}),
                _none_if_blank(correlation_id),
                _now_iso(),
            ),
        )
        conn.commit()
    return event_id


def list_pdca_events(*, task_id: str, limit: int = 100) -> list[dict[str, Any]]:
    safe_limit = _as_int(limit, default=100, minimum=1)
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT event_id, task_id, event_type, payload_json, correlation_id, created_at
            FROM pdca_events
            WHERE task_id = ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (str(task_id or "").strip(), safe_limit),
        ).fetchall()
    return [
        {
            "event_id": str(row[0]),
            "task_id": str(row[1]),
            "event_type": str(row[2]),
            "payload": _parse_json(str(row[3])) or {},
            "correlation_id": str(row[4] or "").strip() or None,
            "created_at": str(row[5] or ""),
        }
        for row in rows
    ]


def has_pdca_event(
    *,
    task_id: str,
    event_type: str,
    correlation_id: str | None = None,
) -> bool:
    task_key = str(task_id or "").strip()
    event_key = str(event_type or "").strip()
    if not task_key or not event_key:
        return False
    query = (
        "SELECT 1 FROM pdca_events WHERE task_id = ? AND event_type = ? "
        + ("AND correlation_id = ? " if correlation_id else "")
        + "LIMIT 1"
    )
    params: list[Any] = [task_key, event_key]
    if correlation_id:
        params.append(str(correlation_id or "").strip())
    with _connect() as conn:
        row = conn.execute(query, tuple(params)).fetchone()
    return row is not None


def get_pdca_queue_metrics(
    *,
    now: str | None = None,
    lookback_minutes: int = 15,
    top_owners: int = 5,
) -> dict[str, Any]:
    now_dt = _parse_or_now(now)
    window_minutes = max(int(lookback_minutes or 15), 1)
    top_limit = max(int(top_owners or 5), 1)
    window_start = (now_dt - timedelta(minutes=window_minutes)).isoformat()
    metrics: dict[str, Any] = {
        "generated_at": now_dt.isoformat(),
        "lookback_minutes": window_minutes,
        "queue_depth_total": 0,
        "queue_depth_by_status": {},
        "queue_depth_by_owner": {},
        "oldest_wait_seconds": {"global": 0, "by_owner": {}},
        "dispatch_rate_per_minute": {"global": 0.0, "by_owner": {}},
        "budget_exhaustions_total": {"all": 0, "by_reason": {}},
        "starvation_warnings_total": 0,
        "owner_fairness_ratio": {"by_owner": {}, "max_skew": 0.0},
        "terminal_outcomes_total": {"done": 0, "waiting_user": 0, "failed": 0},
    }
    try:
        with _connect() as conn:
            status_rows = conn.execute(
                """
                SELECT status, COUNT(*) AS cnt
                FROM pdca_tasks
                GROUP BY status
                """
            ).fetchall()
            by_status = {str(row[0]): int(row[1] or 0) for row in status_rows if row[0]}
            metrics["queue_depth_by_status"] = by_status
            metrics["queue_depth_total"] = sum(by_status.values())

            owner_rows = conn.execute(
                """
                SELECT owner_id, COUNT(*) AS cnt
                FROM pdca_tasks
                WHERE status IN ('queued', 'running')
                GROUP BY owner_id
                ORDER BY cnt DESC, owner_id ASC
                LIMIT ?
                """,
                (top_limit,),
            ).fetchall()
            queue_by_owner = {str(row[0]): int(row[1] or 0) for row in owner_rows if row[0]}
            metrics["queue_depth_by_owner"] = queue_by_owner

            wait_rows = conn.execute(
                """
                SELECT owner_id, next_run_at, created_at
                FROM pdca_tasks
                WHERE status IN ('queued', 'running')
                """
            ).fetchall()
            wait_by_owner: dict[str, int] = {}
            oldest_global = 0
            for row in wait_rows:
                owner_id = str(row[0] or "").strip() or "unknown"
                baseline = _parse_iso_utc(str(row[1] or "")) or _parse_iso_utc(str(row[2] or ""))
                if baseline is None:
                    continue
                wait_seconds = max(int((now_dt - baseline).total_seconds()), 0)
                oldest_global = max(oldest_global, wait_seconds)
                wait_by_owner[owner_id] = max(wait_by_owner.get(owner_id, 0), wait_seconds)
            metrics["oldest_wait_seconds"] = {"global": oldest_global, "by_owner": wait_by_owner}

            dispatch_total = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM pdca_events WHERE event_type = 'slice.requested'
                    """
                ).fetchone()[0]
                or 0
            )
            dispatch_window = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM pdca_events
                    WHERE event_type = 'slice.requested' AND created_at >= ?
                    """,
                    (window_start,),
                ).fetchone()[0]
                or 0
            )
            dispatch_by_owner_rows = conn.execute(
                """
                SELECT t.owner_id, COUNT(*) AS cnt
                FROM pdca_events e
                JOIN pdca_tasks t ON t.task_id = e.task_id
                WHERE e.event_type = 'slice.requested' AND e.created_at >= ?
                GROUP BY t.owner_id
                """,
                (window_start,),
            ).fetchall()
            dispatch_by_owner = {str(row[0]): int(row[1] or 0) for row in dispatch_by_owner_rows if row[0]}
            metrics["dispatch_rate_per_minute"] = {
                "global": round(dispatch_window / float(window_minutes), 4),
                "by_owner": {owner: round(count / float(window_minutes), 4) for owner, count in dispatch_by_owner.items()},
                "dispatch_total": dispatch_total,
                "dispatch_in_window": dispatch_window,
            }

            budget_rows = conn.execute(
                """
                SELECT payload_json
                FROM pdca_events
                WHERE event_type = 'slice.blocked.budget_exhausted'
                """
            ).fetchall()
            by_reason: dict[str, int] = {}
            for row in budget_rows:
                payload = _parse_json(str(row[0] or "")) if row and row[0] else {}
                reason = str((payload or {}).get("reason") or "unknown").strip() or "unknown"
                by_reason[reason] = by_reason.get(reason, 0) + 1
            metrics["budget_exhaustions_total"] = {"all": len(budget_rows), "by_reason": by_reason}

            starvation_total = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM pdca_events WHERE event_type = 'queue.starvation_warning'
                    """
                ).fetchone()[0]
                or 0
            )
            metrics["starvation_warnings_total"] = starvation_total

            terminal_rows = conn.execute(
                """
                SELECT event_type
                FROM pdca_events
                WHERE event_type IN (
                  'slice.completed.done',
                  'slice.completed.waiting_user',
                  'slice.blocked.missing_text',
                  'slice.completed.failed',
                  'slice.failed',
                  'slice.blocked.budget_exhausted'
                )
                """
            ).fetchall()
            terminal = {"done": 0, "waiting_user": 0, "failed": 0}
            for row in terminal_rows:
                event_type = str(row[0] or "").strip()
                if event_type == "slice.completed.done":
                    terminal["done"] += 1
                elif event_type in {"slice.completed.waiting_user", "slice.blocked.missing_text"}:
                    terminal["waiting_user"] += 1
                elif event_type in {"slice.completed.failed", "slice.failed", "slice.blocked.budget_exhausted"}:
                    terminal["failed"] += 1
            metrics["terminal_outcomes_total"] = terminal

            active_owner_rows = conn.execute(
                """
                SELECT owner_id, COUNT(*) AS cnt
                FROM pdca_tasks
                WHERE status IN ('queued', 'running')
                GROUP BY owner_id
                """
            ).fetchall()
            active_by_owner = {str(row[0]): int(row[1] or 0) for row in active_owner_rows if row[0]}
            total_active = sum(active_by_owner.values())
            total_dispatched_window = sum(dispatch_by_owner.values())
            fairness: dict[str, float] = {}
            max_skew = 0.0
            for owner_id in set(active_by_owner.keys()) | set(dispatch_by_owner.keys()):
                queue_share = (active_by_owner.get(owner_id, 0) / float(total_active)) if total_active > 0 else 0.0
                dispatch_share = (
                    dispatch_by_owner.get(owner_id, 0) / float(total_dispatched_window)
                    if total_dispatched_window > 0
                    else 0.0
                )
                if queue_share <= 0:
                    continue
                ratio = round(dispatch_share / queue_share, 4)
                fairness[owner_id] = ratio
                max_skew = max(max_skew, abs(ratio - 1.0))
            metrics["owner_fairness_ratio"] = {"by_owner": fairness, "max_skew": round(max_skew, 4)}
    except sqlite3.OperationalError:
        return metrics
    return metrics


def _connect() -> sqlite3.Connection:
    path = resolve_nervous_system_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_task(row: sqlite3.Row | tuple) -> dict[str, Any]:
    if not isinstance(row, tuple):
        row = tuple(row)
    return {
        "task_id": str(row[0]),
        "owner_id": str(row[1]),
        "conversation_key": str(row[2]),
        "session_id": str(row[3] or "").strip() or None,
        "status": str(row[4]),
        "priority": int(row[5] or 0),
        "next_run_at": str(row[6] or "").strip() or None,
        "lease_until": str(row[7] or "").strip() or None,
        "worker_id": str(row[8] or "").strip() or None,
        "slice_cycles": int(row[9] or 0),
        "max_cycles": int(row[10]) if row[10] is not None else None,
        "max_runtime_seconds": int(row[11]) if row[11] is not None else None,
        "token_budget_remaining": int(row[12]) if row[12] is not None else None,
        "failure_streak": int(row[13] or 0),
        "last_error": str(row[14] or "").strip() or None,
        "metadata": _parse_json(str(row[15])) or {},
        "created_at": str(row[16] or ""),
        "updated_at": str(row[17] or ""),
    }


def _normalize_status(value: str) -> str:
    rendered = str(value or "").strip().lower()
    if rendered not in _ALL_TASK_STATUSES:
        return "queued"
    return rendered


def _as_int(value: Any, *, default: int, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    if minimum is not None and parsed < minimum:
        return int(minimum)
    return parsed


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _parse_json(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _parse_iso_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_or_now(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _none_if_blank(value: Any) -> str | None:
    rendered = str(value or "").strip()
    return rendered or None
