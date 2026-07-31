"""Retention and storage metrics for the unified Alphonse v2 database."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from alphonse.agent_v2.database import DEFAULT_BUSY_TIMEOUT_MS, connect_database, lock_retry_count

DEFAULT_OPERATIONAL_RETENTION_DAYS = 30


def prune_operational_data(
    db_path: str | Path,
    *,
    now: datetime | None = None,
    retention_days: int = DEFAULT_OPERATIONAL_RETENTION_DAYS,
) -> dict[str, int]:
    current = now or datetime.now(timezone.utc)
    cutoff = (current - timedelta(days=max(1, int(retention_days)))).isoformat()
    deleted: dict[str, int] = {}
    connection = connect_database(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        tables = _tables(connection)
        if "v2_task_dependencies" in tables and "v2_questions" in tables:
            cursor = connection.execute(
                """
                DELETE FROM v2_task_dependencies
                WHERE question_id IN (
                  SELECT question_id FROM v2_questions
                  WHERE status IN ('answered','expired','cancelled') AND updated_at < ?
                )
                """,
                (cutoff,),
            )
            deleted["task_dependencies"] = int(cursor.rowcount)
        statements = (
            (
                "questions",
                "v2_questions",
                "DELETE FROM v2_questions WHERE status IN ('answered','expired','cancelled') AND updated_at < ?",
            ),
            (
                "task_checkpoints",
                "v2_task_checkpoints",
                "DELETE FROM v2_task_checkpoints WHERE status IN ('done','failed','cancelled') AND updated_at < ?",
            ),
            (
                "failed_inbound",
                "v2_inbound_messages",
                "DELETE FROM v2_inbound_messages WHERE status='failed' AND queued_at < ?",
            ),
            (
                "outbound",
                "v2_outbox",
                "DELETE FROM v2_outbox WHERE status IN ('delivered','failed') AND updated_at < ?",
            ),
            (
                "scheduled_executions",
                "v2_scheduled_task_executions",
                "DELETE FROM v2_scheduled_task_executions WHERE status IN ('delivered','failed','delivery_failed') AND updated_at < ?",
            ),
            (
                "automation_executions",
                "v2_automation_executions",
                "DELETE FROM v2_automation_executions WHERE status='enqueued' AND updated_at < ?",
            ),
            (
                "communication_threads",
                "v2_communication_threads",
                "DELETE FROM v2_communication_threads WHERE status IN ('replied','expired','failed') AND updated_at < ?",
            ),
        )
        for label, table, statement in statements:
            if table not in tables:
                continue
            deleted[label] = int(connection.execute(statement, (cutoff,)).rowcount)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return deleted


def storage_metrics(db_path: str | Path) -> dict[str, Any]:
    path = Path(db_path).expanduser()
    if not path.exists():
        return {"database_path": str(path), "database_bytes": 0}
    with connect_database(path) as connection:
        tables = _tables(connection)
        metrics: dict[str, Any] = {
            "database_path": str(path),
            "database_bytes": path.stat().st_size,
            "wal_bytes": Path(f"{path}-wal").stat().st_size if Path(f"{path}-wal").exists() else 0,
            "journal_mode": str(connection.execute("PRAGMA journal_mode").fetchone()[0]),
            "busy_timeout_ms": DEFAULT_BUSY_TIMEOUT_MS,
            "lock_retries": lock_retry_count(),
        }
        metrics["database_total_bytes"] = int(metrics["database_bytes"]) + int(metrics["wal_bytes"])
        if "v2_task_checkpoints" in tables:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count,
                       COALESCE(ROUND(AVG(LENGTH(task_state_json))),0) AS average_bytes,
                       COALESCE(MAX(LENGTH(task_state_json)),0) AS maximum_bytes
                FROM v2_task_checkpoints
                """
            ).fetchone()
            metrics["checkpoints"] = {
                "count": int(row["count"]),
                "average_bytes": int(row["average_bytes"]),
                "maximum_bytes": int(row["maximum_bytes"]),
            }
        for label, table, timestamp in (
            ("inbound", "v2_inbound_messages", "queued_at"),
            ("outbound", "v2_outbox", "created_at"),
        ):
            if table not in tables:
                continue
            row = connection.execute(
                f"SELECT COUNT(*) AS count, COALESCE(MIN({timestamp}),'') AS oldest FROM {table}"
            ).fetchone()
            oldest = str(row["oldest"])
            metrics[label] = {
                "count": int(row["count"]),
                "oldest": oldest,
                "oldest_age_seconds": _age_seconds(oldest),
            }
        terminal_rows: dict[str, int] = {}
        for label, table, condition in (
            ("questions", "v2_questions", "status IN ('answered','expired','cancelled')"),
            ("checkpoints", "v2_task_checkpoints", "status IN ('done','failed','cancelled')"),
            ("outbound", "v2_outbox", "status IN ('delivered','failed')"),
            (
                "scheduled_executions",
                "v2_scheduled_task_executions",
                "status IN ('delivered','failed','delivery_failed')",
            ),
            ("communication_threads", "v2_communication_threads", "status IN ('replied','expired','failed')"),
        ):
            if table in tables:
                terminal_rows[label] = int(
                    connection.execute(f"SELECT COUNT(*) FROM {table} WHERE {condition}").fetchone()[0]
                )
        metrics["retained_terminal_rows"] = terminal_rows
        return metrics


def _tables(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {str(row[0]) for row in rows}


def _age_seconds(timestamp: str) -> int:
    if not timestamp:
        return 0
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0, int((datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds()))
    except ValueError:
        return 0
