from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


def insert_gap_task(record: dict[str, Any]) -> str:
    task_id = str(record.get("id") or uuid.uuid4())
    created_at = record.get("created_at") or _now_iso()
    payload = record.get("payload")
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO gap_tasks (
              id, proposal_id, type, status, created_at, payload
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                record.get("proposal_id"),
                record.get("type"),
                record.get("status") or "open",
                created_at,
                _to_json(payload),
            ),
        )
        conn.commit()
    return task_id


def list_gap_tasks(*, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    filters: list[str] = []
    values: list[Any] = []
    if status:
        filters.append("status = ?")
        values.append(status)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = (
        "SELECT id, proposal_id, type, status, created_at, payload "
        f"FROM gap_tasks {where} ORDER BY created_at DESC LIMIT ?"
    )
    values.append(limit)
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        rows = conn.execute(query, tuple(values)).fetchall()
    return [_row_to_task(row) for row in rows]


def get_gap_task(task_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            "SELECT id, proposal_id, type, status, created_at, payload FROM gap_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
    return _row_to_task(row) if row else None


def update_gap_task_status(task_id: str, status: str) -> bool:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        cur = conn.execute(
            "UPDATE gap_tasks SET status = ? WHERE id = ?",
            (status, task_id),
        )
        conn.commit()
    return cur.rowcount > 0


def _row_to_task(row: sqlite3.Row | tuple | None) -> dict[str, Any]:
    if row is None:
        return {}
    if not isinstance(row, tuple):
        row = tuple(row)
    return {
        "id": row[0],
        "proposal_id": row[1],
        "type": row[2],
        "status": row[3],
        "created_at": row[4],
        "payload": _parse_json(row[5]) or {},
    }


def _to_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value)


def _parse_json(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
