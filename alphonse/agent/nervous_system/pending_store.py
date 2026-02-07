from __future__ import annotations

import json
import sqlite3
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


def get_pending_plan(person_id: str | None, channel_type: str) -> dict[str, Any] | None:
    query = """
        SELECT * FROM pending_plans
        WHERE status = 'pending' AND channel_type = ? AND (person_id = ? OR ? IS NULL)
        ORDER BY created_at DESC
        LIMIT 1
    """
    return _fetch_one(query, (channel_type, person_id, person_id))


def create_pending_plan(payload: dict[str, Any]) -> None:
    _execute(
        """
        INSERT INTO pending_plans
          (pending_id, person_id, channel_type, correlation_id, plan_json, status, expires_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload.get("pending_id"),
            payload.get("person_id"),
            payload.get("channel_type"),
            payload.get("correlation_id"),
            json.dumps(payload.get("plan_json") or {}),
            payload.get("status", "pending"),
            payload.get("expires_at"),
        ),
    )


def update_pending_status(pending_id: str, status: str) -> None:
    _execute(
        "UPDATE pending_plans SET status = ? WHERE pending_id = ?",
        (status, pending_id),
    )


def _fetch_one(query: str, params: tuple) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(query, params).fetchone()
    return dict(row) if row else None


def _execute(query: str, params: tuple) -> None:
    with _connect() as conn:
        conn.execute(query, params)
        conn.commit()


def _connect() -> sqlite3.Connection:
    path = resolve_nervous_system_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn
