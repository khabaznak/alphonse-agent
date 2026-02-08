from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any
import uuid

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


def list_timed_signals(limit: int = 200) -> list[dict[str, Any]]:
    query = (
        "SELECT id, trigger_at, next_trigger_at, rrule, timezone, status, fired_at, attempt_count, attempts, "
        "last_error, signal_type, payload, target, origin, correlation_id, created_at, updated_at "
        "FROM timed_signals ORDER BY COALESCE(next_trigger_at, trigger_at) DESC LIMIT ?"
    )
    with _connect() as conn:
        rows = conn.execute(query, (limit,)).fetchall()
    return [_row_to_timed_signal(row) for row in rows]


def list_upcoming_timed_signals(limit: int = 10) -> list[dict[str, Any]]:
    query = (
        "SELECT id, trigger_at, next_trigger_at, rrule, timezone, status, fired_at, attempt_count, attempts, "
        "last_error, signal_type, payload, target, origin, correlation_id, created_at, updated_at "
        "FROM timed_signals "
        "WHERE status IN ('pending', 'processing') "
        "ORDER BY COALESCE(next_trigger_at, trigger_at) ASC LIMIT ?"
    )
    with _connect() as conn:
        rows = conn.execute(query, (limit,)).fetchall()
    return [_row_to_timed_signal(row) for row in rows]


def insert_timed_signal(
    *,
    trigger_at: str,
    timezone: str,
    signal_type: str,
    payload: dict[str, Any],
    target: str | None,
    origin: str | None,
    correlation_id: str | None,
    signal_id: str | None = None,
    next_trigger_at: str | None = None,
    rrule: str | None = None,
) -> str:
    timed_signal_id = signal_id or str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO timed_signals
              (id, trigger_at, next_trigger_at, rrule, timezone, status, fired_at, attempt_count, attempts, last_error,
               signal_type, payload, target, origin, correlation_id)
            VALUES
              (?, ?, ?, ?, ?, 'pending', NULL, 0, 0, NULL, ?, ?, ?, ?, ?)
            """,
            (
                timed_signal_id,
                trigger_at,
                next_trigger_at,
                rrule,
                timezone,
                signal_type,
                json.dumps(payload),
                target,
                origin,
                correlation_id,
            ),
        )
        conn.commit()
    return timed_signal_id


def _connect() -> sqlite3.Connection:
    path = resolve_nervous_system_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def _row_to_timed_signal(row: sqlite3.Row | tuple | None) -> dict[str, Any]:
    if row is None:
        return {}
    if not isinstance(row, tuple):
        row = tuple(row)
    return {
        "id": row[0],
        "trigger_at": row[1],
        "next_trigger_at": row[2],
        "rrule": row[3],
        "timezone": row[4],
        "status": row[5],
        "fired_at": row[6],
        "attempt_count": row[7],
        "attempts": row[8],
        "last_error": row[9],
        "signal_type": row[10],
        "payload": _parse_payload(row[11]),
        "target": row[12],
        "origin": row[13],
        "correlation_id": row[14],
        "created_at": row[15],
        "updated_at": row[16],
    }


def _parse_payload(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}
