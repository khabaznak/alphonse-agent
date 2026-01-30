from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from alphonse.nervous_system.paths import resolve_nervous_system_db_path


def list_timed_signals(limit: int = 200) -> list[dict[str, Any]]:
    query = (
        "SELECT id, trigger_at, next_trigger_at, rrule, timezone, status, signal_type, "
        "payload, target, origin, correlation_id, created_at, updated_at "
        "FROM timed_signals ORDER BY COALESCE(next_trigger_at, trigger_at) DESC LIMIT ?"
    )
    with _connect() as conn:
        rows = conn.execute(query, (limit,)).fetchall()
    return [_row_to_timed_signal(row) for row in rows]


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
        "signal_type": row[6],
        "payload": _parse_payload(row[7]),
        "target": row[8],
        "origin": row[9],
        "correlation_id": row[10],
        "created_at": row[11],
        "updated_at": row[12],
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
