from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


def upsert_tool_config(record: dict[str, Any]) -> str:
    config_id = str(record.get("config_id") or uuid.uuid4())
    tool_key = str(record.get("tool_key") or "").strip()
    if not tool_key:
        raise ValueError("tool_key is required")
    now = _now_iso()
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO tool_configs (
              config_id, tool_key, name, config_json, is_active, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(config_id) DO UPDATE SET
              tool_key = excluded.tool_key,
              name = excluded.name,
              config_json = excluded.config_json,
              is_active = excluded.is_active,
              updated_at = excluded.updated_at
            """,
            (
                config_id,
                tool_key,
                record.get("name"),
                _to_json(record.get("config") or {}),
                1 if bool(record.get("is_active", True)) else 0,
                now,
                now,
            ),
        )
        conn.commit()
    return config_id


def list_tool_configs(
    *,
    tool_key: str | None = None,
    active_only: bool = False,
    limit: int = 100,
) -> list[dict[str, Any]]:
    filters: list[str] = []
    values: list[Any] = []
    if tool_key:
        filters.append("tool_key = ?")
        values.append(tool_key)
    if active_only:
        filters.append("is_active = 1")
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = (
        "SELECT config_id, tool_key, name, config_json, is_active, created_at, updated_at "
        f"FROM tool_configs {where} ORDER BY updated_at DESC LIMIT ?"
    )
    values.append(limit)
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        rows = conn.execute(query, tuple(values)).fetchall()
    return [_row_to_tool_config(row) for row in rows]


def get_tool_config(config_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT config_id, tool_key, name, config_json, is_active, created_at, updated_at
            FROM tool_configs
            WHERE config_id = ?
            """,
            (config_id,),
        ).fetchone()
    return _row_to_tool_config(row) if row else None


def delete_tool_config(config_id: str) -> bool:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        cur = conn.execute(
            "DELETE FROM tool_configs WHERE config_id = ?",
            (config_id,),
        )
        conn.commit()
    return cur.rowcount > 0


def _row_to_tool_config(row: sqlite3.Row | tuple | None) -> dict[str, Any]:
    if row is None:
        return {}
    if not isinstance(row, tuple):
        row = tuple(row)
    return {
        "config_id": row[0],
        "tool_key": row[1],
        "name": row[2],
        "config": _parse_json(row[3]) or {},
        "is_active": bool(row[4]),
        "created_at": row[5],
        "updated_at": row[6],
    }


def _to_json(value: Any) -> str:
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

