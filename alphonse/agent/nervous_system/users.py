from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


def list_users(*, active_only: bool = False, limit: int = 200) -> list[dict[str, Any]]:
    filters: list[str] = []
    values: list[Any] = []
    if active_only:
        filters.append("is_active = 1")
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = (
        "SELECT user_id, principal_id, display_name, role, relationship, is_admin, is_active, "
        "onboarded_at, created_at, updated_at "
        f"FROM users {where} ORDER BY updated_at DESC LIMIT ?"
    )
    values.append(limit)
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        rows = conn.execute(query, tuple(values)).fetchall()
    return [_row_to_user(row) for row in rows]


def get_user(user_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            "SELECT user_id, principal_id, display_name, role, relationship, is_admin, is_active, "
            "onboarded_at, created_at, updated_at FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return _row_to_user(row) if row else None


def get_user_by_display_name(display_name: str) -> dict[str, Any] | None:
    name = str(display_name or "").strip()
    if not name:
        return None
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            "SELECT user_id, principal_id, display_name, role, relationship, is_admin, is_active, "
            "onboarded_at, created_at, updated_at FROM users WHERE lower(display_name) = lower(?) "
            "ORDER BY updated_at DESC LIMIT 1",
            (name,),
        ).fetchone()
    return _row_to_user(row) if row else None


def upsert_user(record: dict[str, Any]) -> str:
    user_id = str(record.get("user_id") or record.get("principal_id") or uuid.uuid4())
    display_name = str(record.get("display_name") or "").strip()
    if not display_name:
        raise ValueError("display_name is required")
    now = _now_iso()
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO users (
              user_id, principal_id, display_name, role, relationship, is_admin, is_active,
              onboarded_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
              principal_id = excluded.principal_id,
              display_name = excluded.display_name,
              role = excluded.role,
              relationship = excluded.relationship,
              is_admin = excluded.is_admin,
              is_active = excluded.is_active,
              onboarded_at = excluded.onboarded_at,
              updated_at = excluded.updated_at
            """,
            (
                user_id,
                record.get("principal_id"),
                display_name,
                record.get("role"),
                record.get("relationship"),
                1 if bool(record.get("is_admin", False)) else 0,
                1 if bool(record.get("is_active", True)) else 0,
                record.get("onboarded_at"),
                now,
                now,
            ),
        )
        conn.commit()
    return user_id


def patch_user(user_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    current = get_user(user_id)
    if not current:
        return None
    merged = dict(current)
    merged.update({k: v for k, v in updates.items() if v is not None})
    merged["user_id"] = user_id
    upsert_user(merged)
    return get_user(user_id)


def delete_user(user_id: str) -> bool:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        cur = conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.commit()
    return cur.rowcount > 0


def _row_to_user(row: sqlite3.Row | tuple | None) -> dict[str, Any]:
    if row is None:
        return {}
    if not isinstance(row, tuple):
        row = tuple(row)
    return {
        "user_id": row[0],
        "principal_id": row[1],
        "display_name": row[2],
        "role": row[3],
        "relationship": row[4],
        "is_admin": bool(row[5]),
        "is_active": bool(row[6]),
        "onboarded_at": row[7],
        "created_at": row[8],
        "updated_at": row[9],
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
