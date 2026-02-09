from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


def upsert_invite(record: dict[str, Any]) -> str:
    chat_id = str(record.get("chat_id") or "").strip()
    if not chat_id:
        raise ValueError("chat_id is required")
    now = _now_iso()
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO telegram_pending_invites (
              chat_id, from_user_id, from_user_name, last_message, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
              from_user_id = excluded.from_user_id,
              from_user_name = excluded.from_user_name,
              last_message = excluded.last_message,
              status = excluded.status,
              updated_at = excluded.updated_at
            """,
            (
                chat_id,
                record.get("from_user_id"),
                record.get("from_user_name"),
                record.get("last_message"),
                record.get("status") or "pending",
                now,
                now,
            ),
        )
        conn.commit()
    return chat_id


def list_invites(status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    filters: list[str] = []
    params: list[Any] = []
    if status:
        filters.append("status = ?")
        params.append(status)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = (
        "SELECT chat_id, from_user_id, from_user_name, last_message, status, created_at, updated_at "
        f"FROM telegram_pending_invites {where} ORDER BY updated_at DESC LIMIT ?"
    )
    params.append(limit)
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [_row_to_invite(row) for row in rows]


def get_invite(chat_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT chat_id, from_user_id, from_user_name, last_message, status, created_at, updated_at
            FROM telegram_pending_invites
            WHERE chat_id = ?
            """,
            (chat_id,),
        ).fetchone()
    return _row_to_invite(row) if row else None


def update_invite_status(chat_id: str, status: str) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            """
            UPDATE telegram_pending_invites
            SET status = ?, updated_at = ?
            WHERE chat_id = ?
            """,
            (status, _now_iso(), chat_id),
        )
        conn.commit()
    return get_invite(chat_id)


def _row_to_invite(row: sqlite3.Row | tuple | None) -> dict[str, Any]:
    if row is None:
        return {}
    if not isinstance(row, tuple):
        row = tuple(row)
    return {
        "chat_id": row[0],
        "from_user_id": row[1],
        "from_user_name": row[2],
        "last_message": row[3],
        "status": row[4],
        "created_at": row[5],
        "updated_at": row[6],
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
