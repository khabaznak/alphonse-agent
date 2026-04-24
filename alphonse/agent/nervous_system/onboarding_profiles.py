from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


def upsert_onboarding_profile(record: dict[str, Any]) -> str:
    user_id = str(record.get("user_id") or "")
    if not user_id:
        raise ValueError("user_id is required")
    now = _now_iso()
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO onboarding_profiles (
              user_id, state, primary_role, next_steps_json, resume_token,
              completed_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
              state = excluded.state,
              primary_role = excluded.primary_role,
              next_steps_json = excluded.next_steps_json,
              resume_token = excluded.resume_token,
              completed_at = excluded.completed_at,
              updated_at = excluded.updated_at
            """,
            (
                user_id,
                record.get("state") or "not_started",
                record.get("primary_role"),
                _to_json(record.get("next_steps") or []),
                record.get("resume_token"),
                record.get("completed_at"),
                now,
                now,
            ),
        )
        conn.commit()
    return user_id


def list_onboarding_profiles(*, state: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    filters: list[str] = []
    values: list[Any] = []
    if state:
        filters.append("state = ?")
        values.append(state)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = (
        "SELECT user_id, state, primary_role, next_steps_json, resume_token, "
        "completed_at, created_at, updated_at "
        f"FROM onboarding_profiles {where} ORDER BY updated_at DESC LIMIT ?"
    )
    values.append(limit)
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        rows = conn.execute(query, tuple(values)).fetchall()
    return [_row_to_onboarding_profile(row) for row in rows]


def get_onboarding_profile(user_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT user_id, state, primary_role, next_steps_json, resume_token,
                   completed_at, created_at, updated_at
            FROM onboarding_profiles
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
    return _row_to_onboarding_profile(row) if row else None


def delete_onboarding_profile(user_id: str) -> bool:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        cur = conn.execute(
            "DELETE FROM onboarding_profiles WHERE user_id = ?",
            (user_id,),
        )
        conn.commit()
    return cur.rowcount > 0


def _row_to_onboarding_profile(row: sqlite3.Row | tuple | None) -> dict[str, Any]:
    if row is None:
        return {}
    if not isinstance(row, tuple):
        row = tuple(row)
    return {
        "user_id": row[0],
        "state": row[1],
        "primary_role": row[2],
        "next_steps": _parse_json(row[3]) or [],
        "resume_token": row[4],
        "completed_at": row[5],
        "created_at": row[6],
        "updated_at": row[7],
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
