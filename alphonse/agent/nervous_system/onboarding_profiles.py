from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


def upsert_onboarding_profile(record: dict[str, Any]) -> str:
    principal_id = str(record.get("principal_id") or "")
    if not principal_id:
        raise ValueError("principal_id is required")
    now = _now_iso()
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        _ensure_principal_exists(conn, principal_id, now)
        conn.execute(
            """
            INSERT INTO onboarding_profiles (
              principal_id, state, primary_role, next_steps_json, resume_token,
              completed_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(principal_id) DO UPDATE SET
              state = excluded.state,
              primary_role = excluded.primary_role,
              next_steps_json = excluded.next_steps_json,
              resume_token = excluded.resume_token,
              completed_at = excluded.completed_at,
              updated_at = excluded.updated_at
            """,
            (
                principal_id,
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
    return principal_id


def _ensure_principal_exists(conn: sqlite3.Connection, principal_id: str, now: str) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO principals (
          principal_id, principal_type, created_at, updated_at
        ) VALUES (?, 'person', ?, ?)
        """,
        (principal_id, now, now),
    )


def list_onboarding_profiles(*, state: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    filters: list[str] = []
    values: list[Any] = []
    if state:
        filters.append("state = ?")
        values.append(state)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = (
        "SELECT principal_id, state, primary_role, next_steps_json, resume_token, "
        "completed_at, created_at, updated_at "
        f"FROM onboarding_profiles {where} ORDER BY updated_at DESC LIMIT ?"
    )
    values.append(limit)
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        rows = conn.execute(query, tuple(values)).fetchall()
    return [_row_to_onboarding_profile(row) for row in rows]


def get_onboarding_profile(principal_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT principal_id, state, primary_role, next_steps_json, resume_token,
                   completed_at, created_at, updated_at
            FROM onboarding_profiles
            WHERE principal_id = ?
            """,
            (principal_id,),
        ).fetchone()
    return _row_to_onboarding_profile(row) if row else None


def delete_onboarding_profile(principal_id: str) -> bool:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        cur = conn.execute(
            "DELETE FROM onboarding_profiles WHERE principal_id = ?",
            (principal_id,),
        )
        conn.commit()
    return cur.rowcount > 0


def _row_to_onboarding_profile(row: sqlite3.Row | tuple | None) -> dict[str, Any]:
    if row is None:
        return {}
    if not isinstance(row, tuple):
        row = tuple(row)
    return {
        "principal_id": row[0],
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
