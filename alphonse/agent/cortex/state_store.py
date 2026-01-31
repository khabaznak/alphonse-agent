from __future__ import annotations

import json
import sqlite3
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


def load_state(chat_id: str) -> dict[str, Any] | None:
    db_path = resolve_nervous_system_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cortex_sessions (
              chat_id TEXT PRIMARY KEY,
              state_json TEXT NOT NULL,
              updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        row = conn.execute(
            "SELECT state_json FROM cortex_sessions WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()
    if not row:
        return None
    try:
        parsed = json.loads(row[0])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def save_state(chat_id: str, state: dict[str, Any]) -> None:
    db_path = resolve_nervous_system_db_path()
    payload = json.dumps(state)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO cortex_sessions (chat_id, state_json, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(chat_id) DO UPDATE
            SET state_json = excluded.state_json,
                updated_at = datetime('now')
            """,
            (chat_id, payload),
        )
        conn.commit()
