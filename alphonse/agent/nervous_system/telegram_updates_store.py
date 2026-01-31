from __future__ import annotations

import sqlite3
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


def mark_update_processed(update_id: int, chat_id: str | None) -> bool:
    db_path = resolve_nervous_system_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO telegram_updates (update_id, chat_id) VALUES (?, ?)",
            (update_id, chat_id),
        )
        conn.commit()
        return cursor.rowcount == 1


def is_update_processed(update_id: int) -> bool:
    db_path = resolve_nervous_system_db_path()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT update_id FROM telegram_updates WHERE update_id = ?",
            (update_id,),
        ).fetchone()
    return row is not None
