from __future__ import annotations

import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


def create_prompt_artifact(
    *,
    user_id: str,
    source_instruction: str,
    agent_internal_prompt: str,
    language: str | None,
    artifact_kind: str,
) -> str:
    artifact_id = f"pa_{secrets.token_hex(6)}"
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_artifacts (
              artifact_id       TEXT PRIMARY KEY,
              user_id           TEXT NOT NULL,
              source_instruction TEXT NOT NULL,
              agent_internal_prompt TEXT NOT NULL,
              language          TEXT,
              artifact_kind     TEXT NOT NULL,
              created_at        TEXT NOT NULL DEFAULT (datetime('now')),
              updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
            ) STRICT
            """
        )
        conn.execute(
            """
            INSERT INTO prompt_artifacts (
              artifact_id, user_id, source_instruction, agent_internal_prompt, language, artifact_kind, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                str(user_id),
                str(source_instruction or ""),
                str(agent_internal_prompt or ""),
                str(language or "").strip() or None,
                str(artifact_kind or "general"),
                now,
                now,
            ),
        )
        conn.commit()
    return artifact_id


def get_prompt_artifact(artifact_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT artifact_id, user_id, source_instruction, agent_internal_prompt, language, artifact_kind, created_at, updated_at
            FROM prompt_artifacts
            WHERE artifact_id = ?
            """,
            (str(artifact_id),),
        ).fetchone()
    if not row:
        return None
    return {
        "artifact_id": row[0],
        "user_id": row[1],
        "source_instruction": row[2],
        "agent_internal_prompt": row[3],
        "language": row[4],
        "artifact_kind": row[5],
        "created_at": row[6],
        "updated_at": row[7],
    }
