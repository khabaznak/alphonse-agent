from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path

DEFAULT_SANDBOX_ALIAS = "telegram_files"


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(resolve_nervous_system_db_path())


def ensure_default_sandbox_aliases() -> None:
    root = Path(os.getenv("ALPHONSE_SANDBOX_ROOT") or "/tmp/alphonse-sandbox").resolve()
    ensure_sandbox_alias(
        alias=DEFAULT_SANDBOX_ALIAS,
        base_path=str((root / DEFAULT_SANDBOX_ALIAS).resolve()),
        description="Downloaded Telegram files sandbox",
    )


def ensure_sandbox_alias(*, alias: str, base_path: str, description: str = "") -> None:
    normalized_alias = str(alias or "").strip()
    if not normalized_alias:
        raise ValueError("sandbox alias is required")
    normalized_path = str(Path(base_path).resolve())
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sandbox_directories (
              alias       TEXT PRIMARY KEY,
              base_path   TEXT NOT NULL,
              description TEXT,
              enabled     INTEGER NOT NULL DEFAULT 1,
              created_at  TEXT NOT NULL DEFAULT (datetime('now')),
              updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
              CHECK (enabled IN (0,1))
            ) STRICT
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO sandbox_directories (alias, base_path, description, enabled)
            VALUES (?, ?, ?, 1)
            """,
            (normalized_alias, normalized_path, str(description or "").strip()),
        )
        conn.commit()


def get_sandbox_alias(alias: str) -> dict[str, Any] | None:
    normalized_alias = str(alias or "").strip()
    if not normalized_alias:
        return None
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT alias, base_path, description, enabled
            FROM sandbox_directories
            WHERE alias = ?
            """,
            (normalized_alias,),
        ).fetchone()
    if not row:
        return None
    return {
        "alias": str(row[0]),
        "base_path": str(row[1]),
        "description": str(row[2] or ""),
        "enabled": bool(row[3]),
    }


def resolve_sandbox_path(*, alias: str, relative_path: str) -> Path:
    ensure_default_sandbox_aliases()
    record = get_sandbox_alias(alias)
    if not isinstance(record, dict) or not record.get("enabled"):
        raise RuntimeError(f"sandbox_alias_not_found:{alias}")
    base = Path(str(record["base_path"])).resolve()
    base.mkdir(parents=True, exist_ok=True)

    rel = Path(str(relative_path or "").strip())
    if rel.is_absolute():
        raise ValueError("relative_path_must_be_relative")
    parts = [part for part in rel.parts if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise ValueError("relative_path_invalid")
    safe_rel = Path(*parts) if parts else Path("file.bin")
    resolved = (base / safe_rel).resolve()

    if os.path.commonpath([str(base), str(resolved)]) != str(base):
        raise ValueError("sandbox_path_escape")
    return resolved
