from __future__ import annotations

import os
from pathlib import Path
import shutil
import sqlite3
from datetime import datetime, timezone
from typing import Any
import uuid

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
from alphonse.agent.nervous_system.sandbox_dirs import default_sandbox_root


def create_voice_profile(record: dict[str, Any]) -> str:
    profile_id = str(record.get("profile_id") or uuid.uuid4())
    name = str(record.get("name") or "").strip()
    if not name:
        raise ValueError("name is required")
    sample_path = str(record.get("source_sample_path") or "").strip()
    if not sample_path:
        raise ValueError("source_sample_path is required")
    now = _now_iso()
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO voice_profiles (
              profile_id, name, source_sample_path, backend, speaker_hint, instruct,
              is_default, status, last_error, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile_id,
                name,
                sample_path,
                str(record.get("backend") or "qwen"),
                str(record.get("speaker_hint") or "").strip() or None,
                str(record.get("instruct") or "").strip() or None,
                1 if bool(record.get("is_default")) else 0,
                str(record.get("status") or "pending"),
                str(record.get("last_error") or "").strip() or None,
                now,
                now,
            ),
        )
        if bool(record.get("is_default")):
            _set_default_locked(conn, profile_id)
        conn.commit()
    return profile_id


def list_voice_profiles(*, limit: int = 100) -> list[dict[str, Any]]:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        rows = conn.execute(
            """
            SELECT profile_id, name, source_sample_path, backend, speaker_hint, instruct,
                   is_default, status, last_error, created_at, updated_at
            FROM voice_profiles
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()
    return [_row_to_voice_profile(row) for row in rows]


def get_voice_profile(profile_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT profile_id, name, source_sample_path, backend, speaker_hint, instruct,
                   is_default, status, last_error, created_at, updated_at
            FROM voice_profiles
            WHERE profile_id = ?
            """,
            (str(profile_id or "").strip(),),
        ).fetchone()
    return _row_to_voice_profile(row) if row else None


def get_default_voice_profile() -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT profile_id, name, source_sample_path, backend, speaker_hint, instruct,
                   is_default, status, last_error, created_at, updated_at
            FROM voice_profiles
            WHERE is_default = 1
            ORDER BY updated_at DESC
            LIMIT 1
            """
        ).fetchone()
    return _row_to_voice_profile(row) if row else None


def resolve_voice_profile(ref: str) -> dict[str, Any] | None:
    resolved = str(ref or "").strip()
    if not resolved:
        return None
    item = get_voice_profile(resolved)
    if item:
        return item
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT profile_id, name, source_sample_path, backend, speaker_hint, instruct,
                   is_default, status, last_error, created_at, updated_at
            FROM voice_profiles
            WHERE name = ?
            COLLATE NOCASE
            LIMIT 1
            """,
            (resolved,),
        ).fetchone()
    return _row_to_voice_profile(row) if row else None


def set_default_voice_profile(profile_id: str) -> bool:
    target = str(profile_id or "").strip()
    if not target:
        return False
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            "SELECT profile_id FROM voice_profiles WHERE profile_id = ?",
            (target,),
        ).fetchone()
        if not row:
            return False
        _set_default_locked(conn, target)
        conn.commit()
    return True


def set_voice_profile_status(profile_id: str, *, status: str, last_error: str | None = None) -> bool:
    target = str(profile_id or "").strip()
    if not target:
        return False
    now = _now_iso()
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        cur = conn.execute(
            """
            UPDATE voice_profiles
            SET status = ?, last_error = ?, updated_at = ?
            WHERE profile_id = ?
            """,
            (
                str(status or "pending"),
                str(last_error or "").strip() or None,
                now,
                target,
            ),
        )
        conn.commit()
        return int(cur.rowcount or 0) > 0


def delete_voice_profile(profile_id: str) -> dict[str, Any] | None:
    item = get_voice_profile(profile_id)
    if not item:
        return None
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute("DELETE FROM voice_profiles WHERE profile_id = ?", (str(item["profile_id"]),))
        conn.commit()
    return item


def store_voice_profile_sample(*, profile_id: str, sample_path: str) -> str:
    source = Path(str(sample_path or "").strip()).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"sample_not_found:{source}")
    suffix = source.suffix.lower() or ".wav"
    profile_dir = (voice_profiles_root() / str(profile_id)).resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    target = (profile_dir / f"source{suffix}").resolve()
    shutil.copy2(source, target)
    return str(target)


def purge_voice_profile_sample(source_sample_path: str) -> bool:
    raw = str(source_sample_path or "").strip()
    if not raw:
        return False
    path = Path(raw).expanduser().resolve()
    try:
        if not path.exists():
            return False
        path.unlink(missing_ok=True)
        parent = path.parent
        if parent != voice_profiles_root():
            try:
                if not any(parent.iterdir()):
                    parent.rmdir()
            except Exception:
                pass
        return True
    except Exception:
        return False


def voice_profiles_root() -> Path:
    configured = str(os.getenv("ALPHONSE_VOICE_PROFILES_ROOT") or "").strip()
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        root = (default_sandbox_root() / "voice_profiles").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _set_default_locked(conn: sqlite3.Connection, profile_id: str) -> None:
    now = _now_iso()
    conn.execute(
        "UPDATE voice_profiles SET is_default = 0, updated_at = ? WHERE is_default = 1",
        (now,),
    )
    conn.execute(
        "UPDATE voice_profiles SET is_default = 1, updated_at = ? WHERE profile_id = ?",
        (now, profile_id),
    )


def _row_to_voice_profile(row: sqlite3.Row | tuple | None) -> dict[str, Any]:
    if row is None:
        return {}
    if not isinstance(row, tuple):
        row = tuple(row)
    return {
        "profile_id": row[0],
        "name": row[1],
        "source_sample_path": row[2],
        "backend": row[3],
        "speaker_hint": row[4],
        "instruct": row[5],
        "is_default": bool(row[6]),
        "status": row[7],
        "last_error": row[8],
        "created_at": row[9],
        "updated_at": row[10],
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
