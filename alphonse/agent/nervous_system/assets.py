from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
from alphonse.agent.nervous_system.sandbox_dirs import ensure_sandbox_alias
from alphonse.agent.nervous_system.sandbox_dirs import resolve_sandbox_path

ASSET_SANDBOX_ALIAS = "ingested_assets"


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(resolve_nervous_system_db_path())


def ensure_asset_storage() -> None:
    root = Path(os.getenv("ALPHONSE_SANDBOX_ROOT") or "/tmp/alphonse-sandbox").resolve()
    ensure_sandbox_alias(
        alias=ASSET_SANDBOX_ALIAS,
        base_path=str((root / ASSET_SANDBOX_ALIAS).resolve()),
        description="Ingested provider assets sandbox",
    )


def register_uploaded_asset(
    *,
    content: bytes,
    kind: str,
    mime_type: str | None,
    owner_user_id: str | None,
    provider: str | None,
    channel_type: str | None,
    channel_target: str | None,
    original_filename: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_asset_storage()
    blob = bytes(content or b"")
    if not blob:
        raise ValueError("asset_content_empty")

    normalized_kind = str(kind or "audio").strip().lower() or "audio"
    normalized_mime = str(mime_type or "").strip().lower()
    if not normalized_mime:
        guessed = mimetypes.guess_type(str(original_filename or ""))[0]
        normalized_mime = guessed or "application/octet-stream"

    asset_id = str(uuid.uuid4())
    digest = hashlib.sha256(blob).hexdigest()
    byte_size = len(blob)
    user_slug = _safe_slug(owner_user_id) or "anonymous"
    suffix = _choose_suffix(original_filename=original_filename, mime_type=normalized_mime)
    relative_path = f"users/{user_slug}/stt/inbox/{asset_id}{suffix}"
    resolved = resolve_sandbox_path(alias=ASSET_SANDBOX_ALIAS, relative_path=relative_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_bytes(blob)

    metadata_json = json.dumps(metadata or {}, ensure_ascii=False, separators=(",", ":"))
    with _connect() as conn:
        _ensure_assets_table(conn)
        conn.execute(
            """
            INSERT INTO assets (
              asset_id, kind, mime_type, byte_size, sha256, sandbox_alias, relative_path,
              owner_user_id, provider, channel_type, channel_target, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset_id,
                normalized_kind,
                normalized_mime,
                int(byte_size),
                digest,
                ASSET_SANDBOX_ALIAS,
                relative_path,
                _optional(owner_user_id),
                _optional(provider),
                _optional(channel_type),
                _optional(channel_target),
                metadata_json,
            ),
        )
        conn.commit()

    return {
        "asset_id": asset_id,
        "kind": normalized_kind,
        "mime": normalized_mime,
        "bytes": int(byte_size),
        "sha256": digest,
    }


def get_asset(asset_id: str) -> dict[str, Any] | None:
    normalized_id = str(asset_id or "").strip()
    if not normalized_id:
        return None
    with _connect() as conn:
        _ensure_assets_table(conn)
        row = conn.execute(
            """
            SELECT asset_id, kind, mime_type, byte_size, sha256, sandbox_alias, relative_path,
                   owner_user_id, provider, channel_type, channel_target, metadata_json, created_at
            FROM assets
            WHERE asset_id = ?
            """,
            (normalized_id,),
        ).fetchone()
    if not row:
        return None
    metadata_raw = str(row[11] or "").strip()
    parsed_meta: dict[str, Any] = {}
    if metadata_raw:
        try:
            payload = json.loads(metadata_raw)
            if isinstance(payload, dict):
                parsed_meta = payload
        except Exception:
            parsed_meta = {}
    return {
        "asset_id": str(row[0]),
        "kind": str(row[1]),
        "mime": str(row[2]),
        "bytes": int(row[3]),
        "sha256": str(row[4]),
        "sandbox_alias": str(row[5]),
        "relative_path": str(row[6]),
        "owner_user_id": _optional(row[7]),
        "provider": _optional(row[8]),
        "channel_type": _optional(row[9]),
        "channel_target": _optional(row[10]),
        "metadata": parsed_meta,
        "created_at": str(row[12] or ""),
    }


def resolve_asset_path(asset_id: str) -> Path:
    record = get_asset(asset_id)
    if not isinstance(record, dict):
        raise ValueError("asset_not_found")
    return resolve_sandbox_path(
        alias=str(record.get("sandbox_alias") or ASSET_SANDBOX_ALIAS),
        relative_path=str(record.get("relative_path") or ""),
    )


def _safe_slug(raw: str | None) -> str:
    candidate = str(raw or "").strip()
    if not candidate:
        return ""
    out = []
    for ch in candidate:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:120]


def _choose_suffix(*, original_filename: str | None, mime_type: str) -> str:
    base_suffix = Path(str(original_filename or "")).suffix.strip().lower()
    if base_suffix and len(base_suffix) <= 10:
        return base_suffix
    guessed = mimetypes.guess_extension(str(mime_type or "").strip().lower()) or ""
    guessed = str(guessed).strip().lower()
    if guessed and len(guessed) <= 10:
        return guessed
    return ".bin"


def _optional(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _ensure_assets_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS assets (
          asset_id        TEXT PRIMARY KEY,
          kind            TEXT NOT NULL,
          mime_type       TEXT NOT NULL,
          byte_size       INTEGER NOT NULL,
          sha256          TEXT NOT NULL,
          sandbox_alias   TEXT NOT NULL,
          relative_path   TEXT NOT NULL,
          owner_user_id   TEXT,
          provider        TEXT,
          channel_type    TEXT,
          channel_target  TEXT,
          metadata_json   TEXT,
          created_at      TEXT NOT NULL DEFAULT (datetime('now'))
        ) STRICT
        """
    )
