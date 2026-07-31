"""Provider-neutral durable attachment storage and adapter boundary for v2."""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4

from alphonse.agent_v2.database import connect_database, default_database_path

MAX_ATTACHMENT_BYTES = 50 * 1024 * 1024
SUPPORTED_MIME_PREFIXES = ("image/", "audio/")
SUPPORTED_MIME_TYPES = {"application/pdf", "audio/ogg"}


@dataclass(frozen=True)
class AttachmentDescriptor:
    filename: str
    mime_type: str
    size_bytes: int
    provider_file_id: str = ""
    caption: str = ""
    kind: str = "document"


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str; owner_user_id: str; filename: str; mime_type: str; size_bytes: int; sha256: str; path: str; source: str; extracted_text: str = ""; processing_status: str = "unindexed"; created_at: str = ""
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()


class AttachmentAdapter(Protocol):
    """The only provider-specific attachment operations."""
    def describe_inbound(self, raw: dict[str, Any]) -> list[AttachmentDescriptor]: ...
    def download(self, descriptor: AttachmentDescriptor) -> bytes: ...
    def send(self, *, target: str, asset: AssetRecord, caption: str = "") -> str: ...


class SQLiteAssetStore:
    def __init__(self, db_path: str | Path = ":memory:", root: str | Path | None = None) -> None:
        self.db_path = str(db_path); self.root = Path(root or Path.home() / ".alphonse" / "assets")
        self._memory = sqlite3.connect(":memory:", check_same_thread=False) if self.db_path == ":memory:" else None
        if self._memory is not None: self._memory.row_factory = sqlite3.Row
        self._ensure_schema()
    @classmethod
    def default(cls) -> "SQLiteAssetStore":
        base = Path(os.getenv("ALPHONSE_V2_ASSETS_ROOT") or Path.home() / ".alphonse" / "assets")
        return cls(default_database_path(), base)
    def register_bytes(self, *, owner_user_id: str, descriptor: AttachmentDescriptor, content: bytes, source: str) -> AssetRecord:
        if len(content) > MAX_ATTACHMENT_BYTES: raise ValueError("attachment_too_large")
        if not _supported(descriptor.mime_type): raise ValueError("attachment_type_unsupported")
        asset_id = str(uuid4()); digest = hashlib.sha256(content).hexdigest(); suffix = Path(descriptor.filename).suffix or mimetypes.guess_extension(descriptor.mime_type) or ""
        folder = self.root / str(owner_user_id) / asset_id; folder.mkdir(parents=True, exist_ok=False); path = folder / f"original{suffix}"
        path.write_bytes(content); record = AssetRecord(asset_id, str(owner_user_id), _safe_name(descriptor.filename), descriptor.mime_type, len(content), digest, str(path), str(source), created_at=_now())
        with self._connect() as conn: conn.execute("INSERT INTO v2_assets(asset_id,owner_user_id,filename,mime_type,size_bytes,sha256,path,source,extracted_text,processing_status,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (*record.__dict__.values(),))
        return record
    def get(self, asset_id: str, *, requester_user_id: str, is_admin: Callable[[str], bool] | None = None) -> AssetRecord | None:
        with self._connect() as conn: row = conn.execute("SELECT * FROM v2_assets WHERE asset_id=?", (str(asset_id),)).fetchone()
        record = _record(row)
        if record is None or (record.owner_user_id != requester_user_id and not bool(is_admin and is_admin(requester_user_id))): return None
        return record
    def system_get(self, asset_id: str) -> AssetRecord | None:
        with self._connect() as conn: row = conn.execute("SELECT * FROM v2_assets WHERE asset_id=?", (str(asset_id),)).fetchone()
        return _record(row)
    def set_extraction(self, asset_id: str, *, text: str = "", status: str = "indexed") -> None:
        with self._connect() as conn: conn.execute("UPDATE v2_assets SET extracted_text=?, processing_status=? WHERE asset_id=?", (str(text), str(status), str(asset_id)))
    def delete(self, asset_id: str, *, requester_user_id: str, is_admin: Callable[[str], bool] | None = None) -> bool:
        record = self.get(asset_id, requester_user_id=requester_user_id, is_admin=is_admin)
        if record is None: return False
        with self._connect() as conn: conn.execute("DELETE FROM v2_assets WHERE asset_id=?", (record.asset_id,))
        shutil.rmtree(Path(record.path).parent, ignore_errors=True); return True
    def _connect(self):
        if self._memory is not None: return _Connection(self._memory)
        return connect_database(self.db_path)
    def _ensure_schema(self) -> None:
        with self._connect() as conn: conn.execute("CREATE TABLE IF NOT EXISTS v2_assets (asset_id TEXT PRIMARY KEY, owner_user_id TEXT NOT NULL, filename TEXT NOT NULL, mime_type TEXT NOT NULL, size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, path TEXT NOT NULL, source TEXT NOT NULL, extracted_text TEXT NOT NULL DEFAULT '', processing_status TEXT NOT NULL DEFAULT 'unindexed', created_at TEXT NOT NULL) STRICT")


class _Connection:
    def __init__(self, conn): self.conn=conn
    def __enter__(self): return self.conn
    def __exit__(self, t,v,b): self.conn.commit() if t is None else self.conn.rollback()
def _supported(mime: str) -> bool: return mime in SUPPORTED_MIME_TYPES or any(mime.startswith(prefix) for prefix in SUPPORTED_MIME_PREFIXES)
def _safe_name(name: str) -> str: return Path(str(name or "attachment")).name[:180] or "attachment"
def _now() -> str: return datetime.now(timezone.utc).isoformat()
def _record(row: Any) -> AssetRecord | None: return AssetRecord(**dict(row)) if row is not None else None
