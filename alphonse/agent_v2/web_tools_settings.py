"""Persistent, daemon-owned configuration for the optional V2 web tools."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from alphonse.agent_v2.database import connect_database, default_database_path


@dataclass(frozen=True)
class WebToolsSettings:
    enabled: bool = False
    searxng_base_url: str = ""
    search_timeout_seconds: float = 10.0
    fetch_timeout_seconds: float = 10.0
    fetch_max_chars: int = 12000
    updated_at: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.searxng_base_url)

    @property
    def available(self) -> bool:
        return self.enabled and self.configured

    def to_dict(self) -> dict[str, object]:
        return {**self.__dict__, "configured": self.configured, "available": self.available}


class SQLiteWebToolsSettingsStore:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory = sqlite3.connect(":memory:", check_same_thread=False) if self.db_path == ":memory:" else None
        if self._memory is not None:
            self._memory.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls) -> "SQLiteWebToolsSettingsStore":
        return cls(default_database_path())

    def get(self) -> WebToolsSettings:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM v2_web_tools_settings WHERE settings_id=1").fetchone()
        if row is None:
            return WebToolsSettings()
        return WebToolsSettings(bool(row["enabled"]), str(row["searxng_base_url"]), float(row["search_timeout_seconds"]), float(row["fetch_timeout_seconds"]), int(row["fetch_max_chars"]), str(row["updated_at"]))

    def save(self, settings: WebToolsSettings) -> WebToolsSettings:
        base_url = _base_url(settings.searxng_base_url)
        search_timeout = _timeout(settings.search_timeout_seconds, "search_timeout_seconds")
        fetch_timeout = _timeout(settings.fetch_timeout_seconds, "fetch_timeout_seconds")
        fetch_max_chars = _max_chars(settings.fetch_max_chars)
        if settings.enabled and not base_url:
            raise ValueError("web_tools_searxng_base_url_required")
        with self._connect() as conn:
            conn.execute("""INSERT OR REPLACE INTO v2_web_tools_settings
                (settings_id,enabled,searxng_base_url,search_timeout_seconds,fetch_timeout_seconds,fetch_max_chars,updated_at)
                VALUES (1,?,?,?,?,?,?)""", (int(bool(settings.enabled)), base_url, search_timeout, fetch_timeout, fetch_max_chars, _now()))
        return self.get()

    def _connect(self):
        if self._memory is not None:
            return _Connection(self._memory)
        path = Path(self.db_path); path.parent.mkdir(parents=True, exist_ok=True)
        return connect_database(path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""CREATE TABLE IF NOT EXISTS v2_web_tools_settings (
              settings_id INTEGER PRIMARY KEY CHECK(settings_id=1), enabled INTEGER NOT NULL DEFAULT 0 CHECK(enabled IN (0,1)),
              searxng_base_url TEXT NOT NULL DEFAULT '', search_timeout_seconds REAL NOT NULL DEFAULT 10,
              fetch_timeout_seconds REAL NOT NULL DEFAULT 10, fetch_max_chars INTEGER NOT NULL DEFAULT 12000,
              updated_at TEXT NOT NULL
            ) STRICT;""")


class _Connection:
    def __init__(self, connection: sqlite3.Connection) -> None: self.connection = connection
    def __enter__(self): return self.connection
    def __exit__(self, typ, value, traceback): self.connection.commit() if typ is None else self.connection.rollback()


def _base_url(value: object) -> str:
    from urllib.parse import urlparse
    rendered = str(value or "").strip().rstrip("/")
    if not rendered:
        return ""
    parsed = urlparse(rendered)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("web_tools_searxng_base_url_invalid")
    return rendered


def _timeout(value: object, field: str) -> float:
    try: rendered = float(value)
    except (TypeError, ValueError) as exc: raise ValueError(f"web_tools_{field}_invalid") from exc
    if not 0 < rendered <= 60: raise ValueError(f"web_tools_{field}_invalid")
    return rendered


def _max_chars(value: object) -> int:
    try: rendered = int(value)
    except (TypeError, ValueError) as exc: raise ValueError("web_tools_fetch_max_chars_invalid") from exc
    if not 100 <= rendered <= 100000: raise ValueError("web_tools_fetch_max_chars_invalid")
    return rendered


def _now() -> str: return datetime.now(timezone.utc).isoformat()
