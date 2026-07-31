"""Persistent administrator settings for v2 conversation ledgers."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from alphonse.agent_v2.database import connect_database, default_database_path


DEFAULT_MAX_LEDGER_BYTES = 500 * 1024
DEFAULT_COMPACTION_SUMMARY_MAX_WORDS = 500


@dataclass(frozen=True)
class MemorySettings:
    max_ledger_bytes: int = DEFAULT_MAX_LEDGER_BYTES
    compaction_summary_max_words: int = DEFAULT_COMPACTION_SUMMARY_MAX_WORDS
    updated_at: str = ""

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


class SQLiteMemorySettingsStore:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory = sqlite3.connect(":memory:", check_same_thread=False) if self.db_path == ":memory:" else None
        if self._memory is not None:
            self._memory.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls) -> "SQLiteMemorySettingsStore":
        return cls(default_database_path())

    def get(self) -> MemorySettings:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM v2_memory_settings WHERE settings_id=1").fetchone()
        if row is None:
            return MemorySettings()
        return MemorySettings(int(row["max_ledger_bytes"]), int(row["compaction_summary_max_words"]), str(row["updated_at"]))

    def save(self, settings: MemorySettings) -> MemorySettings:
        size = _integer(settings.max_ledger_bytes, "memory_max_ledger_bytes_invalid", 1024, 100 * 1024 * 1024)
        words = _integer(settings.compaction_summary_max_words, "memory_compaction_summary_max_words_invalid", 1, 100_000)
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO v2_memory_settings (settings_id,max_ledger_bytes,compaction_summary_max_words,updated_at) VALUES (1,?,?,?)", (size, words, _now()))
        return self.get()

    def _connect(self):
        if self._memory is not None:
            return _Connection(self._memory)
        path = Path(self.db_path); path.parent.mkdir(parents=True, exist_ok=True)
        return connect_database(path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS v2_memory_settings (
              settings_id INTEGER PRIMARY KEY CHECK(settings_id=1),
              max_ledger_bytes INTEGER NOT NULL,
              compaction_summary_max_words INTEGER NOT NULL,
              updated_at TEXT NOT NULL
            ) STRICT""")


class _Connection:
    def __init__(self, connection: sqlite3.Connection) -> None: self.connection = connection
    def __enter__(self): return self.connection
    def __exit__(self, typ, value, traceback): self.connection.commit() if typ is None else self.connection.rollback()


def _integer(value: object, error: str, low: int, high: int) -> int:
    try: result = int(value)
    except (TypeError, ValueError) as exc: raise ValueError(error) from exc
    if not low <= result <= high: raise ValueError(error)
    return result


def _now() -> str: return datetime.now(timezone.utc).isoformat()
