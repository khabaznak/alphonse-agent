"""Shared SQLite configuration and schema migration bookkeeping for Alphonse v2."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

DEFAULT_BUSY_TIMEOUT_MS = 5_000
DEFAULT_DATABASE_NAME = "alphonse-v2.sqlite3"
_LOCK_RETRY_COUNT = 0
_LOCK_RETRY_GUARD = threading.Lock()


class AlphonseSQLiteConnection(sqlite3.Connection):
    """Connection with bounded telemetry-producing retries after SQLite's busy wait."""

    def execute(self, sql: str, parameters=(), /):  # type: ignore[override]
        for attempt in range(3):
            try:
                return super().execute(sql, parameters)
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 2:
                    raise
                _record_lock_retry()
                time.sleep(0.025 * (2**attempt))
        raise AssertionError("unreachable")


def default_database_path() -> Path:
    configured = str(os.getenv("ALPHONSE_V2_DB_PATH") or "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".alphonse" / DEFAULT_DATABASE_NAME


def connect_database(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        path,
        timeout=DEFAULT_BUSY_TIMEOUT_MS / 1000,
        factory=AlphonseSQLiteConnection,
    )
    return configure_connection(connection, persistent=True)


def configure_connection(connection: sqlite3.Connection, *, persistent: bool) -> sqlite3.Connection:
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {DEFAULT_BUSY_TIMEOUT_MS}")
    if persistent:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
    ensure_schema_migrations(connection)
    return connection


def ensure_schema_migrations(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS v2_schema_migrations (
          migration_id TEXT PRIMARY KEY,
          applied_at TEXT NOT NULL,
          details_json TEXT NOT NULL DEFAULT '{}'
        ) STRICT
        """
    )


def migration_applied(connection: sqlite3.Connection, migration_id: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM v2_schema_migrations WHERE migration_id = ?",
        (str(migration_id or "").strip(),),
    ).fetchone()
    return row is not None


def record_migration(connection: sqlite3.Connection, migration_id: str, *, details_json: str = "{}") -> None:
    connection.execute(
        """
        INSERT OR IGNORE INTO v2_schema_migrations (migration_id, applied_at, details_json)
        VALUES (?, ?, ?)
        """,
        (
            str(migration_id or "").strip(),
            datetime.now(timezone.utc).isoformat(),
            str(details_json or "{}"),
        ),
    )


def lock_retry_count() -> int:
    with _LOCK_RETRY_GUARD:
        return _LOCK_RETRY_COUNT


def _record_lock_retry() -> None:
    global _LOCK_RETRY_COUNT
    with _LOCK_RETRY_GUARD:
        _LOCK_RETRY_COUNT += 1


@contextmanager
def transaction(db_path: str | Path, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
    connection = connect_database(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
