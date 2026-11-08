"""Persistent configuration and explicit per-actor browser grants.

This is a storage boundary, not a browser sandbox or a substitute for checking
current grants immediately before and after each browser operation.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from alphonse.agent_v2.database import connect_database, default_database_path


@dataclass(frozen=True)
class BrowserAccess:
    enabled: bool = False
    profile_mode: str = "separate"
    existing_session_consent: bool = False
    revision: int = 0


class BrowserAccessStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._memory = sqlite3.connect(":memory:", check_same_thread=False) if self.path == ":memory:" else None
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS browser_access_settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
                    profile_mode TEXT NOT NULL CHECK (profile_mode IN ('separate', 'existing')),
                    existing_session_consent INTEGER NOT NULL CHECK (existing_session_consent IN (0, 1)),
                    revision INTEGER NOT NULL
                ) STRICT;
                INSERT OR IGNORE INTO browser_access_settings VALUES (1, 0, 'separate', 0, 0);
                CREATE TABLE IF NOT EXISTS browser_task_attempts (
                    actor_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    settings_revision INTEGER NOT NULL,
                    PRIMARY KEY (actor_id, task_id)
                ) STRICT;
                CREATE TABLE IF NOT EXISTS browser_actor_grants (
                    actor_id TEXT PRIMARY KEY,
                    profile_mode TEXT NOT NULL CHECK (profile_mode IN ('separate', 'existing')),
                    settings_revision INTEGER NOT NULL,
                    grant_id TEXT NOT NULL
                ) STRICT;
            """)

    @classmethod
    def default(cls) -> "BrowserAccessStore":
        return cls(default_database_path())

    @contextmanager
    def _connect(self):
        with self._lock:
            conn = self._memory if self._memory is not None else connect_database(self.path)
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                if self._memory is None:
                    conn.close()

    def get(self) -> BrowserAccess:
        with self._connect() as conn:
            row = conn.execute("SELECT enabled, profile_mode, existing_session_consent, revision FROM browser_access_settings WHERE id = 1").fetchone()
        return BrowserAccess(bool(row[0]), row[1], bool(row[2]), row[3])

    def configure(self, *, enabled: bool, profile_mode: str, existing_session_consent: bool = False) -> BrowserAccess:
        if type(enabled) is not bool or type(existing_session_consent) is not bool:
            raise ValueError("browser_settings_invalid")
        if profile_mode not in ("separate", "existing"):
            raise ValueError("browser_profile_mode_invalid")
        if profile_mode == "existing" and enabled and not existing_session_consent:
            raise ValueError("browser_existing_consent_required")
        # Consent is scoped to the existing mode and never survives switching.
        consent = profile_mode == "existing" and enabled and existing_session_consent
        with self._connect() as conn:
            conn.execute("UPDATE browser_access_settings SET enabled=?, profile_mode=?, existing_session_consent=?, revision=revision+1 WHERE id=1", (int(enabled), profile_mode, int(consent)))
            conn.execute("DELETE FROM browser_actor_grants")
        return self.get()

    def grant(self, actor_id: str) -> tuple[int, str]:
        if not isinstance(actor_id, str) or not actor_id.strip() or actor_id != actor_id.strip():
            raise ValueError("browser_actor_invalid")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT enabled, profile_mode, existing_session_consent, revision FROM browser_access_settings WHERE id=1").fetchone()
            if not row[0] or (row[1] == "existing" and not row[2]):
                raise PermissionError("browser_access_disabled")
            grant_id = uuid4().hex
            conn.execute("INSERT OR REPLACE INTO browser_actor_grants VALUES (?, ?, ?, ?)", (actor_id, row[1], row[3], grant_id))
            return int(row[3]), grant_id

    def reserve_task(self, actor_id: str, task_id: str, revision: int, grant_id: str) -> bool:
        """Reserve one launch attempt durably; never adopt a tab after restart."""
        with self._connect() as conn:
            cursor = conn.execute("""INSERT OR IGNORE INTO browser_task_attempts (actor_id, task_id, settings_revision)
                SELECT ?, ?, s.revision FROM browser_access_settings s
                JOIN browser_actor_grants g ON g.profile_mode=s.profile_mode AND g.settings_revision=s.revision
                WHERE s.id=1 AND s.enabled=1 AND g.actor_id=? AND s.revision=? AND g.grant_id=?
                AND (s.profile_mode='separate' OR s.existing_session_consent=1)""",
                (actor_id, task_id, actor_id, revision, grant_id))
            return cursor.rowcount == 1

    def grant_snapshot(self, actor_id: str) -> tuple[int, str] | None:
        with self._connect() as conn:
            row = conn.execute("""SELECT s.revision, g.grant_id FROM browser_access_settings s
                JOIN browser_actor_grants g ON g.profile_mode=s.profile_mode AND g.settings_revision=s.revision
                WHERE s.id=1 AND s.enabled=1 AND g.actor_id=?
                AND (s.profile_mode='separate' OR s.existing_session_consent=1)""", (actor_id,)).fetchone()
        return (int(row[0]), str(row[1])) if row else None

    def granted_actors(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("""SELECT g.actor_id FROM browser_actor_grants g
                JOIN browser_access_settings s ON s.id=1 AND s.enabled=1
                AND g.profile_mode=s.profile_mode AND g.settings_revision=s.revision
                ORDER BY g.actor_id""").fetchall()
        return [str(row[0]) for row in rows]

    def revoke(self, actor_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM browser_actor_grants WHERE actor_id=?", (actor_id,))

    def authorized(self, actor_id: str, revision: int, grant_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("""SELECT 1 FROM browser_access_settings s
                JOIN browser_actor_grants g ON g.profile_mode=s.profile_mode AND g.settings_revision=s.revision
                WHERE s.id=1 AND s.enabled=1 AND g.actor_id=? AND s.revision=?
                AND g.grant_id=? AND (s.profile_mode='separate' OR s.existing_session_consent=1)""", (actor_id, revision, grant_id)).fetchone()
        return row is not None
