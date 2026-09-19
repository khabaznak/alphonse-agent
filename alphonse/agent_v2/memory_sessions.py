"""Durable project-shared memory sessions and channel bindings."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from alphonse.agent_v2.database import connect_database, default_database_path


@dataclass(frozen=True)
class MemorySessionRecord:
    session_id: str
    project_id: str
    name: str
    created_by_user_id: str
    status: str
    ledger_relative_path: str
    created_at: str
    updated_at: str
    closed_at: str = ""
    is_general: bool = False
    is_system: bool = False

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class MemorySessionBindingKey:
    user_id: str
    integration_id: str
    channel_target: str
    thread_id: str
    project_id: str


class SQLiteMemorySessionStore:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory = sqlite3.connect(":memory:", check_same_thread=False) if self.db_path == ":memory:" else None
        if self._memory is not None:
            self._memory.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls) -> "SQLiteMemorySessionStore":
        return cls(default_database_path())

    def create(
        self,
        *,
        project_id: str,
        name: str,
        created_by_user_id: str,
        is_general: bool = False,
        is_system: bool = False,
        session_id: str = "",
    ) -> MemorySessionRecord:
        project = _required(project_id, "project_id")
        display_name = _required(name, "session_name")[:120]
        creator = _required(created_by_user_id, "created_by_user_id")
        identifier = str(session_id or uuid4()).strip()
        now = _now()
        relative = f"sessions/{_safe_segment(identifier)}"
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO v2_memory_sessions(
                     session_id,project_id,name,created_by_user_id,status,ledger_relative_path,
                     created_at,updated_at,closed_at,is_general,is_system
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (identifier, project, display_name, creator, "open", relative, now, now, "", int(is_general), int(is_system)),
            )
        return self.get(identifier)  # type: ignore[return-value]

    def get(self, session_id: str) -> MemorySessionRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM v2_memory_sessions WHERE session_id=?", (str(session_id),)).fetchone()
        return _record(row)

    def list(self, project_id: str, *, include_closed: bool = False, include_system: bool = False) -> list[MemorySessionRecord]:
        clauses = ["project_id=?"]
        values: list[object] = [str(project_id)]
        if not include_closed:
            clauses.append("status='open'")
        if not include_system:
            clauses.append("is_system=0")
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM v2_memory_sessions WHERE {' AND '.join(clauses)} ORDER BY is_general DESC, updated_at DESC, name",
                tuple(values),
            ).fetchall()
        return [_record(row) for row in rows if row is not None]  # type: ignore[misc]

    def resolve_open(self, project_id: str, value: str) -> MemorySessionRecord:
        needle = str(value or "").strip()
        sessions = self.list(project_id)
        exact = [item for item in sessions if item.session_id == needle]
        if exact:
            return exact[0]
        named = [item for item in sessions if item.name.casefold() == needle.casefold()]
        if len(named) == 1:
            return named[0]
        if len(named) > 1:
            raise LookupError("Session name is ambiguous; use its id.")
        raise LookupError(f"Open session not found: {needle}.")

    def ensure_general(self, *, project_id: str, created_by_user_id: str) -> MemorySessionRecord:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM v2_memory_sessions WHERE project_id=? AND is_general=1 AND status='open' ORDER BY created_at DESC LIMIT 1",
                (str(project_id),),
            ).fetchone()
        existing = _record(row)
        return existing or self.create(project_id=project_id, name="General", created_by_user_id=created_by_user_id, is_general=True)

    def ensure_system(self, *, project_id: str, identity: str, created_by_user_id: str) -> MemorySessionRecord:
        identifier = f"automation-{_safe_segment(project_id)[:32]}-{_safe_segment(identity)[:48]}"
        existing = self.get(identifier)
        if existing is not None:
            return existing
        try:
            return self.create(project_id=project_id, name=f"Automation: {identity}"[:120], created_by_user_id=created_by_user_id, is_system=True, session_id=identifier)
        except sqlite3.IntegrityError:
            return self.get(identifier)  # type: ignore[return-value]

    def close(self, session_id: str) -> MemorySessionRecord:
        now = _now()
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE v2_memory_sessions SET status='closed',closed_at=?,updated_at=? WHERE session_id=? AND status='open'",
                (now, now, str(session_id)),
            )
        if cursor.rowcount != 1:
            raise ValueError("memory_session_not_open")
        return self.get(session_id)  # type: ignore[return-value]

    def touch(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE v2_memory_sessions SET updated_at=? WHERE session_id=?", (_now(), str(session_id)))

    def get_binding(self, key: MemorySessionBindingKey) -> MemorySessionRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT s.* FROM v2_memory_session_bindings b
                   JOIN v2_memory_sessions s ON s.session_id=b.session_id
                   WHERE b.user_id=? AND b.integration_id=? AND b.channel_target=? AND b.thread_id=? AND b.project_id=?""",
                _binding_values(key),
            ).fetchone()
        session = _record(row)
        return session if session is not None and session.status == "open" else None

    def bind(self, key: MemorySessionBindingKey, session: MemorySessionRecord) -> None:
        if session.project_id != key.project_id or session.status != "open":
            raise ValueError("memory_session_not_bindable")
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO v2_memory_session_bindings(user_id,integration_id,channel_target,thread_id,project_id,session_id,updated_at)
                   VALUES (?,?,?,?,?,?,?) ON CONFLICT(user_id,integration_id,channel_target,thread_id,project_id)
                   DO UPDATE SET session_id=excluded.session_id,updated_at=excluded.updated_at""",
                (*_binding_values(key), session.session_id, _now()),
            )

    def bindings_for_session(self, session_id: str) -> list[MemorySessionBindingKey]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM v2_memory_session_bindings WHERE session_id=?", (str(session_id),)).fetchall()
        return [MemorySessionBindingKey(str(r["user_id"]), str(r["integration_id"]), str(r["channel_target"]), str(r["thread_id"]), str(r["project_id"])) for r in rows]

    def migration_status(self, project_id: str) -> dict[str, str]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM v2_memory_session_migrations WHERE project_id=?", (str(project_id),)).fetchone()
        if row is None:
            return {"project_id": str(project_id), "status": "pending", "error": "", "updated_at": ""}
        return {"project_id": str(row["project_id"]), "status": str(row["status"]), "error": str(row["error"]), "updated_at": str(row["updated_at"])}

    def set_migration_status(self, project_id: str, status: str, error: str = "") -> dict[str, str]:
        value = str(status or "").strip()
        if value not in {"pending", "running", "complete", "failed"}: raise ValueError("memory_migration_status_invalid")
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO v2_memory_session_migrations(project_id,status,error,updated_at) VALUES (?,?,?,?)
                   ON CONFLICT(project_id) DO UPDATE SET status=excluded.status,error=excluded.error,updated_at=excluded.updated_at""",
                (str(project_id), value, str(error or "")[:2000], _now()),
            )
        return self.migration_status(project_id)

    def _connect(self):
        if self._memory is not None:
            return _Connection(self._memory)
        path = Path(self.db_path); path.parent.mkdir(parents=True, exist_ok=True)
        return connect_database(path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS v2_memory_sessions (
                  session_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, name TEXT NOT NULL,
                  created_by_user_id TEXT NOT NULL, status TEXT NOT NULL,
                  ledger_relative_path TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                  closed_at TEXT NOT NULL DEFAULT '', is_general INTEGER NOT NULL DEFAULT 0,
                  is_system INTEGER NOT NULL DEFAULT 0, CHECK(status IN ('open','closed'))
                ) STRICT;
                CREATE INDEX IF NOT EXISTS idx_v2_memory_sessions_project ON v2_memory_sessions(project_id,status,updated_at);
                CREATE TABLE IF NOT EXISTS v2_memory_session_bindings (
                  user_id TEXT NOT NULL, integration_id TEXT NOT NULL, channel_target TEXT NOT NULL,
                  thread_id TEXT NOT NULL DEFAULT '', project_id TEXT NOT NULL, session_id TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY(user_id,integration_id,channel_target,thread_id,project_id),
                  FOREIGN KEY(session_id) REFERENCES v2_memory_sessions(session_id)
                ) STRICT;
                CREATE TABLE IF NOT EXISTS v2_memory_session_migrations (
                  project_id TEXT PRIMARY KEY, status TEXT NOT NULL, error TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL,
                  CHECK(status IN ('pending','running','complete','failed'))
                ) STRICT;
                """
            )


class _Connection:
    def __init__(self, connection: sqlite3.Connection) -> None: self.connection = connection
    def __enter__(self): return self.connection
    def __exit__(self, typ, value, traceback): self.connection.commit() if typ is None else self.connection.rollback()


def _record(row: sqlite3.Row | None) -> MemorySessionRecord | None:
    if row is None: return None
    return MemorySessionRecord(
        session_id=str(row["session_id"]), project_id=str(row["project_id"]), name=str(row["name"]),
        created_by_user_id=str(row["created_by_user_id"]), status=str(row["status"]),
        ledger_relative_path=str(row["ledger_relative_path"]), created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]), closed_at=str(row["closed_at"] or ""),
        is_general=bool(row["is_general"]), is_system=bool(row["is_system"]),
    )


def _binding_values(key: MemorySessionBindingKey) -> tuple[str, ...]:
    return (key.user_id, key.integration_id, key.channel_target, key.thread_id, key.project_id)


def _safe_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-.") or uuid4().hex


def _required(value: str, error: str) -> str:
    rendered = str(value or "").strip()
    if not rendered: raise ValueError(error)
    return rendered


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
