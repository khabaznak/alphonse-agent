"""Durable, channel-neutral conversation timelines for v2 projects."""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from alphonse.agent_v2.database import connect_database, default_database_path


@dataclass(frozen=True)
class ConversationEvent:
    sequence: int
    event_id: str
    owner_user_id: str
    project_id: str
    role: str
    content: str
    source: str
    source_message_id: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class SQLiteConversationStore:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory = sqlite3.connect(":memory:", check_same_thread=False) if self.db_path == ":memory:" else None
        if self._memory is not None:
            self._memory.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls) -> "SQLiteConversationStore":
        return cls(default_database_path())

    def record(
        self,
        *,
        owner_user_id: str,
        project_id: str,
        role: str,
        content: str,
        source: str,
        source_message_id: str,
        created_at: str = "",
        connection: sqlite3.Connection | None = None,
    ) -> ConversationEvent | None:
        owner, message = str(owner_user_id or "").strip(), str(content or "").strip()
        normalized_role = str(role or "").strip().lower()
        if not owner or not message or normalized_role not in {"user", "assistant"}:
            return None
        event_id = str(uuid4())
        project = str(project_id or "").strip()
        source_value = str(source or "unknown").strip() or "unknown"
        source_id = str(source_message_id or "").strip()
        timestamp = _canonical_timestamp(created_at or _now())
        if connection is not None:
            return self._record(
                connection,
                event_id=event_id,
                owner=owner,
                project=project,
                role=normalized_role,
                message=message,
                source=source_value,
                source_id=source_id,
                timestamp=timestamp,
            )
        with self._connect() as conn:
            return self._record(
                conn,
                event_id=event_id,
                owner=owner,
                project=project,
                role=normalized_role,
                message=message,
                source=source_value,
                source_id=source_id,
                timestamp=timestamp,
            )

    @staticmethod
    def _record(
        conn: sqlite3.Connection,
        *,
        event_id: str,
        owner: str,
        project: str,
        role: str,
        message: str,
        source: str,
        source_id: str,
        timestamp: str,
    ) -> ConversationEvent | None:
        if source_id:
            exists = conn.execute(
                "SELECT * FROM v2_conversation_events WHERE source_message_id=?",
                (source_id,),
            ).fetchone()
            if exists is not None:
                return None
        cursor = conn.execute(
            "INSERT INTO v2_conversation_events(event_id,owner_user_id,project_id,role,content,source,source_message_id,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (event_id, owner, project, role, message, source, source_id, timestamp),
        )
        sequence = int(cursor.lastrowid)
        return ConversationEvent(sequence, event_id, owner, project, role, message, source, source_id, timestamp)

    def list(self, *, owner_user_id: str, project_id: str = "", limit: int = 100) -> list[ConversationEvent]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM v2_conversation_events WHERE owner_user_id=? AND project_id=? ORDER BY sequence DESC LIMIT ?", (str(owner_user_id or "").strip(), str(project_id or "").strip(), max(1, min(int(limit or 100), 500)))).fetchall()
        return [_event(row) for row in reversed(rows)]

    def sequence_for_source_message_id(self, source_message_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT sequence FROM v2_conversation_events WHERE source_message_id=?",
                (str(source_message_id or "").strip(),),
            ).fetchone()
        return int(row["sequence"] or 0) if row is not None else 0

    def mark_project_seen(self, *, owner_user_id: str, project_id: str, through_sequence: int | None = None) -> int:
        owner, project = str(owner_user_id or "").strip(), str(project_id or "").strip()
        with self._connect() as conn:
            if through_sequence is None:
                row = conn.execute(
                    "SELECT COALESCE(MAX(sequence), 0) AS latest FROM v2_conversation_events WHERE owner_user_id=? AND project_id=?",
                    (owner, project),
                ).fetchone()
                through_sequence = int(row["latest"] or 0)
            applied = max(0, int(through_sequence or 0))
            conn.execute(
                """
                INSERT INTO v2_desktop_project_cursors(owner_user_id,project_id,last_seen_sequence,updated_at)
                VALUES (?,?,?,?)
                ON CONFLICT(owner_user_id,project_id) DO UPDATE SET
                  last_seen_sequence=MAX(last_seen_sequence,excluded.last_seen_sequence),
                  updated_at=excluded.updated_at
                """,
                (owner, project, applied, _now()),
            )
        return applied

    def project_unread_counts(self, *, owner_user_id: str) -> dict[str, int]:
        owner = str(owner_user_id or "").strip()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.project_id, COUNT(*) AS unread
                FROM v2_conversation_events e
                LEFT JOIN v2_desktop_project_cursors c
                  ON c.owner_user_id=e.owner_user_id AND c.project_id=e.project_id
                WHERE e.owner_user_id=? AND e.sequence>COALESCE(c.last_seen_sequence,0)
                GROUP BY e.project_id
                """,
                (owner,),
            ).fetchall()
        return {str(row["project_id"] or ""): int(row["unread"] or 0) for row in rows}

    def legacy_import_completed(self, *, owner_user_id: str, project_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM v2_conversation_imports
                WHERE owner_user_id=? AND project_id=? AND source_kind='markdown_ledger'
                """,
                (str(owner_user_id or "").strip(), str(project_id or "").strip()),
            ).fetchone()
        return row is not None

    def import_legacy_events(
        self,
        *,
        owner_user_id: str,
        project_id: str,
        events: list[dict[str, Any]],
    ) -> int:
        owner = str(owner_user_id or "").strip()
        project = str(project_id or "").strip()
        imported = 0
        with self._connect() as conn:
            already = conn.execute(
                """
                SELECT 1 FROM v2_conversation_imports
                WHERE owner_user_id=? AND project_id=? AND source_kind='markdown_ledger'
                """,
                (owner, project),
            ).fetchone()
            if already is not None:
                return 0
            for index, event in enumerate(events):
                role = str(event.get("role") or "").strip().lower()
                content = str(event.get("content") or "").strip()
                if role not in {"user", "assistant"} or not content:
                    continue
                source_id = str(event.get("id") or f"legacy:{owner}:{project}:{index}").strip()
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO v2_conversation_events(
                      event_id,owner_user_id,project_id,role,content,source,source_message_id,created_at
                    ) VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        str(uuid4()),
                        owner,
                        project,
                        role,
                        content,
                        str(event.get("source") or "ledger").strip() or "ledger",
                        source_id,
                        _canonical_timestamp(str(event.get("created_at") or "").strip()) or _now(),
                    ),
                )
                imported += int(cursor.rowcount)
            conn.execute(
                """
                INSERT INTO v2_conversation_imports(
                  owner_user_id,project_id,source_kind,events_imported,imported_at
                ) VALUES (?,?,?,?,?)
                """,
                (owner, project, "markdown_ledger", imported, _now()),
            )
        return imported

    def _connect(self):
        if self._memory is not None:
            return _Connection(self._memory)
        path = Path(self.db_path); path.parent.mkdir(parents=True, exist_ok=True)
        return connect_database(path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            existed = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_conversation_events'").fetchone() is not None
            cursor_existed = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_desktop_project_cursors'").fetchone() is not None
            if existed:
                columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(v2_conversation_events)").fetchall()}
                if "sequence" not in columns:
                    rows = conn.execute("SELECT * FROM v2_conversation_events ORDER BY created_at,rowid").fetchall()
                    conn.execute("ALTER TABLE v2_conversation_events RENAME TO v2_conversation_events_legacy")
                    self._create_events_table(conn)
                    for row in rows:
                        conn.execute(
                            "INSERT INTO v2_conversation_events(event_id,owner_user_id,project_id,role,content,source,source_message_id,created_at) VALUES (?,?,?,?,?,?,?,?)",
                            (row["event_id"], row["owner_user_id"], row["project_id"], row["role"], row["content"], row["source"], row["source_message_id"], row["created_at"]),
                        )
                    conn.execute("DROP TABLE v2_conversation_events_legacy")
            else:
                self._create_events_table(conn)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_v2_conversation_events_scope ON v2_conversation_events(owner_user_id, project_id, sequence)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS v2_desktop_project_cursors (
                  owner_user_id TEXT NOT NULL,
                  project_id TEXT NOT NULL DEFAULT '',
                  last_seen_sequence INTEGER NOT NULL DEFAULT 0,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY(owner_user_id,project_id)
                ) STRICT
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS v2_conversation_imports (
                  owner_user_id TEXT NOT NULL,
                  project_id TEXT NOT NULL DEFAULT '',
                  source_kind TEXT NOT NULL,
                  events_imported INTEGER NOT NULL DEFAULT 0,
                  imported_at TEXT NOT NULL,
                  PRIMARY KEY(owner_user_id,project_id,source_kind)
                ) STRICT
                """
            )
            if existed and not cursor_existed:
                conn.execute(
                    """
                    INSERT INTO v2_desktop_project_cursors(owner_user_id,project_id,last_seen_sequence,updated_at)
                    SELECT owner_user_id,project_id,MAX(sequence),? FROM v2_conversation_events
                    GROUP BY owner_user_id,project_id
                    """,
                    (_now(),),
                )
            self._normalize_existing_timestamps(conn)

    @staticmethod
    def _create_events_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE v2_conversation_events (
              sequence INTEGER PRIMARY KEY AUTOINCREMENT,
              event_id TEXT NOT NULL UNIQUE,
              owner_user_id TEXT NOT NULL,
              project_id TEXT NOT NULL DEFAULT '',
              role TEXT NOT NULL CHECK(role IN ('user','assistant')),
              content TEXT NOT NULL,
              source TEXT NOT NULL,
              source_message_id TEXT NOT NULL DEFAULT '' UNIQUE,
              created_at TEXT NOT NULL
            ) STRICT
            """
        )

    @staticmethod
    def _normalize_existing_timestamps(conn: sqlite3.Connection) -> None:
        """Canonicalize legacy ISO instants without touching malformed values."""
        rows = conn.execute("SELECT event_id, created_at FROM v2_conversation_events").fetchall()
        for row in rows:
            original = str(row["created_at"] or "")
            normalized = _canonical_timestamp(original)
            if normalized != original:
                conn.execute(
                    "UPDATE v2_conversation_events SET created_at=? WHERE event_id=?",
                    (normalized, str(row["event_id"])),
                )


def legacy_ledger_events(content: str, *, owner_user_id: str, project_id: str, limit: int = 100) -> list[dict[str, str]]:
    """Recover visible turns from pre-timeline ledgers without surfacing tool internals."""
    events: list[dict[str, str]] = []
    for index, block in enumerate(re.split(r"(?=^### Task )", str(content or ""), flags=re.MULTILINE)):
        user = re.search(r"^#### Conversation\n- User: (.*?)(?=\n#### |\Z)", block, flags=re.MULTILINE | re.DOTALL)
        if user and user.group(1).strip():
            events.append({"id": f"legacy:{owner_user_id}:{project_id}:{index}:user", "role": "user", "content": user.group(1).strip(), "source": "ledger", "created_at": ""})
        for assistant_index, match in enumerate(re.finditer(r"^#### Conversation\n- Alphonse: (.*?)(?=\n#### |\Z)", block, flags=re.MULTILINE | re.DOTALL)):
            if match.group(1).strip():
                events.append({"id": f"legacy:{owner_user_id}:{project_id}:{index}:assistant:{assistant_index}", "role": "assistant", "content": match.group(1).strip(), "source": "telegram", "created_at": ""})
    return events[-max(1, min(int(limit or 100), 500)):]


class _Connection:
    def __init__(self, connection: sqlite3.Connection) -> None: self.connection = connection
    def __enter__(self): return self.connection
    def __exit__(self, typ, value, traceback): self.connection.commit() if typ is None else self.connection.rollback()


def _event(row: Any) -> ConversationEvent: return ConversationEvent(**dict(row))
def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _canonical_timestamp(value: str) -> str:
    """Return an aware ISO timestamp in UTC, preserving non-ISO legacy data."""
    text = str(value or "").strip()
    if not text:
        return text
    try:
        parsed = datetime.fromisoformat(f"{text[:-1]}+00:00" if text.endswith("Z") else text)
    except ValueError:
        return text
    if parsed.tzinfo is None:
        return text
    return parsed.astimezone(timezone.utc).isoformat()
