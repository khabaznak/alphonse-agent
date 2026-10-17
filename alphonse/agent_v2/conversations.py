"""Durable, channel-neutral conversation timelines for v2 projects."""

from __future__ import annotations

import os
import json
import hashlib
import re
import sqlite3
from contextlib import nullcontext
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
    memory_session_id: str
    role: str
    content: str
    source: str
    source_message_id: str
    created_at: str

    schema_version: int = 1
    recorded_at: str = ""
    observed_at: str = ""
    speaker_id: str = ""
    visibility: str = "private"
    shared_with: tuple[str, ...] = ()
    provenance: str = "recorded"

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class SQLiteConversationStore:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory = sqlite3.connect(":memory:", check_same_thread=False) if self.db_path == ":memory:" else None
        self._daily_ledger_projector: Any | None = None
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
        project_id: str = "",
        memory_session_id: str = "",
        role: str,
        content: str,
        source: str,
        source_message_id: str,
        created_at: str = "",
        connection: sqlite3.Connection | None = None,
        visibility: str = "private",
        shared_with: tuple[str, ...] = (),
    ) -> ConversationEvent | None:
        owner, message = str(owner_user_id or "").strip(), str(content or "").strip()
        normalized_role = str(role or "").strip().lower()
        if not owner or not message or normalized_role not in {"user", "assistant"}:
            return None
        from alphonse.agent_v2.core.memory.contracts import AccessScope
        AccessScope(owner, visibility, shared_with)
        event_id = str(uuid4())
        project = str(project_id or "").strip()
        session = str(memory_session_id or "").strip()
        source_value = str(source or "unknown").strip() or "unknown"
        source_id = str(source_message_id or "").strip()
        timestamp = _canonical_timestamp(created_at or _now())
        if connection is not None:
            event = self._record(
                connection,
                event_id=event_id,
                owner=owner,
                project=project,
                session=session,
                role=normalized_role,
                message=message,
                source=source_value,
                source_id=source_id,
                timestamp=timestamp, visibility=visibility, shared_with=shared_with,
            )
            return event
        with self._connect() as conn:
            event = self._record(
                conn,
                event_id=event_id,
                owner=owner,
                project=project,
                session=session,
                role=normalized_role,
                message=message,
                source=source_value,
                source_id=source_id,
                timestamp=timestamp, visibility=visibility, shared_with=shared_with,
            )
        self._project_daily_ledger(event)
        return event

    def set_daily_ledger_projector(self, projector: Any | None) -> None:
        self._daily_ledger_projector = projector

    def _project_daily_ledger(self, event: ConversationEvent | None) -> None:
        if event is None or self._daily_ledger_projector is None:
            return
        try:
            self._daily_ledger_projector.refresh_for_event(event)
        except Exception:
            # The event store is authoritative; a failed generated view can be
            # rebuilt on the next conversation write without blocking chat.
            import logging
            logging.getLogger(__name__).exception("daily conversation ledger projection failed")

    @staticmethod
    def _record(
        conn: sqlite3.Connection,
        *,
        event_id: str,
        owner: str,
        project: str,
        session: str,
        role: str,
        message: str,
        source: str,
        source_id: str,
        timestamp: str,
        visibility: str = "private",
        shared_with: tuple[str, ...] = (),
    ) -> ConversationEvent | None:
        if source_id:
            digest = hashlib.sha256(source_id.encode()).hexdigest()
            if conn.execute("SELECT 1 FROM v2_conversation_tombstones WHERE source_digest=?", (digest,)).fetchone():
                return None
            exists = conn.execute(
                "SELECT * FROM v2_conversation_events WHERE source_message_id=?",
                (source_id,),
            ).fetchone()
            if exists is not None:
                return None
        recorded_at = _now()
        speaker = owner if role == "user" else "alphonse"
        conn.execute(
            """INSERT INTO v2_conversation_events(event_id,owner_user_id,project_id,memory_session_id,
               role,content,source,source_message_id,created_at,recorded_at,observed_at,speaker_id,
               visibility,shared_with,provenance,schema_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
            (event_id, owner, project, session, role, message, source, source_id, timestamp,
             recorded_at, _known_instant(timestamp), speaker, visibility, json.dumps(list(shared_with)), "recorded"),
        )
        return _event(conn.execute("SELECT * FROM v2_conversation_events WHERE event_id=?", (event_id,)).fetchone())

    def list(self, *, owner_user_id: str, project_id: str = "", memory_session_id: str | None = None, limit: int = 100) -> list[ConversationEvent]:
        with self._connect() as conn:
            if memory_session_id is None:
                rows = conn.execute("SELECT * FROM v2_conversation_events WHERE owner_user_id=? AND project_id=? ORDER BY sequence DESC LIMIT ?", (str(owner_user_id or "").strip(), str(project_id or "").strip(), max(1, min(int(limit or 100), 500)))).fetchall()
            else:
                rows = conn.execute("SELECT * FROM v2_conversation_events WHERE owner_user_id=? AND project_id=? AND memory_session_id=? ORDER BY sequence DESC LIMIT ?", (str(owner_user_id or "").strip(), str(project_id or "").strip(), str(memory_session_id or "").strip(), max(1, min(int(limit or 100), 500)))).fetchall()
        return [_event(row) for row in reversed(rows)]

    def list_recent(self, *, owner_user_id: str, limit: int = 30) -> list[ConversationEvent]:
        """Return the account's recent conversation flow across project boundaries."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM v2_conversation_events WHERE owner_user_id=? ORDER BY sequence DESC LIMIT ?",
                (str(owner_user_id or "").strip(), max(1, min(int(limit or 30), 200))),
            ).fetchall()
        return [_event(row) for row in reversed(rows)]

    def list_between(self, *, owner_user_id: str, start_utc: str, end_utc: str) -> list[ConversationEvent]:
        """List all events in a UTC half-open interval, ordered by conversation sequence."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM v2_conversation_events WHERE owner_user_id=? AND observed_at>=? AND observed_at<? ORDER BY sequence",
                (str(owner_user_id or "").strip(), str(start_utc), str(end_utc)),
            ).fetchall()
        return [_event(row) for row in rows]

    def list_all(self, *, owner_user_id: str) -> list[ConversationEvent]:
        """Unbounded owner export for rebuild/retention, never a prompt API."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM v2_conversation_events WHERE owner_user_id=? ORDER BY sequence",
                                (owner_user_id,)).fetchall()
        return [_event(row) for row in rows]

    def list_owner(self, *, owner_user_id: str, limit: int = 100) -> list[ConversationEvent]:
        """Return the user's conversation across all project associations."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM v2_conversation_events WHERE owner_user_id=? ORDER BY sequence DESC LIMIT ?",
                (str(owner_user_id or "").strip(), max(1, min(int(limit or 100), 500))),
            ).fetchall()
        return [_event(row) for row in reversed(rows)]

    def mark_user_seen(self, *, owner_user_id: str, local_date: str = "", through_sequence: int | None = None, start_utc: str = "", end_utc: str = "") -> int:
        owner = str(owner_user_id or "").strip()
        day = str(local_date or "").strip()
        with self._connect() as conn:
            if through_sequence is None:
                query = "SELECT COALESCE(MAX(sequence),0) AS latest FROM v2_conversation_events WHERE owner_user_id=?"
                values: tuple[Any, ...] = (owner,)
                if start_utc and end_utc:
                    query += " AND observed_at>=? AND observed_at<?"
                    values += (start_utc, end_utc)
                row = conn.execute(query, values).fetchone()
                through_sequence = int(row["latest"] or 0)
            applied = max(0, int(through_sequence or 0))
            conn.execute(
                """INSERT INTO v2_desktop_user_cursors(owner_user_id,local_date,last_seen_sequence,updated_at) VALUES (?,?,?,?)
                   ON CONFLICT(owner_user_id,local_date) DO UPDATE SET last_seen_sequence=MAX(last_seen_sequence,excluded.last_seen_sequence),updated_at=excluded.updated_at""",
                (owner, day, applied, _now()),
            )
        return applied

    def user_has_unseen_messages(self, *, owner_user_id: str, local_date: str = "", start_utc: str = "", end_utc: str = "") -> bool:
        with self._connect() as conn:
            interval = " AND e.observed_at>=? AND e.observed_at<?" if start_utc and end_utc else ""
            values: tuple[Any, ...] = (str(owner_user_id or "").strip(), str(local_date or "").strip())
            if interval:
                values += (start_utc, end_utc)
            row = conn.execute(
                f"""SELECT 1 FROM v2_conversation_events e LEFT JOIN v2_desktop_user_cursors c ON c.owner_user_id=e.owner_user_id AND c.local_date=?
                   WHERE e.owner_user_id=? AND e.sequence>COALESCE(c.last_seen_sequence,0){interval} LIMIT 1""",
                (values[1], values[0], *values[2:]),
            ).fetchone()
        return row is not None

    def search(self, *, owner_user_id: str, query: str, limit: int = 8) -> list[ConversationEvent]:
        """Search an account's full event history using FTS5 BM25 when available."""
        owner = str(owner_user_id or "").strip()
        terms = [term for term in re.findall(r"[\w'-]+", str(query or "")) if len(term) > 1][:12]
        if not owner or not terms:
            return []
        with self._connect() as conn:
            try:
                match = " AND ".join('"' + term.replace('"', '""') + '"' for term in terms)
                rows = conn.execute(
                    """SELECT e.* FROM v2_conversation_fts f JOIN v2_conversation_events e ON e.event_id=f.event_id
                       WHERE f.owner_user_id=? AND v2_conversation_fts MATCH ? ORDER BY bm25(v2_conversation_fts) LIMIT ?""",
                    (owner, match, max(1, min(int(limit or 8), 30))),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = conn.execute(
                    "SELECT * FROM v2_conversation_events WHERE owner_user_id=? AND lower(content) LIKE ? ORDER BY sequence DESC LIMIT ?",
                    (owner, f"%{terms[0].lower()}%", max(1, min(int(limit or 8), 30))),
                ).fetchall()
        return [_event(row) for row in rows]

    def read(self, *, event_id: str, requester_user_id: str) -> ConversationEvent | None:
        """Resolve a historical ID using current access, never historical permission."""
        from alphonse.agent_v2.core.memory.contracts import conversation_item
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM v2_conversation_events WHERE event_id=?", (event_id,)).fetchone()
        event = _event(row) if row is not None else None
        return event if event is not None and conversation_item(event, requester_user_id) is not None else None

    def delete(self, *, event_id: str, owner_user_id: str) -> bool:
        """Owner-scoped erasure; references retain identity but no longer resolve."""
        guard = self._daily_ledger_projector.mutation_guard() if self._daily_ledger_projector is not None else nullcontext()
        with guard:
            return self._delete(event_id=event_id, owner_user_id=owner_user_id)

    def _delete(self, *, event_id: str, owner_user_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM v2_conversation_events WHERE event_id=? AND owner_user_id=?",
                               (event_id, owner_user_id)).fetchone()
            if row is None:
                return False
            if self._daily_ledger_projector is not None:
                # Remove cached content before committing erasure. A failed
                # rebuild then leaves no generated copy of the deleted text.
                self._daily_ledger_projector.invalidate_owner(owner_user_id)
            conn.execute("INSERT OR IGNORE INTO v2_conversation_tombstones VALUES (?,?,?,?)",
                         (event_id, owner_user_id, _now(), hashlib.sha256(row["source_message_id"].encode()).hexdigest()))
            conn.execute("DELETE FROM v2_conversation_events WHERE event_id=?", (event_id,))
            try:
                conn.execute("DELETE FROM v2_conversation_fts WHERE event_id=?", (event_id,))
            except sqlite3.OperationalError:
                pass
        if self._daily_ledger_projector is not None:
            self._daily_ledger_projector.rebuild_owner(owner_user_id)
        return True

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

    def project_has_unseen_messages(self, *, owner_user_id: str, project_id: str) -> bool:
        owner, project = str(owner_user_id or "").strip(), str(project_id or "").strip()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM v2_conversation_events e
                LEFT JOIN v2_desktop_project_cursors c
                  ON c.owner_user_id=e.owner_user_id AND c.project_id=e.project_id
                WHERE e.owner_user_id=? AND e.project_id=? AND e.sequence>COALESCE(c.last_seen_sequence,0)
                LIMIT 1
                """,
                (owner, project),
            ).fetchone()
        return row is not None

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
                if conn.execute("SELECT 1 FROM v2_conversation_tombstones WHERE source_digest=?",
                                (hashlib.sha256(source_id.encode()).hexdigest(),)).fetchone():
                    continue
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
                        _canonical_timestamp(str(event.get("created_at") or "").strip()),
                    ),
                )
                if cursor.rowcount:
                    conn.execute("""UPDATE v2_conversation_events SET recorded_at=?, observed_at=?,
                                 provenance='legacy_import', speaker_id='', schema_version=1 WHERE source_message_id=?""",
                                 (_now(), _known_instant(str(event.get("created_at") or "")), source_id))
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
                            "INSERT INTO v2_conversation_events(event_id,owner_user_id,project_id,memory_session_id,role,content,source,source_message_id,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                            (row["event_id"], row["owner_user_id"], row["project_id"], "", row["role"], row["content"], row["source"], row["source_message_id"], row["created_at"]),
                        )
                    conn.execute("DROP TABLE v2_conversation_events_legacy")
                    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(v2_conversation_events)").fetchall()}
                if "memory_session_id" not in columns:
                    conn.execute("ALTER TABLE v2_conversation_events ADD COLUMN memory_session_id TEXT NOT NULL DEFAULT ''")
            else:
                self._create_events_table(conn)
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(v2_conversation_events)")}
            additions = {
                "schema_version": "INTEGER NOT NULL DEFAULT 1",
                "recorded_at": "TEXT NOT NULL DEFAULT ''",
                "observed_at": "TEXT NOT NULL DEFAULT ''",
                "speaker_id": "TEXT NOT NULL DEFAULT ''",
                "visibility": "TEXT NOT NULL DEFAULT 'private'",
                "shared_with": "TEXT NOT NULL DEFAULT '[]'",
                "provenance": "TEXT NOT NULL DEFAULT 'legacy_unverified'",
            }
            upgrading = "provenance" not in columns
            for name, definition in additions.items():
                if name not in columns:
                    conn.execute(f"ALTER TABLE v2_conversation_events ADD COLUMN {name} {definition}")
            if upgrading:
                conn.execute("""UPDATE v2_conversation_events SET observed_at=created_at,
                    provenance=CASE WHEN source='ledger' OR source_message_id LIKE 'legacy:%'
                    THEN 'legacy_import' ELSE 'legacy_unverified' END""")
                # Existing import dates may be fabricated by the old importer. Keep
                # created_at for recovery, but never promote it to observed time.
                conn.execute("UPDATE v2_conversation_events SET observed_at='' WHERE provenance='legacy_import'")
            conn.execute("""CREATE TABLE IF NOT EXISTS v2_conversation_tombstones (
                event_id TEXT PRIMARY KEY, owner_user_id TEXT NOT NULL, deleted_at TEXT NOT NULL, source_digest TEXT NOT NULL
            ) STRICT""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_v2_conversation_events_scope ON v2_conversation_events(owner_user_id, project_id, sequence)")
            try:
                conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS v2_conversation_fts USING fts5(event_id UNINDEXED, owner_user_id UNINDEXED, content, tokenize='unicode61')")
                conn.execute("CREATE TRIGGER IF NOT EXISTS v2_conversation_fts_insert AFTER INSERT ON v2_conversation_events BEGIN INSERT INTO v2_conversation_fts(event_id,owner_user_id,content) VALUES (new.event_id,new.owner_user_id,new.content); END")
                conn.execute("INSERT INTO v2_conversation_fts(event_id,owner_user_id,content) SELECT e.event_id,e.owner_user_id,e.content FROM v2_conversation_events e LEFT JOIN v2_conversation_fts f ON f.event_id=e.event_id WHERE f.event_id IS NULL")
            except sqlite3.OperationalError:
                pass
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
            conn.execute("CREATE TABLE IF NOT EXISTS v2_desktop_user_cursors (owner_user_id TEXT NOT NULL,local_date TEXT NOT NULL DEFAULT '',last_seen_sequence INTEGER NOT NULL DEFAULT 0,updated_at TEXT NOT NULL,PRIMARY KEY(owner_user_id,local_date)) STRICT")
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
            for row in conn.execute("SELECT event_id,observed_at FROM v2_conversation_events WHERE observed_at<>''").fetchall():
                observed = _known_instant(row["observed_at"])
                if observed != row["observed_at"]:
                    conn.execute("UPDATE v2_conversation_events SET observed_at=? WHERE event_id=?", (observed, row["event_id"]))

    @staticmethod
    def _create_events_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE v2_conversation_events (
              sequence INTEGER PRIMARY KEY AUTOINCREMENT,
              event_id TEXT NOT NULL UNIQUE,
              owner_user_id TEXT NOT NULL,
              project_id TEXT NOT NULL DEFAULT '',
              memory_session_id TEXT NOT NULL DEFAULT '',
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
                events.append({"id": f"legacy:{owner_user_id}:{project_id}:{index}:assistant:{assistant_index}", "role": "assistant", "content": match.group(1).strip(), "source": "ledger", "created_at": ""})
    return events[-max(1, min(int(limit or 100), 10_000)):]


class _Connection:
    def __init__(self, connection: sqlite3.Connection) -> None: self.connection = connection
    def __enter__(self): return self.connection
    def __exit__(self, typ, value, traceback): self.connection.commit() if typ is None else self.connection.rollback()


def _event(row: Any) -> ConversationEvent:
    values = dict(row)
    recipients = json.loads(values.get("shared_with") or "[]")
    if not isinstance(recipients, list):
        raise ValueError("invalid_conversation_recipients")
    values["shared_with"] = tuple(recipients)
    return ConversationEvent(**values)
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


def _known_instant(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc).isoformat() if parsed.tzinfo is not None else ""
    except ValueError:
        return ""
