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


@dataclass(frozen=True)
class ConversationEvent:
    event_id: str
    owner_user_id: str
    project_id: str
    role: str
    content: str
    source: str
    source_message_id: str
    created_at: str

    def to_dict(self) -> dict[str, str]:
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
        return cls(os.getenv("ALPHONSE_V2_CONVERSATIONS_DB_PATH") or os.getenv("ALPHONSE_V2_DB_PATH") or str(Path.home() / ".alphonse" / "v2-conversations.sqlite3"))

    def record(self, *, owner_user_id: str, project_id: str, role: str, content: str, source: str, source_message_id: str, created_at: str = "") -> ConversationEvent | None:
        owner, message = str(owner_user_id or "").strip(), str(content or "").strip()
        normalized_role = str(role or "").strip().lower()
        if not owner or not message or normalized_role not in {"user", "assistant"}:
            return None
        event = ConversationEvent(str(uuid4()), owner, str(project_id or "").strip(), normalized_role, message, str(source or "unknown").strip() or "unknown", str(source_message_id or "").strip(), _canonical_timestamp(created_at or _now()))
        with self._connect() as conn:
            if event.source_message_id:
                exists = conn.execute("SELECT 1 FROM v2_conversation_events WHERE source_message_id=?", (event.source_message_id,)).fetchone()
                if exists is not None:
                    return None
            conn.execute("INSERT INTO v2_conversation_events(event_id,owner_user_id,project_id,role,content,source,source_message_id,created_at) VALUES (?,?,?,?,?,?,?,?)", tuple(event.__dict__.values()))
        return event

    def list(self, *, owner_user_id: str, project_id: str = "", limit: int = 100) -> list[ConversationEvent]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM v2_conversation_events WHERE owner_user_id=? AND project_id=? ORDER BY created_at DESC, event_id DESC LIMIT ?", (str(owner_user_id or "").strip(), str(project_id or "").strip(), max(1, min(int(limit or 100), 500)))).fetchall()
        return [_event(row) for row in reversed(rows)]

    def _connect(self):
        if self._memory is not None:
            return _Connection(self._memory)
        path = Path(self.db_path); path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path); conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS v2_conversation_events (event_id TEXT PRIMARY KEY, owner_user_id TEXT NOT NULL, project_id TEXT NOT NULL DEFAULT '', role TEXT NOT NULL CHECK(role IN ('user','assistant')), content TEXT NOT NULL, source TEXT NOT NULL, source_message_id TEXT NOT NULL DEFAULT '' UNIQUE, created_at TEXT NOT NULL) STRICT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_v2_conversation_events_scope ON v2_conversation_events(owner_user_id, project_id, created_at)")
            self._normalize_existing_timestamps(conn)

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
            events.append({"id": f"legacy:{project_id}:{index}:user", "role": "user", "content": user.group(1).strip(), "source": "ledger", "created_at": ""})
        for assistant_index, match in enumerate(re.finditer(r"^#### Conversation\n- Alphonse: (.*?)(?=\n#### |\Z)", block, flags=re.MULTILINE | re.DOTALL)):
            if match.group(1).strip():
                events.append({"id": f"legacy:{project_id}:{index}:assistant:{assistant_index}", "role": "assistant", "content": match.group(1).strip(), "source": "telegram", "created_at": ""})
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
