"""Deterministic project-session routing for v2 communication channels."""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from alphonse.agent_v2.core.io import ChannelAddress
from alphonse.agent_v2.core.io import SQLiteOutboundStore
from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.core.messages.queue import QueuedMessage
from alphonse.agent_v2.core.projects import ProjectRecord
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.database import connect_database, default_database_path


@dataclass(frozen=True)
class ProjectSessionKey:
    alphonse_user_id: str
    integration_id: str
    channel_target: str
    thread_id: str = ""


@dataclass(frozen=True)
class ProjectSession:
    key: ProjectSessionKey
    active_project_id: str
    project_name: str
    updated_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "alphonse_user_id": self.key.alphonse_user_id,
            "integration_id": self.key.integration_id,
            "channel_target": self.key.channel_target,
            "thread_id": self.key.thread_id,
            "active_project_id": self.active_project_id,
            "project_name": self.project_name,
            "updated_at": self.updated_at,
        }


class SQLiteProjectSessionStore:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory_connection: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._memory_connection = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_connection.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls) -> "SQLiteProjectSessionStore":
        return cls(default_database_path())

    def get(self, key: ProjectSessionKey) -> ProjectSession | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM v2_project_sessions
                WHERE alphonse_user_id = ? AND integration_id = ? AND channel_target = ? AND thread_id = ?
                """,
                _key_values(key),
            ).fetchone()
        return _session_from_row(row)

    def set(self, key: ProjectSessionKey, project: ProjectRecord) -> ProjectSession:
        normalized = _normalize_key(key)
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO v2_project_sessions (
                    alphonse_user_id, integration_id, channel_target, thread_id,
                    active_project_id, project_name, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (*_key_values(normalized), project.project_id, project.name, now),
            )
        return ProjectSession(normalized, project.project_id, project.name, now)

    def clear(self, key: ProjectSessionKey) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM v2_project_sessions
                WHERE alphonse_user_id = ? AND integration_id = ? AND channel_target = ? AND thread_id = ?
                """,
                _key_values(_normalize_key(key)),
            )
        return cursor.rowcount > 0

    def migrate_user(self, old_user_id: str, new_user_id: str) -> int:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM v2_project_sessions WHERE alphonse_user_id=?", (str(old_user_id),)).fetchall()
            for row in rows:
                conn.execute("DELETE FROM v2_project_sessions WHERE alphonse_user_id=? AND integration_id=? AND channel_target=? AND thread_id=?", (str(old_user_id), row["integration_id"], row["channel_target"], row["thread_id"]))
                conn.execute("INSERT OR REPLACE INTO v2_project_sessions (alphonse_user_id,integration_id,channel_target,thread_id,active_project_id,project_name,updated_at) VALUES (?,?,?,?,?,?,?)", (str(new_user_id), row["integration_id"], row["channel_target"], row["thread_id"], row["active_project_id"], row["project_name"], _now_iso()))
        return len(rows)

    def delete_user(self, user_id: str) -> int:
        with self._connect() as conn:
            return conn.execute("DELETE FROM v2_project_sessions WHERE alphonse_user_id=?", (str(user_id),)).rowcount

    def clear_project(self, project_id: str) -> int:
        with self._connect() as conn:
            return conn.execute("DELETE FROM v2_project_sessions WHERE active_project_id=?", (str(project_id),)).rowcount

    def _connect(self) -> sqlite3.Connection | "_ConnectionProxy":
        if self._memory_connection is not None:
            return _ConnectionProxy(self._memory_connection)
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return connect_database(path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS v2_project_sessions (
                    alphonse_user_id TEXT NOT NULL,
                    integration_id TEXT NOT NULL,
                    channel_target TEXT NOT NULL,
                    thread_id TEXT NOT NULL DEFAULT '',
                    active_project_id TEXT NOT NULL,
                    project_name TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (alphonse_user_id, integration_id, channel_target, thread_id)
                ) STRICT;
                """
            )


@dataclass(frozen=True)
class InboundRouteResult:
    queued: QueuedMessage | None = None
    handled_command: bool = False
    project_id: str = ""


class ProjectInboundRouter:
    """Resolves project sessions and handles universal project commands."""

    def __init__(
        self,
        *,
        channel: CommunicationChannel,
        outbox: SQLiteOutboundStore,
        projects: ProjectStore,
        sessions: SQLiteProjectSessionStore,
        is_admin: Any | None = None,
        managed_root: Any | None = None,
        communication_router: Any | None = None,
    ) -> None:
        self.channel = channel
        self.outbox = outbox
        self.projects = projects
        self.sessions = sessions
        self.is_admin = is_admin or (lambda _user: False)
        self.managed_root = managed_root
        self.communication_router = communication_router

    def ingest(
        self,
        *,
        prompt: str,
        user: str,
        integration_id: str,
        provider_key: str,
        channel_target: str,
        provider_user_id: str = "",
        provider_message_id: str = "",
        reply_to_provider_message_id: str = "",
        thread_id: str = "",
        project_id: str = "",
        tag: str = "",
        correlation_id: str = "",
        metadata: dict[str, Any] | None = None,
        message_id: str | None = None,
    ) -> InboundRouteResult:
        address = ChannelAddress(
            integration_id=str(integration_id or "tui").strip() or "tui",
            provider_key=str(provider_key or "tui").strip().lower() or "tui",
            channel_target=str(channel_target or user).strip() or str(user).strip(),
            alphonse_user_id=str(user).strip(),
            provider_user_id=str(provider_user_id or user).strip(),
            provider_message_id=str(provider_message_id).strip(),
            reply_to_provider_message_id=str(reply_to_provider_message_id).strip(),
            thread_id=str(thread_id).strip(),
        )
        key = ProjectSessionKey(address.alphonse_user_id, address.integration_id, address.channel_target, address.thread_id)
        if self.communication_router is not None and self.communication_router.relay_inbound(
            sender_user_id=address.alphonse_user_id,
            address=address,
            text=prompt,
        ):
            return InboundRouteResult(handled_command=True)
        explicit_project = str(project_id or "").strip()
        if not explicit_project:
            command_reply = self._handle_command(prompt, key, address)
            if command_reply is not None:
                self._reply(address, command_reply, correlation_id=correlation_id)
                return InboundRouteResult(handled_command=True)
            active = self.active_project(key)
            explicit_project = active.project_id if active is not None else ""
        queued = self.channel.queue_message(
            prompt=prompt,
            user=address.alphonse_user_id,
            project_id=explicit_project,
            tag=tag,
            correlation_id=correlation_id,
            metadata=metadata,
            integration_id=address.integration_id,
            provider_key=address.provider_key,
            provider_user_id=address.provider_user_id,
            channel_target=address.channel_target,
            provider_message_id=address.provider_message_id,
            reply_to_provider_message_id=address.reply_to_provider_message_id,
            thread_id=address.thread_id,
            message_id=message_id,
        )
        return InboundRouteResult(queued=queued, project_id=explicit_project)

    def active_project(self, key: ProjectSessionKey) -> ProjectRecord | None:
        session = self.sessions.get(key)
        if session is None:
            return None
        project = self.projects.get_project(session.active_project_id, requester_user_id=key.alphonse_user_id, requester_is_admin=bool(self.is_admin(key.alphonse_user_id)))
        if project is None:
            self.sessions.clear(key)
        return project

    def select_project(self, key: ProjectSessionKey, project_id_or_name: str) -> ProjectRecord:
        project = self._resolve_visible_project(key.alphonse_user_id, project_id_or_name)
        self.sessions.set(key, project)
        return project

    def _handle_command(self, prompt: str, key: ProjectSessionKey, address: ChannelAddress) -> str | None:
        raw = str(prompt or "")
        if not raw.startswith("/"):
            return None
        text = raw.strip()
        command, _, arguments = text.partition(" ")
        command = command.lower()
        arguments = arguments.strip()
        if command == "/projects":
            return self._list_projects(key)
        if command != "/project":
            return None
        if not arguments:
            return self._project_status(key)
        if arguments.casefold() == "none":
            self.sessions.clear(key)
            return "Active project cleared for this conversation."
        if arguments.casefold() == "create" or arguments.casefold().startswith("create "):
            name = arguments[6:].strip()
            if not name:
                return "Usage: /project create <name>"
            project = self._create_managed_project(key.alphonse_user_id, name)
            self.sessions.set(key, project)
            return f"Created and activated project: {project.name}."
        if arguments.casefold() == "context":
            project = self.active_project(key)
            if project is None:
                return "No active project. Select one with /project <name>."
            content = self.projects.read_project_context(project.project_id, requester_user_id=key.alphonse_user_id).strip()
            return content or f"Project context for {project.name} is empty."
        if arguments.casefold().startswith("context set "):
            project = self.active_project(key)
            if project is None:
                return "No active project. Select one with /project <name>."
            if project.owner_user_id != key.alphonse_user_id:
                return "Only the project owner can update this project context."
            content = arguments[len("context set ") :].strip()
            if not content:
                return "Usage: /project context set <markdown>"
            self.projects.write_project_context(project.project_id, content, requester_user_id=key.alphonse_user_id)
            return f"Updated context for {project.name}."
        try:
            project = self.select_project(key, arguments)
        except LookupError as exc:
            return str(exc)
        return f"Active project: {project.name}."

    def _list_projects(self, key: ProjectSessionKey) -> str:
        projects = self.projects.list_visible_projects(key.alphonse_user_id, requester_is_admin=bool(self.is_admin(key.alphonse_user_id)))
        active = self.active_project(key)
        if not projects:
            return "No visible projects. Create one with /project create <name>."
        lines = ["Projects:"]
        for project in projects:
            prefix = "* " if active is not None and project.project_id == active.project_id else "- "
            lines.append(f"{prefix}{project.name} ({project.project_id})")
        return "\n".join(lines)

    def _project_status(self, key: ProjectSessionKey) -> str:
        active = self.active_project(key)
        prefix = f"Active project: {active.name}." if active is not None else "No active project."
        return f"{prefix}\n{self._list_projects(key)}"

    def _resolve_visible_project(self, user: str, value: str) -> ProjectRecord:
        rendered = str(value or "").strip()
        visible = self.projects.list_visible_projects(user, requester_is_admin=bool(self.is_admin(user)))
        for project in visible:
            if project.project_id == rendered:
                return project
        matches = [project for project in visible if project.name.casefold() == rendered.casefold()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            choices = ", ".join(f"{project.name} ({project.project_id})" for project in matches)
            raise LookupError(f"Project name is ambiguous. Use an id: {choices}")
        raise LookupError(f"Project not found or not visible: {rendered}. Use /projects.")

    def _create_managed_project(self, user: str, name: str) -> ProjectRecord:
        slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-") or "project"
        project_id = f"{slug}-{uuid4().hex[:8]}"
        root = (Path(self.managed_root(user)) if self.managed_root is not None else managed_projects_root() / _safe_segment(user)) / slug
        return self.projects.create_project(
            name=name,
            root_path=str(root),
            visibility="private",
            owner_user_id=user,
            project_id=project_id,
        )

    def _reply(self, address: ChannelAddress, message: str, *, correlation_id: str) -> None:
        self.outbox.enqueue(
            address=address,
            message=message,
            kind="channel_command",
            correlation_id=correlation_id,
            metadata={"source": "project_command"},
        )


def managed_projects_root() -> Path:
    configured = os.getenv("ALPHONSE_V2_MANAGED_PROJECTS_DIR")
    return Path(configured) if configured else Path.home() / ".alphonse" / "projects"


def _normalize_key(key: ProjectSessionKey) -> ProjectSessionKey:
    user = str(key.alphonse_user_id or "").strip()
    integration = str(key.integration_id or "").strip()
    target = str(key.channel_target or "").strip()
    if not user or not integration or not target:
        raise ValueError("project_session_key_required")
    return ProjectSessionKey(user, integration, target, str(key.thread_id or "").strip())


def _key_values(key: ProjectSessionKey) -> tuple[str, str, str, str]:
    normalized = _normalize_key(key)
    return normalized.alphonse_user_id, normalized.integration_id, normalized.channel_target, normalized.thread_id


def _session_from_row(row: sqlite3.Row | None) -> ProjectSession | None:
    if row is None:
        return None
    return ProjectSession(
        ProjectSessionKey(str(row["alphonse_user_id"]), str(row["integration_id"]), str(row["channel_target"]), str(row["thread_id"])),
        str(row["active_project_id"]),
        str(row["project_name"]),
        str(row["updated_at"]),
    )


def _safe_segment(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", str(value or "").casefold()).strip("-") or "user"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _ConnectionProxy:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def __enter__(self) -> sqlite3.Connection:
        return self.conn

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        return False
