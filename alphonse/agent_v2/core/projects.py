"""Project registry for Alphonse agent v2."""

from __future__ import annotations

import os
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from alphonse.agent_v2.database import connect_database, default_database_path

ProjectVisibility = Literal["private", "shared"]
ProjectStatus = Literal["active", "paused", "completed", "archived"]
PROJECT_CONTEXT_FILENAME = "project_context.md"
PROJECT_CONFIG_FILENAME = "project_config.json"


def default_project_config(*, owner_user_id: str, name: str = "", description: str = "") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": description,
        "goals": [],
        "owner_user_id": str(owner_user_id),
        "duration": {"ongoing": True, "start_date": None, "end_date": None},
        "directory_structure": [],
    }


@dataclass(frozen=True)
class ProjectRecord:
    project_id: str
    name: str
    description: str
    root_path: str
    visibility: ProjectVisibility
    owner_user_id: str
    status: ProjectStatus
    archived_at: str | None
    created_at: str
    updated_at: str
    is_system_home: bool = False

    @property
    def config_path(self) -> str:
        return str(Path(self.root_path) / PROJECT_CONFIG_FILENAME)

    @property
    def context_path(self) -> str:
        return str(Path(self.root_path) / PROJECT_CONTEXT_FILENAME)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "root_path": self.root_path,
            "visibility": self.visibility,
            "owner_user_id": self.owner_user_id,
            "status": self.status,
            "archived_at": self.archived_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "context_path": self.context_path,
            "config_path": self.config_path,
            "is_system_home": self.is_system_home,
        }


class ProjectStore:
    """SQLite-backed v2 project registry."""

    def __init__(self, db_path: str | None = ":memory:") -> None:
        self.db_path = str(db_path or ":memory:")
        self._memory_connection: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._memory_connection = sqlite3.connect(":memory:")
            self._memory_connection.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls) -> "ProjectStore":
        return cls(_default_project_db_path())

    def create_project(
        self,
        *,
        name: str,
        description: str = "",
        root_path: str,
        visibility: ProjectVisibility = "private",
        owner_user_id: str,
        project_id: str | None = None,
        is_system_home: bool = False,
    ) -> ProjectRecord:
        name_value = str(name or "").strip()
        owner_value = str(owner_user_id or "").strip()
        if not name_value:
            raise ValueError("project_name_required")
        if not owner_value:
            raise ValueError("project_owner_required")
        visibility_value = _normalize_visibility(visibility)
        root = _resolve_project_root(root_path)
        root.mkdir(parents=True, exist_ok=True)
        self._ensure_project_config(root, owner_value, description)
        now = _now_iso()
        project_id_value = str(project_id or "").strip() or _project_id_from_name(name_value)
        record = ProjectRecord(
            project_id=project_id_value,
            name=name_value,
            description=str(description or "").strip(),
            root_path=str(root),
            visibility=visibility_value,
            owner_user_id=owner_value,
            status="active",
            archived_at=None,
            created_at=now,
            updated_at=now,
            is_system_home=bool(is_system_home),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_projects (
                  project_id, name, description, root_path, visibility,
                  owner_user_id, status, archived_at, created_at, updated_at, is_system_home
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.project_id,
                    record.name,
                    record.description,
                    record.root_path,
                    record.visibility,
                    record.owner_user_id,
                    record.status,
                    record.archived_at,
                    record.created_at,
                    record.updated_at,
                    int(bool(record.is_system_home)),
                ),
            )
        return record

    def ensure_home_project(self, user_id: str, *, root_path: str) -> ProjectRecord:
        """Return the user's protected Home project, creating it once if needed."""
        owner = str(user_id or "").strip()
        if not owner:
            raise ValueError("project_owner_required")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM v2_projects WHERE owner_user_id=? AND is_system_home=1",
                (owner,),
            ).fetchone()
        existing = _project_from_row(row)
        if existing is not None:
            return existing
        # The uniqueness constraint resolves concurrent startup reconciliation.
        project_id = f"home-{uuid4().hex}"
        try:
            return self.create_project(
                name="Home",
                description="Default project for personal tasks.",
                root_path=root_path,
                visibility="private",
                owner_user_id=owner,
                project_id=project_id,
                is_system_home=True,
            )
        except sqlite3.IntegrityError:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM v2_projects WHERE owner_user_id=? AND is_system_home=1",
                    (owner,),
                ).fetchone()
            resolved = _project_from_row(row)
            if resolved is None:
                raise
            return resolved

    def home_project(self, user_id: str) -> ProjectRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM v2_projects WHERE owner_user_id=? AND is_system_home=1 AND status='active'",
                (str(user_id or "").strip(),),
            ).fetchone()
        return _project_from_row(row)

    def get_project(self, project_id: str, *, requester_user_id: str | None = None, requester_is_admin: bool = False, include_archived: bool = False) -> ProjectRecord | None:
        project_id_value = str(project_id or "").strip()
        if not project_id_value:
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM v2_projects WHERE project_id = ?", (project_id_value,)).fetchone()
        record = _project_from_row(row)
        if record is None:
            return None
        if record.status == "archived" and not include_archived:
            return None
        requester = str(requester_user_id or "").strip()
        if requester and not requester_is_admin and record.owner_user_id != requester and not (record.visibility == "shared" and self.is_member(record.project_id, requester)):
            return None
        return record

    def list_visible_projects(self, user_id: str, *, requester_is_admin: bool = False) -> list[ProjectRecord]:
        user_value = str(user_id or "").strip()
        with self._connect() as conn:
            if requester_is_admin:
                rows = conn.execute("SELECT * FROM v2_projects WHERE status != 'archived' ORDER BY lower(name), created_at").fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM v2_projects
                    WHERE status != 'archived' AND (owner_user_id = ? OR (visibility='shared' AND project_id IN (SELECT project_id FROM v2_project_members WHERE user_id = ?)))
                    ORDER BY lower(name), created_at
                    """,
                    (user_value, user_value),
                ).fetchall()
        return [_project_from_row(row) for row in rows if _project_from_row(row) is not None]

    def list_manageable_projects(self, user_id: str, *, requester_is_admin: bool = False, status: ProjectStatus | None = None) -> list[ProjectRecord]:
        filters = ["owner_user_id = ?"]
        values: list[object] = [str(user_id or "").strip()]
        if requester_is_admin:
            filters = []
            values = []
        if status is not None:
            filters.append("status = ?")
            values.append(_normalize_status(status))
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM v2_projects {where} ORDER BY status, lower(name), created_at", tuple(values)).fetchall()
        return [_project_from_row(row) for row in rows if _project_from_row(row) is not None]

    def find_project_by_root(self, root_path: str) -> ProjectRecord | None:
        root = str(_resolve_project_root(root_path))
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM v2_projects WHERE root_path = ?", (root,)).fetchone()
        return _project_from_row(row)

    def update_project(self, project_id: str, *, name: str, description: str, visibility: ProjectVisibility, requester_user_id: str, requester_is_admin: bool = False, member_user_ids: list[str] | None = None) -> ProjectRecord:
        project = self._require_manageable(project_id, requester_user_id=requester_user_id, requester_is_admin=requester_is_admin)
        if project.is_system_home:
            raise ValueError("system_home_project_protected")
        name_value = str(name or "").strip()
        if not name_value:
            raise ValueError("project_name_required")
        now = _now_iso()
        with self._connect() as conn:
            conn.execute("UPDATE v2_projects SET name=?, description=?, visibility=?, updated_at=? WHERE project_id=?", (name_value, str(description or "").strip(), _normalize_visibility(visibility), now, project.project_id))
            if member_user_ids is not None:
                conn.execute("DELETE FROM v2_project_members WHERE project_id=?", (project.project_id,))
                if _normalize_visibility(visibility) == "shared":
                    conn.executemany("INSERT OR IGNORE INTO v2_project_members (project_id,user_id,created_at) VALUES (?,?,?)", [(project.project_id, member, now) for member in dict.fromkeys(str(item).strip() for item in member_user_ids if str(item).strip() and str(item).strip() != project.owner_user_id)])
        return self._replace(project, name=name_value, description=str(description or "").strip(), visibility=_normalize_visibility(visibility), updated_at=now)

    def archive_project(self, project_id: str, *, requester_user_id: str, requester_is_admin: bool = False) -> ProjectRecord:
        project = self._require_manageable(project_id, requester_user_id=requester_user_id, requester_is_admin=requester_is_admin)
        if project.is_system_home:
            raise ValueError("system_home_project_protected")
        if project.status == "archived": raise ValueError("project_not_active")
        now = _now_iso()
        with self._connect() as conn:
            conn.execute("UPDATE v2_projects SET status='archived', archived_at=?, updated_at=? WHERE project_id=?", (now, now, project.project_id))
        return self._replace(project, status="archived", archived_at=now, updated_at=now)

    def restore_project(self, project_id: str, *, requester_user_id: str, requester_is_admin: bool = False) -> ProjectRecord:
        project = self._require_manageable(project_id, requester_user_id=requester_user_id, requester_is_admin=requester_is_admin)
        if project.status != "archived": raise ValueError("project_not_archived")
        now = _now_iso()
        with self._connect() as conn:
            conn.execute("UPDATE v2_projects SET status='active', archived_at=NULL, updated_at=? WHERE project_id=?", (now, project.project_id))
        return self._replace(project, status="active", archived_at=None, updated_at=now)

    def set_status(self, project_id: str, status: ProjectStatus, *, requester_user_id: str, requester_is_admin: bool = False) -> ProjectRecord:
        project = self._require_manageable(project_id, requester_user_id=requester_user_id, requester_is_admin=requester_is_admin)
        status_value = _normalize_status(status)
        if project.is_system_home and status_value != "active":
            raise ValueError("system_home_project_protected")
        now = _now_iso()
        archived_at = now if status_value == "archived" else None
        with self._connect() as conn:
            conn.execute("UPDATE v2_projects SET status=?, archived_at=?, updated_at=? WHERE project_id=?", (status_value, archived_at, now, project.project_id))
        return self._replace(project, status=status_value, archived_at=archived_at, updated_at=now)

    def read_project_config(self, project_id: str, *, requester_user_id: str | None = None, requester_is_admin: bool = False) -> dict[str, Any]:
        project = self.get_project(project_id, requester_user_id=requester_user_id, requester_is_admin=requester_is_admin, include_archived=True)
        if project is None:
            raise KeyError(f"project_not_found: {project_id}")
        path = Path(project.config_path)
        if not path.exists():
            config = default_project_config(owner_user_id=project.owner_user_id, description=project.description)
            self._write_config_file(path, config)
            return config
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("project_config_invalid") from exc
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("project_config_invalid")
        return _validate_project_config(value, project.owner_user_id)

    def write_project_config(self, project_id: str, config: dict[str, Any], *, requester_user_id: str, requester_is_admin: bool = False) -> ProjectRecord:
        project = self._require_manageable(project_id, requester_user_id=requester_user_id, requester_is_admin=requester_is_admin)
        if not isinstance(config, dict):
            raise ValueError("project_config_invalid")
        value = dict(config)
        value["schema_version"] = 1
        value["owner_user_id"] = project.owner_user_id
        value.setdefault("scope", "")
        value.setdefault("goals", [])
        value.setdefault("duration", {"ongoing": True, "start_date": None, "end_date": None})
        value.setdefault("directory_structure", [])
        value = _validate_project_config(value, project.owner_user_id)
        self._write_config_file(Path(project.config_path), value)
        now = _now_iso()
        with self._connect() as conn:
            conn.execute("UPDATE v2_projects SET updated_at=? WHERE project_id=?", (now, project.project_id))
        return self._replace(project, updated_at=now)

    @staticmethod
    def _write_config_file(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _ensure_project_config(self, root: Path, owner_user_id: str, description: str) -> None:
        path = root / PROJECT_CONFIG_FILENAME
        if not path.exists():
            self._write_config_file(path, default_project_config(owner_user_id=owner_user_id, description=description))

    def delete_project(self, project_id: str, *, requester_user_id: str, requester_is_admin: bool = False) -> ProjectRecord:
        project = self._require_manageable(project_id, requester_user_id=requester_user_id, requester_is_admin=requester_is_admin)
        if project.is_system_home:
            raise ValueError("system_home_project_protected")
        with self._connect() as conn:
            conn.execute("DELETE FROM v2_project_members WHERE project_id=?", (project.project_id,))
            conn.execute("DELETE FROM v2_projects WHERE project_id=?", (project.project_id,))
        return project

    def read_project_context(self, project_id: str, *, requester_user_id: str | None = None) -> str:
        project = self.get_project(project_id, requester_user_id=requester_user_id)
        if project is None:
            return ""
        path = Path(project.context_path)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def write_project_context(
        self,
        project_id: str,
        content: str,
        *,
        requester_user_id: str | None = None,
        requester_is_admin: bool = False,
    ) -> ProjectRecord:
        # Legacy project_context.md is retained as user data for manual review.
        # New edits belong in project_config.json.
        raise ValueError("legacy_project_context_read_only")

    def render_project_context(self, project_id: str, *, requester_user_id: str | None = None) -> str:
        project = self.get_project(project_id, requester_user_id=requester_user_id)
        if project is None:
            return ""
        lines = [
            f"- Project Name: {project.name}",
            f"- Project ID: {project.project_id}",
            f"- Project Directory: {project.root_path}",
            f"- Visibility: {project.visibility}",
        ]
        if project.description:
            lines.append(f"- Description: {project.description}")
        try:
            config = self.read_project_config(project.project_id, requester_user_id=requester_user_id, requester_is_admin=requester_is_admin)
        except (KeyError, ValueError, OSError):
            config = {}
        lines.extend(["", json.dumps(config, ensure_ascii=False, indent=2)])
        return "\n".join(lines).strip()

    def add_member(self, project_id: str, user_id: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO v2_project_members (project_id, user_id, created_at) VALUES (?, ?, ?)", (str(project_id), str(user_id), _now_iso()))

    def remove_member(self, project_id: str, user_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute("DELETE FROM v2_project_members WHERE project_id=? AND user_id=?", (str(project_id), str(user_id))).rowcount > 0

    def list_members(self, project_id: str) -> list[str]:
        with self._connect() as conn:
            return [str(row[0]) for row in conn.execute("SELECT user_id FROM v2_project_members WHERE project_id=? ORDER BY user_id", (str(project_id),)).fetchall()]

    def is_member(self, project_id: str, user_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute("SELECT 1 FROM v2_project_members WHERE project_id=? AND user_id=?", (str(project_id), str(user_id))).fetchone() is not None

    def migrate_legacy_shared_access(self, user_ids: list[str]) -> int:
        """Freeze legacy family-wide sharing as explicit current-member grants once."""
        migration_id = "shared-projects-to-explicit-members-v1"
        active_users = list(dict.fromkeys(str(item).strip() for item in user_ids if str(item).strip()))
        now = _now_iso()
        with self._connect() as conn:
            if conn.execute("SELECT 1 FROM v2_project_migrations WHERE migration_id=?", (migration_id,)).fetchone():
                return 0
            projects = conn.execute(
                "SELECT project_id, owner_user_id FROM v2_projects WHERE visibility='shared'"
            ).fetchall()
            migrated = 0
            for project in projects:
                recipients = [user_id for user_id in active_users if user_id != str(project["owner_user_id"])]
                conn.executemany(
                    "INSERT OR IGNORE INTO v2_project_members (project_id,user_id,created_at) VALUES (?,?,?)",
                    [(str(project["project_id"]), user_id, now) for user_id in recipients],
                )
                migrated += len(recipients)
            conn.execute("INSERT INTO v2_project_migrations (migration_id, completed_at) VALUES (?,?)", (migration_id, now))
        return migrated

    def migrate_owner(self, old_user_id: str, new_user_id: str) -> int:
        with self._connect() as conn:
            rows = conn.execute("SELECT project_id, root_path FROM v2_projects WHERE owner_user_id=?", (str(old_user_id),)).fetchall()
            updated = conn.execute("UPDATE v2_projects SET owner_user_id=?, updated_at=? WHERE owner_user_id=?", (str(new_user_id), _now_iso(), str(old_user_id))).rowcount
        for row in rows:
            config_path = Path(str(row["root_path"])) / PROJECT_CONFIG_FILENAME
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if isinstance(config, dict) and config.get("schema_version") == 1:
                    config["owner_user_id"] = str(new_user_id)
                    self._write_config_file(config_path, config)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
        return updated

    def delete_owned_by(self, user_id: str) -> list[str]:
        with self._connect() as conn:
            roots = [str(row[0]) for row in conn.execute("SELECT root_path FROM v2_projects WHERE owner_user_id=?", (str(user_id),)).fetchall()]
            conn.execute("DELETE FROM v2_project_members WHERE user_id=? OR project_id IN (SELECT project_id FROM v2_projects WHERE owner_user_id=?)", (str(user_id), str(user_id)))
            conn.execute("DELETE FROM v2_projects WHERE owner_user_id=?", (str(user_id),))
        return roots

    def _require_manageable(self, project_id: str, *, requester_user_id: str, requester_is_admin: bool) -> ProjectRecord:
        project = self.get_project(project_id, requester_user_id=requester_user_id, requester_is_admin=requester_is_admin, include_archived=True)
        if project is None: raise KeyError(f"project_not_found: {project_id}")
        if not requester_is_admin and project.owner_user_id != str(requester_user_id or ""):
            raise PermissionError("project_owner_required")
        return project

    def _replace(self, project: ProjectRecord, **values: object) -> ProjectRecord:
        return ProjectRecord(**{**project.__dict__, **values})

    def _connect(self) -> sqlite3.Connection:
        if self._memory_connection is not None:
            return _ConnectionProxy(self._memory_connection)
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return connect_database(path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS v2_projects (
                  project_id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  description TEXT NOT NULL DEFAULT '',
                  root_path TEXT NOT NULL,
                  visibility TEXT NOT NULL,
                  owner_user_id TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'active',
                  archived_at TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  is_system_home INTEGER NOT NULL DEFAULT 0,
                  CHECK (visibility IN ('private', 'shared'))
                ) STRICT;

                CREATE INDEX IF NOT EXISTS idx_v2_projects_owner_visibility
                  ON v2_projects (owner_user_id, visibility, lower(name));
                CREATE TABLE IF NOT EXISTS v2_project_members (
                  project_id TEXT NOT NULL, user_id TEXT NOT NULL, created_at TEXT NOT NULL,
                  PRIMARY KEY (project_id, user_id)
                ) STRICT;
                CREATE TABLE IF NOT EXISTS v2_project_migrations (
                  migration_id TEXT PRIMARY KEY, completed_at TEXT NOT NULL
                ) STRICT;
                """
            )
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(v2_projects)").fetchall()}
            if "status" not in columns:
                conn.execute("ALTER TABLE v2_projects ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
            if "archived_at" not in columns:
                conn.execute("ALTER TABLE v2_projects ADD COLUMN archived_at TEXT")
            if "is_system_home" not in columns:
                conn.execute("ALTER TABLE v2_projects ADD COLUMN is_system_home INTEGER NOT NULL DEFAULT 0")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_v2_projects_one_system_home "
                "ON v2_projects(owner_user_id) WHERE is_system_home=1"
            )
            # Add the structured config beside existing files without opening,
            # rewriting, renaming, or removing the legacy context Markdown.
            existing = conn.execute("SELECT root_path, owner_user_id, description FROM v2_projects").fetchall()
            for row in existing:
                self._ensure_project_config(Path(str(row["root_path"])), str(row["owner_user_id"]), str(row["description"] or ""))


class _ConnectionProxy:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def __enter__(self) -> sqlite3.Connection:
        return self._conn

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()


def _project_from_row(row: sqlite3.Row | None) -> ProjectRecord | None:
    if row is None:
        return None
    return ProjectRecord(
        project_id=str(row["project_id"]),
        name=str(row["name"]),
        description=str(row["description"] or ""),
        root_path=str(row["root_path"]),
        visibility=_normalize_visibility(row["visibility"]),
        owner_user_id=str(row["owner_user_id"]),
        status=_normalize_status(row["status"] if "status" in row.keys() else "active"),
        archived_at=str(row["archived_at"] or "") or None,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        is_system_home=bool(row["is_system_home"]) if "is_system_home" in row.keys() else False,
    )


def _normalize_visibility(value: object) -> ProjectVisibility:
    rendered = str(value or "").strip().lower()
    if rendered not in {"private", "shared"}:
        raise ValueError(f"invalid_project_visibility: {value}")
    return rendered  # type: ignore[return-value]


def _normalize_status(value: object) -> ProjectStatus:
    rendered = str(value or "").strip().lower()
    if rendered not in {"active", "paused", "completed", "archived"}:
        raise ValueError(f"invalid_project_status: {value}")
    return rendered  # type: ignore[return-value]


def _validate_project_config(value: dict[str, Any], owner_user_id: str) -> dict[str, Any]:
    config = dict(value)
    config["owner_user_id"] = str(owner_user_id)
    if not isinstance(config.get("scope"), str) or not isinstance(config.get("goals"), list):
        raise ValueError("project_config_invalid")
    duration = config.get("duration")
    if not isinstance(duration, dict) or not isinstance(duration.get("ongoing"), bool):
        raise ValueError("project_config_invalid")
    for field in ("start_date", "end_date"):
        date_value = duration.get(field)
        if date_value is not None:
            if not isinstance(date_value, str):
                raise ValueError("project_config_invalid")
            try:
                datetime.fromisoformat(date_value + "T00:00:00")
            except ValueError as exc:
                raise ValueError("project_config_invalid") from exc
    structure = config.get("directory_structure")
    if not isinstance(structure, list) or any(not isinstance(item, str) for item in structure):
        raise ValueError("project_config_invalid")
    if any(Path(item).is_absolute() or ".." in Path(item).parts for item in structure):
        raise ValueError("project_config_invalid")
    return config


def _resolve_project_root(value: str) -> Path:
    rendered = str(value or "").strip()
    if not rendered:
        raise ValueError("project_root_path_required")
    return Path(rendered).expanduser().resolve()


def _project_id_from_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(name or "").strip().lower()).strip("-")
    suffix = uuid4().hex[:8]
    return f"{slug or 'project'}-{suffix}"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _default_project_db_path() -> str:
    return str(default_database_path())
