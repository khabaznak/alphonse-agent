"""Durable registered executable artifacts for Alphonse v2."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from alphonse.agent_v2.database import connect_database, default_database_path

DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_TIMEOUT_SECONDS = 120.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    name: str
    description: str
    project_id: str
    owner_user_id: str
    entrypoint_path: str
    argument_schema: dict[str, Any]
    timeout_seconds: float
    enabled: bool
    created_at: str
    updated_at: str
    skill_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SQLiteArtifactStore:
    """Catalog only; artifact programs and their data remain in projects."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._memory = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls) -> "SQLiteArtifactStore":
        return cls(default_database_path())

    def register(self, *, artifact_id: str, name: str, description: str, project_id: str, owner_user_id: str, entrypoint_path: str, argument_schema: dict[str, Any], timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS, skill_id: str = "") -> ArtifactRecord:
        _validate_schema(argument_schema)
        record = ArtifactRecord(
            artifact_id=_artifact_id(artifact_id), name=_required(name, "artifact_name_required"),
            description=_required(description, "artifact_description_required"), project_id=_required(project_id, "artifact_project_required"),
            owner_user_id=_required(owner_user_id, "artifact_owner_required"), entrypoint_path=_required(entrypoint_path, "artifact_entrypoint_required"),
            argument_schema=dict(argument_schema), timeout_seconds=_timeout(timeout_seconds), enabled=True,
            created_at=_now(), updated_at=_now(), skill_id=str(skill_id or ""),
        )
        with self._connect() as conn:
            if conn.execute("SELECT 1 FROM v2_artifacts WHERE artifact_id=?", (record.artifact_id,)).fetchone():
                raise ValueError("artifact_id_already_registered")
            conn.execute("""INSERT INTO v2_artifacts(artifact_id,name,description,project_id,owner_user_id,entrypoint_path,argument_schema_json,timeout_seconds,enabled,created_at,updated_at,skill_id)
                         VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", _values(record))
        return record

    def get(self, artifact_id: str) -> ArtifactRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM v2_artifacts WHERE artifact_id=?", (_artifact_id(artifact_id),)).fetchone()
        return _record(row)

    def list(self, *, enabled_only: bool = False, owner_user_id: str = "") -> list[ArtifactRecord]:
        clauses: list[str] = []
        values: list[Any] = []
        if enabled_only: clauses.append("enabled=1")
        if owner_user_id: clauses.append("owner_user_id=?"); values.append(str(owner_user_id))
        query = "SELECT * FROM v2_artifacts" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY lower(name), artifact_id"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(values)).fetchall()
        return [item for row in rows if (item := _record(row))]

    def update_metadata(self, artifact_id: str, *, name: str, description: str) -> ArtifactRecord:
        with self._connect() as conn:
            conn.execute("UPDATE v2_artifacts SET name=?,description=?,updated_at=? WHERE artifact_id=?", (_required(name, "artifact_name_required"), _required(description, "artifact_description_required"), _now(), _artifact_id(artifact_id)))
        result = self.get(artifact_id)
        if result is None: raise KeyError("artifact_not_found")
        return result

    def set_enabled(self, artifact_id: str, enabled: bool) -> ArtifactRecord:
        with self._connect() as conn:
            conn.execute("UPDATE v2_artifacts SET enabled=?,updated_at=? WHERE artifact_id=?", (int(enabled), _now(), _artifact_id(artifact_id)))
        result = self.get(artifact_id)
        if result is None: raise KeyError("artifact_not_found")
        return result

    def set_skill(self, artifact_id: str, skill_id: str) -> ArtifactRecord:
        with self._connect() as conn:
            conn.execute("UPDATE v2_artifacts SET skill_id=?,updated_at=? WHERE artifact_id=?", (str(skill_id or "").strip(), _now(), _artifact_id(artifact_id)))
        result = self.get(artifact_id)
        if result is None:
            raise KeyError("artifact_not_found")
        return result

    def delete(self, artifact_id: str) -> None:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM v2_artifacts WHERE artifact_id=?", (_artifact_id(artifact_id),))
        if not cursor.rowcount: raise KeyError("artifact_not_found")

    def _connect(self) -> sqlite3.Connection:
        if self._memory is not None: return self._memory
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        return connect_database(self.db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS v2_artifacts(
                artifact_id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
                project_id TEXT NOT NULL, owner_user_id TEXT NOT NULL, entrypoint_path TEXT NOT NULL,
                argument_schema_json TEXT NOT NULL, timeout_seconds REAL NOT NULL, enabled INTEGER NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, skill_id TEXT NOT NULL DEFAULT '')""")
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(v2_artifacts)")}
            if "skill_id" not in columns:
                conn.execute("ALTER TABLE v2_artifacts ADD COLUMN skill_id TEXT NOT NULL DEFAULT ''")


def build_artifact_tool_definitions(store: SQLiteArtifactStore, project_store: Any) -> list[Any]:
    """Compatibility shim: artifacts are catalog records owned by skills, not tools."""
    return []


def _values(record: ArtifactRecord) -> tuple[Any, ...]:
    return (record.artifact_id, record.name, record.description, record.project_id, record.owner_user_id, record.entrypoint_path, json.dumps(record.argument_schema, sort_keys=True), record.timeout_seconds, int(record.enabled), record.created_at, record.updated_at, record.skill_id)

def _record(row: sqlite3.Row | None) -> ArtifactRecord | None:
    if row is None: return None
    return ArtifactRecord(str(row["artifact_id"]), str(row["name"]), str(row["description"]), str(row["project_id"]), str(row["owner_user_id"]), str(row["entrypoint_path"]), dict(json.loads(row["argument_schema_json"])), float(row["timeout_seconds"]), bool(row["enabled"]), str(row["created_at"]), str(row["updated_at"]), str(row["skill_id"] or ""))
def _required(value: Any, error: str) -> str:
    text = str(value or "").strip()
    if not text: raise ValueError(error)
    return text
def _artifact_id(value: Any) -> str:
    text = _required(value, "artifact_id_required")
    if not text.startswith("artifact.") or not all(char.islower() or char.isdigit() or char in ".-_" for char in text): raise ValueError("artifact_id_invalid")
    return text
def _timeout(value: Any) -> float:
    try: result = float(value)
    except (TypeError, ValueError) as exc: raise ValueError("artifact_timeout_invalid") from exc
    if result <= 0: raise ValueError("artifact_timeout_invalid")
    return min(result, MAX_TIMEOUT_SECONDS)
def _validate_schema(schema: dict[str, Any]) -> None:
    if not isinstance(schema, dict): raise ValueError("artifact_argument_schema_required")
    try: Draft202012Validator.check_schema(schema)
    except SchemaError as exc: raise ValueError(f"artifact_argument_schema_invalid: {exc.message}") from exc
