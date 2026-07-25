"""Durable registered executable artifacts for Alphonse v2."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition

DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_TIMEOUT_SECONDS = 120.0
MAX_OUTPUT_CHARS = 12000


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
        return cls(os.getenv("ALPHONSE_V2_ARTIFACTS_DB_PATH") or str(Path.home() / ".alphonse" / "v2-artifacts.sqlite3"))

    def register(self, *, artifact_id: str, name: str, description: str, project_id: str, owner_user_id: str, entrypoint_path: str, argument_schema: dict[str, Any], timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> ArtifactRecord:
        _validate_schema(argument_schema)
        record = ArtifactRecord(
            artifact_id=_artifact_id(artifact_id), name=_required(name, "artifact_name_required"),
            description=_required(description, "artifact_description_required"), project_id=_required(project_id, "artifact_project_required"),
            owner_user_id=_required(owner_user_id, "artifact_owner_required"), entrypoint_path=_required(entrypoint_path, "artifact_entrypoint_required"),
            argument_schema=dict(argument_schema), timeout_seconds=_timeout(timeout_seconds), enabled=True,
            created_at=_now(), updated_at=_now(),
        )
        with self._connect() as conn:
            if conn.execute("SELECT 1 FROM v2_artifacts WHERE artifact_id=?", (record.artifact_id,)).fetchone():
                raise ValueError("artifact_id_already_registered")
            conn.execute("""INSERT INTO v2_artifacts(artifact_id,name,description,project_id,owner_user_id,entrypoint_path,argument_schema_json,timeout_seconds,enabled,created_at,updated_at)
                         VALUES(?,?,?,?,?,?,?,?,?,?,?)""", _values(record))
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

    def delete(self, artifact_id: str) -> None:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM v2_artifacts WHERE artifact_id=?", (_artifact_id(artifact_id),))
        if not cursor.rowcount: raise KeyError("artifact_not_found")

    def _connect(self) -> sqlite3.Connection:
        if self._memory is not None: return self._memory
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS v2_artifacts(
                artifact_id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
                project_id TEXT NOT NULL, owner_user_id TEXT NOT NULL, entrypoint_path TEXT NOT NULL,
                argument_schema_json TEXT NOT NULL, timeout_seconds REAL NOT NULL, enabled INTEGER NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")


def build_artifact_tool_definitions(store: SQLiteArtifactStore, project_store: Any) -> list[ToolDefinition]:
    definitions: list[ToolDefinition] = []
    for record in store.list(enabled_only=True):
        project = project_store.get_project(record.project_id, requester_is_admin=True, include_archived=True)
        if project is None: continue
        root = Path(project.root_path).resolve()
        entrypoint = (root / record.entrypoint_path).resolve()
        if not _within(entrypoint, root): continue
        descriptor = ToolDescriptor(record.artifact_id, record.name, ToolKind.ARTIFACT, record.description, dict(record.argument_schema), ("artifact", "local_execution"), ("artifact",), {"project_id": record.project_id, "entrypoint_path": record.entrypoint_path})
        definitions.append(ToolDefinition(descriptor, lambda arguments, r=record, p=entrypoint, cwd=root: execute_artifact(r, p, cwd, arguments), dict(record.argument_schema)))
    return definitions


def execute_artifact(record: ArtifactRecord, entrypoint: Path, project_root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    if not entrypoint.is_file(): raise ValueError("artifact_entrypoint_missing")
    if not os.access(entrypoint, os.X_OK): raise ValueError("artifact_entrypoint_not_executable")
    try: Draft202012Validator(record.argument_schema).validate(arguments)
    except ValidationError as exc: raise ValueError(f"artifact_arguments_invalid: {exc.message}") from exc
    try:
        completed = subprocess.run([str(entrypoint)], input=json.dumps(dict(arguments)), capture_output=True, text=True, cwd=str(project_root), timeout=record.timeout_seconds, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"artifact_timed_out: {record.timeout_seconds:g}s") from exc
    stdout, stderr = _truncate(completed.stdout), _truncate(completed.stderr)
    if completed.returncode != 0: raise RuntimeError(f"artifact_exit_{completed.returncode}: {stderr or stdout}")
    try: result = json.loads(stdout)
    except (TypeError, ValueError) as exc: raise ValueError("artifact_result_invalid_json") from exc
    if not isinstance(result, dict): raise ValueError("artifact_result_must_be_object")
    return result


def _values(record: ArtifactRecord) -> tuple[Any, ...]:
    return (record.artifact_id, record.name, record.description, record.project_id, record.owner_user_id, record.entrypoint_path, json.dumps(record.argument_schema, sort_keys=True), record.timeout_seconds, int(record.enabled), record.created_at, record.updated_at)

def _record(row: sqlite3.Row | None) -> ArtifactRecord | None:
    if row is None: return None
    return ArtifactRecord(str(row["artifact_id"]), str(row["name"]), str(row["description"]), str(row["project_id"]), str(row["owner_user_id"]), str(row["entrypoint_path"]), dict(json.loads(row["argument_schema_json"])), float(row["timeout_seconds"]), bool(row["enabled"]), str(row["created_at"]), str(row["updated_at"]))
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
def _within(path: Path, root: Path) -> bool:
    try: path.relative_to(root); return True
    except ValueError: return False
def _truncate(value: Any) -> str:
    return str(value or "")[:MAX_OUTPUT_CHARS]
