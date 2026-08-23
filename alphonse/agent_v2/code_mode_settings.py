"""Persistent administrator configuration for Docker-backed program execution."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from alphonse.agent_v2.database import connect_database, default_database_path


@dataclass(frozen=True)
class CodeModeSettings:
    enabled: bool = False
    docker_bin: str = "docker"
    image: str = "python:3.11-slim"
    timeout_seconds: float = 60.0
    max_tool_calls: int = 16
    max_parallel_calls: int = 4
    memory_mb: int = 256
    cpu_count: float = 0.5
    pid_limit: int = 64
    tmpfs_mb: int = 64
    network_disabled: bool = True
    read_only_filesystem: bool = True
    run_as_non_root: bool = True
    drop_all_capabilities: bool = True
    no_new_privileges: bool = True
    verification_ready: bool = False
    verification_error: str = ""
    verified_at: str = ""
    updated_at: str = ""

    @property
    def available(self) -> bool:
        return self.enabled and self.verification_ready

    @property
    def weakened_protections(self) -> tuple[str, ...]:
        labels = []
        if not self.network_disabled: labels.append("network access")
        if not self.read_only_filesystem: labels.append("a writable container filesystem")
        if not self.run_as_non_root: labels.append("root container user")
        if not self.drop_all_capabilities: labels.append("Linux capabilities")
        if not self.no_new_privileges: labels.append("privilege escalation protection")
        return tuple(labels)

    def to_dict(self) -> dict[str, object]:
        return {**self.__dict__, "available": self.available, "weakened_protections": list(self.weakened_protections)}


class SQLiteCodeModeSettingsStore:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory = sqlite3.connect(":memory:", check_same_thread=False) if self.db_path == ":memory:" else None
        if self._memory is not None: self._memory.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls) -> "SQLiteCodeModeSettingsStore":
        return cls(default_database_path())

    def get(self) -> CodeModeSettings:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM v2_code_mode_settings WHERE settings_id=1").fetchone()
        if row is None: return CodeModeSettings()
        return CodeModeSettings(**{field: row[field] for field in CodeModeSettings.__dataclass_fields__ if field in row.keys() and field not in _BOOL_FIELDS}, **{field: bool(row[field]) for field in _BOOL_FIELDS})

    def save(self, settings: CodeModeSettings) -> CodeModeSettings:
        checked = _validate(settings)
        current = self.get()
        verification_ready = current.verification_ready if (current.docker_bin, current.image) == (checked.docker_bin, checked.image) else False
        verification_error = current.verification_error if verification_ready else ""
        verified_at = current.verified_at if verification_ready else ""
        with self._connect() as conn:
            conn.execute("""INSERT OR REPLACE INTO v2_code_mode_settings
                (settings_id,enabled,docker_bin,image,timeout_seconds,max_tool_calls,max_parallel_calls,memory_mb,cpu_count,pid_limit,tmpfs_mb,network_disabled,read_only_filesystem,run_as_non_root,drop_all_capabilities,no_new_privileges,verification_ready,verification_error,verified_at,updated_at)
                VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                int(checked.enabled), checked.docker_bin, checked.image, checked.timeout_seconds, checked.max_tool_calls, checked.max_parallel_calls, checked.memory_mb, checked.cpu_count, checked.pid_limit, checked.tmpfs_mb,
                *[int(getattr(checked, field)) for field in _BOOL_FIELDS if field not in {"enabled", "verification_ready"}], int(verification_ready), verification_error, verified_at, _now(),
            ))
        return self.get()

    def mark_verification(self, *, ready: bool, error: str = "") -> CodeModeSettings:
        if self.get() == CodeModeSettings():
            self.save(CodeModeSettings())
        with self._connect() as conn:
            conn.execute("UPDATE v2_code_mode_settings SET verification_ready=?, verification_error=?, verified_at=?, updated_at=? WHERE settings_id=1", (int(ready), str(error or ""), _now() if ready else "", _now()))
        return self.get()

    def _connect(self):
        if self._memory is not None: return _Connection(self._memory)
        path = Path(self.db_path); path.parent.mkdir(parents=True, exist_ok=True)
        return connect_database(path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""CREATE TABLE IF NOT EXISTS v2_code_mode_settings (
              settings_id INTEGER PRIMARY KEY CHECK(settings_id=1), enabled INTEGER NOT NULL DEFAULT 0 CHECK(enabled IN (0,1)), docker_bin TEXT NOT NULL, image TEXT NOT NULL,
              timeout_seconds REAL NOT NULL, max_tool_calls INTEGER NOT NULL, max_parallel_calls INTEGER NOT NULL, memory_mb INTEGER NOT NULL, cpu_count REAL NOT NULL, pid_limit INTEGER NOT NULL, tmpfs_mb INTEGER NOT NULL,
              network_disabled INTEGER NOT NULL CHECK(network_disabled IN (0,1)), read_only_filesystem INTEGER NOT NULL CHECK(read_only_filesystem IN (0,1)), run_as_non_root INTEGER NOT NULL CHECK(run_as_non_root IN (0,1)), drop_all_capabilities INTEGER NOT NULL CHECK(drop_all_capabilities IN (0,1)), no_new_privileges INTEGER NOT NULL CHECK(no_new_privileges IN (0,1)),
              verification_ready INTEGER NOT NULL DEFAULT 0 CHECK(verification_ready IN (0,1)), verification_error TEXT NOT NULL DEFAULT '', verified_at TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL
            ) STRICT;""")


_BOOL_FIELDS = ("enabled", "network_disabled", "read_only_filesystem", "run_as_non_root", "drop_all_capabilities", "no_new_privileges", "verification_ready")


class _Connection:
    def __init__(self, connection: sqlite3.Connection) -> None: self.connection = connection
    def __enter__(self): return self.connection
    def __exit__(self, typ, value, traceback): self.connection.commit() if typ is None else self.connection.rollback()


def _validate(settings: CodeModeSettings) -> CodeModeSettings:
    docker_bin = _text(settings.docker_bin, "docker_bin")
    image = _text(settings.image, "image")
    timeout = _float(settings.timeout_seconds, "timeout_seconds", 1, 300)
    calls = _int(settings.max_tool_calls, "max_tool_calls", 1, 64)
    parallel = _int(settings.max_parallel_calls, "max_parallel_calls", 1, 16)
    if parallel > calls: raise ValueError("code_mode_max_parallel_calls_exceeds_tool_calls")
    return CodeModeSettings(
        enabled=bool(settings.enabled), docker_bin=docker_bin, image=image, timeout_seconds=timeout, max_tool_calls=calls, max_parallel_calls=parallel,
        memory_mb=_int(settings.memory_mb, "memory_mb", 32, 4096), cpu_count=_float(settings.cpu_count, "cpu_count", 0.1, 8),
        pid_limit=_int(settings.pid_limit, "pid_limit", 16, 1024), tmpfs_mb=_int(settings.tmpfs_mb, "tmpfs_mb", 16, 1024),
        network_disabled=bool(settings.network_disabled), read_only_filesystem=bool(settings.read_only_filesystem), run_as_non_root=bool(settings.run_as_non_root),
        drop_all_capabilities=bool(settings.drop_all_capabilities), no_new_privileges=bool(settings.no_new_privileges),
    )


def _text(value: object, field: str) -> str:
    result = str(value or "").strip()
    if not result or any(character.isspace() for character in result): raise ValueError(f"code_mode_{field}_invalid")
    return result

def _int(value: object, field: str, minimum: int, maximum: int) -> int:
    try: result = int(value)
    except (TypeError, ValueError) as exc: raise ValueError(f"code_mode_{field}_invalid") from exc
    if not minimum <= result <= maximum: raise ValueError(f"code_mode_{field}_invalid")
    return result

def _float(value: object, field: str, minimum: float, maximum: float) -> float:
    try: result = float(value)
    except (TypeError, ValueError) as exc: raise ValueError(f"code_mode_{field}_invalid") from exc
    if not minimum <= result <= maximum: raise ValueError(f"code_mode_{field}_invalid")
    return result

def _now() -> str: return datetime.now(timezone.utc).isoformat()
