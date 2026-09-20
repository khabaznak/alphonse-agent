"""Persistent opt-in and rollback settings for the hierarchical V3 engine."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from alphonse.agent_v2.database import connect_database, default_database_path

TACTICAL_V2 = "tactical_v2"
HIERARCHICAL_V3 = "hierarchical_v3"
VALID_ENGINES = {TACTICAL_V2, HIERARCHICAL_V3}


@dataclass(frozen=True)
class IntelligenceEngineSettings:
    default_engine: str = TACTICAL_V2
    v3_project_ids: tuple[str, ...] = ()
    updated_at: str = ""

    def __post_init__(self) -> None:
        if self.default_engine not in VALID_ENGINES:
            raise ValueError("intelligence_engine_invalid")

    def engine_for(self, project_id: str) -> str:
        return HIERARCHICAL_V3 if str(project_id or "") in set(self.v3_project_ids) else self.default_engine

    def to_dict(self) -> dict[str, object]:
        return {"default_engine": self.default_engine, "v3_project_ids": list(self.v3_project_ids), "updated_at": self.updated_at}


class SQLiteIntelligenceEngineSettingsStore:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory = sqlite3.connect(":memory:", check_same_thread=False) if self.db_path == ":memory:" else None
        if self._memory is not None:
            self._memory.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls) -> "SQLiteIntelligenceEngineSettingsStore":
        return cls(default_database_path())

    def get(self) -> IntelligenceEngineSettings:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM v2_intelligence_engine_settings WHERE settings_id=1").fetchone()
        if row is None:
            return IntelligenceEngineSettings()
        return IntelligenceEngineSettings(
            default_engine=str(row["default_engine"]),
            v3_project_ids=tuple(str(item) for item in json.loads(str(row["v3_project_ids_json"]))),
            updated_at=str(row["updated_at"]),
        )

    def save(self, settings: IntelligenceEngineSettings) -> IntelligenceEngineSettings:
        validated = IntelligenceEngineSettings(settings.default_engine, tuple(dict.fromkeys(settings.v3_project_ids)))
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO v2_intelligence_engine_settings(settings_id,default_engine,v3_project_ids_json,updated_at) VALUES(1,?,?,?)",
                (validated.default_engine, json.dumps(list(validated.v3_project_ids)), datetime.now(timezone.utc).isoformat()),
            )
        return self.get()

    def _connect(self):
        if self._memory is not None:
            return _Connection(self._memory)
        path = Path(self.db_path); path.parent.mkdir(parents=True, exist_ok=True)
        return connect_database(path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS v2_intelligence_engine_settings(
                settings_id INTEGER PRIMARY KEY CHECK(settings_id=1),
                default_engine TEXT NOT NULL DEFAULT 'tactical_v2',
                v3_project_ids_json TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT NOT NULL
            ) STRICT""")


class _Connection:
    def __init__(self, connection): self.connection = connection
    def __enter__(self): return self.connection
    def __exit__(self, typ, value, traceback): self.connection.commit() if typ is None else self.connection.rollback()
