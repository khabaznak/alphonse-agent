"""SQLite-backed optional integration configuration for Alphonse v2."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alphonse.agent_v2.database import connect_database, default_database_path


@dataclass(frozen=True)
class IntegrationConfigRecord:
    integration_id: str
    provider_key: str
    display_name: str
    enabled: bool
    config: dict[str, Any] = field(default_factory=dict)
    secrets: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self, *, mask_secrets: bool = True) -> dict[str, Any]:
        return {
            "integration_id": self.integration_id,
            "provider_key": self.provider_key,
            "display_name": self.display_name,
            "enabled": self.enabled,
            "config": dict(self.config),
            "secrets": _masked_secrets(self.secrets) if mask_secrets else dict(self.secrets),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class SQLiteIntegrationStore:
    """Local integration config store.

    Tests and helper runtimes default to in-memory storage. The Textual app uses
    `default()` so enabled integrations can persist across launches.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory_connection: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._memory_connection = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_connection.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls) -> "SQLiteIntegrationStore":
        return cls(_default_integrations_db_path())

    def upsert(
        self,
        *,
        integration_id: str,
        provider_key: str,
        display_name: str,
        enabled: bool = False,
        config: dict[str, Any] | None = None,
        secrets: dict[str, Any] | None = None,
    ) -> IntegrationConfigRecord:
        integration = str(integration_id or "").strip()
        provider = str(provider_key or "").strip().lower()
        name = str(display_name or "").strip()
        if not integration:
            raise ValueError("integration_id_required")
        if not provider:
            raise ValueError("provider_key_required")
        if not name:
            raise ValueError("display_name_required")
        now = _now_iso()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT created_at FROM v2_integrations WHERE integration_id = ?",
                (integration,),
            ).fetchone()
            created_at = str(existing["created_at"]) if existing is not None else now
            conn.execute(
                """
                INSERT OR REPLACE INTO v2_integrations (
                  integration_id, provider_key, display_name, enabled,
                  config_json, secret_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    integration,
                    provider,
                    name,
                    1 if enabled else 0,
                    json.dumps(dict(config or {}), sort_keys=True),
                    json.dumps(dict(secrets or {}), sort_keys=True),
                    created_at,
                    now,
                ),
            )
        record = self.get(integration)
        if record is None:
            raise RuntimeError("integration_upsert_failed")
        return record

    def get(self, integration_id: str) -> IntegrationConfigRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM v2_integrations WHERE integration_id = ?",
                (str(integration_id or "").strip(),),
            ).fetchone()
        return _record_from_row(row)

    def get_by_provider(self, provider_key: str) -> IntegrationConfigRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM v2_integrations
                WHERE provider_key = ?
                ORDER BY enabled DESC, updated_at DESC
                LIMIT 1
                """,
                (str(provider_key or "").strip().lower(),),
            ).fetchone()
        return _record_from_row(row)

    def list(self, *, provider_key: str | None = None) -> list[IntegrationConfigRecord]:
        values: list[Any] = []
        where = ""
        if provider_key is not None:
            where = "WHERE provider_key = ?"
            values.append(str(provider_key or "").strip().lower())
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM v2_integrations
                {where}
                ORDER BY provider_key, display_name, integration_id
                """,
                tuple(values),
            ).fetchall()
        return [_record_from_row(row) for row in rows if _record_from_row(row) is not None]

    def list_enabled(self) -> list[IntegrationConfigRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM v2_integrations
                WHERE enabled = 1
                ORDER BY provider_key, display_name, integration_id
                """
            ).fetchall()
        return [_record_from_row(row) for row in rows if _record_from_row(row) is not None]

    def set_enabled(self, integration_id: str, enabled: bool) -> IntegrationConfigRecord:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE v2_integrations
                SET enabled = ?, updated_at = ?
                WHERE integration_id = ?
                """,
                (1 if enabled else 0, _now_iso(), str(integration_id or "").strip()),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"integration_not_found: {integration_id}")
        record = self.get(integration_id)
        if record is None:
            raise KeyError(f"integration_not_found: {integration_id}")
        return record

    def remove_secret(self, integration_id: str, secret_name: str) -> IntegrationConfigRecord:
        record = self.get(integration_id)
        if record is None:
            raise KeyError(f"integration_not_found: {integration_id}")
        secrets = dict(record.secrets)
        secrets.pop(str(secret_name or "").strip(), None)
        return self.upsert(
            integration_id=record.integration_id,
            provider_key=record.provider_key,
            display_name=record.display_name,
            enabled=record.enabled,
            config=record.config,
            secrets=secrets,
        )

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
                CREATE TABLE IF NOT EXISTS v2_integrations (
                  integration_id TEXT PRIMARY KEY,
                  provider_key TEXT NOT NULL,
                  display_name TEXT NOT NULL,
                  enabled INTEGER NOT NULL DEFAULT 0,
                  config_json TEXT NOT NULL DEFAULT '{}',
                  secret_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  CHECK (enabled IN (0, 1))
                ) STRICT;

                CREATE INDEX IF NOT EXISTS idx_v2_integrations_provider
                  ON v2_integrations (provider_key, enabled);
                """
            )


def _record_from_row(row: sqlite3.Row | None) -> IntegrationConfigRecord | None:
    if row is None:
        return None
    return IntegrationConfigRecord(
        integration_id=str(row["integration_id"]),
        provider_key=str(row["provider_key"]),
        display_name=str(row["display_name"]),
        enabled=bool(row["enabled"]),
        config=_json_object(row["config_json"]),
        secrets=_json_object(row["secret_json"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


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


def _json_object(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except ValueError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _masked_secrets(secrets: dict[str, Any]) -> dict[str, str]:
    return {key: "***" for key, value in secrets.items() if str(value or "").strip()}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_integrations_db_path() -> str:
    return str(default_database_path())
