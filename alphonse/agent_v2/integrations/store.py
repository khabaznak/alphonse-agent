"""SQLite-backed optional integration configuration for Alphonse v2."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .credentials import CredentialError, CredentialVault

from alphonse.agent_v2.database import connect_database, default_database_path


@dataclass(frozen=True)
class IntegrationConfigRecord:
    integration_id: str
    provider_key: str
    display_name: str
    enabled: bool
    config: dict[str, Any] = field(default_factory=dict)
    secrets: dict[str, Any] = field(default_factory=dict, repr=False)
    connection: dict[str, Any] = field(default_factory=dict)
    credential_ref: str = ""
    credential_generation: int = 0
    revocation_pending: bool = False
    credential_deletion_pending: bool = False
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
            "connection": dict(self.connection),
            "credential_ref": self.credential_ref,
            "credential_generation": self.credential_generation,
            "revocation_pending": self.revocation_pending,
            "credential_deletion_pending": self.credential_deletion_pending,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class SQLiteIntegrationStore:
    """Local integration config store.

    Tests and helper runtimes default to in-memory storage. The Textual app uses
    `default()` so enabled integrations can persist across launches.
    """

    def __init__(self, db_path: str | Path = ":memory:", *, credential_vault: CredentialVault | None = None) -> None:
        self.credential_vault = credential_vault or CredentialVault()
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
        # New OAuth writes must use secure installation, never secret_json.
        if set(secrets or {}) & {"access_token", "refresh_token", "client_secret", "id_token"}:
            raise CredentialError("credential_secure_storage_required")
        now = _now_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM v2_integrations WHERE integration_id = ?",
                (integration,),
            ).fetchone()
            if existing is not None and existing["provider_key"] != provider:
                if existing['credential_generation'] and not _json_object(existing['connection_json']):
                    raise CredentialError('credential_managed_account_protected')
                raise ValueError("integration_identity_immutable")
            if existing is not None and _json_object(existing['connection_json']):
                if (bool(existing['enabled']) != bool(enabled)
                        or _json_object(existing['config_json']) != dict(config or {}) or secrets):
                    raise PermissionError('managed_connection_service_required')
            if existing is not None and existing["credential_generation"]:
                if provider != existing["provider_key"] or secrets or (enabled and not existing["enabled"]):
                    raise CredentialError("credential_managed_account_protected")
            created_at = str(existing["created_at"]) if existing is not None else now
            conn.execute(
                """
                INSERT INTO v2_integrations (
                  integration_id, provider_key, display_name, enabled,
                  config_json, secret_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(integration_id) DO UPDATE SET
                  provider_key=excluded.provider_key, display_name=excluded.display_name,
                  enabled=excluded.enabled, config_json=excluded.config_json,
                  secret_json=excluded.secret_json, updated_at=excluded.updated_at
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

    def install_credentials(self, integration_id: str, credentials: dict[str, str], *,
                            expected_generation: int) -> IntegrationConfigRecord:
        """Trusted opt-in setup/refresh. Readback before atomic metadata switch.

        Refresh workers must persist the returned rotated token before using it.
        A CAS loser must discard its response and resolve again, never overwrite.
        """
        with self._connect() as conn:
            snapshot = conn.execute("SELECT * FROM v2_integrations WHERE integration_id=?", (integration_id,)).fetchone()
        record = _record_from_row(snapshot)
        if record is None or record.credential_generation != expected_generation:
            raise CredentialError("credential_generation_conflict")
        if expected_generation and not record.enabled:
            raise CredentialError("credential_disconnected")
        generation = expected_generation + 1
        reference = self.credential_vault.put(integration_id, generation, credentials)
        with self._connect() as conn:
            cursor = conn.execute("""UPDATE v2_integrations SET credential_ref=?,
                credential_generation=?, secret_json='{}', updated_at=?
                WHERE integration_id=? AND credential_generation=? AND credential_ref=?
                AND enabled=? AND provider_key=? AND secret_json=?""",
                (reference, generation, _now_iso(), integration_id, expected_generation,
                 record.credential_ref, int(record.enabled), record.provider_key, snapshot["secret_json"]))
        if cursor.rowcount != 1:
            self.credential_vault.delete(reference)
            raise CredentialError("credential_generation_conflict")
        if record.credential_ref:
            # Failure leaves an unreachable secure orphan; never roll back to old tokens.
            try:
                self.credential_vault.delete(record.credential_ref)
            except CredentialError:
                pass
        return self.get(integration_id)

    def migrate_credentials(self, integration_id: str) -> IntegrationConfigRecord:
        """Explicit legacy OAuth migration; never touch unrelated messaging secrets."""
        record = self.get(integration_id)
        if record is None or record.provider_key in {"telegram", "discord"}:
            raise CredentialError("credential_migration_denied")
        if record.credential_generation:
            return record
        return self.install_credentials(integration_id, record.secrets, expected_generation=0)

    def resolve_credentials(self, record: IntegrationConfigRecord) -> dict[str, Any]:
        """Private managed consumer only, after binding authorization."""
        current = self.get(record.integration_id)
        if current is None or not current.enabled:
            raise CredentialError("credential_disconnected")
        if current.credential_generation:
            if not current.credential_ref or current.credential_deletion_pending:
                raise CredentialError("credential_disconnected")
            credentials = self.credential_vault.get(current.credential_ref, current.integration_id,
                                                     current.credential_generation)
            latest = self.get(current.integration_id)
            if latest is None or not latest.enabled or latest.credential_generation != current.credential_generation:
                raise CredentialError("credential_generation_conflict")
            return credentials
        return dict(current.secrets)  # Explicitly unmigrated protected-host legacy support.

    def disconnect_credentials(self, integration_id: str, *, expected_generation: int) -> IntegrationConfigRecord:
        """Disable locally first, without network/key access; provider revoke is pending.

        A locked backend retains a deletion-only reference for an explicit retry.
        Provider revocation is a separate adapter concern; never claim it succeeded.
        """
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            cursor = conn.execute("""UPDATE v2_integrations SET enabled=0,
                credential_generation=credential_generation+1, revocation_pending=1,
                credential_deletion_pending=1, secret_json='{}', updated_at=?
                WHERE integration_id=? AND credential_generation=?""",
                (_now_iso(), integration_id, expected_generation))
            if cursor.rowcount != 1:
                raise CredentialError("credential_generation_conflict")
            row = conn.execute('SELECT connection_json FROM v2_integrations WHERE integration_id=?', (integration_id,)).fetchone()
            connection = _json_object(row['connection_json'])
            if connection:
                now = _now_iso()
                if connection['status'] != 'disconnected':
                    connection.update(status='disconnected', revision=connection['revision'] + 1, updated_at=now)
                    conn.execute('UPDATE v2_integrations SET connection_json=? WHERE integration_id=?',
                                 (json.dumps(connection, sort_keys=True), integration_id))
                for grant_row in conn.execute('SELECT * FROM v2_account_grants WHERE connection_id=?', (integration_id,)).fetchall():
                    grant = json.loads(grant_row['definition'])
                    if not grant['revoked_at']:
                        grant.update(revoked_at=now, updated_at=now, revision=grant_row['revision'] + 1)
                        conn.execute('UPDATE v2_account_grants SET definition=?, revision=? WHERE grant_id=?',
                                     (json.dumps(grant), grant['revision'], grant_row['grant_id']))
        return self.retry_credential_deletion(integration_id)

    def retry_credential_deletion(self, integration_id: str) -> IntegrationConfigRecord:
        record = self.get(integration_id)
        if record is None or record.enabled or not record.credential_deletion_pending:
            raise CredentialError("credential_deletion_denied")
        try:
            if record.credential_ref:
                self.credential_vault.delete(record.credential_ref)
        except CredentialError:
            return record
        with self._connect() as conn:
            conn.execute("""UPDATE v2_integrations SET credential_ref='', credential_deletion_pending=0
                WHERE integration_id=? AND credential_generation=? AND enabled=0""",
                (integration_id, record.credential_generation))
        return self.get(integration_id)

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
        record = self.get(integration_id)
        if record and record.connection:
            raise PermissionError('managed_connection_service_required')
        if enabled and record and record.credential_generation and not record.enabled:
            raise CredentialError("credential_reconnect_required")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE v2_integrations
                SET enabled = ?, updated_at = ?
                WHERE integration_id = ? AND credential_generation = ?
                """,
                (1 if enabled else 0, _now_iso(), str(integration_id or "").strip(),
                 record.credential_generation if record else -1),
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

            conn.execute("BEGIN IMMEDIATE")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(v2_integrations)")}
            for name, definition in (("connection_json", "TEXT NOT NULL DEFAULT '{}'"),
                                     ("credential_ref", "TEXT NOT NULL DEFAULT ''"),
                                     ("credential_generation", "INTEGER NOT NULL DEFAULT 0"),
                                     ("revocation_pending", "INTEGER NOT NULL DEFAULT 0"),
                                     ("credential_deletion_pending", "INTEGER NOT NULL DEFAULT 0")):
                if name not in columns:
                    conn.execute(f"ALTER TABLE v2_integrations ADD COLUMN {name} {definition}")


def _record_from_row(row: sqlite3.Row | None) -> IntegrationConfigRecord | None:
    if row is None:
        return None
    return IntegrationConfigRecord(
        integration_id=str(row["integration_id"]),
        provider_key=str(row["provider_key"]),
        display_name=str(row["display_name"]),
        enabled=bool(row["enabled"]),
        connection=_json_object(row["connection_json"]),
        config=_json_object(row["config_json"]),
        secrets=_json_object(row["secret_json"]),
        credential_ref=str(row["credential_ref"]),
        credential_generation=int(row["credential_generation"]),
        revocation_pending=bool(row["revocation_pending"]),
        credential_deletion_pending=bool(row["credential_deletion_pending"]),
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
