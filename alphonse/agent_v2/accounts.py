"""Member-owned optional accounts and grants over existing integration IDs.

Only trusted session/task code constructs AccountSession. It is never deserialized
from tool arguments or IPC payloads. No credential bytes or provider I/O here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any

from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore, _now_iso
from alphonse.agent_v2.users import V2UserStore


@dataclass(frozen=True)
class AccountSession:
    actor_user_id: str


@dataclass(frozen=True)
class AccountHandle:
    connection_id: str
    binding_id: str
    grant_id: str
    connection_revision: int
    grant_revision: int
    provider_key: str
    operation: str
    resources: tuple[str, ...]
    recipient_user_ids: tuple[str, ...]


def _strings(values) -> list[str]:
    if not isinstance(values, (tuple, list)) or any(
            not isinstance(v, str) or not v.strip() or v != v.strip() or '*' in v for v in values):
        raise ValueError('account_constraints_invalid')
    return sorted(set(values))


def _deadline(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if result.tzinfo is None:
            raise ValueError()
        return result.astimezone(timezone.utc)
    except (ValueError, TypeError, AttributeError):
        raise ValueError('account_expiry_invalid') from None


class AccountService:
    def __init__(self, integrations: SQLiteIntegrationStore, users: V2UserStore):
        self.integrations, self.users = integrations, users
        with integrations._connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS v2_account_grants (
                grant_id TEXT PRIMARY KEY, connection_id TEXT NOT NULL,
                revision INTEGER NOT NULL, definition TEXT NOT NULL)''')

    def _active(self, user_id: str) -> None:
        user = self.users.get_user(user_id)
        if not user or not user.is_active:
            raise PermissionError('account_principal_inactive')

    def _owner(self, session: AccountSession, connection) -> None:
        self._active(session.actor_user_id)
        self._active(connection['owner_user_id'])
        if session.actor_user_id != connection['owner_user_id']:
            raise PermissionError('account_owner_required')

    def save_connection(self, session: AccountSession, *, connection_id: str,
                        expected_revision: int, provider_subject: str, capabilities: list[str],
                        granted_scopes: list[str], credential_ref: str,
                        credential_generation: int = 1, status: str = 'connected',
                        health_code: str = 'healthy', last_success_at: str = '',
                        revocation_pending: bool = False) -> dict[str, Any]:
        """Explicitly adopt a secret-free integration or revise its managed state.

        Credential provisioning/migration belongs to the secure credential service.
        Existing binding IDs, audiences and role choices are never rewritten.
        """
        self._active(session.actor_user_id)
        record = self.integrations.get(connection_id)
        if record is None:
            raise ValueError('account_connection_missing')
        old = record.connection
        if old:
            self._owner(session, old)
            if provider_subject != old['provider_subject']:
                raise ValueError('account_identity_immutable')
        elif record.secrets:
            raise ValueError('account_secure_migration_required')
        if (not provider_subject.strip() or not credential_ref.strip() or credential_generation < 1
                or status not in {'connected', 'paused', 'reauth_required', 'disconnected'}):
            raise ValueError('account_connection_invalid')
        if old and credential_generation < old['credential_generation']:
            raise ValueError('account_credential_generation_stale')
        if old and credential_ref != old['credential_ref'] and credential_generation <= old['credential_generation']:
            raise ValueError('account_credential_generation_stale')
        revision = old.get('revision', 0)
        if expected_revision != revision:
            raise ValueError('account_revision_conflict')
        now = _now_iso()
        value = dict(schema_version=1, owner_user_id=session.actor_user_id,
                     provider_subject=provider_subject, capabilities=_strings(capabilities),
                     granted_scopes=_strings(granted_scopes), credential_ref=credential_ref,
                     credential_generation=credential_generation, status=status, health_code=health_code,
                     last_success_at=last_success_at, revocation_pending=bool(revocation_pending),
                     revision=revision + 1, created_at=old.get('created_at', now), updated_at=now)
        with self.integrations._connect() as conn:
            cursor = conn.execute('''UPDATE v2_integrations SET connection_json=?, updated_at=?
                WHERE integration_id=? AND connection_json=?''',
                (json.dumps(value, sort_keys=True), now, connection_id, json.dumps(old, sort_keys=True)))
            if cursor.rowcount != 1:
                raise ValueError('account_revision_conflict')
            if status == 'disconnected':
                for row in conn.execute('SELECT * FROM v2_account_grants WHERE connection_id=?', (connection_id,)).fetchall():
                    grant = json.loads(row['definition'])
                    if not grant['revoked_at']:
                        grant.update(revoked_at=now, updated_at=now, revision=row['revision'] + 1)
                        conn.execute('UPDATE v2_account_grants SET definition=?, revision=? WHERE grant_id=?',
                                     (json.dumps(grant), grant['revision'], row['grant_id']))
        return value

    def save_grant(self, session: AccountSession, *, grant_id: str, connection_id: str,
                   expected_revision: int, beneficiary_user_id: str, operations: list[str],
                   resources: list[str], recipient_user_ids: list[str], expires_at: str,
                   project_id: str | None = None) -> dict[str, Any]:
        record = self.integrations.get(connection_id)
        if not record or not record.connection:
            raise ValueError('account_connection_missing')
        self._owner(session, record.connection)
        self._active(beneficiary_user_id)
        operations, resources, recipients = map(_strings, (operations, resources, recipient_user_ids))
        for recipient in recipients:
            self._active(recipient)
        if (not grant_id.strip() or not operations or not resources
                or not set(operations) <= set(record.connection['capabilities'])
                or _deadline(expires_at) <= datetime.now(timezone.utc)
                or (project_id is not None and not project_id.strip())):
            raise ValueError('account_grant_invalid')
        now = _now_iso()
        with self.integrations._connect() as conn:
            row = conn.execute('SELECT * FROM v2_account_grants WHERE grant_id=?', (grant_id,)).fetchone()
            old = json.loads(row['definition']) if row else {}
            if old and (old['connection_id'] != connection_id or old['beneficiary_user_id'] != beneficiary_user_id
                        or old['project_id'] != project_id or old['revoked_at']):
                raise ValueError('account_grant_identity_immutable')
            if expected_revision != (row['revision'] if row else 0):
                raise ValueError('account_revision_conflict')
            value = dict(schema_version=1, grant_id=grant_id, connection_id=connection_id,
                         grantor_user_id=session.actor_user_id, beneficiary_user_id=beneficiary_user_id,
                         project_id=project_id, operations=operations, resources=resources,
                         recipient_user_ids=recipients, expires_at=expires_at, revoked_at='',
                         revision=expected_revision + 1, created_at=old.get('created_at', now), updated_at=now)
            if row:
                cursor = conn.execute('UPDATE v2_account_grants SET revision=?, definition=? WHERE grant_id=? AND revision=?',
                    (value['revision'], json.dumps(value), grant_id, expected_revision))
                if cursor.rowcount != 1:
                    raise ValueError('account_revision_conflict')
            else:
                conn.execute('INSERT INTO v2_account_grants VALUES (?, ?, ?, ?)',
                             (grant_id, connection_id, value['revision'], json.dumps(value)))
        return value

    def revoke_grant(self, session: AccountSession, *, grant_id: str, expected_revision: int) -> None:
        with self.integrations._connect() as conn:
            row = conn.execute('SELECT * FROM v2_account_grants WHERE grant_id=?', (grant_id,)).fetchone()
            if not row:
                raise ValueError('account_grant_missing')
            grant = json.loads(row['definition'])
            record = self.integrations.get(grant['connection_id'])
            if not record or not record.connection:
                raise ValueError('account_connection_missing')
            self._owner(session, record.connection)
            grant.update(revoked_at=_now_iso(), updated_at=_now_iso(), revision=expected_revision + 1)
            cursor = conn.execute('UPDATE v2_account_grants SET revision=?, definition=? WHERE grant_id=? AND revision=?',
                                 (expected_revision + 1, json.dumps(grant), grant_id, expected_revision))
            if cursor.rowcount != 1:
                raise ValueError('account_revision_conflict')

    def authorize(self, session: AccountSession, *, connection_id: str, grant_id: str,
                  operation: str, resources: list[str], recipient_user_ids: list[str],
                  required_scopes: list[str], project_id: str | None = None,
                  expected_grant_revision: int | None = None):
        """Called at each managed use/page; config/project membership grants nothing."""
        self._active(session.actor_user_id)
        record = self.integrations.get(connection_id)
        if not record or not record.enabled or not record.connection:
            raise PermissionError('account_connection_unavailable')
        connection = record.connection
        self._active(connection['owner_user_id'])
        if (connection['schema_version'] != 1 or connection['status'] != 'connected'
                or connection['health_code'] != 'healthy' or not connection['credential_ref']):
            raise PermissionError('account_connection_unavailable')
        with self.integrations._connect() as conn:
            row = conn.execute('SELECT definition FROM v2_account_grants WHERE grant_id=?', (grant_id,)).fetchone()
        if not row:
            raise PermissionError('account_grant_missing')
        grant = json.loads(row['definition'])
        resources, recipients, scopes = map(_strings, (resources, recipient_user_ids, required_scopes))
        if (grant['schema_version'] != 1 or grant['connection_id'] != connection_id
                or grant['grantor_user_id'] != connection['owner_user_id']
                or grant['beneficiary_user_id'] != session.actor_user_id or grant['project_id'] != project_id
                or grant['revoked_at'] or _deadline(grant['expires_at']) <= datetime.now(timezone.utc)
                or (expected_grant_revision is not None and grant['revision'] != expected_grant_revision)
                or operation not in grant['operations'] or operation not in connection['capabilities']
                or not resources or not set(resources) <= set(grant['resources'])
                or not set(recipients) <= set(grant['recipient_user_ids'])
                or not set(scopes) <= set(connection['granted_scopes'])):
            raise PermissionError('account_grant_denied')
        for recipient in recipients:
            self._active(recipient)
        return record, grant
