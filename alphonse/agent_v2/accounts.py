"""Member-owned optional accounts and grants over existing integration IDs.

Only trusted session/task code constructs AccountSession. It is never deserialized
from tool arguments or IPC payloads. No credential bytes or provider I/O here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from typing import Any

from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore, _now_iso
from alphonse.agent_v2.users import V2UserStore

GMAIL_CONTENT_OPERATION = 'mail_content_read'
GMAIL_READ_SCOPE = 'https://www.googleapis.com/auth/gmail.readonly'
CALENDAR_WRITE_OPERATION = 'calendar_write'
CALENDAR_WRITE_SCOPE = 'https://www.googleapis.com/auth/calendar.events'
OUTLOOK_INBOX_OPERATION = 'outlook_inbox_unread'
OUTLOOK_INBOX_RESOURCE = 'outlook:inbox'
OUTLOOK_INBOX_SCOPE = 'https://graph.microsoft.com/Mail.ReadBasic'
_GMAIL_SYSTEM_LABEL_IDS = frozenset({
    'INBOX', 'SPAM', 'TRASH', 'UNREAD', 'STARRED', 'IMPORTANT', 'SENT', 'DRAFT',
    'CATEGORY_PERSONAL', 'CATEGORY_SOCIAL', 'CATEGORY_PROMOTIONS', 'CATEGORY_UPDATES', 'CATEGORY_FORUMS',
})
_GMAIL_USER_LABEL_ID = re.compile(r'Label_[A-Za-z0-9_-]{1,120}')


def exact_gmail_label_id(value: object) -> bool:
    # Only the documented system IDs and concrete user label IDs are valid here.
    return isinstance(value, str) and (value in _GMAIL_SYSTEM_LABEL_IDS
                                       or _GMAIL_USER_LABEL_ID.fullmatch(value) is not None)


def valid_gmail_label_name(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 256 and value == value.strip() and value.isprintable()


def exact_calendar_id(value: object) -> bool:
    return (isinstance(value, str) and 0 < len(value) <= 512 and value == value.strip()
            and value.casefold() not in {'primary', '.', '..'} and '*' not in value and value.isprintable())


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


def account_grant_is_live(grant: dict[str, Any], now: datetime) -> bool:
    """Honor standing Gmail content grants while retaining expiry for other grants."""
    if grant['revoked_at']:
        return False
    expires_at = grant['expires_at']
    if expires_at is None:
        return grant['operations'] == [GMAIL_CONTENT_OPERATION]
    return _deadline(expires_at) > now


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
        if connection['schema_version'] != 1:
            raise ValueError('account_schema_unsupported')
        if session.actor_user_id != connection['owner_user_id']:
            raise PermissionError('account_owner_required')

    def _gmail_content_scope(self, record, actor, *, beneficiary, operations, resources, recipients, project_id,
                             gmail_label_name):
        owner = self.users.admin_user()
        return (owner is not None and owner.user_id == actor == beneficiary == record.connection['owner_user_id']
                and owner.is_active and owner.role in ('owner', 'admin')
                and record.provider_key == 'google' and 'mail_read' in record.connection['capabilities']
                and GMAIL_READ_SCOPE in record.connection['granted_scopes'] and not record.revocation_pending
                and operations == [GMAIL_CONTENT_OPERATION] and len(resources) == 1
                and exact_gmail_label_id(resources[0])
                and ((resources == ['INBOX'] and gmail_label_name is None) or valid_gmail_label_name(gmail_label_name))
                and recipients == [actor] and project_id is None)

    def _outlook_inbox_scope(self, record, actor, *, beneficiary, operations, resources, recipients, project_id):
        owner = self.users.admin_user()
        return (owner is not None and owner.user_id == actor == beneficiary == record.connection['owner_user_id']
                and owner.is_active and owner.role in ('owner', 'admin')
                and self.users.is_admin(actor)
                and record.provider_key == 'microsoft' and record.connection['capabilities'] == [OUTLOOK_INBOX_OPERATION]
                and OUTLOOK_INBOX_SCOPE in record.connection['granted_scopes'] and not record.revocation_pending
                and operations == [OUTLOOK_INBOX_OPERATION] and resources == [OUTLOOK_INBOX_RESOURCE]
                and recipients == [actor] and project_id is None)

    def _calendar_write_scope(self, record, actor, *, beneficiary, operations, resources, recipients, project_id):
        owner = self.users.admin_user()
        return (owner is not None and owner.user_id == actor == beneficiary == record.connection['owner_user_id']
                and owner.is_active and owner.role in ('owner', 'admin') and self.users.is_admin(actor)
                and record.provider_key == 'google' and record.connection['capabilities'] == [CALENDAR_WRITE_OPERATION]
                and CALENDAR_WRITE_SCOPE in record.connection['granted_scopes'] and not record.revocation_pending
                and operations == [CALENDAR_WRITE_OPERATION] and len(resources) == 1 and exact_calendar_id(resources[0])
                and recipients == [actor] and project_id is None)

    def create_connection(self, session: AccountSession, *, connection_id: str,
                          provider_key: str, provider_subject: str, display_name: str) -> dict[str, Any]:
        """Reserve a new exact ID with its authenticated owner before consent setup."""
        self._active(session.actor_user_id)
        if not all(isinstance(v, str) and v.strip() for v in
                   (connection_id, provider_key, provider_subject, display_name)):
            raise ValueError('account_connection_invalid')
        now = _now_iso()
        value = dict(schema_version=1, owner_user_id=session.actor_user_id,
                     provider_subject=provider_subject, capabilities=[], granted_scopes=[],
                     status='reauth_required', health_code='consent_required', last_success_at='',
                     revision=1, created_at=now, updated_at=now)
        with self.integrations._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            if conn.execute('SELECT 1 FROM v2_integrations WHERE integration_id=?', (connection_id,)).fetchone():
                raise ValueError('account_connection_exists')
            conn.execute('''INSERT INTO v2_integrations
                (integration_id, provider_key, display_name, enabled, connection_json, created_at, updated_at)
                VALUES (?, ?, ?, 1, ?, ?, ?)''',
                (connection_id, provider_key.strip().lower(), display_name, json.dumps(value, sort_keys=True), now, now))
        return value

    def save_connection(self, session: AccountSession, *, connection_id: str,
                        expected_revision: int, provider_subject: str, capabilities: list[str],
                        granted_scopes: list[str], status: str = 'connected',
                        health_code: str = 'healthy', last_success_at: str = '',
                        reconcile_owner_user_id: str | None = None) -> dict[str, Any]:
        """Explicitly adopt a secret-free integration or revise its managed state.

        Credential provisioning/migration belongs to the secure credential service.
        Existing binding IDs, audiences and role choices are never rewritten.
        """
        self._active(session.actor_user_id)
        record = self.integrations.get(connection_id)
        if record is None:
            raise ValueError('account_connection_missing')
        old = record.connection
        owner = reconcile_owner_user_id or session.actor_user_id
        self._active(owner)
        if old:
            if owner != old['owner_user_id']:
                raise ValueError('account_identity_immutable')
            self._owner(session, old)
            if provider_subject != old['provider_subject']:
                raise ValueError('account_identity_immutable')
        elif not self.users.is_admin(session.actor_user_id):
            raise PermissionError('account_legacy_reconciliation_admin_required')
        elif record.secrets:
            raise ValueError('account_secure_migration_required')
        if (not provider_subject.strip() or status not in {'connected', 'paused', 'reauth_required', 'disconnected'}):
            raise ValueError('account_connection_invalid')
        if last_success_at:
            _deadline(last_success_at)
        if status == 'connected' and (not record.enabled or not record.credential_ref or record.credential_deletion_pending):
            raise ValueError('account_secure_credentials_required')
        revision = old.get('revision', 0)
        if expected_revision != revision:
            raise ValueError('account_revision_conflict')
        now = _now_iso()
        value = dict(schema_version=1, owner_user_id=owner,
                     provider_subject=provider_subject, capabilities=_strings(capabilities),
                     granted_scopes=_strings(granted_scopes), status=status, health_code=health_code,
                     last_success_at=last_success_at,
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
        if status == 'disconnected':
            self.integrations.disconnect_credentials(connection_id, expected_generation=record.credential_generation)
        return value

    def save_grant(self, session: AccountSession, *, grant_id: str, connection_id: str,
                   expected_revision: int, beneficiary_user_id: str, operations: list[str],
                   resources: list[str], recipient_user_ids: list[str], expires_at: str | None,
                   project_id: str | None = None, gmail_label_name: str | None = None,
                   expected_connection_revision: int | None = None) -> dict[str, Any]:
        record = self.integrations.get(connection_id)
        if not record or not record.connection:
            raise ValueError('account_connection_missing')
        self._owner(session, record.connection)
        self._active(beneficiary_user_id)
        if (expected_connection_revision is not None
                and expected_connection_revision != record.connection['revision']):
            raise ValueError('oauth_revision_conflict')
        requested_resource_count = len(resources) if isinstance(resources, (list, tuple)) else 0
        operations, resources, recipients = map(_strings, (operations, resources, recipient_user_ids))
        for recipient in recipients:
            self._active(recipient)
        content_grant = GMAIL_CONTENT_OPERATION in operations
        outlook_grant = OUTLOOK_INBOX_OPERATION in operations
        calendar_write_grant = CALENDAR_WRITE_OPERATION in operations
        compatible_operations = set(operations) <= set(record.connection['capabilities'])
        if content_grant:
            compatible_operations = requested_resource_count == 1 and self._gmail_content_scope(record, session.actor_user_id,
                beneficiary=beneficiary_user_id, operations=operations, resources=resources,
                recipients=recipients, project_id=project_id, gmail_label_name=gmail_label_name)
        if outlook_grant:
            compatible_operations = requested_resource_count == 1 and self._outlook_inbox_scope(
                record, session.actor_user_id, beneficiary=beneficiary_user_id, operations=operations,
                resources=resources, recipients=recipients, project_id=project_id)
        if calendar_write_grant:
            compatible_operations = requested_resource_count == 1 and self._calendar_write_scope(
                record, session.actor_user_id, beneficiary=beneficiary_user_id, operations=operations,
                resources=resources, recipients=recipients, project_id=project_id)
        if (not grant_id.strip() or not operations or not resources
                or not compatible_operations
                or (not content_grant and gmail_label_name is not None)
                or (expires_at is None and not content_grant)
                or (expires_at is not None and _deadline(expires_at) <= datetime.now(timezone.utc))
                or (project_id is not None and not project_id.strip())):
            raise ValueError('account_grant_invalid')
        now = _now_iso()
        with self.integrations._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            current = conn.execute('SELECT enabled, connection_json FROM v2_integrations WHERE integration_id=?', (connection_id,)).fetchone()
            if (not current or not current['enabled'] or json.loads(current['connection_json']) != record.connection
                    or record.connection['status'] != 'connected'):
                raise PermissionError('account_connection_unavailable')
            row = conn.execute('SELECT * FROM v2_account_grants WHERE grant_id=?', (grant_id,)).fetchone()
            old = json.loads(row['definition']) if row else {}
            if old and (old['connection_id'] != connection_id or old['beneficiary_user_id'] != beneficiary_user_id
                        or old['project_id'] != project_id or old['revoked_at']
                        or (GMAIL_CONTENT_OPERATION in old['operations']) != content_grant
                        or (OUTLOOK_INBOX_OPERATION in old['operations']) != outlook_grant
                        or (CALENDAR_WRITE_OPERATION in old['operations']) != calendar_write_grant
                        or ((content_grant or calendar_write_grant) and old['resources'] != resources)):
                raise ValueError('account_grant_identity_immutable')
            if expected_revision != (row['revision'] if row else 0):
                raise ValueError('account_revision_conflict')
            if content_grant and row is None and conn.execute('''SELECT 1 FROM v2_account_grants
                    WHERE connection_id=? AND json_extract(definition, '$.grantor_user_id')=?
                      AND json_extract(definition, '$.operations')=json_array(?)
                      AND json_extract(definition, '$.resources')=json_array(?)
                      AND json_extract(definition, '$.revoked_at')='' LIMIT 1''',
                    (connection_id, session.actor_user_id, GMAIL_CONTENT_OPERATION, resources[0])).fetchone():
                raise ValueError('gmail_label_grant_exists')
            if outlook_grant:
                if not self._outlook_inbox_scope(record, session.actor_user_id,
                        beneficiary=beneficiary_user_id, operations=operations, resources=resources,
                        recipients=recipients, project_id=project_id):
                    raise PermissionError('owner_only_account_scope_unavailable')
                other_grants = conn.execute('''SELECT definition FROM v2_account_grants
                    WHERE grant_id<>? AND json_extract(definition, '$.grantor_user_id')=?
                      AND json_extract(definition, '$.operations')=json_array(?)''',
                    (grant_id, session.actor_user_id, OUTLOOK_INBOX_OPERATION))
                at = datetime.now(timezone.utc)
                if any(account_grant_is_live(json.loads(candidate['definition']), at)
                       for candidate in other_grants):
                    raise ValueError('outlook_inbox_grant_exists')
            value = dict(schema_version=1, grant_id=grant_id, connection_id=connection_id,
                         grantor_user_id=session.actor_user_id, beneficiary_user_id=beneficiary_user_id,
                         project_id=project_id, operations=operations, resources=resources,
                         recipient_user_ids=recipients, expires_at=expires_at, revoked_at='',
                         revision=expected_revision + 1, created_at=old.get('created_at', now), updated_at=now)
            if gmail_label_name is not None:
                value['gmail_label_name'] = gmail_label_name
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
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT * FROM v2_account_grants WHERE grant_id=?', (grant_id,)).fetchone()
            if not row:
                raise ValueError('account_grant_missing')
            grant = json.loads(row['definition'])
            record = conn.execute('SELECT connection_json FROM v2_integrations WHERE integration_id=?',
                                  (grant['connection_id'],)).fetchone()
            if not record or not json.loads(record['connection_json']):
                raise ValueError('account_connection_missing')
            self._owner(session, json.loads(record['connection_json']))
            grant.update(revoked_at=_now_iso(), updated_at=_now_iso(), revision=expected_revision + 1)
            cursor = conn.execute('UPDATE v2_account_grants SET revision=?, definition=? WHERE grant_id=? AND revision=?',
                                 (expected_revision + 1, json.dumps(grant), grant_id, expected_revision))
            if cursor.rowcount != 1:
                raise ValueError('account_revision_conflict')

    def authorize(self, session: AccountSession, *, connection_id: str, grant_id: str,
                  operation: str, resources: list[str], recipient_user_ids: list[str],
                  required_scopes: list[str], project_id: str | None = None,
                  expected_grant_revision: int | None = None, owner_gmail_content: bool = False,
                  owner_outlook_inbox: bool = False, owner_calendar_write: bool = False):
        """Called at each managed use/page; config/project membership grants nothing."""
        self._active(session.actor_user_id)
        record = self.integrations.get(connection_id)
        if not record or not record.enabled or not record.connection:
            raise PermissionError('account_connection_unavailable')
        connection = record.connection
        self._active(connection['owner_user_id'])
        if (connection['schema_version'] != 1 or connection['status'] != 'connected'
                or connection['health_code'] != 'healthy' or not record.credential_ref
                or record.credential_deletion_pending):
            raise PermissionError('account_connection_unavailable')
        with self.integrations._connect() as conn:
            row = conn.execute('SELECT revision, definition FROM v2_account_grants WHERE grant_id=?', (grant_id,)).fetchone()
        if not row:
            raise PermissionError('account_grant_missing')
        grant = json.loads(row['definition'])
        requested_resource_count = len(resources) if isinstance(resources, (list, tuple)) else 0
        resources, recipients, scopes = map(_strings, (resources, recipient_user_ids, required_scopes))
        capable = operation in connection['capabilities']
        if operation == GMAIL_CONTENT_OPERATION:
            capable = (owner_gmail_content is True and scopes == [GMAIL_READ_SCOPE]
                and requested_resource_count == 1 and resources == grant['resources']
                and grant.get('grant_id') == grant_id and grant.get('revision') == row['revision']
                and recipients == [session.actor_user_id] and self._gmail_content_scope(record, session.actor_user_id,
                    beneficiary=grant['beneficiary_user_id'], operations=grant['operations'], resources=grant['resources'],
                    recipients=grant['recipient_user_ids'], project_id=project_id,
                    gmail_label_name=grant.get('gmail_label_name')))
        if operation == OUTLOOK_INBOX_OPERATION:
            capable = (owner_outlook_inbox is True and scopes == [OUTLOOK_INBOX_SCOPE] and requested_resource_count == 1
                and resources == [OUTLOOK_INBOX_RESOURCE] and recipients == [session.actor_user_id]
                and grant.get('grant_id') == grant_id and grant.get('revision') == row['revision']
                and self._outlook_inbox_scope(record, session.actor_user_id,
                    beneficiary=grant['beneficiary_user_id'], operations=grant['operations'],
                    resources=grant['resources'], recipients=grant['recipient_user_ids'],
                    project_id=grant['project_id']))
        if operation == CALENDAR_WRITE_OPERATION:
            capable = (owner_calendar_write is True and scopes == [CALENDAR_WRITE_SCOPE]
                and requested_resource_count == 1 and resources == grant['resources']
                and grant.get('grant_id') == grant_id and grant.get('revision') == row['revision']
                and recipients == [session.actor_user_id] and self._calendar_write_scope(record, session.actor_user_id,
                    beneficiary=grant['beneficiary_user_id'], operations=grant['operations'], resources=grant['resources'],
                    recipients=grant['recipient_user_ids'], project_id=grant['project_id']))
        if (grant['schema_version'] != 1 or grant['connection_id'] != connection_id
                or grant['grantor_user_id'] != connection['owner_user_id']
                or grant['beneficiary_user_id'] != session.actor_user_id or grant['project_id'] != project_id
                or not account_grant_is_live(grant, datetime.now(timezone.utc))
                or (expected_grant_revision is not None and grant['revision'] != expected_grant_revision)
                or operation not in grant['operations'] or not capable
                or not resources or not set(resources) <= set(grant['resources'])
                or not set(recipients) <= set(grant['recipient_user_ids'])
                or not set(scopes) <= set(connection['granted_scopes'])):
            raise PermissionError('account_grant_denied')
        for recipient in recipients:
            self._active(recipient)
        if self.integrations.credential_vault.health(record.credential_ref, connection_id,
                                                      record.credential_generation) != 'available':
            raise PermissionError('account_credentials_unavailable')
        return record, grant
