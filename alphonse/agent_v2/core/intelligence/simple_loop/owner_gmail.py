"""One owner-scoped Gmail INBOX count with offline pickup reauthorization."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from typing import Any

from alphonse.agent_v2.accounts import AccountSession, GMAIL_CONTENT_OPERATION
from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.tools.registry import ToolDefinition
from alphonse.agent_v2.integrations.gmail import ALL_MAIL, SCOPE
from alphonse.agent_v2.integrations.gmail_unread import GmailUnreadService
from alphonse.agent_v2.services.project_check_worker import CheckStopped
from .owner_direct_claim import verify_owner_execution_claim

OWNER_GMAIL_TOOL_ID = 'native.owner_gmail_unread'
_FAILURE_MESSAGES = {
    'inbox_access_required': ('I couldn’t check Gmail. In Settings → Integrations → Gmail, grant Inbox access '
                              'for unread counts.'),
    'account_ambiguous': ('I couldn’t check Gmail because multiple Gmail accounts have Inbox access. '
                          'Keep Inbox access enabled for the account you want to use.'),
    'renewal_required': ('I couldn’t check Gmail. In Settings → Integrations → Gmail, refresh access or '
                         'reconnect your account, then try again.'),
    'reconnect_required': ('I couldn’t check Gmail. Reconnect your Gmail account in Settings → Integrations '
                           'and try again.'),
    'read_unavailable': ('I couldn’t verify the Gmail Inbox unread count. Check the account and Inbox '
                         'access in Settings → Integrations and try again.'),
}
_SCHEMA = {'type': 'object', 'properties': {
    'unit': {'type': 'string', 'enum': ['messages', 'threads'], 'default': 'messages',
             'description': 'Count unread messages by default; use threads only when asked about conversations or threads.'},
}, 'additionalProperties': False}


def owner_gmail_unit(arguments: Any) -> str:
    """Validate the count unit without accepting model-supplied authority."""
    if (not isinstance(arguments, dict) or set(arguments) - {'unit'}
            or arguments.get('unit', 'messages') not in ('messages', 'threads')):
        raise ValueError('owner_gmail_arguments_invalid')
    return arguments.get('unit', 'messages')


class _NoRefreshOAuth:
    def __init__(self, service):
        self.service, self.accounts = service, service.accounts

    def token(self, session, *, authorization):
        return self.service.token_without_refresh(session, authorization=authorization)


def _failure(reason: str) -> dict[str, Any]:
    return {'schema_version': 1, 'status': 'unavailable', 'provider': 'gmail', 'reason': reason}


def owner_gmail_failure_message(payload: Any) -> str | None:
    if (isinstance(payload, dict) and set(payload) == {'schema_version', 'status', 'provider', 'reason'}
            and type(payload['schema_version']) is int and payload['schema_version'] == 1
            and payload['status'] == 'unavailable' and payload['provider'] == 'gmail'
            and isinstance(payload['reason'], str)):
        return _FAILURE_MESSAGES.get(payload['reason'])
    return None


class OwnerGmailService:
    """Bounded label count; never activates account selection or token refresh."""
    def __init__(self, accounts, oauth_factory, *, transport=None, clock=None):
        self.accounts = accounts
        self.reader = GmailUnreadService(accounts, lambda session: _NoRefreshOAuth(oauth_factory(session)),
                                         transport=transport, clock=clock)

    def binding(self, actor: str) -> dict[str, Any]:
        candidates = self.reader._candidates(AccountSession(actor), 'INBOX', None)
        accounts = {row[0].integration_id for row in candidates}
        if not accounts:
            raise PermissionError('owner_gmail_account_unavailable')
        if len(accounts) > 1:
            raise PermissionError('owner_gmail_account_ambiguous')
        record, grant, resource = sorted(candidates, key=lambda row: (row[2] == ALL_MAIL, row[1]['grant_id']))[0]
        return {'connection_id': record.integration_id, 'connection_revision': record.connection['revision'],
                'credential_generation': record.credential_generation,
                'grant_id': grant['grant_id'], 'grant_revision': grant['revision'], 'resource': resource}

    def observe(self, actor: str, *, guard, unit: str = 'messages') -> dict[str, Any]:
        guard()
        try:
            before = self.binding(actor)
        except PermissionError as exc:
            guard()
            if exc.args == ('owner_gmail_account_unavailable',):
                return _failure('inbox_access_required')
            if exc.args == ('owner_gmail_account_ambiguous',):
                return _failure('account_ambiguous')
            raise

        def recheck():
            guard()
            if self.binding(actor) != before:
                raise CheckStopped('blocked', 'owner_gmail_authority_changed')
        recheck()
        result = self.reader.count(AccountSession(actor), label_id='INBOX', unit=unit, guard=recheck)
        recheck()
        if result.get('status') in ('blocked', 'partial', 'unknown'):
            return _failure({'gmail_reauth_required': 'reconnect_required', 'gmail_access_expired': 'renewal_required'}
                            .get(result.get('reason'), 'read_unavailable'))
        if (result.get('status') != 'complete' or result.get('connection_id') != before['connection_id']
                or result.get('label_id') != 'INBOX' or result.get('unit') != unit):
            raise ValueError('owner_gmail_unavailable')
        return {'schema_version': 1, 'status': 'complete', 'provider': 'gmail', 'label_id': 'INBOX',
                'unit': unit, 'count': result['count'], 'observed_at': result['observed_at'],
                'authority': before}


def _authorized(actor: str, context: Any) -> bool:
    task = getattr(context, 'task', None)
    queue = getattr(context, 'messages', None)
    cancelled = getattr(context, 'cancellation_checker', None)
    return bool(actor and actor == getattr(task, 'user', None) and type(queue) is SQLiteMessageQueue
                and callable(cancelled) and cancelled() is False and getattr(task, 'status', '') != 'cancelled'
                and verify_owner_execution_claim(queue, getattr(context, 'claim_receipt', None), task))


def execute_owner_gmail(arguments: dict[str, Any], *, context: Any) -> dict[str, Any]:
    actor = str(getattr(getattr(context, 'task', None), 'user', '') or '')
    service = getattr(context, 'owner_gmail_service', None)
    unit = owner_gmail_unit(arguments)
    if not isinstance(service, OwnerGmailService) or not _authorized(actor, context):
        raise PermissionError('owner_gmail_unavailable')
    # Refuse before provider I/O if pickup cannot later fence this exact
    # account authority within the same claim-bound database transaction.
    try:
        if (not os.path.samefile(context.messages.db_path, service.accounts.integrations.db_path)
                or not os.path.samefile(context.messages.db_path, service.accounts.users.db_path)):
            raise PermissionError('owner_gmail_unavailable')
    except OSError:
        raise PermissionError('owner_gmail_unavailable') from None
    def guard():
        if not _authorized(actor, context):
            raise CheckStopped('cancelled', 'owner_gmail_claim_unavailable')
    result = service.observe(actor, guard=guard, unit=unit)
    if result.get('reason') == 'inbox_access_required':
        from .owner_gmail_search import count_with_inbox_content_grant
        return count_with_inbox_content_grant(actor, unit, context=context, otherwise=result)
    return result


def valid_owner_gmail(actor: str, arguments: dict[str, Any], payload: dict[str, Any], *, context: Any) -> bool:
    """Recheck authority and freshness without recontacting Gmail or refreshing credentials."""
    try:
        service = getattr(context, 'owner_gmail_service', None)
        if not isinstance(service, OwnerGmailService) or not _authorized(actor, context):
            return False
        if owner_gmail_failure_message(payload) is not None:
            owner_gmail_unit(arguments)
            return True
        if (not isinstance(payload, dict) or set(payload) != {'schema_version', 'status', 'provider',
                    'label_id', 'unit', 'count', 'observed_at', 'authority'}
                or type(payload['schema_version']) is not int or payload['schema_version'] != 1
                or payload['status'] != 'complete' or payload['provider'] != 'gmail'
                or payload['label_id'] != 'INBOX' or payload['unit'] != owner_gmail_unit(arguments)
                or type(payload['count']) is not int or payload['count'] < 0
                or not isinstance(payload['observed_at'], str)):
            return False
        observed = datetime.fromisoformat(payload['observed_at'])
        now = datetime.now(timezone.utc)
        if observed.tzinfo is None or not timedelta(0) <= now - observed <= timedelta(minutes=5):
            return False
        authority = payload['authority']
        if isinstance(authority, dict) and authority.get('operation') == GMAIL_CONTENT_OPERATION:
            from .owner_gmail_search import valid_content_grant_count
            return valid_content_grant_count(actor, authority, context=context)
        return service.binding(actor) == authority and _authorized(actor, context)
    except Exception:
        return False


def valid_owner_gmail_locked(conn: Any, actor: str, payload: dict[str, Any], *, context: Any,
                            desktop_pickup: bool = False) -> bool:
    """Fence the DB-backed grant and connection in the owner pickup transaction.

    Credential-vault state is external to SQLite and is checked immediately
    before this transaction by valid_owner_gmail, not locked by this method.
    """
    try:
        service = context.owner_gmail_service
        queue = context.messages
        if (not isinstance(service, OwnerGmailService) or type(queue) is not SQLiteMessageQueue
                or not os.path.samefile(queue.db_path, service.accounts.integrations.db_path)
                or not os.path.samefile(queue.db_path, service.accounts.users.db_path)):
            return False
        actor_row = conn.execute('SELECT is_active FROM v2_users WHERE user_id=?', (actor,)).fetchone()
        if actor_row is None or actor_row['is_active'] != 1:
            return False
        if owner_gmail_failure_message(payload) is not None:
            return True
        if (not isinstance(payload, dict) or payload.get('status') != 'complete'
                or payload.get('provider') != 'gmail' or payload.get('label_id') != 'INBOX'
                or payload.get('unit') not in ('messages', 'threads') or type(payload.get('count')) is not int
                or payload['count'] < 0 or not isinstance(payload.get('authority'), dict)):
            return False
        authority = payload['authority']
        observed = datetime.fromisoformat(payload['observed_at'])
        now = datetime.now(timezone.utc)
        if observed.tzinfo is None or not timedelta(0) <= now - observed <= timedelta(minutes=5):
            return False
        if authority.get('operation') == GMAIL_CONTENT_OPERATION:
            from .owner_gmail_search import valid_content_grant_count_locked
            return desktop_pickup and valid_content_grant_count_locked(conn, actor, authority, context=context)
        if (set(authority) != {'connection_id', 'connection_revision', 'credential_generation',
                               'grant_id', 'grant_revision', 'resource'}
                or any(type(authority.get(key)) is not int or authority[key] < 1
                       for key in ('connection_revision', 'credential_generation', 'grant_revision'))
                or any(not isinstance(authority.get(key), str) or not authority[key]
                       for key in ('connection_id', 'grant_id', 'resource'))):
            return False
        eligible = []
        rows = conn.execute('''SELECT g.grant_id,g.connection_id,g.revision,g.definition,
            i.provider_key,i.enabled,i.connection_json,i.credential_ref,i.credential_generation,
            i.revocation_pending,i.credential_deletion_pending
            FROM v2_account_grants g JOIN v2_integrations i ON i.integration_id=g.connection_id''').fetchall()
        for row in rows:
            try:
                grant, connection = json.loads(row['definition']), json.loads(row['connection_json'])
                if (grant.get('beneficiary_user_id') != actor or grant.get('project_id') is not None
                        or grant.get('revoked_at') or grant.get('schema_version') != 1
                        or grant.get('grant_id') != row['grant_id'] or grant.get('connection_id') != row['connection_id']
                        or type(grant.get('revision')) is not int or grant['revision'] != row['revision']
                        or not isinstance(grant.get('operations'), list) or not isinstance(grant.get('resources'), list)
                        or 'mail_read' not in grant.get('operations', [])
                        or row['provider_key'] != 'google' or row['enabled'] != 1
                        or not row['credential_ref'] or row['credential_generation'] < 1
                        or row['revocation_pending'] or row['credential_deletion_pending']
                        or connection.get('schema_version') != 1 or connection.get('status') != 'connected'
                        or type(connection.get('revision')) is not int or connection['revision'] < 1
                        or not isinstance(connection.get('capabilities'), list)
                        or not isinstance(connection.get('granted_scopes'), list)
                        or connection.get('health_code') != 'healthy' or 'mail_read' not in connection.get('capabilities', [])
                        or SCOPE not in connection.get('granted_scopes', [])
                        or connection.get('owner_user_id') != grant.get('grantor_user_id')):
                    continue
                expires = datetime.fromisoformat(grant['expires_at'].replace('Z', '+00:00'))
                if expires.tzinfo is None or expires <= now:
                    continue
                owner = conn.execute('SELECT is_active FROM v2_users WHERE user_id=?',
                                     (connection['owner_user_id'],)).fetchone()
                if owner is None or owner['is_active'] != 1:
                    continue
                resources = grant.get('resources', [])
                resource = 'INBOX' if 'INBOX' in resources else ALL_MAIL if ALL_MAIL in resources else None
                if resource is None:
                    continue
                eligible.append({'connection_id': row['connection_id'], 'connection_revision': connection['revision'],
                                 'credential_generation': row['credential_generation'],
                                 'grant_id': row['grant_id'], 'grant_revision': grant['revision'], 'resource': resource})
            except Exception:
                return False
        if len({item['connection_id'] for item in eligible}) != 1:
            return False
        selected = sorted(eligible, key=lambda item: (item['resource'] == ALL_MAIL, item['grant_id']))[0]
        return selected == payload['authority']
    except Exception:
        return False


def build_owner_gmail_tool_definition() -> ToolDefinition:
    descriptor = ToolDescriptor(OWNER_GMAIL_TOOL_ID, 'owner_gmail_unread', ToolKind.NATIVE,
        'Read the current unread message or thread count in the owner’s Gmail INBOX when exactly one authorized account exists. '
        'No message contents, new grants, account selection, credential refresh, or mailbox mutation. '
        'Report the observed time; inaccessible or ambiguous accounts are unavailable.',
        _SCHEMA, ('mail_read',), ('native', 'gmail'), read_only=True)
    return ToolDefinition(descriptor, execute_owner_gmail, _SCHEMA, accepts_context=True)
