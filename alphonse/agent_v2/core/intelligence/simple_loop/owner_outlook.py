"""An owner Desktop read of the connected Microsoft account's Inbox metadata."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from alphonse.agent_v2.accounts import (
    AccountSession, OUTLOOK_INBOX_OPERATION, OUTLOOK_INBOX_RESOURCE, OUTLOOK_INBOX_SCOPE,
    account_grant_is_live,
)
from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.tools.registry import ToolDefinition
from alphonse.agent_v2.integrations.graph_mail import ROOT, GraphHTTP
from alphonse.agent_v2.integrations.oauth import OAuthError
from alphonse.agent_v2.services.project_check_worker import CheckStopped

OWNER_OUTLOOK_TOOL_ID = 'native.owner_outlook_unread'
_URL = ROOT + 'inbox?$select=id,unreadItemCount'
_MAX_RESPONSE_BYTES = 8192
_SCHEMA = {'type': 'object', 'properties': {}, 'additionalProperties': False}
_FAILURES = {
    'inbox_access_required': ('To check Outlook Inbox, open Settings → Integrations → Outlook / Hotmail. '
                              'Connect your account and grant Inbox access.'),
    'account_ambiguous': ('More than one Outlook account has Inbox access. In Settings → Integrations → '
                          'Outlook / Hotmail, keep Inbox access for the account you want to use.'),
    'paused': 'Outlook access is paused. Reconnect in Settings → Integrations → Outlook / Hotmail and try again.',
    'refresh_required': 'Refresh Outlook Inbox access in Settings → Integrations → Outlook / Hotmail and try again.',
    'reconnect_required': 'Reconnect Outlook in Settings → Integrations → Outlook / Hotmail, then try again.',
    'provider_denied': 'Microsoft denied the Outlook Inbox check. Review access in Outlook / Hotmail settings.',
    'provider_busy': 'Microsoft is temporarily limiting Outlook Inbox checks. Try again in a moment.',
    'read_unavailable': 'The Outlook Inbox unread count could not be verified. Try the check again.',
}


def _failure(reason):
    return {'schema_version': 1, 'status': 'unavailable', 'provider': 'outlook', 'reason': reason}


def owner_outlook_failure_message(payload):
    if (isinstance(payload, dict) and set(payload) == {'schema_version', 'status', 'provider', 'reason'}
            and type(payload['schema_version']) is int and payload['schema_version'] == 1
            and payload['status'] == 'unavailable' and payload['provider'] == 'outlook'
            and isinstance(payload['reason'], str)):
        return _FAILURES.get(payload['reason'])
    return None


def _binding(conn, actor, now):
    """Select one exact self-grant, using the same query during locked pickup."""
    matches = []
    for row in conn.execute('''SELECT g.grant_id, g.connection_id, g.revision, g.definition,
            i.enabled, i.connection_json, i.credential_ref, i.credential_generation,
            i.revocation_pending, i.credential_deletion_pending
            FROM v2_account_grants g
            JOIN v2_integrations i ON i.integration_id = g.connection_id
            JOIN v2_users u ON u.user_id = ? AND u.is_active = 1 AND u.role IN ('admin', 'owner')
            JOIN v2_user_settings s ON s.key = 'admin_user_id' AND s.value = u.user_id
            WHERE i.provider_key = 'microsoft' ORDER BY g.grant_id''', (actor,)):
        try:
            grant, connection = json.loads(row['definition']), json.loads(row['connection_json'])
            if (grant['beneficiary_user_id'] != actor or grant['grantor_user_id'] != actor
                    or grant['recipient_user_ids'] != [actor] or grant['project_id'] is not None
                    or grant['schema_version'] != 1 or grant['grant_id'] != row['grant_id']
                    or grant['connection_id'] != row['connection_id'] or grant['revision'] != row['revision']
                    or grant['operations'] != [OUTLOOK_INBOX_OPERATION] or grant['resources'] != [OUTLOOK_INBOX_RESOURCE]
                    or not account_grant_is_live(grant, now)
                    or connection['owner_user_id'] != actor or connection['schema_version'] != 1
                    or connection['status'] == 'disconnected'
                    or connection['capabilities'] != [OUTLOOK_INBOX_OPERATION]
                    or OUTLOOK_INBOX_SCOPE not in connection['granted_scopes']):
                continue
            status = connection['status']
            if status == 'paused':
                reason = 'paused'
            elif status == 'reauth_required':
                reason = 'reconnect_required'
            elif (status != 'connected' or connection['health_code'] != 'healthy' or row['enabled'] != 1
                    or not row['credential_ref'] or row['credential_generation'] < 1
                    or row['revocation_pending'] or row['credential_deletion_pending']):
                reason = 'read_unavailable'
            else:
                reason = None
            matches.append(({
                'connection_id': row['connection_id'], 'connection_revision': connection['revision'],
                'credential_generation': row['credential_generation'], 'grant_id': row['grant_id'],
                'grant_revision': row['revision'], 'resource': OUTLOOK_INBOX_RESOURCE,
            }, reason))
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    if not matches:
        return None, 'inbox_access_required'
    if len(matches) != 1:
        return None, 'account_ambiguous'
    binding, reason = matches[0]
    return (binding, None) if reason is None else (None, reason)


def _authorization(actor, binding):
    return dict(connection_id=binding['connection_id'], grant_id=binding['grant_id'],
                operation=OUTLOOK_INBOX_OPERATION, resources=[OUTLOOK_INBOX_RESOURCE],
                recipient_user_ids=[actor], required_scopes=[OUTLOOK_INBOX_SCOPE],
                expected_grant_revision=binding['grant_revision'], owner_outlook_inbox=True)


def _valid_folder_id(value):
    return isinstance(value, str) and 0 < len(value) <= 1024 and value.strip() == value and value.isprintable()


class OwnerOutlookService:
    def __init__(self, accounts, oauth_factory, *, transport=None, clock=None):
        self.accounts = accounts
        self.oauth_factory = oauth_factory
        self.transport = transport or GraphHTTP()
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def binding(self, actor):
        with self.accounts.integrations._connect() as conn:
            return _binding(conn, actor, self.clock())

    def authorized_record(self, actor, binding):
        record, _ = self.accounts.authorize(AccountSession(actor), **_authorization(actor, binding))
        if (record.provider_key != 'microsoft' or record.connection['revision'] != binding['connection_revision']
                or record.credential_generation != binding['credential_generation']):
            raise PermissionError('owner_outlook_authority_changed')
        return record

    def observe(self, actor, *, guard):
        guard()
        binding, reason = self.binding(actor)
        guard()
        if binding is None:
            return _failure(reason)

        def check():
            guard()
            if self.binding(actor)[0] != binding:
                raise CheckStopped('blocked', 'owner_outlook_authority_changed')
            record = self.authorized_record(actor, binding)
            guard()
            return record

        session = AccountSession(actor)
        oauth = self.oauth_factory(session)
        try:
            record = check()
        except PermissionError as exc:
            if exc.args == ('account_credentials_unavailable',):
                guard()
                return _failure('reconnect_required')
            raise
        try:
            token = oauth.token_without_refresh(session, authorization=_authorization(actor, binding))
        except OAuthError as exc:
            check()
            reason = 'refresh_required' if exc.args == ('oauth_refresh_required',) else 'read_unavailable'
            return _failure(reason)
        check()
        deadline = self.clock() + timedelta(seconds=15)
        try:
            status, _, raw = self.transport.get(_URL, token=token, deadline=deadline,
                max_bytes=_MAX_RESPONSE_BYTES, headers={'Accept': 'application/json'})
        except (CheckStopped, ImportError, OSError, ValueError):
            check()
            return _failure('read_unavailable')
        check()
        if self.clock() >= deadline:
            return _failure('read_unavailable')
        if status == 401:
            if not oauth.require_outlook_reconnect(session, record,
                    grant_id=binding['grant_id'], grant_revision=binding['grant_revision']):
                raise CheckStopped('blocked', 'owner_outlook_authority_changed')
            guard()
            return _failure('reconnect_required')
        if status in (429, 503):
            return _failure('provider_busy')
        if status == 403:
            return _failure('provider_denied')
        if status != 200 or not isinstance(raw, bytes) or len(raw) > _MAX_RESPONSE_BYTES:
            return _failure('read_unavailable')
        try:
            data = json.loads(raw)
        except (ValueError, UnicodeError):
            return _failure('read_unavailable')
        if not isinstance(data, dict):
            return _failure('read_unavailable')
        folder_id, count = data.get('id'), data.get('unreadItemCount')
        if not _valid_folder_id(folder_id) or type(count) is not int or not 0 <= count <= 2**31 - 1:
            return _failure('read_unavailable')
        check()
        return {'schema_version': 1, 'status': 'complete', 'provider': 'outlook',
                'resource': OUTLOOK_INBOX_RESOURCE, 'folder_id': folder_id, 'count': count,
                'observed_at': self.clock().isoformat(), 'authority': binding}


def _same_database(context, service):
    return (isinstance(service, OwnerOutlookService) and type(context.messages) is SQLiteMessageQueue
            and os.path.samefile(context.messages.db_path, service.accounts.integrations.db_path)
            and os.path.samefile(context.messages.db_path, service.accounts.users.db_path))


def _authorized(actor, context):
    return (bool(actor) and actor == context.task.user and context.cancellation_checker() is False
            and _same_database(context, context.owner_outlook_service)
            and context.messages.verify_trusted_local_owner_claim(context.claim_receipt, context.task))


def execute_owner_outlook(arguments, *, context):
    if not isinstance(arguments, dict) or arguments:
        raise ValueError('owner_outlook_arguments_invalid')
    actor = context.task.user

    def guard():
        if not _authorized(actor, context):
            raise CheckStopped('cancelled', 'owner_outlook_claim_unavailable')

    guard()
    return context.owner_outlook_service.observe(actor, guard=guard)


def _valid_payload(arguments, payload, now):
    if not isinstance(arguments, dict) or arguments:
        return False
    if owner_outlook_failure_message(payload):
        return True
    if (not isinstance(payload, dict) or set(payload) != {
            'schema_version', 'status', 'provider', 'resource', 'folder_id', 'count', 'observed_at', 'authority'}
            or type(payload['schema_version']) is not int or payload['schema_version'] != 1
            or payload['status'] != 'complete' or payload['provider'] != 'outlook'
            or payload['resource'] != OUTLOOK_INBOX_RESOURCE or not _valid_folder_id(payload['folder_id'])
            or type(payload['count']) is not int or not 0 <= payload['count'] <= 2**31 - 1
            or not isinstance(payload['authority'], dict)):
        return False
    observed = datetime.fromisoformat(payload['observed_at'])
    return observed.tzinfo is not None and timedelta(0) <= now - observed <= timedelta(minutes=5)


def valid_owner_outlook(actor, arguments, payload, *, context):
    try:
        if not _authorized(actor, context) or not _valid_payload(arguments, payload, datetime.now(timezone.utc)):
            return False
        if owner_outlook_failure_message(payload):
            return True
        service = context.owner_outlook_service
        binding, _ = service.binding(actor)
        if binding != payload['authority']:
            return False
        service.authorized_record(actor, binding)
        return _authorized(actor, context)
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        return False


def valid_owner_outlook_locked(conn, actor, arguments, payload, *, context):
    try:
        now = datetime.now(timezone.utc)
        return (_same_database(context, context.owner_outlook_service) and _valid_payload(arguments, payload, now)
                and (owner_outlook_failure_message(payload) is not None or _binding(conn, actor, now)[0] == payload['authority']))
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        return False


def outlook_display_locale(value):
    """The model may supply an untrusted presentation hint, never visible copy."""
    if isinstance(value, str) and re.fullmatch(r'[a-zA-Z]{2,8}(?:-[a-zA-Z0-9]{1,8}){0,4}', value):
        return value
    return None


def _display_zone(timezone_name):
    try:
        return ZoneInfo(timezone_name)
    except (TypeError, ZoneInfoNotFoundError, ValueError):
        return ZoneInfo('UTC')


def outlook_inbox_result(payload, *, locale_hint=None, timezone_name=None):
    result = {'count': payload['count'], 'observed_at': payload['observed_at'],
              'timezone': _display_zone(timezone_name).key}
    locale = outlook_display_locale(locale_hint)
    if locale is not None:
        result['locale'] = locale
    return result


def format_outlook_observation(payload, *, private_delivery=False, timezone_name=None):
    failure = owner_outlook_failure_message(payload)
    if failure is not None:
        return failure
    if not private_delivery:
        return 'The Outlook Inbox unread count was displayed privately in Desktop.'
    observed = datetime.fromisoformat(payload['observed_at']).astimezone(_display_zone(timezone_name))
    label = 'unread message' if payload['count'] == 1 else 'unread messages'
    return f"Outlook Inbox has **{payload['count']} {label}**. Checked {observed:%b %d, %Y at %H:%M %Z}."


def build_owner_outlook_tool_definition():
    descriptor = ToolDescriptor(OWNER_OUTLOOK_TOOL_ID, 'owner_outlook_unread', ToolKind.NATIVE,
        'Check how many unread messages are in the owner’s authorized Outlook.com or Hotmail Inbox. '
        'Use only when asked for an unread Inbox count. No message content, other folders, account changes, or scheduled checks. '
        'The host delivers the verified result privately in Desktop.',
        _SCHEMA, (OUTLOOK_INBOX_OPERATION,), ('native', 'outlook'), read_only=True,
        metadata={'sensitive_content': True})
    return ToolDefinition(descriptor, execute_owner_outlook, _SCHEMA, accepts_context=True)
