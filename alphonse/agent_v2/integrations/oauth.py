"""Opt-in managed OAuth. No imports of provider libraries or network work at startup.

Public methods require a trusted AccountSession, never a payload-supplied actor.
Only managed executors may consume token(); it is not an IPC/tool response.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import base64
import json
import logging
import math
import re
import secrets
import threading
import time
from urllib.parse import parse_qs, urlsplit, urlencode

from alphonse.agent_v2.accounts import (
    AccountService, AccountSession, CALENDAR_WRITE_OPERATION, CALENDAR_WRITE_SCOPE,
    GMAIL_CONTENT_OPERATION, GMAIL_READ_SCOPE,
    OUTLOOK_INBOX_OPERATION, OUTLOOK_INBOX_RESOURCE, OUTLOOK_INBOX_SCOPE, account_grant_is_live,
    exact_gmail_label_id, valid_gmail_label_name,
)


class OAuthError(ValueError):
    """Static safe code: never include provider errors, URLs or token responses."""


def _exchange_error_code(exc: Exception) -> str:
    """Keep only a small allowlist of OAuth protocol error categories."""
    try:
        from oauthlib.oauth2 import OAuth2Error
        if isinstance(exc, OAuth2Error):
            category = getattr(exc, 'error', None)
            if type(category) is str and category in {
                    'invalid_request', 'invalid_client', 'invalid_grant',
                    'unauthorized_client', 'access_denied', 'temporarily_unavailable'}:
                return 'oauth_exchange_' + category
    except ImportError:
        pass
    return 'oauth_exchange_failed'


def safe_callback_message(exc: Exception) -> str:
    """Return fixed text only; never echo an exception or provider description."""
    generic = 'Authorization could not be completed. Return to Alphonse and reconnect.'
    if not isinstance(exc, OAuthError) or len(exc.args) != 1 or type(exc.args[0]) is not str:
        return generic
    messages = {
        'oauth_exchange_invalid_request': 'The provider rejected the token request (invalid_request). Check the OAuth client configuration, then reconnect.',
        'oauth_exchange_invalid_client': 'The provider rejected the OAuth client (invalid_client). Check the OAuth client configuration, then reconnect.',
        'oauth_exchange_invalid_grant': 'The provider rejected the authorization grant (invalid_grant). Start a new connection and try again.',
        'oauth_exchange_unauthorized_client': 'The provider rejected this client for authorization (unauthorized_client). Check the OAuth client configuration.',
        'oauth_exchange_access_denied': 'The provider declined the token request (access_denied). Start a new connection and try again.',
        'oauth_exchange_temporarily_unavailable': 'The provider is temporarily unavailable. Start a new connection and try again.',
        'oauth_exchange_failed': 'The token exchange failed. Check the connection and OAuth client configuration, then reconnect.',
        'oauth_client_credential_unavailable': 'The configured Google client credential is unavailable. Check credential storage and reconnect.',
        'oauth_identity_failed': 'The provider account identity could not be checked. Start a new connection and try again.',
        'oauth_offline_consent_required': 'The provider did not return offline access. Start a new connection and grant the requested access.',
        'oauth_scopes_missing': 'The provider did not grant the requested access. Start a new connection and review the requested permissions.',
        'oauth_account_mismatch': 'This is a different provider account from the saved connection. Reconnect the original account or add another.',
        'oauth_state_expired': 'This connection attempt expired. Start a new connection and try again.',
        'oauth_revision_conflict': 'This connection changed during authorization. Refresh Alphonse and start a new connection.',
        'oauth_consent_cancelled': 'Microsoft sign-in was cancelled. Return to Alphonse to try again.',
    }
    return messages.get(exc.args[0], generic)


SCOPES = {
    'google': {
        'mail_read': GMAIL_READ_SCOPE,
        'calendar_discover': 'https://www.googleapis.com/auth/calendar.calendarlist.readonly',
        'calendar_read': 'https://www.googleapis.com/auth/calendar.events.readonly',
        CALENDAR_WRITE_OPERATION: CALENDAR_WRITE_SCOPE,
        'classroom_read': 'https://www.googleapis.com/auth/classroom.coursework.me.readonly',
        'classroom_announcements_read': 'https://www.googleapis.com/auth/classroom.announcements.readonly',
    },
    'microsoft': {'mail_read': 'https://graph.microsoft.com/Mail.Read',
                  OUTLOOK_INBOX_OPERATION: OUTLOOK_INBOX_SCOPE,
                  'calendar_read': 'https://graph.microsoft.com/Calendars.Read'},
}


def capability_scopes(provider, capability):
    scopes = [SCOPES[provider][capability]]
    if provider == 'google' and capability in {'classroom_read', 'classroom_announcements_read'}:
        # Course metadata and proof of the connected student's own enrollment.
        scopes += ['https://www.googleapis.com/auth/classroom.courses.readonly',
                   'https://www.googleapis.com/auth/classroom.rosters.readonly']
    return scopes


ENDPOINTS = {
    'google': ('https://accounts.google.com/o/oauth2/v2/auth',
               'https://oauth2.googleapis.com/token', 'https://openidconnect.googleapis.com/v1/userinfo'),
    'microsoft': ('https://login.microsoftonline.com/common/oauth2/v2.0/authorize',
                  'https://login.microsoftonline.com/common/oauth2/v2.0/token',
                  'https://graph.microsoft.com/oidc/userinfo'),
}


@dataclass(frozen=True)
class ClientConfig:
    provider: str
    client_id: str
    redirect_uri: str
    client_secret_ref: str = field(default='', repr=False)

    def __post_init__(self):
        uri = urlsplit(self.redirect_uri)
        if (self.provider not in SCOPES or not self.client_id.strip() or uri.query or uri.fragment
                or uri.username or uri.password or not uri.path
                or not isinstance(self.client_secret_ref, str)
                or (self.client_secret_ref and (self.provider != 'google' or re.fullmatch(r'[0-9a-f]{32}', self.client_secret_ref) is None))
                or not ((uri.scheme == 'http' and uri.hostname == '127.0.0.1' and uri.port)
                        or (uri.scheme == 'https' and uri.netloc))):
            raise OAuthError('oauth_client_configuration_invalid')


class ProviderClient:
    """requests-oauthlib authorization-code client with explicit S256 verifier.

    No implicit flow, token-in-query or automatic refresh hooks. An optional
    Google Desktop client secret is resolved from the credential vault only
    during token exchange and refresh, and never included in the config repr.
    Identity is fetched from a fixed TLS userinfo endpoint, never decoded from an
    unverified ID token. Microsoft sub is scoped to this app's registration.
    """
    def __init__(self, config, *, secret_resolver=None):
        self.config = config
        self._secret_resolver = secret_resolver

    def _client_secret(self):
        if not self.config.client_secret_ref:
            return None
        try:
            secret = self._secret_resolver(self.config.client_secret_ref, self.config.client_id)
            if not isinstance(secret, str) or not secret:
                raise ValueError()
            return secret
        except Exception:
            raise OAuthError('oauth_client_credential_unavailable') from None

    def _session(self, scopes=None, token=None):
        try:
            from requests_oauthlib import OAuth2Session
        except ImportError:
            raise OAuthError('oauth_dependency_unavailable') from None
        # These libraries log request/response bodies at DEBUG. Disable their
        # diagnostics before any credential operation; application logs stay on.
        for name in list(logging.Logger.manager.loggerDict):
            if name.startswith(('requests_oauthlib', 'oauthlib')):
                logging.getLogger(name).disabled = True
        return OAuth2Session(self.config.client_id, scope=scopes,
                             redirect_uri=self.config.redirect_uri, token=token)

    def _endpoint(self, index, scopes):
        endpoint = ENDPOINTS[self.config.provider][index]
        if self.config.provider == 'microsoft' and OUTLOOK_INBOX_SCOPE in scopes:
            return endpoint.replace('/common/', '/consumers/')
        return endpoint

    def begin(self, scopes, state, verifier):
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode('ascii')).digest()).rstrip(b'=').decode('ascii')
        options = dict(code_challenge=challenge, code_challenge_method='S256', prompt='select_account')
        if self.config.provider == 'google':
            options.update(access_type='offline', prompt='consent select_account')
        with self._session(scopes=scopes) as client:
            return client.authorization_url(self._endpoint(0, scopes), state=state, **options)[0]

    def exchange(self, code, verifier, scopes):
        with self._session() as client:
            return dict(client.fetch_token(self._endpoint(1, scopes), code=code,
                        code_verifier=verifier, include_client_id=True,
                        client_secret=self._client_secret(), timeout=20))

    def refresh(self, token, scopes):
        with self._session() as client:
            return dict(client.refresh_token(self._endpoint(1, scopes),
                        refresh_token=token['refresh_token'], client_id=self.config.client_id,
                        client_secret=self._client_secret(),
                        body=urlencode({'scope': ' '.join(scopes)}), timeout=20))

    def subject(self, token):
        with self._session(token=token) as client:
            response = client.get(ENDPOINTS[self.config.provider][2], timeout=20)
            response.raise_for_status()
            subject = response.json().get('sub')
            if not isinstance(subject, str) or not subject:
                raise OAuthError('oauth_identity_invalid')
            return subject

    def revoke(self, token):
        if self.config.provider != 'google':
            # Microsoft has no per-token RFC7009 endpoint. Do not invoke the
            # broad Graph revokeSignInSessions permission/effect.
            return False
        import requests
        response = requests.post('https://oauth2.googleapis.com/revoke',
                                 data={'token': token.get('refresh_token') or token['access_token']}, timeout=20)
        return response.status_code == 200


@dataclass(repr=False)
class _Pending:
    actor: str
    connection_id: str
    revision: int
    generation: int
    capabilities: list[str]
    scopes: list[str]
    verifier: str = field(repr=False)
    expires: float = 0
    attempt_id: str = ''


@dataclass(repr=False)
class _OutlookAttempt:
    actor: str
    connection_id: str
    expires: float
    status: str = 'pending'
    reason: str = ''


class OAuthService:
    def __init__(self, accounts: AccountService, clients: list[ClientConfig], *, client_factory=ProviderClient,
                 session_guard=None, transaction_guard=None):
        self.accounts, self.store = accounts, accounts.integrations
        if session_guard is not None and not callable(session_guard):
            raise ValueError('oauth_session_guard_invalid')
        self.session_guard = session_guard
        if transaction_guard is not None and not callable(transaction_guard):
            raise ValueError('oauth_transaction_guard_invalid')
        self.transaction_guard = transaction_guard
        if len({c.provider for c in clients}) != len(clients):
            raise OAuthError('oauth_client_ambiguous')
        def resolve_client_secret(reference, client_id):
            credentials = self.store.credential_vault.get(reference, 'oauth-client:google:' + client_id, 1)
            if set(credentials) != {'client_secret'}:
                raise ValueError('oauth_client_credential_invalid')
            return credentials['client_secret']
        self.clients = {c.provider: (ProviderClient(c, secret_resolver=resolve_client_secret)
                                     if client_factory is ProviderClient else client_factory(c)) for c in clients}
        self.pending = {}
        self.outlook_attempts = {}
        self.lock = threading.RLock()
        with self.store._connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS v2_oauth_leases (connection_id TEXT PRIMARY KEY, nonce TEXT NOT NULL)')

    def _owned(self, session, connection_id):
        if self.session_guard is not None and self.session_guard(session) is not True:
            raise PermissionError('oauth_session_unavailable')
        record = self.store.get(connection_id)
        if not record or not record.connection or record.provider_key not in SCOPES:
            raise OAuthError('oauth_connection_unavailable')
        self.accounts._owner(session, record.connection)
        return record

    def health(self, session: AccountSession, connection_id: str):
        record = self._owned(session, connection_id)
        value = record.connection
        storage = 'absent'
        access_token_status = 'unavailable'
        if record.credential_ref:
            try:
                token = self._read(record)
                storage = 'available'
                access_token_status = 'current' if self._access_token_current(token) else 'refresh_required'
            except Exception:
                storage = 'credential_backend_unavailable'
        return dict(connection_id=connection_id, provider=record.provider_key, display_name=record.display_name,
                    status=value['status'], health_code=value['health_code'], revision=value['revision'],
                    capabilities=list(value['capabilities']), granted_scopes=list(value['granted_scopes']),
                    credential_health=storage, access_token_status=access_token_status,
                    provider_configured=record.provider_key in self.clients,
                    revocation_pending=record.revocation_pending,
                    credential_deletion_pending=record.credential_deletion_pending)

    def begin(self, session: AccountSession, *, connection_id: str, provider: str,
              capabilities: list[str], expected_revision: int, display_name: str = 'Account'):
        if self.session_guard is not None and self.session_guard(session) is not True:
            raise PermissionError('oauth_session_unavailable')
        if (provider not in self.clients or not capabilities or len(set(capabilities)) != len(capabilities)
                or not set(capabilities) <= SCOPES[provider].keys()):
            raise OAuthError('oauth_capability_invalid')
        outlook = capabilities == [OUTLOOK_INBOX_OPERATION]
        if OUTLOOK_INBOX_OPERATION in capabilities and not outlook:
            raise OAuthError('oauth_capability_invalid')
        calendar_write = capabilities == [CALENDAR_WRITE_OPERATION]
        if CALENDAR_WRITE_OPERATION in capabilities and not calendar_write:
            raise OAuthError('oauth_capability_invalid')
        if outlook or calendar_write:
            owner = self.accounts.users.admin_user()
            if owner is None or not owner.is_active or owner.user_id != session.actor_user_id or owner.role not in ('admin', 'owner'):
                raise PermissionError('owner_only_account_scope_unavailable')
        record = self.store.get(connection_id)
        if record is not None and expected_revision == 0:
            raise OAuthError('oauth_revision_conflict')
        if record is None:
            if expected_revision != 0:
                raise OAuthError('oauth_revision_conflict')
            self.accounts.create_connection(session, connection_id=connection_id, provider_key=provider,
                provider_subject='pending:' + secrets.token_hex(16), display_name=display_name)
        record = self._owned(session, connection_id)
        if record.provider_key != provider or (record.connection['revision'] != expected_revision and expected_revision != 0):
            raise OAuthError('oauth_revision_conflict')
        if expected_revision == 0 and not record.connection['provider_subject'].startswith('pending:'):
            raise OAuthError('oauth_revision_conflict')
        if record.credential_deletion_pending:
            raise OAuthError('oauth_credential_cleanup_required')
        if outlook and record.connection['capabilities'] not in ([], [OUTLOOK_INBOX_OPERATION]):
            raise OAuthError('oauth_capability_invalid')
        if (calendar_write and record.connection['capabilities'] not in ([], [CALENDAR_WRITE_OPERATION])
                or not calendar_write and CALENDAR_WRITE_OPERATION in record.connection['capabilities']):
            raise OAuthError('oauth_capability_invalid')
        scopes = sorted({'openid', *(s for c in capabilities for s in capability_scopes(provider, c))})
        if provider == 'microsoft':
            scopes.append('offline_access')
        # Fence in-flight managed reads/refresh and any earlier consent attempt.
        self.accounts.save_connection(session, connection_id=connection_id,
            expected_revision=record.connection['revision'], provider_subject=record.connection['provider_subject'],
            capabilities=record.connection['capabilities'], granted_scopes=record.connection['granted_scopes'],
            status='reauth_required', health_code='consent_required')
        record = self._owned(session, connection_id)
        # Explicit reconnect is also recovery for a crashed refresh worker. Its
        # previous revision can no longer persist a response or return credentials.
        with self.store._connect() as db:
            db.execute('DELETE FROM v2_oauth_leases WHERE connection_id=?', (connection_id,))
        state, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(64)
        try:
            url = self.clients[provider].begin(scopes, state, verifier)
        except Exception:
            raise OAuthError('oauth_setup_unavailable') from None
        with self.lock:
            now = time.monotonic()
            self.outlook_attempts = {key: value for key, value in self.outlook_attempts.items()
                                     if value.expires + 300 > now}
            self._cancel_outlook_attempts(connection_id)
            self.pending = {k: p for k, p in self.pending.items() if p.expires > now and p.connection_id != connection_id}
            if len(self.pending) >= 128 or (outlook and len(self.outlook_attempts) >= 128):
                raise OAuthError('oauth_setup_busy')
            attempt_id = secrets.token_urlsafe(24) if outlook else ''
            if outlook:
                self.outlook_attempts[attempt_id] = _OutlookAttempt(session.actor_user_id, connection_id, now + 300)
            self.pending[state] = _Pending(session.actor_user_id, connection_id, record.connection['revision'],
                record.credential_generation, list(capabilities), scopes, verifier, now + 300, attempt_id)
        # URL contains state/challenge only; return to the trusted setup surface,
        # never record in event history, prompts or routine logs.
        result = {'authorization_url': url, 'expires_in': 300}
        if outlook:
            result['attempt_id'] = attempt_id
        return result

    def _cancel_outlook_attempts(self, connection_id):
        for attempt in self.outlook_attempts.values():
            if attempt.connection_id == connection_id and attempt.status == 'pending':
                attempt.status = 'cancelled'

    def _outlook_attempt(self, session, provider, attempt_id, *, cancel=False):
        if provider != 'microsoft' or not isinstance(attempt_id, str) or re.fullmatch(r'[A-Za-z0-9_-]{32}', attempt_id) is None:
            raise OAuthError('oauth_attempt_unavailable')
        with self.lock:
            attempt = self.outlook_attempts.get(attempt_id)
            if attempt is None or attempt.actor != session.actor_user_id:
                raise OAuthError('oauth_attempt_unavailable')
            self._owned(session, attempt.connection_id)
            if attempt.status == 'pending':
                if attempt.expires <= time.monotonic():
                    attempt.status = 'expired'
                elif cancel:
                    attempt.status = 'cancelled'
                if attempt.status != 'pending':
                    self.pending = {state: pending for state, pending in self.pending.items()
                                    if pending.attempt_id != attempt_id}
            result = {'status': attempt.status}
            if attempt.reason:
                result['reason'] = attempt.reason
            return result

    def oauth_attempt_status(self, session, *, provider, attempt_id):
        return self._outlook_attempt(session, provider, attempt_id)

    def cancel_oauth_attempt(self, session, *, provider, attempt_id):
        return self._outlook_attempt(session, provider, attempt_id, cancel=True)

    @contextmanager
    def _lease(self, connection_id):
        nonce = secrets.token_hex(16)
        with self.store._connect() as db:
            cursor = db.execute('INSERT OR IGNORE INTO v2_oauth_leases VALUES (?, ?)', (connection_id, nonce))
            if cursor.rowcount != 1:
                raise OAuthError('oauth_operation_in_progress')
        try:
            yield
        finally:
            with self.store._connect() as db:
                db.execute('DELETE FROM v2_oauth_leases WHERE connection_id=? AND nonce=?', (connection_id, nonce))

    def _persist(self, session, record, token, *, subject=None, capabilities=None, scopes=None, outlook_grant=None):
        # Validate trusted principal again after network I/O.
        current = self._owned(session, record.integration_id)
        if current.connection != record.connection or current.credential_generation != record.credential_generation:
            raise OAuthError('oauth_revision_conflict')
        if capabilities is None:
            capabilities = record.connection['capabilities']
        scopes = scopes if scopes is not None else record.connection['granted_scopes']
        # Credential rotation is not a change of account authority. Preserve
        # the revision pinned by project policies on ordinary healthy refresh.
        authority_changed = (subject is not None or capabilities != record.connection['capabilities']
                             or scopes != record.connection['granted_scopes'])
        value = dict(record.connection)
        value.update(provider_subject=subject or value['provider_subject'], capabilities=capabilities,
                     granted_scopes=scopes, status='connected', health_code='healthy', revision=value['revision'] + int(authority_changed))
        from alphonse.agent_v2.integrations.store import _now_iso
        value.update(updated_at=_now_iso(), last_success_at=_now_iso())
        generation = record.credential_generation + 1
        # Store each secret separately so existing managed-execution redaction
        # recognizes token bytes. Never retain unused identity tokens.
        credentials = {key: str(token[key]) for key in ('access_token', 'refresh_token', 'token_type', 'expires_at')}
        granted = token.get('scope', scopes)
        credentials['scope'] = granted if isinstance(granted, str) else ' '.join(granted)
        reference = self.store.credential_vault.put(record.integration_id, generation, credentials)
        try:
            with self.store._connect() as db:
                db.execute('BEGIN IMMEDIATE')
                if self.transaction_guard is not None and self.transaction_guard(db, session) is not True:
                    raise PermissionError('oauth_session_unavailable')
                if outlook_grant is not None:
                    row = db.execute('SELECT revision, definition FROM v2_account_grants WHERE grant_id=? AND connection_id=?',
                                     (outlook_grant['grant_id'], record.integration_id)).fetchone()
                    if (row is None or row['revision'] != outlook_grant['revision']
                            or json.loads(row['definition']) != outlook_grant
                            or not account_grant_is_live(outlook_grant, datetime.now(timezone.utc))):
                        raise PermissionError('account_grant_denied')
                cursor = db.execute('''UPDATE v2_integrations SET connection_json=?, credential_ref=?,
                    credential_generation=?, enabled=1, revocation_pending=0, credential_deletion_pending=0,
                    secret_json='{}', updated_at=? WHERE integration_id=? AND connection_json=?
                    AND credential_generation=?''', (json.dumps(value, sort_keys=True), reference, generation,
                    value['updated_at'], record.integration_id, json.dumps(record.connection, sort_keys=True), record.credential_generation))
                if cursor.rowcount != 1:
                    raise OAuthError('oauth_revision_conflict')
        except Exception:
            self.store.credential_vault.delete(reference)
            raise
        if record.credential_ref:
            try:
                self.store.credential_vault.delete(record.credential_ref)
            except ValueError:
                pass

    @staticmethod
    def _validated(token, scopes, old=None, provider='google'):
        if (not isinstance(token, dict) or not isinstance(token.get('access_token'), str)
                or not token['access_token'] or token.get('token_type', '').lower() != 'bearer'):
            raise OAuthError('oauth_token_invalid')
        token = dict(token)
        if old and not token.get('refresh_token'):
            token['refresh_token'] = old.get('refresh_token')
        if not isinstance(token.get('refresh_token'), str) or not token['refresh_token']:
            raise OAuthError('oauth_offline_consent_required')
        granted = token.get('scope', old.get('scope') if old else None)
        granted = granted.split() if isinstance(granted, str) else granted
        # Graph may return short resource scopes even when full URIs were
        # requested. Canonicalize only our explicit Graph capability allowlist.
        if isinstance(granted, list) and provider == 'microsoft':
            aliases = {s.rsplit('/', 1)[-1].lower(): s for s in SCOPES['microsoft'].values()}
            granted = [aliases.get(s.lower(), s) for s in granted]
        # offline_access controls refresh issuance. openid is verified through
        # the fixed userinfo endpoint, not a resource permission in access scopes.
        if not isinstance(granted, list) or not (set(scopes) - {'offline_access', 'openid'}) <= set(granted):
            raise OAuthError('oauth_scopes_missing')
        try:
            expiry = float(token.get('expires_at', 0))
            if not math.isfinite(expiry) or expiry <= time.time():
                raise ValueError()
        except (TypeError, ValueError):
            raise OAuthError('oauth_token_expiry_invalid') from None
        return token, sorted(set(granted))

    def complete(self, session: AccountSession, callback_url: str):
        uri = urlsplit(callback_url)
        query = parse_qs(uri.query, keep_blank_values=True)
        state = query.get('state', [''])[0]
        with self.lock:
            pending = self.pending.get(state)
            if not pending or pending.actor != session.actor_user_id:
                raise OAuthError('oauth_state_invalid')
            self.pending.pop(state)
        if not pending.attempt_id:
            return self._complete(session, pending, uri, query)
        try:
            return self._complete(session, pending, uri, query)
        except Exception as exc:
            with self.lock:
                attempt = self.outlook_attempts.get(pending.attempt_id)
                if attempt is not None and attempt.status == 'pending':
                    code = (exc.args[0] if isinstance(exc, OAuthError) and len(exc.args) == 1
                            and isinstance(exc.args[0], str) else '')
                    if code == 'oauth_consent_cancelled':
                        attempt.status = 'cancelled'
                    elif code == 'oauth_state_expired':
                        attempt.status = 'expired'
                    else:
                        attempt.status = 'failed'
                        attempt.reason = {
                            'oauth_account_mismatch': 'account_mismatch',
                            'oauth_exchange_access_denied': 'consent_denied',
                            'oauth_consent_denied': 'consent_denied',
                            'oauth_scopes_missing': 'consent_incomplete',
                            'oauth_offline_consent_required': 'consent_incomplete',
                            'oauth_revision_conflict': 'connection_changed',
                        }.get(code, 'provider_unavailable')
            raise

    def _complete(self, session, pending, uri, query):
        if pending.expires <= time.monotonic():
            raise OAuthError('oauth_state_expired')
        record = self._owned(session, pending.connection_id)
        client = self.clients[record.provider_key]
        if (uri._replace(query='').geturl() != client.config.redirect_uri or uri.fragment
                or len(query.get('state', [])) != 1):
            raise OAuthError('oauth_callback_invalid')
        if pending.attempt_id and 'error' in query:
            if query['error'] == ['access_denied']:
                raise OAuthError('oauth_consent_cancelled')
            raise OAuthError('oauth_consent_denied')
        if len(query.get('code', [])) != 1 or not query['code'][0] or 'error' in query:
            raise OAuthError('oauth_callback_invalid')
        if record.connection['revision'] != pending.revision or record.credential_generation != pending.generation:
            raise OAuthError('oauth_revision_conflict')
        with self._lease(record.integration_id):
            try:
                raw_token = client.exchange(query['code'][0], pending.verifier, pending.scopes)
            except OAuthError:
                raise
            except Exception as exc:
                raise OAuthError(_exchange_error_code(exc)) from None
            token, granted = self._validated(raw_token, pending.scopes, provider=record.provider_key)
            if pending.attempt_id:
                granted = [OUTLOOK_INBOX_SCOPE]
            try:
                subject = client.subject(token)
                if not subject or (not record.connection['provider_subject'].startswith('pending:')
                                   and subject != record.connection['provider_subject']):
                    raise OAuthError('oauth_account_mismatch')
            except OAuthError:
                raise
            except Exception:
                raise OAuthError('oauth_identity_failed') from None
            if pending.attempt_id:
                with self.lock:
                    attempt = self.outlook_attempts.get(pending.attempt_id)
                    if attempt is None or attempt.status != 'pending':
                        raise OAuthError('oauth_consent_cancelled')
                    if attempt.expires <= time.monotonic():
                        raise OAuthError('oauth_state_expired')
                    self._persist(session, record, token, subject=subject, capabilities=pending.capabilities, scopes=granted)
                    attempt.status = 'connected'
            else:
                self._persist(session, record, token, subject=subject, capabilities=pending.capabilities, scopes=granted)
        return self.health(session, record.integration_id)

    def pause(self, session: AccountSession, connection_id: str, expected_revision: int):
        record = self._owned(session, connection_id)
        self.accounts.save_connection(session, connection_id=connection_id, expected_revision=expected_revision,
            provider_subject=record.connection['provider_subject'], capabilities=record.connection['capabilities'],
            granted_scopes=record.connection['granted_scopes'], status='paused', health_code='paused')
        return self.health(session, connection_id)

    def disconnect(self, session: AccountSession, connection_id: str, expected_revision: int):
        record = self._owned(session, connection_id)
        if record.connection['revision'] != expected_revision:
            raise OAuthError('oauth_revision_conflict')
        token = None
        try:
            token = self._read(record)
        except Exception:
            pass
        self.accounts.save_connection(session, connection_id=connection_id, expected_revision=expected_revision,
            provider_subject=record.connection['provider_subject'],
            capabilities=[OUTLOOK_INBOX_OPERATION] if record.connection['capabilities'] == [OUTLOOK_INBOX_OPERATION] else [],
            granted_scopes=[],
            status='disconnected', health_code='disconnected')
        with self.lock:
            self._cancel_outlook_attempts(connection_id)
            self.pending = {s: p for s, p in self.pending.items() if p.connection_id != connection_id}
        # Local authority and credentials are invalidated before provider I/O.
        revoked = False
        if token:
            try:
                revoked = self.clients[record.provider_key].revoke(token)
            except Exception:
                pass
        if revoked:
            with self.store._connect() as db:
                db.execute('UPDATE v2_integrations SET revocation_pending=0 WHERE integration_id=? AND enabled=0 AND credential_generation=?',
                           (connection_id, record.credential_generation + 1))
        return self.health(session, connection_id)

    def retry_deletion(self, session: AccountSession, connection_id: str, expected_revision: int):
        record = self._owned(session, connection_id)
        if record.connection['revision'] != expected_revision:
            raise OAuthError('oauth_revision_conflict')
        self.store.retry_credential_deletion(connection_id)
        return self.health(session, connection_id)

    def _read(self, record):
        return self.store.credential_vault.get(record.credential_ref, record.integration_id, record.credential_generation)

    def _refresh_access_token(self, session, record, token, *, fixed_scopes=False, outlook_grant=None):
        scopes = record.connection['granted_scopes']
        outlook = record.connection['capabilities'] == [OUTLOOK_INBOX_OPERATION]
        requested = [OUTLOOK_INBOX_SCOPE] if outlook else scopes
        try:
            token, granted = self._validated(self.clients[record.provider_key].refresh(token, requested),
                                             requested, old=token, provider=record.provider_key)
            if outlook:
                granted = [OUTLOOK_INBOX_SCOPE]
            if (fixed_scopes or outlook) and granted != scopes:
                raise OAuthError('oauth_scopes_changed')
        except Exception as exc:
            # Failed validation may have consumed a rotating refresh token.
            if getattr(exc, 'error', '') in {'temporarily_unavailable', 'server_error'}:
                raise OAuthError('oauth_provider_unavailable') from None
            self._require_reconnect(record)
            raise OAuthError('oauth_reauth_required') from None
        try:
            self._persist(session, record, token, scopes=granted, outlook_grant=outlook_grant)
        except Exception:
            self._require_reconnect(record)
            raise OAuthError('oauth_refresh_not_persisted') from None
        return token

    def token(self, session: AccountSession, *, authorization: dict, expected_connection_revision=None):
        """Managed executor boundary: reauthorize each page/use, including after I/O.

        authorization is the complete explicit AccountService.authorize request.
        Never register this method as a general tool or expose its result via IPC.
        """
        def authorized_record():
            record, grant = self.accounts.authorize(session, **authorization)
            if (expected_connection_revision is not None
                    and record.connection['revision'] != expected_connection_revision):
                raise OAuthError('oauth_revision_conflict')
            return record, grant

        record, _ = authorized_record()
        if record.provider_key not in self.clients:
            raise OAuthError('oauth_provider_unconfigured')
        with self._lease(record.integration_id):
            record, grant = authorized_record()
            token = self._read(record)
            if float(token.get('expires_at', 0)) <= time.time() + 60:
                outlook_grant = grant if authorization['operation'] == OUTLOOK_INBOX_OPERATION else None
                token = self._refresh_access_token(AccountSession(record.connection['owner_user_id']), record, token,
                                                   outlook_grant=outlook_grant)
            authorized_record()
            return token['access_token']

    def list_gmail_labels(self, session: AccountSession, *, connection_id: str, expected_revision: int):
        """List label names and IDs from the connected Gmail account for owner settings."""
        from alphonse.agent_v2.integrations.gmail import GmailHTTPError, GmailTransport
        from alphonse.agent_v2.services.project_check_worker import CheckStopped

        if (type(expected_revision) is not int or expected_revision < 1
                or not isinstance(connection_id, str) or not connection_id.strip()):
            raise ValueError('gmail_label_list_request_invalid')

        def eligible_record():
            record = self._owned(session, connection_id)
            owner = self.accounts.users.admin_user()
            connection = record.connection
            if (owner is None or not owner.is_active or owner.role not in ('owner', 'admin')
                    or owner.user_id != session.actor_user_id or record.provider_key != 'google'
                    or 'mail_read' not in connection['capabilities']
                    or GMAIL_READ_SCOPE not in connection['granted_scopes']):
                raise PermissionError('owner_only_account_scope_unavailable')
            if connection['revision'] != expected_revision:
                raise OAuthError('oauth_revision_conflict')
            if (not record.enabled or connection['status'] != 'connected'
                    or connection['health_code'] != 'healthy' or not record.credential_ref
                    or record.revocation_pending or record.credential_deletion_pending):
                raise OAuthError('oauth_connection_unavailable')
            return record

        eligible_record()
        if 'google' not in self.clients:
            raise OAuthError('oauth_provider_unconfigured')
        with self._lease(connection_id):
            record = eligible_record()
            token = self._read(record)
            if not self._access_token_current(token):
                self._refresh_access_token(session, record, token, fixed_scopes=True)
                record = eligible_record()
                token = self._read(record)
            if not self._access_token_current(token):
                raise OAuthError('oauth_refresh_required')
            try:
                response, _ = GmailTransport().get('labels', params={'fields': 'labels(id,name)'},
                    token=token['access_token'], deadline=datetime.now(timezone.utc) + timedelta(seconds=8),
                    max_bytes=4 * 1024 * 1024)
            except (GmailHTTPError, CheckStopped, ImportError):
                raise OAuthError('gmail_label_list_unavailable') from None
            current = eligible_record()
            if (current.connection != record.connection or current.credential_ref != record.credential_ref
                    or current.credential_generation != record.credential_generation):
                raise OAuthError('oauth_revision_conflict')
        labels = response.get('labels', []) if isinstance(response, dict) else None
        if not isinstance(labels, list) or len(labels) > 10100:
            raise OAuthError('gmail_label_list_unavailable')
        by_id = {}
        for label in labels:
            if not isinstance(label, dict):
                raise OAuthError('gmail_label_list_unavailable')
            label_id, name = label.get('id'), label.get('name')
            if exact_gmail_label_id(label_id) and valid_gmail_label_name(name):
                if label_id in by_id and by_id[label_id] != name:
                    raise OAuthError('gmail_label_list_unavailable')
                by_id[label_id] = name
        return {'labels': [{'id': label_id, 'name': name} for label_id, name in sorted(
            by_id.items(), key=lambda item: (item[1].casefold(), item[1], item[0]))]}

    def refresh_gmail_access(self, session: AccountSession, *, connection_id: str,
                             expected_revision: int, grant_id: str, grant_revision: int):
        """Owner-initiated credential maintenance, separate from Gmail reads."""
        if (type(expected_revision) is not int or type(grant_revision) is not int
                or expected_revision < 1 or grant_revision < 1
                or not isinstance(connection_id, str) or not isinstance(grant_id, str)):
            raise ValueError('oauth_refresh_request_invalid')
        record = self._owned(session, connection_id)
        if record.provider_key != 'google' or 'mail_read' not in record.connection['capabilities']:
            raise PermissionError('owner_only_account_scope_unavailable')
        with self.store._connect() as db:
            grant = db.execute('SELECT definition FROM v2_account_grants WHERE grant_id=? AND connection_id=?',
                               (grant_id, connection_id)).fetchone()
        if grant is None:
            raise PermissionError('account_grant_missing')
        definition = json.loads(grant['definition'])
        content_grant = definition.get('operations') == [GMAIL_CONTENT_OPERATION]
        if not content_grant and record.connection['capabilities'] != ['mail_read']:
            raise PermissionError('owner_only_account_scope_unavailable')
        resources = definition.get('resources') if content_grant else ['INBOX']
        if not isinstance(resources, list) or len(resources) != 1 or not exact_gmail_label_id(resources[0]):
            raise PermissionError('account_grant_denied')
        self.token(session, expected_connection_revision=expected_revision, authorization={
            'connection_id': connection_id, 'grant_id': grant_id,
            'operation': GMAIL_CONTENT_OPERATION if content_grant else 'mail_read', 'owner_gmail_content': content_grant,
            'resources': resources, 'recipient_user_ids': [session.actor_user_id],
            'required_scopes': [SCOPES['google']['mail_read']], 'project_id': None,
            'expected_grant_revision': grant_revision,
        })
        value = self.health(session, connection_id)
        return {'status': value['status'], 'access_token_status': value['access_token_status']}

    def refresh_outlook_inbox_access(self, session: AccountSession, *, connection_id: str,
                                    expected_revision: int, grant_id: str, grant_revision: int):
        """Refresh only a currently granted personal Inbox connection."""
        if (type(expected_revision) is not int or type(grant_revision) is not int
                or expected_revision < 1 or grant_revision < 1
                or not isinstance(connection_id, str) or not isinstance(grant_id, str)):
            raise ValueError('oauth_refresh_request_invalid')
        self._owned(session, connection_id)
        self.token(session, expected_connection_revision=expected_revision, authorization={
            'connection_id': connection_id, 'grant_id': grant_id, 'operation': OUTLOOK_INBOX_OPERATION,
            'resources': [OUTLOOK_INBOX_RESOURCE], 'recipient_user_ids': [session.actor_user_id],
            'required_scopes': [OUTLOOK_INBOX_SCOPE], 'project_id': None, 'expected_grant_revision': grant_revision,
            'owner_outlook_inbox': True,
        })
        value = self.health(session, connection_id)
        return {'status': value['status'], 'access_token_status': value['access_token_status']}

    def token_without_refresh(self, session: AccountSession, *, authorization: dict):
        """Use a currently valid token without any refresh or credential mutation.

        Intended for read-only paths that cannot reconcile a credential refresh.
        Expired or near-expiry tokens require a separately authorized refresh.
        """
        record, _ = self.accounts.authorize(session, **authorization)
        if record.provider_key not in self.clients:
            raise OAuthError('oauth_provider_unconfigured')
        with self._lease(record.integration_id):
            record, _ = self.accounts.authorize(session, **authorization)
            token = self._read(record)
            if not self._access_token_current(token):
                raise OAuthError('oauth_refresh_required')
            self.accounts.authorize(session, **authorization)
            return token['access_token']

    @staticmethod
    def _access_token_current(token):
        import math
        try:
            expires_at = float(token.get('expires_at', 0))
            return (math.isfinite(expires_at) and expires_at > time.time() + 60
                    and isinstance(token.get('access_token'), str) and bool(token['access_token'].strip()))
        except (TypeError, ValueError, AttributeError):
            return False

    def _require_reconnect(self, record):
        current = self.store.get(record.integration_id)
        if current and current.connection == record.connection:
            try:
                self.accounts.save_connection(AccountSession(record.connection['owner_user_id']),
                    connection_id=record.integration_id, expected_revision=record.connection['revision'],
                    provider_subject=record.connection['provider_subject'], capabilities=record.connection['capabilities'],
                    granted_scopes=record.connection['granted_scopes'], status='reauth_required', health_code='consent_required')
            except (ValueError, PermissionError):
                pass  # Concurrent lifecycle mutation already fences the result.

    def require_outlook_reconnect(self, session, record, *, grant_id, grant_revision):
        """Mark rejected Outlook credentials only while that generation remains current."""
        current = self._owned(session, record.integration_id)
        value = current.connection
        if (current.provider_key != 'microsoft' or value != record.connection
                or value['capabilities'] != [OUTLOOK_INBOX_OPERATION] or value['status'] != 'connected'
                or current.credential_generation != record.credential_generation or not current.enabled):
            return False
        from alphonse.agent_v2.integrations.store import _now_iso
        now = _now_iso()
        replacement = dict(value, status='reauth_required', health_code='consent_required',
                           revision=value['revision'] + 1, updated_at=now)
        with self.store._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if self.transaction_guard is not None and self.transaction_guard(db, session) is not True:
                return False
            row = db.execute('SELECT revision, definition FROM v2_account_grants WHERE grant_id=? AND connection_id=?',
                             (grant_id, record.integration_id)).fetchone()
            if (row is None or row['revision'] != grant_revision
                    or not account_grant_is_live(json.loads(row['definition']), datetime.now(timezone.utc))):
                return False
            updated = db.execute('''UPDATE v2_integrations SET connection_json=?, updated_at=?
                WHERE integration_id=? AND enabled=1 AND credential_generation=? AND connection_json=?''',
                (json.dumps(replacement, sort_keys=True), now, record.integration_id,
                 record.credential_generation, json.dumps(value, sort_keys=True)))
            return updated.rowcount == 1
