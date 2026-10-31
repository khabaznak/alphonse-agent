"""Opt-in managed OAuth. No imports of provider libraries or network work at startup.

Public methods require a trusted AccountSession, never a payload-supplied actor.
Only managed executors may consume token(); it is not an IPC/tool response.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import base64
import json
import logging
import math
import secrets
import threading
import time
from urllib.parse import parse_qs, urlsplit, urlencode

from alphonse.agent_v2.accounts import AccountService, AccountSession


class OAuthError(ValueError):
    """Static safe code: never include provider errors, URLs or token responses."""


SCOPES = {
    'google': {
        'mail.read': 'https://www.googleapis.com/auth/gmail.readonly',
        'calendar.read': 'https://www.googleapis.com/auth/calendar.events.readonly',
    },
    'microsoft': {'mail.read': 'https://graph.microsoft.com/Mail.Read',
                  'calendar.read': 'https://graph.microsoft.com/Calendars.Read'},
}
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

    def __post_init__(self):
        uri = urlsplit(self.redirect_uri)
        if (self.provider not in SCOPES or not self.client_id.strip() or uri.query or uri.fragment
                or uri.username or uri.password or not uri.path
                or not ((uri.scheme == 'http' and uri.hostname == '127.0.0.1' and uri.port)
                        or (uri.scheme == 'https' and uri.netloc))):
            raise OAuthError('oauth_client_configuration_invalid')


class ProviderClient:
    """requests-oauthlib authorization-code client with explicit S256 verifier.

    No client secret, implicit flow, token-in-query or automatic refresh hooks.
    Identity is fetched from a fixed TLS userinfo endpoint, never decoded from an
    unverified ID token. Microsoft sub is scoped to this app's registration.
    """
    def __init__(self, config):
        self.config = config

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

    def begin(self, scopes, state, verifier):
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode('ascii')).digest()).rstrip(b'=').decode('ascii')
        options = dict(code_challenge=challenge, code_challenge_method='S256', prompt='select_account')
        if self.config.provider == 'google':
            options.update(access_type='offline', prompt='consent select_account')
        with self._session(scopes=scopes) as client:
            return client.authorization_url(ENDPOINTS[self.config.provider][0], state=state, **options)[0]

    def exchange(self, code, verifier, scopes):
        with self._session() as client:
            return dict(client.fetch_token(ENDPOINTS[self.config.provider][1], code=code,
                        code_verifier=verifier, include_client_id=True, timeout=20))

    def refresh(self, token, scopes):
        with self._session() as client:
            return dict(client.refresh_token(ENDPOINTS[self.config.provider][1],
                        refresh_token=token['refresh_token'], client_id=self.config.client_id,
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


class OAuthService:
    def __init__(self, accounts: AccountService, clients: list[ClientConfig], *, client_factory=ProviderClient):
        self.accounts, self.store = accounts, accounts.integrations
        if len({c.provider for c in clients}) != len(clients):
            raise OAuthError('oauth_client_ambiguous')
        self.clients = {c.provider: client_factory(c) for c in clients}
        self.pending = {}
        self.lock = threading.RLock()
        with self.store._connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS v2_oauth_leases (connection_id TEXT PRIMARY KEY, nonce TEXT NOT NULL)')

    def _owned(self, session, connection_id):
        record = self.store.get(connection_id)
        if not record or not record.connection or record.provider_key not in SCOPES:
            raise OAuthError('oauth_connection_unavailable')
        self.accounts._owner(session, record.connection)
        return record

    def health(self, session: AccountSession, connection_id: str):
        record = self._owned(session, connection_id)
        value = record.connection
        storage = 'absent'
        if record.credential_ref:
            storage = self.store.credential_vault.health(record.credential_ref, connection_id, record.credential_generation)
        return dict(connection_id=connection_id, provider=record.provider_key, display_name=record.display_name,
                    status=value['status'], health_code=value['health_code'], revision=value['revision'],
                    capabilities=list(value['capabilities']), granted_scopes=list(value['granted_scopes']),
                    credential_health=storage, provider_configured=record.provider_key in self.clients,
                    revocation_pending=record.revocation_pending,
                    credential_deletion_pending=record.credential_deletion_pending)

    def begin(self, session: AccountSession, *, connection_id: str, provider: str,
              capabilities: list[str], expected_revision: int, display_name: str = 'Account'):
        if (provider not in self.clients or not capabilities or len(set(capabilities)) != len(capabilities)
                or not set(capabilities) <= SCOPES[provider].keys()):
            raise OAuthError('oauth_capability_invalid')
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
        scopes = sorted({'openid', *(SCOPES[provider][c] for c in capabilities)})
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
            self.pending = {k: p for k, p in self.pending.items() if p.expires > now and p.connection_id != connection_id}
            if len(self.pending) >= 128:
                raise OAuthError('oauth_setup_busy')
            self.pending[state] = _Pending(session.actor_user_id, connection_id, record.connection['revision'],
                record.credential_generation, list(capabilities), scopes, verifier, now + 300)
        # URL contains state/challenge only; return to the trusted setup surface,
        # never record in event history, prompts or routine logs.
        return {'authorization_url': url, 'expires_in': 300}

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

    def _persist(self, session, record, token, *, subject=None, capabilities=None, scopes=None):
        # Validate trusted principal again after network I/O.
        current = self._owned(session, record.integration_id)
        if current.connection != record.connection or current.credential_generation != record.credential_generation:
            raise OAuthError('oauth_revision_conflict')
        if capabilities is None:
            capabilities = record.connection['capabilities']
        scopes = scopes if scopes is not None else record.connection['granted_scopes']
        value = dict(record.connection)
        value.update(provider_subject=subject or value['provider_subject'], capabilities=capabilities,
                     granted_scopes=scopes, status='connected', health_code='healthy', revision=value['revision'] + 1)
        from alphonse.agent_v2.integrations.store import _now_iso
        value.update(updated_at=_now_iso(), last_success_at=_now_iso())
        generation = record.credential_generation + 1
        # Store each secret separately so existing managed-execution redaction
        # recognizes token bytes. Never retain unused identity tokens.
        credentials = {key: str(token[key]) for key in ('access_token', 'refresh_token', 'token_type', 'expires_at')}
        granted = token.get('scope', scopes)
        credentials['scope'] = granted if isinstance(granted, str) else ' '.join(granted)
        reference = self.store.credential_vault.put(record.integration_id, generation, credentials)
        with self.store._connect() as db:
            cursor = db.execute('''UPDATE v2_integrations SET connection_json=?, credential_ref=?,
                credential_generation=?, enabled=1, revocation_pending=0, credential_deletion_pending=0,
                secret_json='{}', updated_at=? WHERE integration_id=? AND connection_json=?
                AND credential_generation=?''', (json.dumps(value, sort_keys=True), reference, generation,
                value['updated_at'], record.integration_id, json.dumps(record.connection, sort_keys=True), record.credential_generation))
        if cursor.rowcount != 1:
            self.store.credential_vault.delete(reference)
            raise OAuthError('oauth_revision_conflict')
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
        if pending.expires <= time.monotonic():
            raise OAuthError('oauth_state_expired')
        record = self._owned(session, pending.connection_id)
        client = self.clients[record.provider_key]
        if (uri._replace(query='').geturl() != client.config.redirect_uri or uri.fragment
                or len(query.get('state', [])) != 1 or len(query.get('code', [])) != 1
                or not query['code'][0] or 'error' in query):
            raise OAuthError('oauth_callback_invalid')
        if record.connection['revision'] != pending.revision or record.credential_generation != pending.generation:
            raise OAuthError('oauth_revision_conflict')
        with self._lease(record.integration_id):
            try:
                token, granted = self._validated(client.exchange(query['code'][0], pending.verifier, pending.scopes), pending.scopes, provider=record.provider_key)
                subject = client.subject(token)
                if not subject or (not record.connection['provider_subject'].startswith('pending:')
                                   and subject != record.connection['provider_subject']):
                    raise OAuthError('oauth_account_mismatch')
            except OAuthError:
                raise
            except Exception:
                raise OAuthError('oauth_consent_failed') from None
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
            provider_subject=record.connection['provider_subject'], capabilities=[], granted_scopes=[],
            status='disconnected', health_code='disconnected')
        with self.lock:
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

    def token(self, session: AccountSession, *, authorization: dict):
        """Managed executor boundary: reauthorize each page/use, including after I/O.

        authorization is the complete explicit AccountService.authorize request.
        Never register this method as a general tool or expose its result via IPC.
        """
        record, _ = self.accounts.authorize(session, **authorization)
        if record.provider_key not in self.clients:
            raise OAuthError('oauth_provider_unconfigured')
        with self._lease(record.integration_id):
            record, _ = self.accounts.authorize(session, **authorization)
            token = self._read(record)
            if float(token.get('expires_at', 0)) <= time.time() + 60:
                try:
                    token, granted = self._validated(self.clients[record.provider_key].refresh(token,
                        record.connection['granted_scopes']), record.connection['granted_scopes'], old=token, provider=record.provider_key)
                except Exception as exc:
                    # A timeout/invalid response may have consumed a rotating
                    # refresh token. Require reconnect rather than replay it.
                    if getattr(exc, 'error', '') in {'temporarily_unavailable', 'server_error'}:
                        raise OAuthError('oauth_provider_unavailable') from None
                    self._require_reconnect(record)
                    raise OAuthError('oauth_reauth_required') from None
                try:
                    self._persist(AccountSession(record.connection['owner_user_id']), record, token, scopes=granted)
                except Exception:
                    self._require_reconnect(record)
                    raise OAuthError('oauth_refresh_not_persisted') from None
            self.accounts.authorize(session, **authorization)
            return token['access_token']

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
