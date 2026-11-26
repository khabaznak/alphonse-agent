"""Synthetic provider only. Never connects to an account or OS credential store."""
import json
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest

from alphonse.agent_v2.accounts import AccountService, AccountSession
from alphonse.agent_v2.integrations.credentials import CredentialVault
from alphonse.agent_v2.integrations.oauth import ClientConfig, OAuthError, OAuthService, SCOPES
from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
from alphonse.agent_v2.users import V2UserStore


class MemoryBackend:
    def __init__(self):
        self.data = {}
    def put(self, ref, value):
        self.data[ref] = value
    def get(self, ref):
        return self.data[ref]
    def delete(self, ref):
        self.data.pop(ref, None)


class FakeProvider:
    def __init__(self, config):
        self.config = config
        self.identity = 'subject-a'
        self.revoked = False
        self.on_refresh = lambda: None
        self.on_exchange = lambda: None
        self.refresh_count = 0
    def begin(self, scopes, state, verifier):
        self.scopes, self.state, self.verifier = scopes, state, verifier
        return 'https://provider.invalid/auth?' + urlencode({'state': state})
    def exchange(self, code, verifier, scopes):
        assert verifier == self.verifier and code == 'disposable-code'
        self.on_exchange()
        return dict(access_token='disposable-access', refresh_token='disposable-refresh',
                    expires_at=time.time() + 3600, token_type='Bearer', scope=self.scopes)
    def subject(self, token):
        return self.identity
    def refresh(self, token, scopes):
        self.refresh_count += 1
        self.on_refresh()
        return dict(access_token='rotated-access', refresh_token='rotated-refresh',
                    expires_at=time.time() + 3600, token_type='Bearer', scope=scopes)
    def revoke(self, token):
        return self.revoked


@pytest.fixture
def f(tmp_path):
    users = V2UserStore(tmp_path / 'users.db')
    users.onboard(display_name='Owner', user_id='owner', users_root=tmp_path / 'profiles')
    users.create_user(display_name='Other', user_id='other')
    backend = MemoryBackend()
    store = SQLiteIntegrationStore(tmp_path / 'accounts.db', credential_vault=CredentialVault(backend))
    accounts = AccountService(store, users)
    configs = [ClientConfig(p, 'public-app', 'http://127.0.0.1:54321/callback') for p in SCOPES]
    service = OAuthService(accounts, configs, client_factory=FakeProvider)
    return SimpleNamespace(service=service, accounts=accounts, store=store, backend=backend,
                           owner=AccountSession('owner'), other=AccountSession('other'), configs=configs)


def begin(f, provider='google', connection='one'):
    record = f.store.get(connection)
    result = f.service.begin(f.owner, connection_id=connection, provider=provider, capabilities=['mail_read'],
                            expected_revision=record.connection['revision'] if record else 0)
    client = f.service.clients[provider]
    return client.config.redirect_uri + '?' + urlencode({'state': client.state, 'code': 'disposable-code'})


def connect(f, provider='google', connection='one'):
    return f.service.complete(f.owner, begin(f, provider, connection))


def revision(f):
    return f.store.get('one').connection['revision']


def authorization(f):
    from datetime import datetime, timedelta, timezone
    f.accounts.save_grant(f.owner, grant_id='grant', connection_id='one', expected_revision=0,
        beneficiary_user_id='owner', operations=['mail_read'], resources=['mailbox'], recipient_user_ids=[],
        expires_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat())
    return dict(connection_id='one', grant_id='grant', operation='mail_read', resources=['mailbox'],
                recipient_user_ids=[], required_scopes=[SCOPES['google']['mail_read']], expected_grant_revision=1)


def expire(f):
    record = f.store.get('one')
    token = f.store.resolve_credentials(record)
    token['expires_at'] = str(time.time() - 1)
    f.store.install_credentials('one', token, expected_generation=record.credential_generation)


def test_no_refresh_token_access_refuses_expired_credentials_without_mutation(f):
    connect(f)
    args = authorization(f)
    assert f.service.token_without_refresh(f.owner, authorization=args) == 'disposable-access'
    expire(f)
    before = dict(f.backend.data)
    before_revision = revision(f)
    with pytest.raises(OAuthError, match='oauth_refresh_required'):
        f.service.token_without_refresh(f.owner, authorization=args)
    assert f.service.clients['google'].refresh_count == 0
    assert f.backend.data == before and revision(f) == before_revision


@pytest.mark.parametrize('operation, resource, label_name', [
    ('mail_read', 'INBOX', None), ('mail_content_read', 'INBOX', None),
    ('mail_content_read', 'Label_17', 'Projects/Planning'),
])
def test_owner_refresh_gmail_access_requires_current_exact_grant_and_returns_no_token(f, operation, resource, label_name):
    from datetime import datetime, timedelta, timezone
    connect(f)
    expire(f)
    before_revision = revision(f)
    args = {'connection_id': 'one', 'expected_revision': before_revision, 'grant_id': 'gmail', 'grant_revision': 1}
    with pytest.raises(PermissionError, match='account_grant_missing'):
        f.service.refresh_gmail_access(f.owner, **args)
    f.accounts.save_grant(f.owner, grant_id='gmail', connection_id='one', expected_revision=0,
        beneficiary_user_id='owner', operations=[operation], resources=[resource], recipient_user_ids=['owner'],
        expires_at=None if label_name else (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
        gmail_label_name=label_name)
    with pytest.raises(OAuthError, match='oauth_revision_conflict'):
        f.service.refresh_gmail_access(f.owner, **{**args, 'expected_revision': before_revision + 1})
    assert f.service.clients['google'].refresh_count == 0
    assert f.service.refresh_gmail_access(f.owner, **args) == {'status': 'connected', 'access_token_status': 'current'}
    assert f.service.clients['google'].refresh_count == 1
    assert revision(f) == before_revision


def test_owner_list_gmail_labels_returns_only_metadata_and_renews_without_a_grant(f, monkeypatch):
    from alphonse.agent_v2.integrations.gmail import GmailTransport
    connect(f)
    expire(f)
    calls = []

    def get(_, path, *, params, token, deadline, max_bytes):
        calls.append((path, params))
        assert token == 'rotated-access' and max_bytes > 0
        return {'labels': [
            {'id': 'Label_17', 'name': 'Projects/Planning', 'messagesTotal': 19},
            {'id': 'INBOX', 'name': 'INBOX'},
            {'id': 'gmail:all', 'name': 'Everything'},
            {'id': 'Label_18', 'name': 'Invalid\nLabel'},
        ]}, 200

    monkeypatch.setattr(GmailTransport, 'get', get)
    before_revision = revision(f)
    args = {'connection_id': 'one', 'expected_revision': before_revision}
    with pytest.raises(PermissionError, match='owner_required'):
        f.service.list_gmail_labels(f.other, **args)
    with pytest.raises(OAuthError, match='oauth_revision_conflict'):
        f.service.list_gmail_labels(f.owner, **{**args, 'expected_revision': before_revision + 1})
    assert calls == [] and f.service.clients['google'].refresh_count == 0
    assert f.service.list_gmail_labels(f.owner, **args) == {'labels': [
        {'id': 'INBOX', 'name': 'INBOX'}, {'id': 'Label_17', 'name': 'Projects/Planning'},
    ]}
    assert calls == [('labels', {'fields': 'labels(id,name)'})]
    assert f.service.clients['google'].refresh_count == 1
    assert revision(f) == before_revision


def test_list_gmail_labels_rejects_authority_change_during_fetch(f, monkeypatch):
    from alphonse.agent_v2.integrations.gmail import GmailTransport
    connect(f)
    before_revision = revision(f)

    def get(*args, **kwargs):
        f.service.pause(f.owner, 'one', before_revision)
        return {'labels': [{'id': 'Label_17', 'name': 'Projects/Planning'}]}, 200

    monkeypatch.setattr(GmailTransport, 'get', get)
    with pytest.raises(OAuthError, match='oauth_revision_conflict'):
        f.service.list_gmail_labels(f.owner, connection_id='one', expected_revision=before_revision)


def test_list_gmail_labels_rejects_unexpected_scopes_during_renewal(f, monkeypatch):
    from alphonse.agent_v2.integrations.gmail import GmailTransport
    connect(f)
    expire(f)
    client = f.service.clients['google']
    refresh = client.refresh

    def expanded_refresh(token, scopes):
        refreshed = refresh(token, scopes)
        refreshed['scope'] = [*scopes, SCOPES['google']['calendar_read']]
        return refreshed

    def unexpected_fetch(*args, **kwargs):
        pytest.fail('Label metadata fetch ran with changed scopes')

    monkeypatch.setattr(client, 'refresh', expanded_refresh)
    monkeypatch.setattr(GmailTransport, 'get', unexpected_fetch)
    before = f.store.get('one')
    with pytest.raises(OAuthError, match='oauth_reauth_required'):
        f.service.list_gmail_labels(f.owner, connection_id='one', expected_revision=revision(f))
    after = f.store.get('one')
    assert after.connection['status'] == 'reauth_required'
    assert after.connection['granted_scopes'] == before.connection['granted_scopes']
    assert after.credential_generation == before.credential_generation


def test_account_health_reports_token_expiry_without_refreshing_or_exposing_tokens(f):
    connect(f)
    assert f.service.health(f.owner, 'one')['access_token_status'] == 'current'
    expire(f)
    before = dict(f.backend.data)
    health = f.service.health(f.owner, 'one')
    assert health['access_token_status'] == 'refresh_required'
    assert health['credential_health'] == 'available'
    assert 'disposable-access' not in json.dumps(health)
    assert 'disposable-refresh' not in json.dumps(health)
    assert f.service.clients['google'].refresh_count == 0 and f.backend.data == before
    with pytest.raises(PermissionError):
        f.service.health(f.other, 'one')


@pytest.mark.parametrize('change', [
    {'expires_at': float('nan')}, {'expires_at': float('inf')},
    {'expires_at': 'not-a-time'}, {'access_token': ''}, {'access_token': None},
])
def test_no_refresh_token_access_rejects_malformed_credentials(f, monkeypatch, change):
    connect(f)
    args = authorization(f)
    original = f.service._read
    monkeypatch.setattr(f.service, '_read', lambda record: {**original(record), **change})
    with pytest.raises(OAuthError, match='oauth_refresh_required'):
        f.service.token_without_refresh(f.owner, authorization=args)
    assert f.service.clients['google'].refresh_count == 0


@pytest.mark.parametrize('provider', ['google', 'microsoft'])
def test_connect_minimal_scopes_and_secret_free_health(f, provider):
    health = connect(f, provider)
    assert health['status'] == 'connected'
    assert health['capabilities'] == ['mail_read']
    assert set(f.service.clients[provider].scopes) == {'openid', SCOPES[provider]['mail_read']} | ({'offline_access'} if provider == 'microsoft' else set())
    assert not {'credential_ref', 'secrets', 'config', 'provider_subject'} & health.keys()
    assert 'disposable-' not in json.dumps(health)
    assert f.store.get('one').secrets == {}


@pytest.mark.parametrize('mutation', ['state', 'path', 'port', 'fragment', 'duplicate', 'error', 'code'])
def test_callback_rejects_failures_and_replay(f, mutation):
    url = begin(f)
    url = {'state': url.replace('state=', 'state=wrong'), 'path': url.replace('/callback', '/other'),
           'port': url.replace(':54321', ':54322'), 'fragment': url + '#x',
           'duplicate': url + '&state=duplicate', 'error': url + '&error=access_denied',
           'code': url.replace('code=', 'missing=')}[mutation]
    with pytest.raises(OAuthError, match='oauth_(state|callback)_invalid'):
        f.service.complete(f.owner, url)
    assert not f.backend.data


def test_state_expiry_owner_and_one_use(f):
    url = begin(f)
    with pytest.raises(OAuthError, match='state_invalid'):
        f.service.complete(f.other, url)
    pending = next(iter(f.service.pending.values()))
    pending.expires = 0
    with pytest.raises(OAuthError, match='state_expired'):
        f.service.complete(f.owner, url)
    url = begin(f)
    f.service.complete(f.owner, url)
    with pytest.raises(OAuthError, match='state_invalid'):
        f.service.complete(f.owner, url)


def test_missing_scope_and_refresh_consent(f):
    for changes, error in [({'scope': ['openid']}, 'scopes_missing'), ({'refresh_token': ''}, 'offline_consent_required')]:
        url = begin(f)
        client = f.service.clients['google']
        original = client.exchange
        client.exchange = lambda *args, original=original, changes=changes: dict(original(*args), **changes)
        with pytest.raises(OAuthError, match=error):
            f.service.complete(f.owner, url)
        client.exchange = original
        assert not f.backend.data


def test_reconnect_preserves_subject_and_does_not_restore_revoked_grants(f):
    connect(f)
    args = authorization(f)
    f.service.disconnect(f.owner, 'one', revision(f))
    assert not f.backend.data
    url = begin(f)
    f.service.clients['google'].identity = 'different-account'
    with pytest.raises(OAuthError, match='account_mismatch'):
        f.service.complete(f.owner, url)
    f.service.clients['google'].identity = 'subject-a'
    connect(f)
    with pytest.raises(PermissionError, match='grant_denied'):
        f.service.token(f.owner, authorization=args)


def test_multiple_accounts_never_guess_or_retarget(f):
    connect(f)
    f.service.clients['google'].identity = 'subject-b'
    connect(f, connection='two')
    assert f.store.get('one').connection['provider_subject'] == 'subject-a'
    assert f.store.get('two').connection['provider_subject'] == 'subject-b'
    with pytest.raises(OAuthError, match='unavailable'):
        f.service.health(f.owner, 'google')
    with pytest.raises(PermissionError, match='owner_required'):
        f.service.health(f.other, 'one')


@pytest.mark.parametrize('action', ['pause', 'disconnect', 'revoke'])
def test_authority_change_during_refresh_never_returns_token(f, action):
    connect(f)
    args = authorization(f)
    expire(f)
    def change():
        if action == 'revoke':
            f.accounts.revoke_grant(f.owner, grant_id='grant', expected_revision=1)
        else:
            getattr(f.service, action)(f.owner, 'one', revision(f))
    f.service.clients['google'].on_refresh = change
    with pytest.raises((OAuthError, PermissionError)):
        f.service.token(f.owner, authorization=args)
    assert f.store.get('one').connection['status'] == ('connected' if action == 'revoke' else ('paused' if action == 'pause' else 'disconnected'))


def test_pause_during_callback_cannot_reenable(f):
    url = begin(f)
    f.service.clients['google'].on_exchange = lambda: f.service.pause(f.owner, 'one', revision(f))
    with pytest.raises(OAuthError, match='revision_conflict'):
        f.service.complete(f.owner, url)
    assert not f.backend.data
    assert f.store.get('one').connection['status'] == 'paused'


def test_guard_rechecks_before_exchange_and_before_credential_persistence(f):
    allowed = {'value': True}
    f.service.session_guard = lambda session: allowed['value'] and session == f.owner
    url = begin(f)
    allowed['value'] = False
    with pytest.raises(PermissionError, match='oauth_session_unavailable'):
        f.service.complete(f.owner, url)
    assert not f.backend.data
    allowed['value'] = True
    url = begin(f)
    f.service.clients['google'].on_exchange = lambda: allowed.update(value=False)
    with pytest.raises(PermissionError, match='oauth_session_unavailable'):
        f.service.complete(f.owner, url)
    assert not f.backend.data
    assert f.store.get('one').connection['status'] == 'reauth_required'


def test_transaction_guard_rejects_owner_change_after_vault_write(f, monkeypatch):
    url = begin(f)
    allowed = {'value': True}
    f.service.transaction_guard = lambda db, session: db.in_transaction and allowed['value']
    original = f.store.credential_vault.put
    def put_then_change(*args, **kwargs):
        reference = original(*args, **kwargs)
        allowed['value'] = False
        return reference
    monkeypatch.setattr(f.store.credential_vault, 'put', put_then_change)
    with pytest.raises(PermissionError, match='oauth_session_unavailable'):
        f.service.complete(f.owner, url)
    assert not f.backend.data
    assert not f.store.get('one').credential_ref


def test_refresh_serialized_across_services_and_rotation_saved(f):
    connect(f)
    args = authorization(f)
    expire(f)
    other = OAuthService(f.accounts, f.configs, client_factory=FakeProvider)
    authority_revision = revision(f)
    started, release = threading.Event(), threading.Event()
    f.service.clients['google'].on_refresh = lambda: (started.set(), release.wait(5))
    result = []
    thread = threading.Thread(target=lambda: result.append(f.service.token(f.owner, authorization=args)))
    thread.start()
    try:
        assert started.wait(5)
        with pytest.raises(OAuthError, match='operation_in_progress'):
            other.token(f.owner, authorization=args)
    finally:
        release.set()
        thread.join(5)
    assert result == ['rotated-access']
    assert revision(f) == authority_revision
    assert other.token(f.owner, authorization=args) == 'rotated-access'
    assert f.service.clients['google'].refresh_count == 1
    assert other.clients['google'].refresh_count == 0
    assert len(f.backend.data) == 1
    assert 'rotated-refresh' in next(iter(f.backend.data.values())).decode()


@pytest.mark.parametrize('provider', ['google', 'microsoft'])
def test_revoked_consent_and_redacted_errors(f, provider, caplog):
    connect(f, provider)
    args = authorization(f)
    args['required_scopes'] = [SCOPES[provider]['mail_read']]
    expire(f)
    class Revoked(Exception):
        error = 'invalid_grant'
    def fail(*args):
        raise Revoked('secret-provider-token-and-description')
    f.service.clients[provider].refresh = fail
    with pytest.raises(OAuthError, match='reauth_required') as exc:
        f.service.token(f.owner, authorization=args)
    assert f.service.health(f.owner, 'one')['status'] == 'reauth_required'
    assert 'secret-provider' not in str(exc.value) + caplog.text


def test_disconnect_offline_pending_local_deletion_and_reconnect(f):
    connect(f)
    args = authorization(f)
    def offline(*args):
        raise OSError('private diagnostic')
    f.service.clients['google'].revoke = offline
    health = f.service.disconnect(f.owner, 'one', revision(f))
    assert health['revocation_pending'] and health['credential_health'] == 'absent'
    assert not f.backend.data
    with pytest.raises(PermissionError):
        f.service.token(f.owner, authorization=args)
    connect(f)
    assert not f.service.health(f.owner, 'one')['revocation_pending']


def test_crash_lease_requires_explicit_reconnect(f):
    connect(f)
    args = authorization(f)
    with f.store._connect() as db:
        db.execute("INSERT INTO v2_oauth_leases VALUES ('one', 'crashed-worker')")
    with pytest.raises(OAuthError, match='operation_in_progress'):
        f.service.token(f.owner, authorization=args)
    connect(f)
    assert f.service.token(f.owner, authorization=args) == 'disposable-access'


@pytest.mark.parametrize('provider', ['google', 'microsoft'])
def test_real_library_transport_pkce_no_secrets_and_no_debug_leak(f, provider, monkeypatch, caplog):
    import hashlib
    import base64
    import logging
    import requests
    from urllib.parse import parse_qs, urlsplit
    from alphonse.agent_v2.integrations.oauth import ProviderClient
    pytest.importorskip('requests_oauthlib')
    client = ProviderClient(next(c for c in f.configs if c.provider == provider))
    f.service.clients[provider] = client
    result = f.service.begin(f.owner, connection_id='one', provider=provider, capabilities=['mail_read'], expected_revision=0)
    params = parse_qs(urlsplit(result['authorization_url']).query)
    state = params['state'][0]
    verifier = f.service.pending[state].verifier
    assert params['code_challenge_method'] == ['S256']
    assert params['code_challenge'] == [base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()]
    assert params['response_type'] == ['code']
    assert 'client_secret' not in params
    calls = []
    def send(session, request, **kwargs):
        calls.append(request)
        response = requests.Response()
        response.status_code, response.request = 200, request
        response.headers['Content-Type'] = 'application/json'
        if request.method == 'POST':
            body = parse_qs(request.body)
            assert 'client_secret' not in body and 'Authorization' not in request.headers
            assert body['client_id'] == ['public-app']
            if body['grant_type'] == ['authorization_code']:
                assert body['code_verifier'] == [verifier]
                assert body['redirect_uri'] == [client.config.redirect_uri]
                assert body['code'] == ['disposable-code']
            else:
                assert body['refresh_token'] == ['transport-refresh-secret']
            scope = 'Mail.Read' if provider == 'microsoft' else SCOPES[provider]['mail_read']
            response._content = json.dumps(dict(access_token='transport-access-secret', refresh_token='transport-refresh-secret',
                expires_in=3600, token_type='Bearer', scope=scope)).encode()
        else:
            assert request.headers['Authorization'] == 'Bearer transport-access-secret'
            response._content = b'{"sub": "verified-subject"}'
        return response
    monkeypatch.setattr(requests.Session, 'send', send)
    caplog.set_level(logging.DEBUG)
    health = f.service.complete(f.owner, client.config.redirect_uri + '?' + urlencode({'state': state, 'code': 'disposable-code'}))
    assert health['status'] == 'connected'
    args = authorization(f)
    args['required_scopes'] = [SCOPES[provider]['mail_read']]
    expire(f)
    assert f.service.token(f.owner, authorization=args) == 'transport-access-secret'
    assert len(calls) == 3  # Exchange, identity, refresh; no monitoring.
    assert 'transport-access-secret' not in caplog.text
    assert 'transport-refresh-secret' not in caplog.text
    assert 'disposable-code' not in caplog.text


def test_missing_dependency_is_opt_in_and_safe(f, monkeypatch):
    import builtins
    from alphonse.agent_v2.integrations.oauth import ProviderClient
    original = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == 'requests_oauthlib':
            raise ImportError('private-path')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', missing)
    service = OAuthService(f.accounts, f.configs)  # No library import at construction.
    with pytest.raises(OAuthError, match='setup_unavailable'):
        service.begin(f.owner, connection_id='one', provider='google', capabilities=['mail_read'], expected_revision=0)
    assert not f.backend.data


def test_google_desktop_client_secret_import_and_transport(f, tmp_path, monkeypatch, caplog):
    import logging
    import requests
    from urllib.parse import parse_qs
    from alphonse.agent_v2.integrations.google_client_setup import import_desktop_client
    pytest.importorskip('requests_oauthlib')
    source = tmp_path / 'client.json'
    source.write_text(json.dumps({'installed': {'client_id': 'public-app', 'client_secret': 'fixture-client-secret'}}))
    ref = import_desktop_client(str(source), 'public-app', f.store.credential_vault)
    assert source.exists() and 'fixture-client-secret' not in repr(ref)
    config = ClientConfig('google', 'public-app', 'http://127.0.0.1:54321/callback', ref)
    assert ref not in repr(config) and 'fixture-client-secret' not in repr(config)
    service = OAuthService(f.accounts, [config])
    client = service.clients['google']
    sent = []
    def send(session, request, **kwargs):
        body = parse_qs(request.body)
        assert body['client_id'] == ['public-app']
        assert body['client_secret'] == ['fixture-client-secret']
        assert 'Authorization' not in request.headers
        sent.append(body['grant_type'][0])
        response = requests.Response()
        response.status_code, response.request = 200, request
        response.headers['Content-Type'] = 'application/json'
        response._content = json.dumps({'access_token': 'fixture-access', 'refresh_token': 'fixture-refresh',
                                        'expires_in': 3600, 'token_type': 'Bearer'}).encode()
        return response
    monkeypatch.setattr(requests.Session, 'send', send)
    caplog.set_level(logging.DEBUG)
    client.exchange('fixture-code', 'x' * 64, [])
    client.refresh({'refresh_token': 'fixture-refresh'}, [])
    assert sent == ['authorization_code', 'refresh_token']
    assert all(value not in caplog.text for value in ('fixture-client-secret', 'fixture-code', 'fixture-refresh'))
    with pytest.raises(ValueError, match='google_client_import_unavailable'):
        import_desktop_client(str(source), 'different-client', f.store.credential_vault)
    link = tmp_path / 'link.json'
    link.symlink_to(source)
    with pytest.raises(ValueError, match='google_client_import_unavailable'):
        import_desktop_client(str(link), 'public-app', f.store.credential_vault)


def test_client_secret_unavailable_fails_closed_with_static_error(f):
    from alphonse.agent_v2.integrations.oauth import ProviderClient
    config = ClientConfig('google', 'public-app', 'http://127.0.0.1:54321/callback', 'a' * 32)
    def fail(*_):
        raise RuntimeError('fixture-secret-not-to-print')
    client = ProviderClient(config, secret_resolver=fail)
    with pytest.raises(OAuthError, match='oauth_client_credential_unavailable') as error:
        client._client_secret()
    assert 'fixture-secret' not in str(error.value)
    with pytest.raises(OAuthError, match='oauth_client_configuration_invalid'):
        ClientConfig('microsoft', 'public-app', 'http://127.0.0.1:54321/callback', 'a' * 32)


def test_google_desktop_client_import_rejects_unsafe_files(f, tmp_path):
    import os
    from alphonse.agent_v2.integrations.google_client_setup import import_desktop_client
    malformed = tmp_path / 'malformed.json'
    malformed.write_text('{')
    oversized = tmp_path / 'oversized.json'
    oversized.write_text('x' * 65537)
    fifo = tmp_path / 'pipe'
    os.mkfifo(fifo)
    for path in (malformed, oversized, fifo):
        with pytest.raises(ValueError, match='google_client_import_unavailable'):
            import_desktop_client(str(path), 'public-app', f.store.credential_vault)
    assert not f.backend.data


def test_google_client_import_cli_never_prints_credentials(f, tmp_path, monkeypatch, capsys):
    import sys
    from alphonse.agent_v2.integrations import google_client_setup
    source = tmp_path / 'client.json'
    source.write_text(json.dumps({'installed': {'client_id': 'public-app', 'client_secret': 'fixture-very-private'}}))
    monkeypatch.setattr(sys, 'argv', ['google_client_setup', str(source)])
    monkeypatch.setenv('ALPHONSE_GOOGLE_OAUTH_CLIENT_ID', 'public-app')
    monkeypatch.setattr(google_client_setup, 'CredentialVault', lambda: f.store.credential_vault)
    assert google_client_setup.main() == 0
    output = capsys.readouterr()
    assert 'fixture-very-private' not in output.out + output.err
    class Unavailable:
        def put(self, *_):
            raise RuntimeError('fixture-very-private')
    monkeypatch.setattr(google_client_setup, 'CredentialVault', Unavailable)
    assert google_client_setup.main() == 1
    output = capsys.readouterr()
    assert 'fixture-very-private' not in output.out + output.err
    assert 'Traceback' not in output.out + output.err


def test_daemon_api_lazy_trusted_session_and_no_clients_still_allows_disconnect(f, monkeypatch):
    from alphonse.agent_v2.daemon import V2Daemon
    for provider in SCOPES:
        for suffix in ['CLIENT_ID', 'REDIRECT_URI']:
            monkeypatch.delenv(f'ALPHONSE_{provider.upper()}_OAUTH_{suffix}', raising=False)
    daemon = SimpleNamespace(runtime=SimpleNamespace(integration_store=f.store, user_store=f.accounts.users),
        _lifecycle_lock=threading.RLock(), _optional_account_service=None)
    with pytest.raises(PermissionError, match='trusted_session_required'):
        V2Daemon.optional_accounts(daemon, {'actor_user_id': 'owner'})
    service = V2Daemon.optional_accounts(daemon, f.owner)
    assert service.clients == {}
    assert V2Daemon.optional_accounts(daemon, f.owner) is service
    connect(f)
    assert not service.health(f.owner, 'one')['provider_configured']
    service.disconnect(f.owner, 'one', revision(f))
    assert not f.backend.data
    assert not f.store.get('one').enabled


def test_refresh_storage_failure_and_unknown_network_outcome_require_reconnect(f, monkeypatch):
    connect(f)
    args = authorization(f)
    expire(f)
    def unavailable(*args):
        raise OSError('disposable private network info')
    f.service.clients['google'].refresh = unavailable
    with pytest.raises(OAuthError, match='reauth_required'):
        f.service.token(f.owner, authorization=args)
    assert f.service.health(f.owner, 'one')['status'] == 'reauth_required'
    f.service.clients['google'] = FakeProvider(f.configs[0])
    connect(f)
    expire(f)
    monkeypatch.setattr(f.backend, 'put', unavailable)
    with pytest.raises(OAuthError, match='refresh_not_persisted'):
        f.service.token(f.owner, authorization=args)
    assert f.service.health(f.owner, 'one')['status'] == 'reauth_required'


def test_successful_provider_revoke_and_locked_local_deletion_retry(f, monkeypatch):
    connect(f)
    f.service.clients['google'].revoked = True
    original = f.backend.delete
    def locked(*args):
        raise OSError('locked-keychain')
    monkeypatch.setattr(f.backend, 'delete', locked)
    health = f.service.disconnect(f.owner, 'one', revision(f))
    assert not health['revocation_pending']
    assert health['credential_deletion_pending']
    assert not f.store.get('one').enabled
    monkeypatch.setattr(f.backend, 'delete', original)
    f.store.retry_credential_deletion('one')
    assert not f.backend.data
    assert f.store.get('one').credential_ref == ''


def test_managed_binding_redacts_individual_tokens(f, tmp_path):
    from alphonse.agent_v2.skills import SkillStore
    from alphonse.agent_v2.skill_bindings import SkillBindings
    connect(f)
    authorization(f)
    skills = SkillStore(tmp_path / 'skills')
    skills.create_skill('fixture', 'Disposable', 'Synthetic only')
    bindings = SkillBindings(f.store, f.accounts.users, skills)
    bindings.configure(actor_id='owner', owner_id='owner', binding_id='binding', skill_id='skill:fixture',
        kind='account', target='one', grant_id='grant', capabilities=('mail_read',),
        required_scopes=(SCOPES['google']['mail_read'],))
    bindings.select(actor_id='owner', skill_id='skill:fixture', name='mail', binding_id='binding')
    result = bindings.use_account(f.owner, request=dict(skill_id='skill:fixture', name='mail',
        operation='mail_read', resources=['mailbox'], recipient_user_ids=[]),
        operation=lambda handle, credentials: {'echo': credentials['access_token'], 'refresh': credentials['refresh_token']})
    assert 'disposable-access' not in json.dumps(result)
    assert 'disposable-refresh' not in json.dumps(result)


def test_reconnect_waits_for_pending_deletion(f, monkeypatch):
    connect(f)
    original = f.backend.delete
    monkeypatch.setattr(f.backend, 'delete', lambda *args: (_ for _ in ()).throw(OSError('locked')))
    f.service.disconnect(f.owner, 'one', revision(f))
    with pytest.raises(OAuthError, match='cleanup_required'):
        begin(f)
    monkeypatch.setattr(f.backend, 'delete', original)
    f.service.retry_deletion(f.owner, 'one', revision(f))
    connect(f)


def test_provider_revocation_transport_is_narrow(f, monkeypatch):
    import requests
    from alphonse.agent_v2.integrations.oauth import ProviderClient
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=200)
    monkeypatch.setattr(requests, 'post', post)
    token = dict(access_token='disposable-access', refresh_token='disposable-refresh')
    assert ProviderClient(f.configs[0]).revoke(token)
    assert calls == [('https://oauth2.googleapis.com/revoke', {'data': {'token': 'disposable-refresh'}, 'timeout': 20})]
    assert not ProviderClient(f.configs[1]).revoke(token)
    assert len(calls) == 1


@pytest.mark.parametrize('action', ['pause', 'disconnect'])
def test_project_policy_survives_rotation_and_stops_on_lifecycle_change(f, tmp_path, action):
    from datetime import datetime, timedelta, timezone
    from alphonse.agent_v2.core.projects import ProjectStore
    from alphonse.agent_v2.skills import SkillStore
    from alphonse.agent_v2.skill_bindings import SkillBindings
    from alphonse.agent_v2.project_checks import ProjectCheckService
    connect(f)
    skills = SkillStore(tmp_path / 'skills')
    skills.create_skill('fixture', 'Disposable', 'Synthetic only')
    bindings = SkillBindings(f.store, f.accounts.users, skills)
    projects = ProjectStore(str(tmp_path / 'projects.db'))
    project = projects.create_project(name='School', root_path=str(tmp_path / 'school'), owner_user_id='owner')
    now = datetime.now(timezone.utc)
    f.accounts.save_grant(f.owner, grant_id='project-grant', connection_id='one', expected_revision=0,
        beneficiary_user_id='owner', operations=['mail_read'], resources=['mailbox'], recipient_user_ids=[],
        expires_at=(now + timedelta(days=2)).isoformat(), project_id=project.project_id)
    bindings.configure(actor_id='owner', owner_id='owner', binding_id='binding', skill_id='skill:fixture',
        kind='account', target='one', grant_id='project-grant', capabilities=('mail_read',),
        required_scopes=(SCOPES['google']['mail_read'],))
    bindings.select(actor_id='owner', skill_id='skill:fixture', name='mail', binding_id='binding')
    checks = ProjectCheckService(projects, bindings)
    policy = checks.save(f.owner, activate=True, authorization_reference='conversation:explicit', definition=dict(
        project_id=project.project_id,
        source_bindings=[dict(skill_id='skill:fixture', role='mail', binding_id='binding', connection_id='one',
                              grant_id='project-grant', operation='mail_read', resources=['mailbox'])],
        query={'text': 'school'}, trigger={'kind': 'schedule', 'weekdays': [0], 'local_time': '08:00'},
        timezone='America/Mexico_City', retrieval_window={'weekdays': [0], 'start': '08:00', 'end': '09:00'},
        duration={'ongoing': False, 'starts_at': (now - timedelta(days=1)).isoformat(), 'ends_at': (now + timedelta(days=1)).isoformat()},
        budgets=dict(max_pages=5, max_items=100, max_bytes=1000000, max_model_calls=2, max_runtime_seconds=60),
        recipient_user_ids=[], skill_hints=['skill:fixture']))
    expire(f)
    args = dict(connection_id='one', grant_id='project-grant', operation='mail_read', resources=['mailbox'],
                recipient_user_ids=[], required_scopes=[SCOPES['google']['mail_read']], project_id=project.project_id)
    assert f.service.token(f.owner, authorization=args) == 'rotated-access'
    assert checks.guard(policy['policy_id'], expected_revision=policy['revision'])
    getattr(f.service, action)(f.owner, 'one', revision(f))
    with pytest.raises(PermissionError):
        checks.guard(policy['policy_id'], expected_revision=policy['revision'])


@pytest.mark.parametrize('capability', ['calendar_discover', 'calendar_read'])
def test_calendar_consent_does_not_request_mail_or_writes(f, capability):
    from urllib.parse import urlencode
    f.service.begin(f.owner, connection_id='calendar', provider='google', capabilities=[capability], expected_revision=0)
    client = f.service.clients['google']
    assert set(client.scopes) == {'openid', SCOPES['google'][capability]}
    url = client.config.redirect_uri + '?' + urlencode({'state': client.state, 'code': 'disposable-code'})
    result = f.service.complete(f.owner, url)
    assert result['capabilities'] == [capability]


@pytest.mark.parametrize('capability', ['classroom_read', 'classroom_announcements_read'])
def test_classroom_consent_is_minimal_and_role_discovery_is_required(f, capability):
    from alphonse.agent_v2.integrations.oauth import capability_scopes
    f.service.begin(f.owner, connection_id='school', provider='google', capabilities=[capability], expected_revision=0)
    provider = f.service.clients['google']
    assert set(provider.scopes) == {'openid', *capability_scopes('google', capability)}
    callback = provider.config.redirect_uri + '?' + urlencode({'state': provider.state, 'code': 'disposable-code'})
    assert f.service.complete(f.owner, callback)['capabilities'] == [capability]
    assert not any('guardian' in s or s.endswith('.students.readonly') for s in provider.scopes)
    with pytest.raises(OAuthError, match='scopes_missing'):
        f.service._validated(dict(access_token='fake', refresh_token='fake-refresh', expires_at=time.time() + 3600,
            token_type='Bearer', scope=[SCOPES['google'][capability]]), provider.scopes)
