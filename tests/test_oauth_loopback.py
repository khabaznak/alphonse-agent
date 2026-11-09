"""Real local callback transport with synthetic provider and memory credentials."""
import socket
from urllib.error import HTTPError
from urllib.request import urlopen
from urllib.parse import urlencode

import pytest

from test_oauth_lifecycle import f, begin
from alphonse.agent_v2.integrations.oauth import ClientConfig
from alphonse.agent_v2.integrations.oauth_loopback import OAuthLoopback


def configure(f, provider, path='/callback'):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    client = f.service.clients[provider]
    client.config = ClientConfig(provider, 'synthetic-public', f'http://127.0.0.1:{port}{path}')
    return client


def test_browser_callback_and_replay_no_private_response_or_logs(f, capsys):
    client = configure(f, 'google')
    transport = OAuthLoopback(f.service)
    transport.start('google')
    try:
        url = begin(f)
        with urlopen(url, timeout=2) as response:
            assert response.status == 200
            body = response.read().decode()
            assert 'Account connected' in body
            assert response.headers['Cache-Control'] == 'no-store'
            assert 'disposable-' not in body and 'state=' not in body
        assert f.service.health(f.owner, 'one')['status'] == 'connected'
        with pytest.raises(HTTPError) as error:
            urlopen(url, timeout=2)
        assert error.value.code == 400
        assert capsys.readouterr().err == ''
    finally:
        transport.stop()


def test_unknown_state_wrong_path_and_denied_consent_fail_closed(f):
    client = configure(f, 'google')
    transport = OAuthLoopback(f.service)
    transport.start('google')
    try:
        callback = begin(f)
        for url in [callback.replace('/callback?', '/wrong?'), client.config.redirect_uri + '?state=unknown&code=disposable-code']:
            with pytest.raises(HTTPError):
                urlopen(url, timeout=2)
        denied = client.config.redirect_uri + '?' + urlencode({'state': client.state, 'error': 'access_denied', 'error_description': 'synthetic-private-detail'})
        with pytest.raises(HTTPError) as error:
            urlopen(denied, timeout=2)
        body = error.value.read().decode()
        assert 'Consent was not granted' in body
        assert 'synthetic-private-detail' not in body
        with pytest.raises(HTTPError):
            urlopen(callback, timeout=2)  # The denied attempt cannot be replayed.
        assert f.service.health(f.owner, 'one')['status'] == 'reauth_required'
    finally:
        transport.stop()


def test_two_provider_paths_share_listener_and_keep_actors(f):
    google = configure(f, 'google', '/google')
    microsoft = f.service.clients['microsoft']
    microsoft.config = ClientConfig('microsoft', 'synthetic-public', google.config.redirect_uri.replace('/google', '/microsoft'))
    transport = OAuthLoopback(f.service)
    transport.start('google')
    transport.start('microsoft')
    try:
        with urlopen(begin(f, provider='microsoft', connection='ms'), timeout=2) as response:
            assert response.status == 200
        with urlopen(begin(f, connection='google'), timeout=2) as response:
            assert response.status == 200
        assert f.service.health(f.owner, 'ms')['status'] == 'connected'
    finally:
        transport.stop()


def test_https_redirect_needs_explicit_non_desktop_transport(f):
    f.service.clients['google'].config = ClientConfig('google', 'public-app', 'https://callback.example/callback')
    with pytest.raises(ValueError, match='loopback_required'):
        OAuthLoopback(f.service).start('google')


def test_busy_callback_port_reports_safe_recovery_error(f):
    transport = OAuthLoopback(f.service)
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1', 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        f.service.clients['google'].config = ClientConfig('google', 'synthetic-public', f'http://127.0.0.1:{port}/callback')
        with pytest.raises(ValueError, match='^oauth_callback_listener_unavailable$'):
            transport.start('google')
    try:
        transport.start('google')
        with urlopen(begin(f), timeout=2) as response:
            assert response.status == 200
        assert f.service.health(f.owner, 'one')['status'] == 'connected'
    finally:
        transport.stop()


@pytest.mark.parametrize(('phase', 'message'), [
    ('exchange', 'provider token exchange failed'),
    ('identity', 'provider identity check failed'),
    ('storage', 'Secure credential storage is unavailable'),
])
def test_callback_reports_only_safe_failure_stage(f, monkeypatch, capsys, phase, message):
    from alphonse.agent_v2.integrations.credentials import CredentialError

    client = configure(f, 'google')
    transport = OAuthLoopback(f.service)
    transport.start('google')
    try:
        callback = begin(f)
        def fail(*args, **kwargs):
            raise OSError('synthetic-private-token-and-provider-details')
        if phase == 'exchange':
            monkeypatch.setattr(client, 'exchange', fail)
        elif phase == 'identity':
            monkeypatch.setattr(client, 'subject', fail)
        else:
            def vault_fail(*args, **kwargs):
                raise CredentialError('credential_backend_unavailable')
            monkeypatch.setattr(f.store.credential_vault, 'put', vault_fail)
        with pytest.raises(HTTPError) as error:
            urlopen(callback, timeout=2)
        body = error.value.read().decode()
        assert message in body
        assert 'synthetic-private' not in body and 'disposable-' not in body
        assert capsys.readouterr().err == ''
    finally:
        transport.stop()
