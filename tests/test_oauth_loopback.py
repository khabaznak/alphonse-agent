"""Real local callback transport with synthetic provider and memory credentials."""
import socket
import threading
from urllib.error import HTTPError
from urllib.request import urlopen
from urllib.parse import urlencode

import pytest

from test_oauth_lifecycle import f, begin
from alphonse.agent_v2.integrations.oauth import ClientConfig, OAuthError, safe_callback_message
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
        for url in [callback.replace('/callback?', '/wrong?'), client.config.redirect_uri + '?state=unknown&code=disposable-code',
                    client.config.redirect_uri + '?' + urlencode({'state': client.state, 'error': 'access_denied'})]:
            with pytest.raises(HTTPError):
                urlopen(url, timeout=2)
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


def test_callback_messages_never_echo_unknown_error_text():
    for error in (OAuthError('disposable-secret'), RuntimeError('disposable-token'),
                  OAuthError('oauth_exchange_invalid_request', 'disposable-code')):
        message = safe_callback_message(error)
        assert 'disposable' not in message
        assert 'could not be completed' in message


def test_token_exchange_failure_reports_only_allowlisted_category(f, capsys, monkeypatch):
    try:
        from oauthlib.oauth2 import InvalidRequestError
    except ImportError:
        import sys
        from types import ModuleType
        oauthlib = ModuleType('oauthlib')
        oauth2 = ModuleType('oauthlib.oauth2')
        class OAuth2Error(Exception):
            error = None
        class InvalidRequestError(OAuth2Error):
            error = 'invalid_request'
            def __init__(self, description):
                super().__init__(description)
        oauth2.OAuth2Error = OAuth2Error
        monkeypatch.setitem(sys.modules, 'oauthlib', oauthlib)
        monkeypatch.setitem(sys.modules, 'oauthlib.oauth2', oauth2)
    client = configure(f, 'google')
    def fail():
        raise InvalidRequestError(description='disposable-token-provider-description')
    client.on_exchange = fail
    transport = OAuthLoopback(f.service)
    transport.start('google')
    try:
        callback = begin(f)
        with pytest.raises(HTTPError) as response:
            urlopen(callback, timeout=2)
        text = response.value.read().decode()
        assert 'invalid_request' in text
        assert 'OAuth client configuration' in text
        assert 'disposable' not in text and 'state=' not in text and 'code=' not in text
        assert capsys.readouterr().err == ''
    finally:
        transport.stop()


def test_stop_tracks_inflight_oauth_callback(f, monkeypatch):
    configure(f, 'google')
    entered, finish = threading.Event(), threading.Event()
    original = f.service.complete
    def complete(*args, **kwargs):
        entered.set()
        assert finish.wait(3)
        return original(*args, **kwargs)
    monkeypatch.setattr(f.service, 'complete', complete)
    transport = OAuthLoopback(f.service)
    transport.start('google')
    callback = begin(f)
    results = []
    def request():
        try:
            with urlopen(callback, timeout=5) as response:
                results.append(response.status)
        except Exception as exc:
            results.append(exc)
    client = threading.Thread(target=request)
    client.start()
    try:
        assert entered.wait(3)
        transport.stop()
        assert not transport.wait_quiescent(0)
        assert not transport.wait_quiescent(0.01)
        finish.set()
        client.join(3)
        assert transport.wait_quiescent(0)
        assert results == [200]
    finally:
        finish.set()
        client.join(3)
        transport.stop()


def test_loopback_thread_start_failure_closes_listener(f, monkeypatch):
    configure(f, 'google')
    transport = OAuthLoopback(f.service)
    original_start = threading.Thread.start
    def fail_loopback(thread):
        if thread.name == 'oauth-loopback':
            raise RuntimeError('synthetic thread failure')
        return original_start(thread)
    monkeypatch.setattr(threading.Thread, 'start', fail_loopback)
    with pytest.raises(RuntimeError, match='synthetic thread failure'):
        transport.start('google')
    assert not transport.servers
    transport.stop()
    assert transport.wait_quiescent(0)
