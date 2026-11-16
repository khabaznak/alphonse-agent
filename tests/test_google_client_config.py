import json
import sqlite3
import sys

import pytest

from alphonse.agent_v2.integrations.credentials import CredentialVault
from alphonse.agent_v2.integrations.google_client_config import resolve_google_config, save_google_config
from alphonse.agent_v2.integrations.oauth import ClientConfig
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def state(tmp_path):
    path = tmp_path / 'state.sqlite'
    users = V2UserStore(path)
    users.onboard(display_name='Owner', user_id='owner', users_root=tmp_path / 'users')
    config = ClientConfig('google', 'public-client', 'http://127.0.0.1:55331/callback', 'a' * 32)
    return path, users, config


def test_persistent_config_default_denied_and_environment_is_indivisible(state):
    path, users, config = state
    assert resolve_google_config(users, {}) is None
    save_google_config(path, config)
    assert resolve_google_config(users, {}) == config
    env = {'ALPHONSE_GOOGLE_OAUTH_CLIENT_ID': 'other',
           'ALPHONSE_GOOGLE_OAUTH_REDIRECT_URI': 'http://127.0.0.1:12345/callback'}
    result = resolve_google_config(users, env)
    assert result.client_id == 'other' and result.client_secret_ref == ''
    with pytest.raises(ValueError):
        resolve_google_config(users, {'ALPHONSE_GOOGLE_OAUTH_CLIENT_ID': 'other'})
    assert resolve_google_config(users, {'ALPHONSE_GOOGLE_OAUTH_CLIENT_ID': ''}) is None


def test_saved_config_validation_and_explicit_replace(state):
    path, users, config = state
    save_google_config(path, config)
    with pytest.raises(FileExistsError):
        save_google_config(path, config)
    replacement = ClientConfig('google', 'public-client', 'http://127.0.0.1:55331/callback', 'b' * 32)
    save_google_config(path, replacement, replace=True)
    assert resolve_google_config(users, {}) == replacement
    users._set_setting('google_desktop_oauth_v1', json.dumps({'schema_version': True}))
    with pytest.raises(ValueError, match='google_client_configuration_invalid'):
        resolve_google_config(users, {})


def test_save_requires_existing_db_and_active_owner(state, tmp_path):
    path, users, config = state
    missing = tmp_path / 'missing.sqlite'
    with pytest.raises(ValueError, match='configuration_unavailable'):
        save_google_config(missing, config)
    assert not missing.exists()
    save_google_config(path, config)
    users.update_user('owner', is_active=False)
    with pytest.raises(ValueError, match='google_client_configuration_invalid'):
        resolve_google_config(users, {})
    with pytest.raises(ValueError, match='configuration_unavailable'):
        save_google_config(path, config)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT 1 FROM v2_user_settings WHERE key='google_desktop_oauth_v1'").fetchone() is not None


def test_helper_persists_reference_without_outputting_secret_or_changing_history(state, tmp_path, monkeypatch, capsys):
    from alphonse.agent_v2.integrations import google_client_setup
    path, users, _ = state
    class Backend:
        def __init__(self): self.records = {}
        def put(self, ref, payload): self.records[ref] = payload
        def get(self, ref): return self.records[ref]
        def delete(self, ref): self.records.pop(ref)
    backend = Backend()
    source = tmp_path / 'client.json'
    source.write_text(json.dumps({'installed': {'client_id': 'public-client', 'client_secret': 'fixture-secret'}}))
    monkeypatch.setattr(google_client_setup, 'CredentialVault', lambda: CredentialVault(backend))
    monkeypatch.setattr(sys, 'argv', ['setup', str(source), '--persist'])
    monkeypatch.setenv('ALPHONSE_GOOGLE_OAUTH_CLIENT_ID', 'public-client')
    monkeypatch.setenv('ALPHONSE_GOOGLE_OAUTH_REDIRECT_URI', 'http://127.0.0.1:55331/callback')
    monkeypatch.setenv('ALPHONSE_V2_DB_PATH', str(path))
    assert google_client_setup.main() == 0
    assert 'fixture-secret' not in capsys.readouterr().out
    saved = resolve_google_config(users, {})
    assert CredentialVault(backend).get(saved.client_secret_ref, 'oauth-client:google:public-client', 1) == {'client_secret': 'fixture-secret'}
    with sqlite3.connect(path) as conn:
        assert 'fixture-secret' not in '\n'.join(conn.iterdump())
    assert google_client_setup.main() == 1  # Existing config is not silently replaced.
    assert len(backend.records) == 2  # Preserve even the new reference when save fails ambiguously.
    assert resolve_google_config(users, {}) == saved


def test_ambiguous_error_after_commit_never_deletes_saved_credential(state, tmp_path, monkeypatch, capsys):
    from alphonse.agent_v2.integrations import google_client_config, google_client_setup
    path, users, _ = state
    class Backend:
        def __init__(self): self.records = {}
        def put(self, ref, payload): self.records[ref] = payload
        def get(self, ref): return self.records[ref]
        def delete(self, ref): self.records.pop(ref)
    backend = Backend()
    vault = CredentialVault(backend)
    source = tmp_path / 'client.json'
    source.write_text(json.dumps({'installed': {'client_id': 'public-client', 'client_secret': 'fixture-secret'}}))
    monkeypatch.setattr(google_client_setup, 'CredentialVault', lambda: vault)
    monkeypatch.setattr(sys, 'argv', ['setup', str(source), '--persist'])
    monkeypatch.setenv('ALPHONSE_GOOGLE_OAUTH_CLIENT_ID', 'public-client')
    monkeypatch.setenv('ALPHONSE_GOOGLE_OAUTH_REDIRECT_URI', 'http://127.0.0.1:55331/callback')
    monkeypatch.setenv('ALPHONSE_V2_DB_PATH', str(path))
    actual_save = google_client_config.save_google_config
    def committed_then_error(*args, **kwargs):
        actual_save(*args, **kwargs)
        raise OSError('synthetic ambiguous failure')
    monkeypatch.setattr(google_client_config, 'save_google_config', committed_then_error)
    assert google_client_setup.main() == 1
    output = capsys.readouterr()
    assert 'fixture-secret' not in output.out + output.err
    saved = resolve_google_config(users, {})
    assert vault.get(saved.client_secret_ref, 'oauth-client:google:public-client', 1) == {'client_secret': 'fixture-secret'}
    assert len(backend.records) == 1


def test_helper_migrates_existing_reference_without_copy_or_overwrite(state, monkeypatch, capsys):
    from alphonse.agent_v2.integrations import google_client_setup
    path, users, _ = state
    class Backend:
        def __init__(self): self.records = {}
        def put(self, ref, payload): self.records[ref] = payload
        def get(self, ref): return self.records[ref]
        def delete(self, ref): self.records.pop(ref)
    backend = Backend()
    vault = CredentialVault(backend)
    ref = vault.put('oauth-client:google:public-client', 1, {'client_secret': 'fixture-private'})
    monkeypatch.setattr(google_client_setup, 'CredentialVault', lambda: vault)
    monkeypatch.setattr(sys, 'argv', ['setup', '--persist-existing'])
    monkeypatch.setenv('ALPHONSE_GOOGLE_OAUTH_CLIENT_ID', 'public-client')
    monkeypatch.setenv('ALPHONSE_GOOGLE_OAUTH_REDIRECT_URI', 'http://127.0.0.1:55331/callback')
    monkeypatch.setenv('ALPHONSE_GOOGLE_OAUTH_CLIENT_SECRET_REF', ref)
    monkeypatch.setenv('ALPHONSE_V2_DB_PATH', str(path))
    assert google_client_setup.main() == 0
    output = capsys.readouterr()
    assert 'fixture-private' not in output.out + output.err and ref not in output.out + output.err
    assert len(backend.records) == 1 and resolve_google_config(users, {}).client_secret_ref == ref
    assert google_client_setup.main() == 1
    assert len(backend.records) == 1  # Existing vault record must never be deleted.
    monkeypatch.setenv('ALPHONSE_GOOGLE_OAUTH_CLIENT_ID', 'wrong-client')
    assert google_client_setup.main() == 1
    assert len(backend.records) == 1
