"""No real accounts, network, provider writes or host keychain access."""
import builtins
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

import pytest

from alphonse.agent_v2.integrations.credentials import (
    CredentialError, CredentialVault, EncryptedFileBackend, MacOSKeychainBackend,
)
from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
from alphonse.agent_v2.skill_bindings import SkillBindings
from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.users import V2UserStore

SECRET = 'disposable-refresh-"\n雪'


@pytest.fixture
def store(tmp_path, secure_vault):
    store = SQLiteIntegrationStore(tmp_path / 'integrations.sqlite', credential_vault=secure_vault)
    for account in ('one', 'two'):
        store.upsert(integration_id=account, provider_key='google-calendar', display_name=account, enabled=True)
    return store


def install(store, account='one', token=SECRET, generation=0):
    return store.install_credentials(account, {'refresh_token': token, 'client_secret': 'disposable-client'},
                                     expected_generation=generation)


def test_encrypted_roundtrip_restart_and_metadata_only_projections(store, tmp_path):
    record = install(store)
    assert record.secrets == {}
    assert store.resolve_credentials(record)['refresh_token'] == SECRET
    reopened = SQLiteIntegrationStore(store.db_path, credential_vault=store.credential_vault)
    assert reopened.resolve_credentials(reopened.get('one'))['refresh_token'] == SECRET
    projections = [record.to_dict(), record.to_dict(mask_secrets=False),
                   [r.to_dict() for r in reopened.list()]]
    assert SECRET not in json.dumps(projections, ensure_ascii=False)
    assert SECRET not in repr(record)
    # Keep a database reader alive while inspecting SQLite sidecars. Otherwise
    # GC can close the last connection and remove WAL between is_file/read_bytes.
    with store._connect() as reader:
        reader.execute('SELECT * FROM v2_integrations').fetchall()
        for path in tmp_path.rglob('*'):
            if path.is_file():
                assert SECRET.encode() not in path.read_bytes()
                assert b'disposable-client' not in path.read_bytes()
    with store._connect() as conn:
        assert conn.execute("SELECT secret_json FROM v2_integrations WHERE integration_id='one'").fetchone()[0] == '{}'


def test_connection_generation_and_reference_are_authenticated(store):
    first = install(store)
    other = install(store, 'two', token='second')
    with pytest.raises(CredentialError):
        store.credential_vault.get(first.credential_ref, 'two', first.credential_generation)
    with pytest.raises(CredentialError):
        store.credential_vault.get(first.credential_ref, 'one', first.credential_generation + 1)
    backend = store.credential_vault.backend
    (backend.root / other.credential_ref).write_bytes((backend.root / first.credential_ref).read_bytes())
    with pytest.raises(CredentialError):
        store.resolve_credentials(other)


@pytest.mark.parametrize('failure', ['missing', 'corrupt', 'locked', 'key_permissions', 'root_permissions', 'symlink', 'wrong_key'])
def test_unavailable_backend_fails_closed(store, failure):
    record = install(store)
    backend = store.credential_vault.backend
    path = backend.root / record.credential_ref
    if failure == 'missing':
        path.unlink()
    elif failure == 'corrupt':
        path.write_bytes(b'bad ciphertext')
    elif failure == 'locked':
        backend.key_file.unlink()
    elif failure == 'key_permissions':
        backend.key_file.chmod(0o644)
    elif failure == 'root_permissions':
        backend.root.chmod(0o755)
    elif failure == 'symlink':
        target = backend.root.parent / 'replacement'
        path.rename(target)
        path.symlink_to(target)
    else:
        backend.key_file.write_bytes(b'x' * 32)
    assert store.credential_vault.health(record.credential_ref, 'one', 1) == 'credential_backend_unavailable'
    with pytest.raises(CredentialError, match='^credential_backend_unavailable$') as exc:
        store.resolve_credentials(record)
    assert SECRET not in str(exc.value)
    assert store.get('one').secrets == {}


def test_key_must_be_separate_and_references_cannot_traverse(store):
    backend = store.credential_vault.backend
    key = backend.root / 'key'
    key.write_bytes(backend.key_file.read_bytes())
    key.chmod(0o600)
    vault = CredentialVault(EncryptedFileBackend(backend.root, key))
    with pytest.raises(CredentialError):
        vault.put('one', 1, {'token': SECRET})
    with pytest.raises(CredentialError):
        store.credential_vault.delete('../credential-key')
    assert backend.key_file.exists()


def test_missing_optional_import_does_not_break_store_or_messaging(tmp_path, monkeypatch):
    original = builtins.__import__
    def unavailable(name, *args, **kwargs):
        if name.startswith(('keyring', 'cryptography')):
            raise ImportError(SECRET)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', unavailable)
    monkeypatch.setenv('ALPHONSE_CREDENTIAL_BACKEND', 'encrypted-file')
    monkeypatch.setenv('ALPHONSE_CREDENTIAL_DIRECTORY', str(tmp_path))
    monkeypatch.setenv('ALPHONSE_CREDENTIAL_KEY_FILE', str(tmp_path / 'key'))
    store = SQLiteIntegrationStore()
    record = store.upsert(integration_id='telegram', provider_key='telegram', display_name='Telegram', secrets={'bot_token': 'legacy'})
    assert record.secrets['bot_token'] == 'legacy'
    with pytest.raises(CredentialError, match='^credential_backend_unavailable$'):
        store.install_credentials('telegram', {'refresh_token': SECRET}, expected_generation=0)
    assert store.get('telegram').credential_generation == 0
    assert store.get('telegram').secrets == {'bot_token': 'legacy'}


def test_backend_unconfigured_and_failure_before_switch(store, monkeypatch):
    monkeypatch.setenv('ALPHONSE_CREDENTIAL_BACKEND', 'invalid')
    store.credential_vault = CredentialVault()
    with pytest.raises(CredentialError, match='^credential_backend_unavailable$'):
        install(store)
    assert store.get('one').credential_ref == ''
    assert store.get('one').credential_generation == 0


def test_rotation_race_across_store_instances_has_one_winner(store):
    old = install(store)
    barrier = threading.Barrier(2)
    backend = store.credential_vault.backend
    class RacingBackend:
        def put(self, reference, payload):
            backend.put(reference, payload)
            barrier.wait(timeout=5)
        def get(self, reference):
            return backend.get(reference)
        def delete(self, reference):
            backend.delete(reference)
    def rotate(token):
        concurrent = SQLiteIntegrationStore(store.db_path, credential_vault=CredentialVault(RacingBackend()))
        try:
            return concurrent.install_credentials('one', {'refresh_token': token}, expected_generation=1)
        except CredentialError as exc:
            return str(exc)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(rotate, ['rotated-a', 'rotated-b']))
    assert sum(isinstance(r, str) for r in results) == 1
    assert 'credential_generation_conflict' in results
    current = store.get('one')
    assert current.credential_generation == 2
    assert store.resolve_credentials(current)['refresh_token'] in {'rotated-a', 'rotated-b'}
    assert not (backend.root / old.credential_ref).exists()
    assert len(list(backend.root.iterdir())) == 1


def test_disconnect_wins_inflight_rotation(store):
    record = install(store)
    backend = store.credential_vault.backend
    original = backend.put
    def interrupted(reference, payload):
        original(reference, payload)
        store.disconnect_credentials('one', expected_generation=1)
    backend.put = interrupted
    with pytest.raises(CredentialError, match='generation_conflict'):
        install(store, token='late', generation=1)
    assert not store.get('one').enabled
    assert list(backend.root.iterdir()) == []


def test_disconnect_without_backend_blocks_use_and_retries_deletion(store):
    record = install(store)
    backend = store.credential_vault.backend
    original = backend.delete
    def locked(reference):
        raise RuntimeError(SECRET)
    backend.delete = locked
    disconnected = store.disconnect_credentials('one', expected_generation=1)
    assert not disconnected.enabled
    assert disconnected.revocation_pending
    assert disconnected.credential_deletion_pending
    assert disconnected.credential_generation == 2
    with pytest.raises(CredentialError):
        store.resolve_credentials(record)
    with pytest.raises(CredentialError):
        store.set_enabled('one', True)
    with pytest.raises(CredentialError):
        install(store, generation=2)
    backend.delete = original
    deleted = store.retry_credential_deletion('one')
    assert not deleted.credential_deletion_pending
    assert not deleted.credential_ref
    assert deleted.revocation_pending  # No provider call was made.
    assert list(backend.root.iterdir()) == []


def test_explicit_migration_preserves_account_and_does_not_touch_messaging(store):
    # Simulate the existing database, not a new plaintext API.
    with store._connect() as conn:
        conn.execute("UPDATE v2_integrations SET secret_json=? WHERE integration_id='one'", (json.dumps({'refresh_token': SECRET}, sort_keys=True),))
    store.upsert(integration_id='telegram', provider_key='telegram', display_name='T', secrets={'bot_token': 'legacy'})
    migrated = store.migrate_credentials('one')
    assert migrated.integration_id == 'one'
    assert migrated.secrets == {}
    assert store.resolve_credentials(migrated) == {'refresh_token': SECRET}
    assert store.migrate_credentials('one') == migrated
    with pytest.raises(CredentialError, match='migration_denied'):
        store.migrate_credentials('telegram')
    assert store.get('telegram').secrets == {'bot_token': 'legacy'}
    with pytest.raises(CredentialError, match='secure_storage_required'):
        store.upsert(integration_id='two', provider_key='google-calendar', display_name='Two', secrets={'refresh_token': SECRET})
    with pytest.raises(CredentialError, match='managed_account_protected'):
        store.upsert(integration_id='one', provider_key='different', display_name='One', enabled=True)
    store.upsert(integration_id='one', provider_key='google-calendar', display_name='Renamed', enabled=True)
    assert store.get('one').credential_ref == migrated.credential_ref


def test_secure_binding_delivery_redaction_and_denied_actor(store, tmp_path):
    install(store)
    users = V2UserStore(tmp_path / 'users.sqlite')
    users.onboard(display_name='Owner', user_id='owner', users_root=tmp_path / 'profiles')
    users.create_user(display_name='Other', user_id='other')
    skills = SkillStore(tmp_path / 'skills')
    skills.create_skill('synthetic', 'Synthetic', 'Use bindings.')
    bindings = SkillBindings(store, users, skills)
    bindings.configure(actor_id='owner', owner_id='owner', binding_id='test', skill_id='skill:synthetic',
                       kind='account', target='one', required_secrets=('refresh_token',))
    bindings.select(actor_id='owner', skill_id='skill:synthetic', name='account', binding_id='test')
    prepared = bindings.prepare(actor_id='owner', skill_id='skill:synthetic', names=['account'])
    assert prepared.values['account']['credentials'] == {'refresh_token': SECRET}
    assert bindings.use(actor_id='owner', skill_id='skill:synthetic', names=['account'],
                        operation=lambda _: {'nested': [SECRET, json.dumps(SECRET)], 'extra': 'disposable-client'}) == {
                            'nested': ['[REDACTED]', '"[REDACTED]"'], 'extra': '[REDACTED]'}
    assert bindings.management_status(actor_id='owner', skill_id='skill:synthetic') == [{'name': 'account', 'status': 'available'}]
    with pytest.raises(PermissionError):
        bindings.select(actor_id='other', skill_id='skill:synthetic', name='account', binding_id='test')
    def failure(_):
        raise RuntimeError(SECRET)
    with pytest.raises(RuntimeError, match='^skill_binding_operation_failed$'):
        bindings.use(actor_id='owner', skill_id='skill:synthetic', names=['account'], operation=failure)
    store.credential_vault.backend.key_file.unlink()
    assert bindings.management_status(actor_id='owner', skill_id='skill:synthetic') == [
        {'name': 'account', 'status': 'credential_backend_unavailable'}]


def test_keychain_selects_only_native_backend_and_sanitizes_locked_errors(monkeypatch):
    import sys
    from types import ModuleType
    calls = []
    module = ModuleType('keyring.backends.macOS')
    class Keyring:
        def set_password(self, service, reference, value):
            calls.append((service, reference))
            raise RuntimeError(SECRET)
    module.Keyring = Keyring
    monkeypatch.setitem(sys.modules, 'keyring.backends.macOS', module)
    monkeypatch.setattr(sys, 'platform', 'darwin')
    with pytest.raises(CredentialError, match='^credential_backend_unavailable$'):
        CredentialVault(MacOSKeychainBackend()).put('one', 1, {'refresh_token': SECRET})
    assert calls[0][0] == 'org.alphonse.oauth.v1'


def test_failed_migration_preserves_legacy_row_for_explicit_recovery(store):
    with store._connect() as conn:
        conn.execute("UPDATE v2_integrations SET secret_json=? WHERE integration_id='one'",
                     (json.dumps({'refresh_token': SECRET}),))
    backend = store.credential_vault.backend
    backend.key_file.unlink()
    with pytest.raises(CredentialError):
        store.migrate_credentials('one')
    assert store.get('one').secrets == {'refresh_token': SECRET}
    assert store.get('one').credential_generation == 0


def test_resolve_rechecks_disconnect_during_backend_read(store):
    record = install(store)
    backend = store.credential_vault.backend
    original = backend.get
    def disconnect(reference):
        payload = original(reference)
        store.disconnect_credentials('one', expected_generation=1)
        return payload
    backend.get = disconnect
    with pytest.raises(CredentialError, match='generation_conflict'):
        store.resolve_credentials(record)


def test_secure_bash_output_and_task_evidence_never_contain_credentials(store, tmp_path):
    from alphonse.agent_v2.core.core import ToolExecutionContext
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry.native.bash import execute_bash
    install(store)
    users = V2UserStore(tmp_path / 'users.sqlite')
    users.onboard(display_name='Owner', user_id='owner', users_root=tmp_path / 'profiles')
    skills = SkillStore(tmp_path / 'skills')
    skills.create_skill('synthetic', 'Synthetic', 'Use bindings.')
    bindings = SkillBindings(store, users, skills)
    bindings.configure(actor_id='owner', owner_id='owner', binding_id='test', skill_id='skill:synthetic',
                       kind='account', target='one', required_secrets=('refresh_token',))
    bindings.select(actor_id='owner', skill_id='skill:synthetic', name='account', binding_id='test')
    task = TaskState(goal='Disposable credential delivery', user='owner',
                     metadata={'v3_skill_selection': {'selected_skill_ids': ['skill:synthetic']}})
    context = ToolExecutionContext(task=task, messages=InMemoryMessageQueue(), skill_bindings=bindings)
    result = execute_bash({'skill_id': 'skill:synthetic', 'binding_names': ['account'],
                           'command': 'printf "%s" "$ALPHONSE_SKILL_BINDINGS"'}, context=context)
    assert result['exit_code'] == 0
    assert '[REDACTED]' in result['stdout']
    assert 'disposable-client' not in result['stdout']
    assert SECRET not in json.dumps(result, ensure_ascii=False)
    assert SECRET not in repr(task)
    assert SECRET not in (skills.skills_dir / 'synthetic' / 'SKILL.md').read_text()
