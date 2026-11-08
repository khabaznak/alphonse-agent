"""Offline recovery uses disposable SQLite files and in-memory credentials only."""
import json
import sqlite3

import pytest

from test_account_grants import f, resolve
from alphonse.agent_v2.account_recovery import fence_restored_accounts
from alphonse.agent_v2.accounts import AccountService, AccountSession
from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
from alphonse.agent_v2.skill_bindings import SkillBindings


def test_restored_pre_revocation_snapshot_is_fenced_before_use(f):
    snapshot = f.tmp / 'offline-restored.sqlite'
    with f.store._connect() as live, sqlite3.connect(snapshot) as copy:
        live.backup(copy)
    # Current state is narrower than the old snapshot.
    f.service.revoke_grant(AccountSession('parent'), grant_id='grant-one', expected_revision=1)
    with sqlite3.connect(snapshot) as copy:
        result = fence_restored_accounts(copy)
    assert result == {'connections': 2, 'grants': 2}
    restored = SQLiteIntegrationStore(snapshot, credential_vault=f.store.credential_vault)
    service = AccountService(restored, f.users)
    bindings = SkillBindings(restored, f.users, f.skills)
    assert not restored.get('one').enabled
    assert restored.get('one').connection['health_code'] == 'restore_reconciliation_required'
    # A restored reference must not be used by ordinary reconnect/deletion paths:
    # it can still identify a credential used by the live database. Preserve it
    # separately for reconciliation without touching the shared vault.
    assert restored.get('one').credential_ref == ''
    with restored._connect() as db:
        escrow = db.execute("SELECT credential_ref, credential_generation FROM v2_recovery_credential_references WHERE integration_id='one'").fetchone()
    assert tuple(escrow) == (f.store.get('one').credential_ref, f.store.get('one').credential_generation)
    with pytest.raises(PermissionError):
        bindings.resolve_account(AccountSession('child'), skill_id='skill:fixture', name='mail',
            operation='read', resources=['folder-a'], recipient_user_ids=[])
    # Old revoked grant IDs cannot be revived through normal grant editing.
    with restored._connect() as db:
        grant = json.loads(db.execute("SELECT definition FROM v2_account_grants WHERE grant_id='grant-one'").fetchone()[0])
    assert grant['revoked_at'] and grant['resources'] == ['folder-a', 'folder-b']
    with pytest.raises((ValueError, PermissionError)):
        service.save_grant(AccountSession('parent'), grant_id='grant-one', connection_id='one',
            expected_revision=grant['revision'], beneficiary_user_id='child', operations=['read'],
            resources=['folder-a'], recipient_user_ids=[], expires_at=grant['expires_at'])


def test_recovery_rejects_corrupt_grant_atomically(f):
    snapshot = f.tmp / 'corrupt-restored.sqlite'
    with f.store._connect() as live, sqlite3.connect(snapshot) as copy:
        live.backup(copy)
    with sqlite3.connect(snapshot) as copy:
        copy.execute("UPDATE v2_account_grants SET definition='{}' WHERE grant_id='grant-one'")
        copy.commit()
        before = copy.execute("SELECT enabled, connection_json FROM v2_integrations WHERE integration_id='one'").fetchone()
        with pytest.raises(ValueError, match='invalid_grant'):
            fence_restored_accounts(copy)
        assert copy.execute("SELECT enabled, connection_json FROM v2_integrations WHERE integration_id='one'").fetchone() == before
        assert not copy.execute("SELECT 1 FROM sqlite_master WHERE name='v2_recovery_credential_references'").fetchone()


def test_recovery_never_overwrites_preserved_credential_references(f):
    snapshot = f.tmp / 'already-fenced.sqlite'
    with f.store._connect() as live, sqlite3.connect(snapshot) as copy:
        live.backup(copy)
        fence_restored_accounts(copy)
        before = copy.execute('SELECT * FROM v2_recovery_credential_references ORDER BY integration_id').fetchall()
        with pytest.raises(ValueError, match='already_fenced'):
            fence_restored_accounts(copy)
        assert copy.execute('SELECT * FROM v2_recovery_credential_references ORDER BY integration_id').fetchall() == before


def test_staged_copy_is_private_fenced_and_never_overwrites(f):
    import os
    import stat
    from alphonse.agent_v2.account_recovery import prepare_restored_database

    destination = f.tmp / 'prepared.sqlite'
    before = f.store.get('one')
    assert prepare_restored_database(f.store.db_path, destination) == {'connections': 2, 'grants': 2}
    assert stat.S_IMODE(os.stat(destination).st_mode) == 0o600
    assert f.store.get('one') == before
    with sqlite3.connect(destination) as copy:
        assert copy.execute('SELECT state FROM v2_recovery_staging').fetchone()[0] == 'isolated'
        assert copy.execute("SELECT enabled FROM v2_integrations WHERE integration_id='one'").fetchone()[0] == 0
    with pytest.raises(ValueError, match='destination_unavailable'):
        prepare_restored_database(f.store.db_path, destination)
    assert not list(f.tmp.glob('.alphonse-recovery-*'))


def test_staged_copy_rejects_bad_source_without_publishing(f):
    from alphonse.agent_v2.account_recovery import prepare_restored_database

    broken = f.tmp / 'bad.sqlite'
    with sqlite3.connect(broken) as db:
        db.execute('CREATE TABLE irrelevant (id INTEGER)')
    destination = f.tmp / 'not-created.sqlite'
    with pytest.raises(ValueError, match='missing_integrations'):
        prepare_restored_database(broken, destination)
    assert not destination.exists()
    assert not list(f.tmp.glob('.alphonse-recovery-*'))
    link = f.tmp / 'source-link.sqlite'
    link.symlink_to(broken)
    with pytest.raises(ValueError, match='source_invalid'):
        prepare_restored_database(link, destination)


def test_staging_marker_blocks_daemon_database_startup(f):
    from alphonse.agent_v2.account_recovery import prepare_restored_database, reject_staged_database

    destination = f.tmp / 'staged.sqlite'
    reject_staged_database(f.store.db_path)
    prepare_restored_database(f.store.db_path, destination)
    with pytest.raises(RuntimeError, match='staged_database_not_activated'):
        reject_staged_database(destination)


def test_staged_copy_requires_private_destination_directory(f):
    from alphonse.agent_v2.account_recovery import prepare_restored_database

    public = f.tmp / 'public'
    public.mkdir(mode=0o755)
    public.chmod(0o755)
    with pytest.raises(ValueError, match='directory_unprotected'):
        prepare_restored_database(f.store.db_path, public / 'restore.sqlite')
    assert not list(public.iterdir())


def test_daemon_refuses_staged_database_before_migration(f, monkeypatch):
    from alphonse.agent_v2.account_recovery import prepare_restored_database
    from alphonse.agent_v2 import daemon

    destination = f.tmp / 'staged-start.sqlite'
    prepare_restored_database(f.store.db_path, destination)
    monkeypatch.setenv('ALPHONSE_V2_DB_PATH', str(destination))
    monkeypatch.setattr(daemon, 'migrate_legacy_databases', lambda: pytest.fail('must not migrate staged recovery'))
    monkeypatch.setattr(daemon, '_claim_single_instance_lock', lambda *args: pytest.fail('must not start staged recovery'))
    with pytest.raises(RuntimeError, match='staged_database_not_activated'):
        daemon.main()


def test_cli_prepares_staged_copy_without_starting_daemon(f, capsys):
    from alphonse.agent_v2.cli import main
    from alphonse.agent_v2.account_recovery import reject_staged_database

    destination = f.tmp / 'cli-stage.sqlite'
    assert main(['prepare-account-recovery', f.store.db_path, str(destination)]) == 0
    message = capsys.readouterr().out
    assert '2 accounts fenced' in message and 'Do not start' in message
    assert 'disposable-' not in message
    with pytest.raises(RuntimeError, match='staged_database_not_activated'):
        reject_staged_database(destination)
