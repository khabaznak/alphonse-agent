"""Offline recovery uses disposable SQLite files and in-memory credentials only."""
import json
import sqlite3

import pytest

from test_account_grants import f, resolve, expiry
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


def test_recovery_detaches_client_secret_without_touching_live_vault(f):
    from alphonse.agent_v2.integrations.oauth import save_operator_client_secret, operator_client_secret

    save_operator_client_secret(f.service, AccountSession('parent'), 'google', 'public-client', 'synthetic-client-secret')
    snapshot = f.tmp / 'client-secret-restore.sqlite'
    with f.store._connect() as live, sqlite3.connect(snapshot) as copy:
        live.backup(copy)
        fence_restored_accounts(copy)
        assert copy.execute('SELECT COUNT(*) FROM v2_oauth_client_secrets').fetchone()[0] == 0
        assert copy.execute('SELECT COUNT(*) FROM v2_recovery_client_secrets').fetchone()[0] == 1
    assert operator_client_secret(f.store, 'google', 'public-client') == 'synthetic-client-secret'


def test_reconsent_on_fenced_copy_preserves_live_credential_and_old_grant_revocation(f):
    from alphonse.agent_v2.integrations.oauth import OAuthService, ClientConfig
    from test_oauth_lifecycle import FakeProvider

    snapshot = f.tmp / 'synthetic-reconsent.sqlite'
    with f.store._connect() as live, sqlite3.connect(snapshot) as copy:
        live.backup(copy)
        fence_restored_accounts(copy)
        # Use a supported provider label only inside this disposable copy.
        copy.execute("UPDATE v2_integrations SET provider_key='google' WHERE integration_id='one'")
    restored = SQLiteIntegrationStore(snapshot, credential_vault=f.store.credential_vault)
    accounts = AccountService(restored, f.users)
    oauth = OAuthService(accounts, [ClientConfig('google', 'synthetic-client', 'http://127.0.0.1:8765/google')], client_factory=FakeProvider)
    owner = AccountSession('parent')
    old_live = f.store.resolve_credentials(f.store.get('one'))
    old_ref = f.store.get('one').credential_ref
    # Simulate a successful provider response without provider I/O. Exercise the
    # same persistence path used after a real consent flow on an isolated copy.
    oauth._persist(owner, restored.get('one'), {
        'access_token': 'synthetic-access', 'refresh_token': 'synthetic-refresh',
        'token_type': 'Bearer', 'expires_at': '9999999999', 'scope': 'scope.read',
    }, subject='subject-one', capabilities=['read'], scopes=['scope.read'])
    assert f.store.get('one').credential_ref == old_ref
    assert f.store.resolve_credentials(f.store.get('one')) == old_live
    assert restored.get('one').credential_ref != old_ref
    accounts.save_grant(owner, grant_id='new-grant', connection_id='one', expected_revision=0,
                        beneficiary_user_id='child', operations=['read'], resources=['folder-a'],
                        recipient_user_ids=['parent'], expires_at=expiry())
    with pytest.raises(PermissionError):
        accounts.authorize(AccountSession('child'), connection_id='one', grant_id='grant-one',
                           operation='read', resources=['folder-a'], recipient_user_ids=['parent'],
                           required_scopes=['scope.read'])
    assert accounts.authorize(AccountSession('child'), connection_id='one', grant_id='new-grant',
                              operation='read', resources=['folder-a'], recipient_user_ids=['parent'],
                              required_scopes=['scope.read'])


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


def test_inspection_counts_staged_work_without_exposing_or_changing_it(f, capsys):
    from alphonse.agent_v2.account_recovery import prepare_restored_database, inspect_staged_database, reject_staged_database
    from alphonse.agent_v2.cli import main

    with f.store._connect() as db:
        db.execute('CREATE TABLE v2_outbox (message TEXT)')
        db.execute("INSERT INTO v2_outbox VALUES ('private-synthetic-content')")
    destination = f.tmp / 'inventory.sqlite'
    prepare_restored_database(f.store.db_path, destination)
    before = destination.read_bytes()
    counts = inspect_staged_database(destination)
    assert counts['managed_connections'] == 2 and counts['grants'] == 2
    assert counts['outbound_messages'] == 1 and counts['scheduled_tasks'] is None
    assert main(['inspect-account-recovery', str(destination)]) == 0
    output = capsys.readouterr().out
    assert 'outbound messages: 1' in output and 'unknown (table absent)' in output
    assert 'private-synthetic-content' not in output
    assert destination.read_bytes() == before
    with pytest.raises(RuntimeError, match='staged_database_not_activated'):
        reject_staged_database(destination)


def test_inspection_rejects_live_unstaged_and_symlinked_databases(f):
    from alphonse.agent_v2.account_recovery import inspect_staged_database, prepare_restored_database

    with pytest.raises(ValueError, match='not_staged'):
        inspect_staged_database(f.store.db_path)
    destination = f.tmp / 'stage.sqlite'
    prepare_restored_database(f.store.db_path, destination)
    link = f.tmp / 'link.sqlite'
    link.symlink_to(destination)
    with pytest.raises(ValueError, match='source_invalid'):
        inspect_staged_database(link)


def _recovery_deletion_fixture(f):
    from alphonse.agent_v2.account_recovery import prepare_restored_database

    evidence = {'connection_id': 'one', 'resource': 'folder-a', 'source_id': 'fixture-source',
                'state': 'available', 'deleted': False, 'content': {'text': 'synthetic private'}}
    finding = {'finding_id': 'f1', 'version': 1, 'policy_id': 'p', 'policy_revision': 1,
               'project_id': 'project', 'executing_user_id': 'child', 'audience_user_ids': ['parent'],
               'state': 'open', 'source_refs': [{'evidence_id': 'e1'}], 'summary': 'synthetic private'}
    with f.store._connect() as db:
        db.executescript('''CREATE TABLE v2_source_evidence(evidence_id TEXT PRIMARY KEY,definition TEXT);
            CREATE TABLE v2_findings(finding_id TEXT PRIMARY KEY,definition TEXT);
            CREATE TABLE v2_finding_versions(finding_id TEXT,version INTEGER,definition TEXT);
            CREATE TABLE v2_outbox(kind TEXT,message TEXT,status TEXT,metadata_json TEXT,
                last_error TEXT,lease_owner TEXT,lease_expires_at TEXT);''')
        db.execute('INSERT INTO v2_source_evidence VALUES (?,?)', ('e1', json.dumps(evidence)))
        db.execute('INSERT INTO v2_findings VALUES (?,?)', ('f1', json.dumps(finding)))
        db.execute('INSERT INTO v2_finding_versions VALUES (?,?,?)', ('f1', 1, json.dumps(finding)))
        db.execute('INSERT INTO v2_outbox VALUES (?,?,?,?,?,?,?)',
                   ('finding', 'synthetic private', 'pending', '{"finding_id":"f1","finding_version":1}', '', '', ''))
    stage = f.tmp / 'restore.sqlite'
    prepare_restored_database(f.store.db_path, stage)
    current = f.tmp / 'current-snapshot.sqlite'
    with f.store._connect() as live, sqlite3.connect(current) as copy:
        live.backup(copy)
        evidence.update(state='deleted', deleted=True, content={})
        copy.execute('UPDATE v2_source_evidence SET definition=?', (json.dumps(evidence),))
        # An item absent in the old backup needs a durable marker as well.
        evidence['source_id'] = 'newer-deletion'
        copy.execute('INSERT INTO v2_source_evidence VALUES (?,?)', ('e2', json.dumps(evidence)))
    return stage, current


def test_current_deletion_reconciles_staged_evidence_findings_and_outbox(f, capsys):
    from alphonse.agent_v2.account_recovery import apply_current_source_deletions, reject_staged_database
    from alphonse.agent_v2.cli import main

    stage, current = _recovery_deletion_fixture(f)
    assert main(['apply-recovery-deletions', str(stage), str(current)]) == 0
    output = capsys.readouterr().out
    assert '2 deletion markers' in output and 'synthetic private' not in output
    with sqlite3.connect(stage) as db:
        evidence = json.loads(db.execute('SELECT definition FROM v2_source_evidence').fetchone()[0])
        finding = json.loads(db.execute('SELECT definition FROM v2_findings').fetchone()[0])
        assert evidence['state'] == 'deleted' and evidence['content'] == {}
        assert finding['state'] == 'redacted' and 'summary' not in finding
        assert db.execute('SELECT COUNT(*) FROM v2_finding_versions').fetchone()[0] == 0
        assert db.execute('SELECT message,status FROM v2_outbox').fetchone() == ('[Finding unavailable]', 'failed')
        assert db.execute('SELECT COUNT(*) FROM v2_recovery_source_tombstones').fetchone()[0] == 2
    assert apply_current_source_deletions(stage, current)['source_tombstones'] == 2
    with pytest.raises(RuntimeError, match='staged_database_not_activated'):
        reject_staged_database(stage)


def test_recovery_deletion_rejects_incomplete_data_without_partial_writes(f):
    from alphonse.agent_v2.account_recovery import apply_current_source_deletions

    stage, current = _recovery_deletion_fixture(f)
    with sqlite3.connect(stage) as db:
        db.execute("UPDATE v2_findings SET definition='{}'")
    with pytest.raises(ValueError, match='invalid_finding'):
        apply_current_source_deletions(stage, current)
    with sqlite3.connect(stage) as db:
        assert json.loads(db.execute('SELECT definition FROM v2_source_evidence').fetchone()[0])['state'] == 'available'
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='v2_recovery_source_tombstones'").fetchone()
    with pytest.raises(ValueError, match='current_matches_stage'):
        apply_current_source_deletions(stage, stage)


def test_recovery_deletion_also_checks_historical_finding_versions(f):
    from alphonse.agent_v2.account_recovery import apply_current_source_deletions

    stage, current = _recovery_deletion_fixture(f)
    with sqlite3.connect(stage) as db:
        finding = json.loads(db.execute('SELECT definition FROM v2_findings').fetchone()[0])
        finding['source_refs'] = []
        db.execute('UPDATE v2_findings SET definition=?', (json.dumps(finding),))
    assert apply_current_source_deletions(stage, current)['findings_redacted'] == 1
    with sqlite3.connect(stage) as db:
        assert json.loads(db.execute('SELECT definition FROM v2_findings').fetchone()[0])['state'] == 'redacted'
        assert db.execute('SELECT COUNT(*) FROM v2_finding_versions').fetchone()[0] == 0


@pytest.mark.parametrize('table', ['v2_findings', 'v2_finding_versions'])
def test_recovery_deletion_rejects_findings_with_missing_source_evidence(f, table):
    from alphonse.agent_v2.account_recovery import apply_current_source_deletions, reject_staged_database

    stage, current = _recovery_deletion_fixture(f)
    with sqlite3.connect(stage) as db:
        value = json.loads(db.execute(f'SELECT definition FROM {table}').fetchone()[0])
        value['source_refs'].append({'evidence_id': 'missing-evidence'})
        db.execute(f'UPDATE {table} SET definition=?', (json.dumps(value),))

    with pytest.raises(ValueError, match='finding_evidence_missing'):
        apply_current_source_deletions(stage, current)
    with sqlite3.connect(stage) as db:
        assert json.loads(db.execute('SELECT definition FROM v2_source_evidence').fetchone()[0])['state'] == 'available'
        assert db.execute('SELECT message,status FROM v2_outbox').fetchone() == ('synthetic private', 'pending')
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='v2_recovery_source_tombstones'").fetchone()
    with pytest.raises(RuntimeError, match='staged_database_not_activated'):
        reject_staged_database(stage)


@pytest.mark.parametrize('state', ['deleted', 'revoked', 'expired'])
def test_recovery_respects_source_state_already_in_staged_copy(f, state):
    from alphonse.agent_v2.account_recovery import apply_current_source_deletions

    stage, current = _recovery_deletion_fixture(f)
    with sqlite3.connect(current) as db:
        evidence = json.loads(db.execute("SELECT definition FROM v2_source_evidence WHERE evidence_id='e1'").fetchone()[0])
        evidence.update(state='available', deleted=False)
        db.execute("UPDATE v2_source_evidence SET definition=? WHERE evidence_id='e1'", (json.dumps(evidence),))
    with sqlite3.connect(stage) as db:
        evidence = json.loads(db.execute("SELECT definition FROM v2_source_evidence WHERE evidence_id='e1'").fetchone()[0])
        evidence.update(state=state, deleted=state == 'deleted')
        db.execute("UPDATE v2_source_evidence SET definition=? WHERE evidence_id='e1'", (json.dumps(evidence),))
        if state == 'deleted':
            # Earlier evidence for the same source must also be invalidated.
            deleted_evidence = dict(evidence)
            db.execute("DELETE FROM v2_source_evidence WHERE evidence_id='e1'")
            evidence.update(state='available', deleted=False)
            db.execute("INSERT INTO v2_source_evidence VALUES (?,?)", ('earlier', json.dumps(evidence)))
            db.execute("INSERT INTO v2_source_evidence VALUES (?,?)", ('e1', json.dumps(deleted_evidence)))
            finding = json.loads(db.execute('SELECT definition FROM v2_findings').fetchone()[0])
            finding['source_refs'] = [{'evidence_id': 'earlier'}]
            db.execute('UPDATE v2_findings SET definition=?', (json.dumps(finding),))

    result = apply_current_source_deletions(stage, current)
    with sqlite3.connect(stage) as db:
        finding = json.loads(db.execute('SELECT definition FROM v2_findings').fetchone()[0])
        if state == 'expired':
            assert finding['state'] == 'open' and result['findings_redacted'] == 0
            assert json.loads(db.execute("SELECT definition FROM v2_source_evidence WHERE evidence_id='e1'").fetchone()[0])['content'] == {}
        else:
            assert finding['state'] == 'redacted' and result['findings_redacted'] == 1
            assert db.execute('SELECT message,status FROM v2_outbox').fetchone() == ('[Finding unavailable]', 'failed')
            assert json.loads(db.execute("SELECT definition FROM v2_source_evidence WHERE evidence_id='e1'").fetchone()[0])['content'] == {}
        if state == 'deleted':
            assert json.loads(db.execute("SELECT definition FROM v2_source_evidence WHERE evidence_id='earlier'").fetchone()[0])['state'] == 'deleted'
            assert db.execute('SELECT COUNT(*) FROM v2_recovery_source_tombstones').fetchone()[0] == 2


@pytest.mark.parametrize('state', ['revoked', 'expired'])
def test_recovery_applies_current_exact_evidence_state(f, state):
    from alphonse.agent_v2.account_recovery import apply_current_source_deletions

    stage, current = _recovery_deletion_fixture(f)
    with sqlite3.connect(current) as db:
        evidence = json.loads(db.execute("SELECT definition FROM v2_source_evidence WHERE evidence_id='e1'").fetchone()[0])
        evidence.update(state=state, deleted=False)
        db.execute("UPDATE v2_source_evidence SET definition=? WHERE evidence_id='e1'", (json.dumps(evidence),))
    result = apply_current_source_deletions(stage, current)
    with sqlite3.connect(stage) as db:
        restored = json.loads(db.execute("SELECT definition FROM v2_source_evidence WHERE evidence_id='e1'").fetchone()[0])
        finding = json.loads(db.execute('SELECT definition FROM v2_findings').fetchone()[0])
        assert restored['state'] == state and restored['content'] == {}
        assert finding['state'] == ('redacted' if state == 'revoked' else 'open')
        assert result['findings_redacted'] == (1 if state == 'revoked' else 0)


def test_recovery_rejects_current_evidence_identity_mismatch(f):
    from alphonse.agent_v2.account_recovery import apply_current_source_deletions

    stage, current = _recovery_deletion_fixture(f)
    with sqlite3.connect(current) as db:
        evidence = json.loads(db.execute("SELECT definition FROM v2_source_evidence WHERE evidence_id='e1'").fetchone()[0])
        evidence.update(state='revoked', deleted=False, source_id='different-source')
        db.execute("UPDATE v2_source_evidence SET definition=? WHERE evidence_id='e1'", (json.dumps(evidence),))
    with pytest.raises(ValueError, match='evidence_identity_mismatch'):
        apply_current_source_deletions(stage, current)
    with sqlite3.connect(stage) as db:
        assert json.loads(db.execute('SELECT definition FROM v2_source_evidence').fetchone()[0])['state'] == 'available'
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='v2_recovery_source_tombstones'").fetchone()


@pytest.mark.parametrize('metadata,error', [('{bad', 'outbox_invalid'), ('{}', 'outbox_finding_missing'),
                                            ('{"finding_id":"missing"}', 'outbox_finding_missing'),
                                            ('{"finding_id":"f1"}', 'outbox_invalid'),
                                            ('{"finding_id":"f1","finding_version":true}', 'outbox_invalid'),
                                            ('{"finding_id":"f1","finding_version":2}', 'outbox_version_missing')])
def test_recovery_rejects_unresolvable_finding_deliveries(f, metadata, error):
    from alphonse.agent_v2.account_recovery import apply_current_source_deletions

    stage, current = _recovery_deletion_fixture(f)
    with sqlite3.connect(stage) as db:
        db.execute('UPDATE v2_outbox SET metadata_json=?', (metadata,))
    with pytest.raises(ValueError, match=error):
        apply_current_source_deletions(stage, current)
    with sqlite3.connect(stage) as db:
        assert json.loads(db.execute('SELECT definition FROM v2_source_evidence').fetchone()[0])['state'] == 'available'
        assert db.execute('SELECT message,status FROM v2_outbox').fetchone() == ('synthetic private', 'pending')
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='v2_recovery_source_tombstones'").fetchone()


def test_recovery_redacts_delivery_of_an_already_redacted_finding(f):
    from alphonse.agent_v2.account_recovery import apply_current_source_deletions

    stage, current = _recovery_deletion_fixture(f)
    with sqlite3.connect(stage) as db:
        finding = json.loads(db.execute('SELECT definition FROM v2_findings').fetchone()[0])
        finding.update(state='redacted', source_refs=[])
        db.execute('UPDATE v2_findings SET definition=?', (json.dumps(finding),))
    result = apply_current_source_deletions(stage, current)
    assert result['findings_redacted'] == 1
    with sqlite3.connect(stage) as db:
        assert db.execute('SELECT message,status FROM v2_outbox').fetchone() == ('[Finding unavailable]', 'failed')
        assert db.execute('SELECT COUNT(*) FROM v2_finding_versions').fetchone()[0] == 0
