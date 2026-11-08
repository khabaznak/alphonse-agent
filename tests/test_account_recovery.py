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
    # IDs, account ownership, credential references and original scopes remain
    # recoverable; neither the stored token nor current vault is touched.
    assert restored.get('one').credential_ref == f.store.get('one').credential_ref
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
