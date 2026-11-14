"""Managed preflight is disposable and is never delivery authorization."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import sqlite3

import pytest

from test_single_loop_delivery_reservation import setup
from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.core.intelligence.v3.single_loop_delivery_reservation import ACTION_KEY
from alphonse.agent_v2.core.intelligence.v3.single_loop_managed_delivery import (
    MANAGED_PREFLIGHT_KEY, reserve_managed_delivery,
)
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable
from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
from alphonse.agent_v2.skill_bindings import SkillBindings
from alphonse.agent_v2.skills import SkillStore


@pytest.fixture
def managed(setup, tmp_path, secure_vault):
    queue, users, questions, store, claim, version, now, action = setup
    integrations = SQLiteIntegrationStore(questions.db_path, credential_vault=secure_vault)
    skills = SkillStore(tmp_path / 'skills')
    skills.create_skill('fixture', 'Disposable skill', 'Synthetic test only.')
    bindings = SkillBindings(integrations, users, skills)
    session = AccountSession('alice')
    bindings.accounts.create_connection(session, connection_id='synthetic', provider_key='fixture-provider',
        provider_subject='fixture', display_name='Fixture')
    integrations.install_credentials('synthetic', {'token': 'fixture-secret'}, expected_generation=0)
    bindings.accounts.save_connection(session, connection_id='synthetic', expected_revision=1,
        provider_subject='fixture', capabilities=['send'], granted_scopes=['scope.send'])
    bindings.accounts.save_grant(session, grant_id='grant-1', connection_id='synthetic', expected_revision=0,
        beneficiary_user_id='alice', operations=['send'], resources=['channel'],
        recipient_user_ids=['alice'], expires_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat())
    bindings.configure(actor_id='alice', owner_id='alice', binding_id='binding-one', skill_id='skill:fixture',
        kind='account', target='synthetic', grant_id='grant-1', required_scopes=('scope.send',), capabilities=('send',))
    bindings.select(actor_id='alice', skill_id='skill:fixture', name='messages', binding_id='binding-one')
    action = replace(action, connection_revision=2)
    return setup, bindings, session, action


def invoke(fixture, **changes):
    (queue, users, questions, store, claim, version, now, _), bindings, session, action = fixture
    args = dict(bindings=bindings, session=session, skill_id='skill:fixture', role='messages', operation='send',
        resources=['channel'], project_id=None, task_id='task', expected_version=version,
        binding=action, message='synthetic message', now_ms=1100, clock=lambda: now)
    args.update(changes)
    return reserve_managed_delivery(store, claim, **args)


def test_real_resolver_pins_scoped_preflight_with_reservation(managed, monkeypatch):
    setup, bindings, _, _ = managed
    calls = []
    original = bindings.resolve_account
    def resolve(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(bindings, 'resolve_account', resolve)
    saved = invoke(managed)
    metadata = setup[2].load_task_checkpoint('task').metadata
    recorded = metadata[MANAGED_PREFLIGHT_KEY]
    assert len(calls) == 1 and calls[0]['expected_binding_id'] == 'binding-one'
    assert recorded['binding_id'] == 'binding-one' and recorded['binding_revision'] == 1
    assert recorded['operation'] == 'send' and recorded['resources'] == ['channel']
    assert recorded['required_scopes'] == ['scope.send'] and recorded['project_id'] is None
    assert recorded['grant_id'] == 'grant-1' and recorded['connection_revision'] == 2
    assert saved.state.inflight == recorded['action_id'] == metadata[ACTION_KEY]['action_id']
    assert metadata[ACTION_KEY]['authority_mode'] == 'managed'
    assert 'fixture-secret' not in json.dumps(metadata)
    ref = bindings.integrations.get('synthetic').credential_ref
    assert ref and ref not in json.dumps(metadata)


def test_lost_managed_preflight_never_becomes_synthetic_recovery(managed):
    from alphonse.agent_v2.core.intelligence.v3.single_loop_recovery_inspection import inspect_recovery
    setup, _, session, _ = managed
    saved = invoke(managed)
    with setup[2]._connect() as conn:
        raw = json.loads(conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0])
        raw['metadata'].pop(MANAGED_PREFLIGHT_KEY)
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(raw),))
    assert inspect_recovery(setup[3], session, task_id='task', expected_version=saved.version).status == 'unavailable'
    assert setup[3].load('task', 'alice', expected_version=saved.version).state.inflight == 'action'


def test_v2_managed_action_stays_pending_for_explicit_reconciliation(managed):
    from alphonse.agent_v2.core.intelligence.v3.single_loop_recovery_inspection import inspect_recovery
    setup, _, session, _ = managed
    saved = invoke(managed)
    with setup[2]._connect() as conn:
        raw = json.loads(conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0])
        raw['metadata'][ACTION_KEY]['schema_version'] = 2
        raw['metadata'][ACTION_KEY].pop('authority_mode')
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(raw),))
    assert inspect_recovery(setup[3], session, task_id='task', expected_version=saved.version).status == 'unavailable'
    assert setup[3].load('task', 'alice', expected_version=saved.version).state.inflight == 'action'


@pytest.mark.parametrize('change', ['grant', 'binding', 'choice', 'user', 'connection'])
def test_database_revocation_or_change_between_resolution_and_reservation_fails_closed(managed, monkeypatch, change):
    setup, bindings, session, _ = managed
    original = bindings.resolve_account
    def resolve(*args, **kwargs):
        handle = original(*args, **kwargs)
        if change == 'grant':
            bindings.accounts.revoke_grant(session, grant_id='grant-1', expected_revision=1)
            return handle
        with bindings.integrations._connect() as conn:
            if change == 'binding':
                conn.execute("UPDATE v2_skill_bindings SET enabled=0 WHERE binding_id='binding-one'")
            elif change == 'choice':
                conn.execute("DELETE FROM v2_skill_binding_choices WHERE actor_id='alice'")
            elif change == 'user':
                conn.execute("UPDATE v2_users SET is_active=0 WHERE user_id='alice'")
            else:
                conn.execute("UPDATE v2_integrations SET enabled=0 WHERE integration_id='synthetic'")
        return handle
    monkeypatch.setattr(bindings, 'resolve_account', resolve)
    with pytest.raises((PermissionError, ClaimFenceUnavailable)):
        invoke(managed)
    raw = setup[2].load_task_checkpoint('task')
    assert ACTION_KEY not in raw.metadata and MANAGED_PREFLIGHT_KEY not in raw.metadata
    assert not setup[3].load('task', 'alice', expected_version=setup[5]).state.inflight


@pytest.mark.parametrize('changes', [dict(operation='delete'), dict(resources=['other']),
                                     dict(project_id='other'), dict(role='missing')])
def test_denied_or_mismatched_scope_never_reserves(managed, changes):
    with pytest.raises((ValueError, PermissionError)):
        invoke(managed, **changes)
    assert ACTION_KEY not in managed[0][2].load_task_checkpoint('task').metadata


def test_read_authority_cannot_be_bound_to_delivery(managed):
    setup, bindings, session, _ = managed
    # Construct a disposable account where the real resolver would accept read.
    # A read capability must not be reinterpreted as permission for delivery.
    with bindings.integrations._connect() as conn:
        conn.execute("UPDATE v2_integrations SET connection_json=json_set(connection_json, '$.capabilities', json('[\"read\"]')) WHERE integration_id='synthetic'")
        conn.execute("UPDATE v2_account_grants SET definition=json_set(definition, '$.operations', json('[\"read\"]')) WHERE grant_id='grant-1'")
        conn.execute("UPDATE v2_skill_bindings SET requirements=json_set(requirements, '$.capabilities', json('[\"read\"]')) WHERE binding_id='binding-one'")
    assert bindings.resolve_account(session, skill_id='skill:fixture', name='messages', operation='read',
        resources=['channel'], recipient_user_ids=['alice']).operation == 'read'
    with pytest.raises(PermissionError, match='operation_unsupported'):
        invoke(managed, operation='read')
    task = setup[2].load_task_checkpoint('task')
    assert ACTION_KEY not in task.metadata and MANAGED_PREFLIGHT_KEY not in task.metadata
    assert not setup[3].load('task', 'alice', expected_version=setup[5]).state.inflight


def test_stored_read_preflight_is_not_a_delivery_binding(managed):
    from alphonse.agent_v2.core.intelligence.v3.single_loop_recovery_inspection import inspect_recovery
    setup, _, session, _ = managed
    saved = invoke(managed)
    with setup[2]._connect() as conn:
        raw = json.loads(conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0])
        raw['metadata'][MANAGED_PREFLIGHT_KEY]['operation'] = 'read'
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(raw),))
    assert inspect_recovery(setup[3], session, task_id='task', expected_version=saved.version).status == 'unavailable'
    assert setup[3].load('task', 'alice', expected_version=saved.version).state.inflight == 'action'


def test_revocation_writer_cannot_race_locked_reservation(managed, monkeypatch):
    from alphonse.agent_v2.core.intelligence.v3 import single_loop_managed_delivery as module
    original = module._capture
    calls = []
    def capture(conn, **kwargs):
        calls.append(1)
        result = original(conn, **kwargs)
        if len(calls) == 2:
            with sqlite3.connect(managed[0][2].db_path, timeout=0) as other:
                with pytest.raises(sqlite3.OperationalError, match='locked'):
                    other.execute("UPDATE v2_account_grants SET revision=revision+1 WHERE grant_id='grant-1'")
        return result
    monkeypatch.setattr(module, '_capture', capture)
    assert invoke(managed).state.inflight == 'action'
    assert len(calls) == 2


def test_separate_identity_store_fails_before_reservation(managed, tmp_path):
    from alphonse.agent_v2.users import V2UserStore
    bindings = managed[1]
    bindings.users = V2UserStore(tmp_path / 'different.sqlite')
    with pytest.raises(PermissionError, match='authority_unavailable'):
        invoke(managed)
    assert ACTION_KEY not in managed[0][2].load_task_checkpoint('task').metadata


def test_preflight_project_must_match_checkpoint(managed):
    questions = managed[0][2]
    with questions._connect() as conn:
        row = conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()
        raw = json.loads(row[0]); raw['project_id'] = 'another-project'
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(raw),))
    with pytest.raises(PermissionError):
        invoke(managed)
    assert ACTION_KEY not in questions.load_task_checkpoint('task').metadata


def test_unavailable_vault_or_denied_skill_prevents_preflight(managed, monkeypatch):
    bindings = managed[1]
    original = bindings.skills.can_use
    monkeypatch.setattr(bindings.skills, 'can_use', lambda *args: False)
    with pytest.raises(PermissionError):
        invoke(managed)
    monkeypatch.setattr(bindings.skills, 'can_use', original)
    monkeypatch.setattr(bindings.integrations.credential_vault, 'health', lambda *args: 'unavailable')
    with pytest.raises(PermissionError):
        invoke(managed)
    assert ACTION_KEY not in managed[0][2].load_task_checkpoint('task').metadata


def test_corrupt_or_boolean_grant_revision_fails_closed(managed):
    with managed[1].integrations._connect() as conn:
        row = conn.execute("SELECT definition FROM v2_account_grants WHERE grant_id='grant-1'").fetchone()
        grant = json.loads(row[0]); grant['revision'] = True
        conn.execute("UPDATE v2_account_grants SET definition=? WHERE grant_id='grant-1'", (json.dumps(grant),))
    with pytest.raises(PermissionError):
        invoke(managed)
    assert ACTION_KEY not in managed[0][2].load_task_checkpoint('task').metadata


def test_unscoped_destination_never_reserves(managed):
    with pytest.raises(ValueError, match='destination_not_in_resources'):
        invoke(managed, resources=['unrelated'])
    assert ACTION_KEY not in managed[0][2].load_task_checkpoint('task').metadata


def test_recovery_rechecks_real_authority_without_settling(managed):
    from alphonse.agent_v2.core.intelligence.v3.single_loop_recovery_inspection import inspect_managed_delivery_authority
    setup, bindings, session, _ = managed
    saved = invoke(managed)
    kwargs = dict(bindings=bindings, task_id='task', expected_version=saved.version, clock=lambda: setup[6])
    result = inspect_managed_delivery_authority(setup[3], session, setup[4], **kwargs)
    assert result.status == 'authority_checked'
    restored = setup[3].load('task', 'alice', expected_version=saved.version)
    assert restored.state.inflight == 'action'
    assert restored.budget.usage()['in_flight'] == ['action']
    assert inspect_managed_delivery_authority(setup[3], AccountSession('bob'), setup[4], **kwargs).status == 'unavailable'
    assert inspect_managed_delivery_authority(setup[3], session, setup[4], **{**kwargs, 'expected_version': saved.version + 1}).status == 'unavailable'


def test_recovery_revocation_during_real_resolution_preserves_marker(managed, monkeypatch):
    from alphonse.agent_v2.core.intelligence.v3.single_loop_recovery_inspection import inspect_managed_delivery_authority
    setup, bindings, session, _ = managed
    saved = invoke(managed)
    original = bindings.resolve_account
    def resolve(*args, **kwargs):
        handle = original(*args, **kwargs)
        bindings.accounts.revoke_grant(session, grant_id='grant-1', expected_revision=1)
        return handle
    monkeypatch.setattr(bindings, 'resolve_account', resolve)
    result = inspect_managed_delivery_authority(setup[3], session, setup[4], bindings=bindings,
        task_id='task', expected_version=saved.version, clock=lambda: setup[6])
    assert result.status == 'unavailable'
    assert setup[3].load('task', 'alice', expected_version=saved.version).state.inflight == 'action'


def test_recovery_denied_skill_and_unavailable_vault_are_opaque(managed, monkeypatch):
    from alphonse.agent_v2.core.intelligence.v3.single_loop_recovery_inspection import inspect_managed_delivery_authority
    setup, bindings, session, _ = managed
    saved = invoke(managed)
    kwargs = dict(bindings=bindings, task_id='task', expected_version=saved.version, clock=lambda: setup[6])
    original = bindings.skills.can_use
    monkeypatch.setattr(bindings.skills, 'can_use', lambda *args: False)
    assert inspect_managed_delivery_authority(setup[3], session, setup[4], **kwargs).status == 'unavailable'
    monkeypatch.setattr(bindings.skills, 'can_use', original)
    monkeypatch.setattr(bindings.integrations.credential_vault, 'health', lambda *args: 'unavailable')
    assert inspect_managed_delivery_authority(setup[3], session, setup[4], **kwargs).status == 'unavailable'
    assert setup[3].load('task', 'alice', expected_version=saved.version).state.inflight == 'action'


def test_recovery_expired_claim_or_changed_checkpoint_fails_closed(managed, monkeypatch):
    from alphonse.agent_v2.core.intelligence.v3.single_loop_recovery_inspection import inspect_managed_delivery_authority
    setup, bindings, session, _ = managed
    saved = invoke(managed)
    args = dict(bindings=bindings, task_id='task', expected_version=saved.version)
    expired = inspect_managed_delivery_authority(setup[3], session, setup[4], **args,
                                                  clock=lambda: setup[6] + timedelta(days=1))
    assert expired.status == 'unavailable'
    original = bindings.resolve_account
    def resolve(*args, **kwargs):
        handle = original(*args, **kwargs)
        with setup[2]._connect() as conn:
            conn.execute("UPDATE v2_task_checkpoints SET version=version+1 WHERE task_id='task'")
        return handle
    monkeypatch.setattr(bindings, 'resolve_account', resolve)
    changed = inspect_managed_delivery_authority(setup[3], session, setup[4], **args, clock=lambda: setup[6])
    assert changed.status == 'unavailable'
    assert setup[3].load('task', 'alice', expected_version=saved.version + 1).state.inflight == 'action'


def test_recovery_rejects_binding_revision_changed_after_reservation(managed):
    from alphonse.agent_v2.core.intelligence.v3.single_loop_recovery_inspection import inspect_managed_delivery_authority
    setup, bindings, session, _ = managed
    saved = invoke(managed)
    with bindings.integrations._connect() as conn:
        conn.execute("UPDATE v2_skill_bindings SET revision=revision+1 WHERE binding_id='binding-one'")
    result = inspect_managed_delivery_authority(setup[3], session, setup[4], bindings=bindings,
        task_id='task', expected_version=saved.version, clock=lambda: setup[6])
    assert result.status == 'unavailable'
    assert setup[3].load('task', 'alice', expected_version=saved.version).state.inflight == 'action'


def test_recovery_rejects_clock_regression_between_locked_checks(managed):
    from alphonse.agent_v2.core.intelligence.v3.single_loop_recovery_inspection import inspect_managed_delivery_authority
    setup, bindings, session, _ = managed
    saved = invoke(managed)
    now = setup[6]
    moments = iter((now, now + timedelta(seconds=2), now + timedelta(seconds=1)))
    result = inspect_managed_delivery_authority(setup[3], session, setup[4], bindings=bindings,
        task_id='task', expected_version=saved.version, clock=lambda: next(moments))
    assert result.status == 'unavailable'
    assert setup[3].load('task', 'alice', expected_version=saved.version).state.inflight == 'action'


@pytest.mark.parametrize('change', ['version', 'scope', 'project', 'action', 'missing_action'])
def test_malformed_managed_record_never_projects_recovery_as_clear(managed, change):
    from alphonse.agent_v2.core.intelligence.v3.single_loop_recovery_inspection import inspect_recovery
    saved = invoke(managed)
    questions, store = managed[0][2], managed[0][3]
    with questions._connect() as conn:
        row = conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()
        raw = json.loads(row[0]); record = raw['metadata'][MANAGED_PREFLIGHT_KEY]
        if change == 'version': record['schema_version'] = True
        if change == 'scope': record['resources'] = []
        if change == 'project': record['project_id'] = 'other'
        if change == 'action': record['message_digest'] = 'x' * 64
        if change == 'missing_action': raw['metadata'].pop(ACTION_KEY)
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(raw),))
    result = inspect_recovery(store, managed[2], task_id='task', expected_version=saved.version)
    assert result.status == 'unavailable'
    assert store.load('task', 'alice', expected_version=saved.version).state.inflight == 'action'
