"""Disposable accounts only: no provider I/O or household credential access."""
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.accounts import AccountService, AccountSession
from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
from alphonse.agent_v2.skill_bindings import SkillBindings
from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.users import V2UserStore


def expiry():
    return (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()


@pytest.fixture
def f(tmp_path, secure_vault):
    users = V2UserStore(tmp_path / 'users.sqlite')
    users.onboard(display_name='Parent', user_id='parent', users_root=tmp_path / 'profiles')
    for user in ['child', 'other']:
        users.create_user(display_name=user, user_id=user)
    store = SQLiteIntegrationStore(tmp_path / 'accounts.sqlite', credential_vault=secure_vault)
    skills = SkillStore(tmp_path / 'skills')
    skills.create_skill('fixture', 'Disposable skill', 'No private configuration.')
    bindings = SkillBindings(store, users, skills)
    service = bindings.accounts
    for connection, owner in [('one', 'parent'), ('two', 'child')]:
        service.create_connection(AccountSession(owner), connection_id=connection, provider_key='synthetic',
                                  provider_subject='subject-' + connection, display_name='Same label')
        store.install_credentials(connection, {'token': 'disposable-' + connection}, expected_generation=0)
        service.save_connection(AccountSession(owner), connection_id=connection, expected_revision=1,
                                provider_subject='subject-' + connection, capabilities=['read', 'search'],
                                granted_scopes=['scope.read'])
        service.save_grant(AccountSession(owner), grant_id='grant-' + connection, connection_id=connection,
                           expected_revision=0, beneficiary_user_id='child', operations=['read', 'search'],
                           resources=['folder-a', 'folder-b'], recipient_user_ids=['parent'], expires_at=expiry())
        bindings.configure(actor_id=owner, owner_id='child', binding_id='binding-' + connection,
                           skill_id='skill:fixture', kind='account', target=connection,
                           grant_id='grant-' + connection, capabilities=('read', 'search'), required_scopes=('scope.read',))
    bindings.select(actor_id='child', skill_id='skill:fixture', name='mail', binding_id='binding-one')
    return SimpleNamespace(users=users, store=store, skills=skills, bindings=bindings, service=service, tmp=tmp_path)


def resolve(f, **changes):
    args = dict(skill_id='skill:fixture', name='mail', operation='read', resources=['folder-a'], recipient_user_ids=[])
    args.update(changes)
    return f.bindings.resolve_account(AccountSession('child'), **args)


def revise_connection(f, **changes):
    old = f.store.get('one').connection
    args = {k: old[k] for k in ['provider_subject', 'capabilities', 'granted_scopes', 'status', 'health_code', 'last_success_at']}
    args.update(changes)
    return f.service.save_connection(AccountSession('parent'), connection_id='one', expected_revision=old['revision'], **args)


def revise_grant(f, **changes):
    args = dict(grant_id='grant-one', connection_id='one', expected_revision=1, beneficiary_user_id='child',
                operations=['read', 'search'], resources=['folder-a', 'folder-b'], recipient_user_ids=['parent'], expires_at=expiry())
    args.update(changes)
    return f.service.save_grant(AccountSession('parent'), **args)


def test_exact_choice_no_provider_fallback_or_guessed_label(f, monkeypatch):
    monkeypatch.setattr(f.store, 'get_by_provider', lambda *a: pytest.fail('provider fallback'))
    assert resolve(f).connection_id == 'one'
    with pytest.raises(ValueError, match='choice_missing'):
        resolve(f, name='synthetic')
    f.bindings.select(actor_id='child', skill_id='skill:fixture', name='mail', binding_id='binding-two')
    assert resolve(f).connection_id == 'two'
    with pytest.raises(PermissionError, match='constraints_denied'):
        resolve(f, expected_binding_id='binding-one', expected_connection_id='one', expected_grant_id='grant-one')
    # Administrators have no automatic role choice.
    with pytest.raises(ValueError, match='choice_missing'):
        f.bindings.resolve_account(AccountSession('parent'), skill_id='skill:fixture', name='mail',
                                   operation='read', resources=['folder-a'], recipient_user_ids=[])


def test_handle_has_no_config_credentials_or_implicit_audience(f):
    handle = asdict(resolve(f))
    assert handle['recipient_user_ids'] == ()
    assert not {'credential_ref', 'secrets', 'config', 'configuration'} & handle.keys()
    assert 'opaque-' not in json.dumps(handle)
    with pytest.raises(PermissionError, match='managed_service_required'):
        f.bindings.prepare(actor_id='child', skill_id='skill:fixture', names=['mail'])


@pytest.mark.parametrize('changes', [dict(resources=['folder-c']), dict(resources=[]), dict(operation='write'),
                                     dict(recipient_user_ids=['other']), dict(project_id='shared-project')])
def test_operation_resource_project_and_audience_intersection(f, changes):
    with pytest.raises(PermissionError, match='denied'):
        resolve(f, **changes)


def test_explicit_project_grant_is_not_ad_hoc_permission(f):
    revise_grant(f, grant_id='project-grant', expected_revision=0, project_id='project')
    f.bindings.attach_grant(AccountSession('parent'), binding_id='binding-one', grant_id='project-grant',
                            expected_revision=1, required_scopes=('scope.read',), capabilities=('read',))
    assert resolve(f, project_id='project').grant_id == 'project-grant'
    with pytest.raises(PermissionError):
        resolve(f)


def test_revocation_and_narrowing_rechecked_after_restart(f):
    assert resolve(f, resources=['folder-b'], recipient_user_ids=['parent'])
    revise_grant(f, resources=['folder-a'], recipient_user_ids=[])
    f.bindings = SkillBindings(SQLiteIntegrationStore(f.store.db_path, credential_vault=f.store.credential_vault), f.users, f.skills)
    with pytest.raises(PermissionError):
        resolve(f, resources=['folder-b'])
    with pytest.raises(PermissionError):
        resolve(f, recipient_user_ids=['parent'])
    with pytest.raises(PermissionError):
        resolve(f, expected_grant_revision=1)
    assert resolve(f, expected_grant_revision=2)
    f.service.revoke_grant(AccountSession('parent'), grant_id='grant-one', expected_revision=2)
    with pytest.raises(PermissionError):
        resolve(f)
    with pytest.raises(ValueError, match='identity_immutable'):
        revise_grant(f, expected_revision=3)


@pytest.mark.parametrize('who', ['child', 'parent'])
def test_inactive_actor_or_connection_owner_denied(f, who):
    f.users.update_user(who, is_active=False)
    with pytest.raises(PermissionError):
        resolve(f)


@pytest.mark.parametrize('changes', [dict(status='paused'), dict(status='reauth_required'),
                                     dict(status='disconnected'), dict(health_code='locked'),
                                     dict(capabilities=['search']), dict(granted_scopes=[])])
def test_connection_status_and_provider_consent_narrowing(f, changes):
    revise_connection(f, **changes)
    with pytest.raises(PermissionError):
        resolve(f)


def test_disconnect_revokes_grants_permanently_even_after_reconnect(f):
    revise_connection(f, status='disconnected')
    with pytest.raises(ValueError, match='secure_credentials_required'):
        revise_connection(f, status='connected')
    with pytest.raises(PermissionError):
        resolve(f)


@pytest.mark.parametrize('table,key', [('v2_integrations', 'integration_id'), ('v2_account_grants', 'grant_id')])
def test_missing_connection_or_grant_never_falls_back(f, table, key):
    with f.store._connect() as conn:
        conn.execute(f'DELETE FROM {table} WHERE {key}=?', ('one' if key == 'integration_id' else 'grant-one',))
    with pytest.raises(PermissionError):
        resolve(f)


def test_expiry_is_enforced_at_use(f):
    with f.store._connect() as conn:
        row = conn.execute("SELECT definition FROM v2_account_grants WHERE grant_id='grant-one'").fetchone()
        grant = json.loads(row['definition'])
        grant['expires_at'] = '2000-01-01T00:00:00Z'
        conn.execute("UPDATE v2_account_grants SET definition=? WHERE grant_id='grant-one'", (json.dumps(grant),))
    with pytest.raises(PermissionError):
        resolve(f)


def test_only_connection_owner_can_issue_and_change_grants(f):
    with pytest.raises(PermissionError, match='owner_required'):
        f.service.save_grant(AccountSession('child'), grant_id='stolen', connection_id='one', expected_revision=0,
                             beneficiary_user_id='child', operations=['read'], resources=['folder-a'],
                             recipient_user_ids=[], expires_at=expiry())
    # Even the household admin cannot issue grants for another member's account.
    with pytest.raises(PermissionError, match='owner_required'):
        f.service.revoke_grant(AccountSession('parent'), grant_id='grant-two', expected_revision=1)


def test_binding_sharing_is_intersected_with_grant(f):
    f.bindings.set_access('binding-one', actor_id='child', shared_with=('other',), expected_revision=1)
    f.bindings.select(actor_id='other', skill_id='skill:fixture', name='mail', binding_id='binding-one')
    with pytest.raises(PermissionError, match='grant_denied'):
        f.bindings.resolve_account(AccountSession('other'), skill_id='skill:fixture', name='mail',
                                   operation='read', resources=['folder-a'], recipient_user_ids=[])
    f.bindings.set_access('binding-one', actor_id='child', enabled=False, expected_revision=2)
    with pytest.raises(PermissionError, match='revoked'):
        resolve(f)


def test_revision_and_identity_guards(f):
    revise_grant(f)
    with pytest.raises(ValueError, match='revision_conflict'):
        revise_grant(f)
    with pytest.raises(ValueError, match='identity_immutable'):
        revise_connection(f, provider_subject='somebody-else')
    f.store.install_credentials('one', {'token': 'rotated'}, expected_generation=1)
    assert resolve(f).connection_id == 'one'
    with pytest.raises(ValueError, match='identity_immutable'):
        f.store.upsert(integration_id='one', provider_key='different', display_name='Changed')
    with pytest.raises(PermissionError, match='service_required'):
        f.store.upsert(integration_id='one', provider_key='synthetic', display_name='Same label', enabled=True,
                       config={'owner': 'child'}, secrets={'token': 'plaintext'})
    with pytest.raises(PermissionError, match='service_required'):
        f.store.set_enabled('one', False)
    with pytest.raises(ValueError, match='revision_required'):
        f.bindings.set_access('binding-one', actor_id='child')


def test_package_lifecycle_preserves_account_grant_and_choice(f):
    target = f.tmp / 'export'
    target.mkdir()
    package = f.skills.export_directory('skill:fixture', target)
    assert 'opaque-one' not in (package / 'SKILL.md').read_text()
    f.skills.replace_directory('skill:fixture', package)
    assert resolve(f).connection_id == 'one'
    f.skills.uninstall('skill:fixture')
    with pytest.raises(ValueError, match='skill_missing'):
        resolve(f)
    f.skills.install_directory(package)
    assert resolve(f).grant_revision == 1


def test_skill_availability_independent_of_selection(f, monkeypatch):
    monkeypatch.setattr(f.skills, 'can_use', lambda *a: False)
    with pytest.raises(PermissionError, match='skill_denied'):
        resolve(f)


def test_legacy_adoption_requires_explicit_secure_migration_and_preserves_choices(f):
    f.store.upsert(integration_id='legacy', provider_key='synthetic', display_name='Legacy', enabled=True,
                   secrets={'token': 'synthetic'})
    f.bindings.configure(actor_id='parent', owner_id='child', binding_id='legacy', skill_id='skill:fixture',
                         kind='account', target='legacy', required_secrets=('token',))
    f.bindings.select(actor_id='child', skill_id='skill:fixture', name='legacy', binding_id='legacy')
    args = dict(connection_id='legacy', expected_revision=0, provider_subject='legacy-subject',
                capabilities=['read'], granted_scopes=['scope.read'])
    with pytest.raises(ValueError, match='secure_migration_required'):
        f.service.save_connection(AccountSession('parent'), **args)
    f.store.migrate_credentials('legacy')
    f.service.save_connection(AccountSession('parent'), **args)
    with pytest.raises(PermissionError, match='constraints_denied'):
        resolve(f, name='legacy')
    f.service.save_grant(AccountSession('parent'), grant_id='legacy-grant', connection_id='legacy',
                         expected_revision=0, beneficiary_user_id='child', operations=['read'], resources=['folder-a'],
                         recipient_user_ids=[], expires_at=expiry())
    f.bindings.attach_grant(AccountSession('parent'), binding_id='legacy', grant_id='legacy-grant', expected_revision=1,
                            required_scopes=('scope.read',), capabilities=('read',))
    assert resolve(f, name='legacy').binding_id == 'legacy'


def test_sharing_removal_blocks_grant_beneficiary(f):
    f.bindings.configure(actor_id='parent', owner_id='parent', binding_id='parent-binding', skill_id='skill:fixture',
                         kind='account', target='one', grant_id='grant-one', capabilities=('read',), required_scopes=('scope.read',))
    f.bindings.set_access('parent-binding', actor_id='parent', shared_with=('child',), expected_revision=1)
    f.bindings.select(actor_id='child', skill_id='skill:fixture', name='mail', binding_id='parent-binding')
    assert resolve(f)
    f.bindings.set_access('parent-binding', actor_id='parent', expected_revision=2)
    with pytest.raises(PermissionError, match='access_denied'):
        resolve(f)


def test_direct_secure_disconnect_also_revokes_grants(f):
    f.store.disconnect_credentials('one', expected_generation=1)
    assert f.store.get('one').connection['status'] == 'disconnected'
    with f.store._connect() as conn:
        grant = json.loads(conn.execute("SELECT definition FROM v2_account_grants WHERE grant_id='grant-one'").fetchone()[0])
    assert grant['revoked_at'] and grant['revision'] == 2
    with pytest.raises(PermissionError):
        resolve(f)


def test_bounded_native_use_redacts_and_rechecks_inflight_revocation(f):
    request = dict(skill_id='skill:fixture', name='mail', operation='read', resources=['folder-a'], recipient_user_ids=[])
    def adapter(handle, credentials):
        assert handle.resources == ('folder-a',)
        return {'private': credentials['token']}
    assert f.bindings.use_account(AccountSession('child'), request=request, operation=adapter) == {'private': '[REDACTED]'}
    def revoke(handle, credentials):
        f.service.revoke_grant(AccountSession('parent'), grant_id='grant-one', expected_revision=1)
        return {'result': 'must not reach caller'}
    with pytest.raises(PermissionError):
        f.bindings.use_account(AccountSession('child'), request=request, operation=revoke)


def test_locked_backend_blocks_before_adapter(f):
    f.store.credential_vault.backend.key_file.unlink()
    with pytest.raises(PermissionError, match='credentials_unavailable'):
        resolve(f)


def test_model_arguments_cannot_override_session_actor(f):
    with pytest.raises(TypeError):
        resolve(f, actor_user_id='parent')
    with pytest.raises(TypeError):
        resolve(f, session=AccountSession('parent'))


def test_concurrent_grant_revision_has_one_winner(f):
    from concurrent.futures import ThreadPoolExecutor
    def update():
        try:
            revise_grant(f, resources=['folder-a'])
            return 'saved'
        except ValueError as exc:
            return str(exc)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: update(), range(2)))
    assert sorted(outcomes) == ['account_revision_conflict', 'saved']


def test_new_connection_owner_reserved_before_consent_and_cannot_be_claimed(f):
    f.service.create_connection(AccountSession('other'), connection_id='new', provider_key='synthetic',
                                provider_subject='new-subject', display_name='New')
    assert f.store.get('new').connection['owner_user_id'] == 'other'
    assert f.store.get('new').connection['status'] == 'reauth_required'
    with pytest.raises(ValueError, match='connection_exists'):
        f.service.create_connection(AccountSession('child'), connection_id='new', provider_key='synthetic',
                                    provider_subject='new-subject', display_name='New')
    f.store.upsert(integration_id='unowned', provider_key='synthetic', display_name='Legacy', enabled=True)
    f.store.install_credentials('unowned', {'token': 'disposable'}, expected_generation=0)
    with pytest.raises(PermissionError, match='reconciliation_admin_required'):
        f.service.save_connection(AccountSession('child'), connection_id='unowned', expected_revision=0,
                                  provider_subject='legacy', capabilities=['read'], granted_scopes=['scope.read'])


def test_editable_binding_config_cannot_expand_resource_or_audience(f):
    with f.store._connect() as conn:
        conn.execute("UPDATE v2_skill_bindings SET configuration=? WHERE binding_id='binding-one'",
                     (json.dumps({'resources': ['private'], 'recipient_user_ids': ['other'], 'owner_user_id': 'child'}),))
    with pytest.raises(PermissionError):
        resolve(f, resources=['private'], recipient_user_ids=['other'])


def test_stale_binding_acl_revision_cannot_restore_sharing(f):
    f.bindings.set_access('binding-one', actor_id='child', expected_revision=1)
    with pytest.raises(ValueError, match='revision_conflict'):
        f.bindings.set_access('binding-one', actor_id='child', shared_with=('other',), expected_revision=1)


def test_admin_explicitly_reconciles_legacy_owner_without_claiming_account(f):
    f.store.upsert(integration_id='old-child', provider_key='synthetic', display_name='Legacy child', enabled=True)
    f.store.install_credentials('old-child', {'token': 'disposable'}, expected_generation=0)
    value = f.service.save_connection(AccountSession('parent'), connection_id='old-child', expected_revision=0,
                                      provider_subject='child-subject', capabilities=['read'], granted_scopes=['scope.read'],
                                      reconcile_owner_user_id='child')
    assert value['owner_user_id'] == 'child'
    with pytest.raises(PermissionError, match='owner_required'):
        f.service.save_grant(AccountSession('parent'), grant_id='no-admin-fallback', connection_id='old-child',
                             expected_revision=0, beneficiary_user_id='parent', operations=['read'], resources=['folder-a'],
                             recipient_user_ids=[], expires_at=expiry())
