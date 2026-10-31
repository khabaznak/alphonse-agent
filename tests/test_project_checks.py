"""Policy/lifecycle authority with disposable users/accounts, never provider I/O."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json

import pytest

from test_account_grants import f, expiry, revise_connection
from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.project_checks import ProjectCheckService


@pytest.fixture
def checks(f):
    projects = ProjectStore(str(f.tmp / 'projects.sqlite'))
    project = projects.create_project(name='School', root_path=str(f.tmp / 'school'), owner_user_id='child', visibility='shared')
    f.service.save_grant(AccountSession('parent'), grant_id='school-grant', connection_id='one', expected_revision=0,
        beneficiary_user_id='child', operations=['read'], resources=['folder-a'], recipient_user_ids=['parent'],
        expires_at=expiry(), project_id=project.project_id)
    f.bindings.attach_grant(AccountSession('parent'), binding_id='binding-one', grant_id='school-grant',
        expected_revision=1, required_scopes=('scope.read',), capabilities=('read',))
    f.checks = ProjectCheckService(projects, f.bindings)
    f.projects, f.project = projects, project
    now = datetime.now(timezone.utc)
    f.definition = dict(project_id=project.project_id,
        source_bindings=[dict(skill_id='skill:fixture', role='mail', binding_id='binding-one',
                             connection_id='one', grant_id='school-grant', operation='read', resources=['folder-a'])],
        query={'text': 'school'}, trigger={'kind': 'schedule', 'weekdays': [0], 'local_time': '08:00'},
        timezone='America/Mexico_City', retrieval_window={'weekdays': [0], 'start': '08:00', 'end': '09:00'},
        duration={'ongoing': False, 'starts_at': (now - timedelta(days=1)).isoformat(), 'ends_at': (now + timedelta(days=30)).isoformat()},
        budgets=dict(max_pages=5, max_items=100, max_bytes=1000000, max_model_calls=2, max_runtime_seconds=60),
        recipient_user_ids=['parent'], skill_hints=['skill:fixture'])
    return f


def save(f, **kw):
    return f.checks.save(AccountSession('child'), definition=f.definition, **kw)


def active(f):
    return save(f, activate=True, authorization_reference='conversation:explicit-instruction')


def guard(f, policy, **kw):
    return f.checks.guard(policy['policy_id'], expected_revision=policy['revision'], **kw)


def test_persisted_default_disabled_and_explicit_authority(checks):
    f = checks
    draft = save(f)
    assert draft['status'] == 'disabled' and draft['revision'] == 1
    assert draft['missed_run'] == {'kind': 'next_allowed_window'}
    assert draft['initial_lookback_days'] == 30
    with pytest.raises(PermissionError):
        guard(f, draft)
    with pytest.raises(PermissionError, match='authorization_required'):
        save(f, activate=True)
    policy = active(f)
    f.checks = ProjectCheckService(ProjectStore(f.projects.db_path), f.bindings)
    restored = f.checks.get(AccountSession('child'), policy['policy_id'])
    assert restored == policy
    assert len(policy['goal_digest']) == len(policy['query_digest']) == 64
    source = policy['source_bindings'][0]
    assert source['grant_revision'] == 1 and source['connection_revision'] == 2
    assert policy['authorized_by'] == 'child'
    for _ in range(3):
        assert guard(f, policy)['revision'] == 1  # no per-run authorization
    assert f.checks.list(AccountSession('child'), f.project.project_id)
    with pytest.raises(PermissionError):
        f.checks.get(AccountSession('parent'), policy['policy_id'])


@pytest.mark.parametrize('change', ['resource', 'recipient', 'connection', 'grant', 'skill', 'actor', 'write'])
def test_no_implicit_source_audience_or_actor_authority(checks, change):
    f = checks
    source = f.definition['source_bindings'][0]
    if change == 'resource': source['resources'] = ['folder-b']
    if change == 'recipient': f.definition['recipient_user_ids'] = ['other']
    if change == 'connection': source['connection_id'] = 'two'
    if change == 'grant': source['grant_id'] = 'grant-two'
    if change == 'skill': f.definition['skill_hints'] = ['skill:missing']
    if change == 'actor': f.definition['executing_user_id'] = 'parent'
    if change == 'write': source['operation'] = 'send'
    with pytest.raises((PermissionError, ValueError)):
        active(f)


@pytest.mark.parametrize('change', ['scope', 'goals', 'duration', 'file'])
def test_goal_edits_suspend_and_need_new_authorization(checks, change):
    f = checks
    policy = active(f)
    config = f.projects.read_project_config(f.project.project_id)
    if change == 'scope': config['scope'] = 'expanded goal'
    if change == 'goals': config['goals'] = ['New goal']
    if change == 'duration': config['duration']['ongoing'] = False; config['duration']['end_date'] = '2099-01-01'
    if change == 'file': config['scope'] = 'Edited directly on disk'
    if change == 'file':
        from pathlib import Path
        Path(f.project.config_path).write_text(json.dumps(config))
    else:
        f.projects.write_project_config(f.project.project_id, config, requester_user_id='child')
    with pytest.raises(PermissionError): guard(f, policy)
    suspended = f.checks.get(AccountSession('child'), policy['policy_id'])
    assert suspended['status'] == 'needs_review' and suspended['revision'] == 2
    with pytest.raises(PermissionError):
        f.checks.set_status(AccountSession('child'), policy['policy_id'], expected_revision=2, status='active')
    revised = save(f, policy_id=policy['policy_id'], expected_revision=2, activate=True, authorization_reference='conversation:revised-goal')
    assert revised['goal_digest'] != policy['goal_digest']
    assert guard(f, revised)
    with pytest.raises(PermissionError): guard(f, policy)


def test_config_spoof_cannot_create_or_activate_policies(checks):
    f = checks
    config = f.projects.read_project_config(f.project.project_id)
    config.update(proactive_checks=[dict(f.definition, status='active', authorized_by='parent')])
    f.projects.write_project_config(f.project.project_id, config, requester_user_id='child')
    assert f.checks.list(AccountSession('child'), f.project.project_id) == []
    policy = save(f)
    config.update(policy_id=policy['policy_id'], status='active', revision=999)
    f.projects.write_project_config(f.project.project_id, config, requester_user_id='child')
    assert f.checks.get(AccountSession('child'), policy['policy_id'])['status'] == 'disabled'


@pytest.mark.parametrize('status', ['paused', 'completed', 'archived'])
def test_lifecycle_round_trip_fences_queued_work(checks, status):
    f = checks
    policy = active(f)
    f.projects.set_status(f.project.project_id, status, requester_user_id='child')
    f.projects.set_status(f.project.project_id, 'active', requester_user_id='child')
    with pytest.raises(PermissionError, match='project_changed'): guard(f, policy)
    assert f.checks.get(AccountSession('child'), policy['policy_id'])['status'] == 'needs_review'


def test_pause_resume_cancel_and_end_never_revive_old_work(checks):
    f = checks
    policy = active(f)
    paused = f.checks.set_status(AccountSession('child'), policy['policy_id'], expected_revision=1, status='paused')
    with pytest.raises(PermissionError): guard(f, policy)
    resumed = f.checks.set_status(AccountSession('child'), policy['policy_id'], expected_revision=2, status='active')
    assert resumed['revision'] == 3 and guard(f, resumed)
    with pytest.raises(PermissionError): guard(f, paused)
    with pytest.raises(PermissionError, match='cancelled'): guard(f, resumed, cancelled=True)
    assert guard(f, resumed)  # cancellation is per run, not silently ending policy
    ended = f.checks.set_status(AccountSession('child'), policy['policy_id'], expected_revision=3, status='ended')
    with pytest.raises(PermissionError): guard(f, resumed)
    with pytest.raises(PermissionError):
        f.checks.set_status(AccountSession('child'), ended['policy_id'], expected_revision=4, status='active')


@pytest.mark.parametrize('change', ['revoke', 'narrow', 'pause', 'choice', 'acl', 'inactive', 'missing', 'membership'])
def test_dependency_changes_fence_and_do_not_expand_sharing(checks, change):
    f = checks
    policy = active(f)
    if change == 'revoke':
        f.service.revoke_grant(AccountSession('parent'), grant_id='school-grant', expected_revision=1)
    if change == 'narrow':
        f.service.save_grant(AccountSession('parent'), grant_id='school-grant', connection_id='one', expected_revision=1,
            beneficiary_user_id='child', operations=['read'], resources=['folder-a'], recipient_user_ids=[],
            expires_at=expiry(), project_id=f.project.project_id)
    if change == 'pause':
        revise_connection(f, status='paused'); revise_connection(f, status='connected')
    if change == 'choice':
        f.bindings.select(actor_id='child', skill_id='skill:fixture', name='mail', binding_id='binding-two')
        f.bindings.select(actor_id='child', skill_id='skill:fixture', name='mail', binding_id='binding-one')
    if change == 'acl':
        f.bindings.set_access('binding-one', actor_id='child', enabled=False, expected_revision=2)
    if change == 'inactive': f.users.update_user('parent', is_active=False)
    if change == 'missing':
        with f.store._connect() as conn:
            conn.execute("DELETE FROM v2_integrations WHERE integration_id='one'")
    if change == 'membership': f.projects.add_member(f.project.project_id, 'other')
    with pytest.raises((PermissionError, ValueError)): guard(f, policy)
    if change == 'membership':
        assert f.checks.get(AccountSession('child'), policy['policy_id'])['recipient_user_ids'] == ['parent']
        f.definition['recipient_user_ids'] = ['other']
        with pytest.raises(PermissionError): active(f)


def test_projectless_queries_preserved_but_checks_require_accessible_project(checks):
    f = checks
    f.bindings.select(actor_id='child', skill_id='skill:fixture', name='adhoc', binding_id='binding-two')
    assert f.bindings.resolve_account(AccountSession('child'), skill_id='skill:fixture', name='adhoc',
                                      operation='read', resources=['folder-a'], recipient_user_ids=[])
    f.definition['project_id'] = ''
    with pytest.raises(ValueError): active(f)
    f.definition['project_id'] = 'missing'
    with pytest.raises(PermissionError): active(f)
    other = f.projects.create_project(name='Private', root_path=str(f.tmp / 'private'), owner_user_id='parent')
    f.definition['project_id'] = other.project_id
    with pytest.raises(PermissionError): active(f)


def test_duration_and_explicit_ongoing_renewal(checks):
    f = checks
    policy = active(f)
    with pytest.raises(PermissionError, match='outside_duration'):
        guard(f, policy, now=datetime.now(timezone.utc) + timedelta(days=31))
    assert f.checks.get(AccountSession('child'), policy['policy_id'])['status'] == 'ended'
    f.definition['duration'].update(ongoing=True, ends_at=None)
    renewal = save(f, policy_id=policy['policy_id'], expected_revision=2, activate=True, authorization_reference='conversation:explicit-ongoing')
    assert guard(f, renewal)
    f.definition['duration'].update(ongoing=False, ends_at=None)
    with pytest.raises(ValueError): active(f)


@pytest.mark.parametrize('field,value', [
    ('timezone', 'Invalid/Zone'), ('initial_lookback_days', 0), ('initial_lookback_days', True),
    ('trigger', {'kind': 'script', 'command': 'anything'}), ('trigger', {'kind': 'schedule', 'weekdays': [7], 'local_time': '08:00'}),
    ('retrieval_window', {'weekdays': [0], 'start': '09:00', 'end': '08:00'}),
    ('budgets', {}), ('missed_run', {'kind': 'all_missed'}), ('notification_mode', 'all_members'),
    ('duration', {'ongoing': True, 'starts_at': '2026-01-01', 'ends_at': None}),
    ('query', {'text': 'school', 'script': 'no'}), ('source_bindings', []),
])
def test_bounded_schema_rejects_unsafe_or_incomplete_inputs(checks, field, value):
    checks.definition[field] = value
    with pytest.raises(ValueError, match='definition_invalid'): active(checks)


def test_concurrent_policy_revision_has_one_winner(checks):
    f = checks
    policy = active(f)
    def pause():
        try:
            f.checks.set_status(AccountSession('child'), policy['policy_id'], expected_revision=1, status='paused')
            return 'saved'
        except ValueError as exc:
            return str(exc)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: pause(), range(2)))
    assert sorted(outcomes) == ['project_check_revision_conflict', 'saved']


@pytest.mark.parametrize('state', ['disabled', 'ended', 'needs_review'])
def test_status_hopping_cannot_bypass_reauthorization(checks, state):
    f = checks
    policy = save(f) if state == 'disabled' else active(f)
    if state == 'ended':
        policy = f.checks.set_status(AccountSession('child'), policy['policy_id'], expected_revision=1, status='ended')
    if state == 'needs_review':
        f.projects.add_member(f.project.project_id, 'other')
        with pytest.raises(PermissionError): guard(f, policy)
        policy = f.checks.get(AccountSession('child'), policy['policy_id'])
    with pytest.raises(PermissionError, match='reauthorization_required'):
        f.checks.set_status(AccountSession('child'), policy['policy_id'], expected_revision=policy['revision'], status='paused')


def test_future_start_is_inactive_without_losing_authorization(checks):
    f = checks
    f.definition['duration']['starts_at'] = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    policy = active(f)
    with pytest.raises(PermissionError, match='not_started'):
        guard(f, policy)
    assert f.checks.get(AccountSession('child'), policy['policy_id'])['status'] == 'active'
    assert guard(f, policy, now=datetime.now(timezone.utc) + timedelta(days=2))


def test_project_duration_independently_limits_ongoing_policy(checks):
    f = checks
    config = f.projects.read_project_config(f.project.project_id)
    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
    config['duration'] = {'ongoing': False, 'start_date': None, 'end_date': tomorrow.date().isoformat()}
    f.projects.write_project_config(f.project.project_id, config, requester_user_id='child')
    f.definition['duration'].update(ongoing=True, ends_at=None)
    policy = active(f)
    assert guard(f, policy)
    with pytest.raises(PermissionError, match='outside_duration'):
        guard(f, policy, now=tomorrow + timedelta(days=2))
    assert f.checks.get(AccountSession('child'), policy['policy_id'])['status'] == 'ended'


def test_revision_change_fences_query_and_duration_and_preserves_identity(checks):
    f = checks
    policy = active(f)
    f.definition['query']['text'] = 'revised school query'
    revised = save(f, policy_id=policy['policy_id'], expected_revision=1,
                   activate=True, authorization_reference='conversation:revised-query')
    assert revised['policy_id'] == policy['policy_id']
    assert revised['query_digest'] != policy['query_digest']
    assert guard(f, revised)
    with pytest.raises(PermissionError): guard(f, policy)
    with pytest.raises(ValueError, match='revision_conflict'):
        save(f, policy_id=policy['policy_id'], expected_revision=1)


def test_empty_recipients_and_bounded_late_policy_are_explicit(checks):
    f = checks
    f.definition['recipient_user_ids'] = []
    f.definition['missed_run'] = {'kind': 'bounded_late', 'max_lateness_seconds': 300}
    policy = active(f)
    assert policy['recipient_user_ids'] == []
    assert guard(f, policy)['missed_run']['max_lateness_seconds'] == 300


def test_removing_executing_member_invalidates_access(checks):
    f = checks
    # Use an accessible shared project belonging to another member.
    other = f.projects.create_project(name='Shared', root_path=str(f.tmp / 'shared'), owner_user_id='parent', visibility='shared')
    f.projects.add_member(other.project_id, 'child')
    f.definition['project_id'] = other.project_id
    f.service.save_grant(AccountSession('parent'), grant_id='shared-grant', connection_id='one', expected_revision=0,
        beneficiary_user_id='child', operations=['read'], resources=['folder-a'], recipient_user_ids=['parent'],
        expires_at=expiry(), project_id=other.project_id)
    f.bindings.attach_grant(AccountSession('parent'), binding_id='binding-one', grant_id='shared-grant',
        expected_revision=2, required_scopes=('scope.read',), capabilities=('read',))
    f.definition['source_bindings'][0]['grant_id'] = 'shared-grant'
    policy = active(f)
    assert guard(f, policy)
    f.projects.remove_member(other.project_id, 'child')
    with pytest.raises(PermissionError): guard(f, policy)


def test_project_deletion_and_id_reuse_never_revive_queued_work(checks):
    f = checks
    policy = active(f)
    f.projects.delete_project(f.project.project_id, requester_user_id='child')
    f.projects.create_project(name='School', root_path=f.project.root_path, owner_user_id='child',
                             project_id=f.project.project_id, visibility='shared')
    with pytest.raises(PermissionError): guard(f, policy)
