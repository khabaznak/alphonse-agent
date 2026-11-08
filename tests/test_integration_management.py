"""Synthetic management principals and resources; no provider access."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest

from test_account_grants import f, expiry, revise_connection
from test_project_checks import checks
from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.integration_management import IntegrationManagement
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.tools.registry.native.integration_manage import build_integration_manage_tool_definition
from alphonse.agent_v2.ipc import V2DaemonServer, PROTOCOL_VERSION


@pytest.fixture
def management(checks):
    f = checks
    runtime = SimpleNamespace(project_checks=f.checks, user_store=f.users, skill_store=f.skills)
    service = IntegrationManagement(runtime)
    runtime.integration_management = service
    f.management, f.runtime = service, runtime
    return f


def review(f, **kw):
    return f.management.review(AccountSession('child'), definition=f.definition, **kw)


def test_owner_member_admin_catalog_isolation(management):
    f = management
    parent = f.management.catalog(AccountSession('parent'))
    child = f.management.catalog(AccountSession('child'))
    other = f.management.catalog(AccountSession('other'))
    assert [c['connection_id'] for c in parent['connections']] == ['one']
    assert [c['connection_id'] for c in child['connections']] == ['two']
    assert not other['connections'] and not other['grants'] and not other['sources']
    assert parent['sources'] == []  # admin is not a selected beneficiary
    assert len(child['sources']) == 1
    assert set(child['sources'][0]['resources']) == {'folder-a'}
    serialized = json.dumps(child)
    assert 'disposable-' not in serialized and 'credential_ref' not in serialized
    assert 'scope.read' not in serialized and 'provider_subject' not in serialized
    assert set(child['members'][0]) == {'user_id', 'display_name'}
    with pytest.raises(PermissionError):
        f.management.catalog({'actor_user_id': 'child'})


def test_review_binds_exact_scope_actor_channel_and_single_use(management):
    f = management
    r = review(f)
    assert r['summary']['source_bindings'] == f.definition['source_bindings']
    assert f.checks.list(AccountSession('child'), f.project.project_id) == []
    with pytest.raises(PermissionError):
        f.management.apply_review(AccountSession('parent'), r['review_id'], activate=True)
    with pytest.raises(PermissionError):
        f.management.apply_review(AccountSession('child'), r['review_id'], activate=True, channel='task:other')
    value = f.management.apply_review(AccountSession('child'), r['review_id'], activate=True)
    assert value['status'] == 'active'
    for _ in range(3):
        assert f.checks.guard(value['policy_id'], expected_revision=value['revision'])
    with pytest.raises(PermissionError):
        f.management.apply_review(AccountSession('child'), r['review_id'], activate=True)


@pytest.mark.parametrize('change', ['grant', 'project', 'connection', 'choice', 'expired'])
def test_stale_reviews_fail_closed(management, change):
    f = management
    r = review(f)
    if change == 'grant':
        f.service.revoke_grant(AccountSession('parent'), grant_id='school-grant', expected_revision=1)
    elif change == 'project':
        config = f.projects.read_project_config(f.project.project_id)
        config['goals'] = ['Different goal']
        f.projects.write_project_config(f.project.project_id, config, requester_user_id='child')
    elif change == 'connection':
        revise_connection(f, status='paused', health_code='paused')
    elif change == 'choice':
        f.bindings.select(actor_id='child', skill_id='skill:fixture', name='mail', binding_id='binding-one')
    else:
        f.management._reviews[r['review_id']]['expires'] = 0
    with pytest.raises((PermissionError, ValueError)):
        f.management.apply_review(AccountSession('child'), r['review_id'], activate=True)
    assert f.checks.list(AccountSession('child'), f.project.project_id) == []


@pytest.mark.parametrize('change', ['recipient', 'source', 'missing', 'duration', 'window'])
def test_invalid_scope_or_missing_details_cannot_review(management, change):
    f = management
    if change == 'recipient': f.definition['recipient_user_ids'] = ['other']
    if change == 'source': f.definition['source_bindings'][0]['resources'] = ['private-other-folder']
    if change == 'missing': del f.definition['timezone']
    if change == 'duration': f.definition['duration']['ends_at'] = None
    if change == 'window': f.definition['retrieval_window']['end'] = '08:00'
    with pytest.raises((ValueError, PermissionError)):
        review(f)


def test_one_minute_window_and_stale_lifecycle_response(management):
    f = management
    f.definition['retrieval_window']['end'] = '08:01'
    r = review(f)
    p = f.management.apply_review(AccountSession('child'), r['review_id'], activate=True)
    paused = f.management.request(AccountSession('child'), 'set_status', dict(policy_id=p['policy_id'], expected_revision=p['revision'], status='paused'))
    with pytest.raises(ValueError, match='revision_conflict'):
        f.management.request(AccountSession('child'), 'set_status', dict(policy_id=p['policy_id'], expected_revision=p['revision'], status='active'))
    resumed = f.management.request(AccountSession('child'), 'set_status', dict(policy_id=p['policy_id'], expected_revision=paused['revision'], status='active'))
    assert resumed['status'] == 'active'
    with pytest.raises(PermissionError):
        f.management.project(AccountSession('parent'), f.project.project_id)


def test_grant_and_explicit_role_selection(management):
    f = management
    result = f.management.request(AccountSession('parent'), 'bind_grant', dict(grant_id='school-grant', skill_id='skill:fixture'))
    choices = f.management.catalog(AccountSession('child'))['bindings']
    assert result['binding_id'] in {b['binding_id'] for b in choices}
    f.management.request(AccountSession('child'), 'select_binding', dict(binding_id=result['binding_id'], skill_id='skill:fixture', name='school'))
    sources = f.management.sources(AccountSession('child'))
    assert any(s['role'] == 'school' for s in sources)
    with pytest.raises(PermissionError):
        f.management.request(AccountSession('other'), 'bind_grant', dict(grant_id='school-grant', skill_id='skill:fixture'))


def test_native_uses_task_identity_and_cancellation(management):
    f = management
    tool = build_integration_manage_tool_definition(f.runtime)
    context = SimpleNamespace(task=SimpleNamespace(user='child', task_id='test-task', status='running'), cancellation_checker=lambda: False)
    result = tool.callable({'action': 'review', 'values': {'definition': f.definition}}, context=context)
    assert f.management._reviews[result['review_id']]['channel'] == 'task:test-task'
    context.cancellation_checker = lambda: True
    with pytest.raises(PermissionError, match='cancelled'):
        tool.callable({'action': 'apply_review', 'values': {'review_id': result['review_id'], 'activate': True}}, context=context)
    with pytest.raises(PermissionError):
        tool.callable({'action': 'catalog', 'values': {}}, context=None)
    context.cancellation_checker = lambda: False
    from alphonse.agent_v2.core.intelligence.v3.act_evidence import observation
    arguments = {'action': 'apply_review', 'values': {'review_id': result['review_id'], 'activate': True}}
    stored = tool.callable(arguments, context=context)
    entry = {'evidence_ref': 'check:created', 'status': 'success', 'tool_id': tool.descriptor.tool_id,
             'arguments': arguments, 'result': stored}
    receipt = observation(entry, ['ac-1'])['effect_receipt']
    assert receipt == {'source': 'integration_management_commit',
                       'target': {'kind': 'project_check', 'id': stored['policy_id']}}
    assert f.checks.get(AccountSession('child'), stored['policy_id'])['status'] == 'active'
    assert 'effect_receipt' not in observation({**entry, 'result': {**stored, 'status': 'paused'}}, ['ac-1'])


def test_legacy_dispatch_cannot_claim_authenticated_management(management):
    server = V2DaemonServer(SimpleNamespace(runtime=management.runtime))
    with pytest.raises(ValueError):
        server._dispatch({'version': PROTOCOL_VERSION, 'method': 'optional_management', 'params': {'actor_user_id': 'parent'}})


def test_peer_identity_not_payload(management, monkeypatch):
    f = management
    f.runtime.user = 'child'
    daemon = SimpleNamespace(runtime=f.runtime, optional_accounts=lambda s: None)
    server = V2DaemonServer(daemon)
    import os
    peer = SimpleNamespace(getpeereid=lambda: (os.getuid(), 0))
    request = {'version': PROTOCOL_VERSION, 'params': {'action': 'catalog'}}
    assert server._optional_management(peer, request)['actor_user_id'] == 'child'
    with pytest.raises(ValueError, match='identity_payload'):
        server._optional_management(peer, dict(request, params={'action': 'catalog', 'user': 'parent'}))
    with pytest.raises(PermissionError, match='peer_denied'):
        server._optional_management(SimpleNamespace(getpeereid=lambda: (os.getuid()+1, 0)), request)


def test_real_unix_peer_is_authenticated_without_python_getpeereid(management):
    import socket
    import os
    f = management
    f.runtime.user = 'child'
    daemon = SimpleNamespace(runtime=f.runtime, optional_accounts=lambda s: None)
    server = V2DaemonServer(daemon)
    left, right = socket.socketpair()
    try:
        result = server._optional_management(left, {'version': PROTOCOL_VERSION, 'params': {'action': 'catalog'}})
        assert result['actor_user_id'] == 'child'
    finally:
        left.close()
        right.close()


def test_review_save_rechecks_authority_after_preflight(management, monkeypatch):
    f = management
    r = review(f)
    original = f.checks.save

    def race(*args, **kwargs):
        config = f.projects.read_project_config(f.project.project_id)
        config['goals'] = ['Concurrent goal change']
        f.projects.write_project_config(f.project.project_id, config, requester_user_id='child')
        return original(*args, **kwargs)

    monkeypatch.setattr(f.checks, 'save', race)
    with pytest.raises(ValueError, match='review_stale'):
        f.management.apply_review(AccountSession('child'), r['review_id'], activate=True)
    assert f.checks.list(AccountSession('child'), f.project.project_id) == []


def test_catalog_reports_provider_readiness_without_exposing_configuration(management):
    f = management
    f.management.oauth = SimpleNamespace(clients={'google': object()},
        provider_status={'google': 'available', 'microsoft': 'invalid_configuration'},
        health=lambda session, connection_id: {'connection_id': connection_id})
    providers = f.management.catalog(AccountSession('parent'))['providers']
    assert [(p['provider'], p['status']) for p in providers] == [
        ('google', 'available'), ('microsoft', 'invalid_configuration')]
    assert 'client_id' not in json.dumps(providers)
