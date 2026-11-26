"""Synthetic management principals and resources; no provider access."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest

from test_account_grants import f, expiry, revise_connection
from test_project_checks import checks
from alphonse.agent_v2.accounts import (
    AccountSession, GMAIL_CONTENT_OPERATION, GMAIL_READ_SCOPE, OUTLOOK_INBOX_OPERATION, OUTLOOK_INBOX_RESOURCE,
)
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


def test_owner_google_setup_uses_canonical_owner_and_limits_actions(management, monkeypatch):
    import os
    f = management
    f.runtime.user = 'child'  # Stale runtime identity must never choose the actor.
    f.runtime.integration_store = f.store
    monkeypatch.setattr(os.path, 'samefile', lambda *_: True)
    captured = []
    oauth = SimpleNamespace(clients={})
    daemon = SimpleNamespace(runtime=f.runtime, experimental_owner_only=True,
                             optional_accounts=lambda session: oauth)
    server = V2DaemonServer(daemon)
    peer = SimpleNamespace(getpeereid=lambda: (os.getuid(), 0))
    def invoke(session, action, values):
        captured.append((session.actor_user_id, action, values))
        return {'providers': [{'provider': 'google', 'capabilities': ['mail_read', 'calendar_read']},
                              {'provider': 'microsoft', 'capabilities': ['mail_read']}],
                'connections': [{'provider': 'google', 'connection_id': 'g'}, {'provider': 'microsoft', 'connection_id': 'm'}],
                'grants': []}
    monkeypatch.setattr(f.management, 'request', invoke)
    req = lambda action, values={}: {'version': PROTOCOL_VERSION, 'params': {'action': action, 'values': values}}
    result = server._optional_management(peer, req('catalog'))
    assert captured[-1][0] == 'parent'
    assert result['owner_gmail_setup_only'] is True
    assert result['owner_google_setup_only'] is True
    assert result['owner_gmail_content_setup'] is True
    assert result['owner_gmail_label_setup'] is True
    assert result['providers'] == [{'provider': 'google', 'capabilities': ['mail_read', 'calendar_read']}]
    assert result['connections'] == [{'provider': 'google', 'connection_id': 'g'}]
    started = []
    server._oauth_loopback = SimpleNamespace(start=started.append)
    server._accepting_handlers = True  # Exercise an accepted live IPC request.
    values = {'connection_id': 'new', 'provider': 'google', 'capabilities': ['mail_read'],
              'expected_revision': 0, 'display_name': 'Fixture'}
    server._optional_management(peer, req('begin', values))
    assert captured[-1] == ('parent', 'begin', values)
    assert started == ['google']
    calendar = {**values, 'capabilities': ['calendar_read']}
    server._optional_management(peer, req('begin', calendar))
    assert captured[-1] == ('parent', 'begin', calendar)
    server._accepting_handlers = False
    with pytest.raises(RuntimeError, match='daemon_stopping'):
        server._optional_management(peer, req('begin', values))
    assert started == ['google', 'google']
    for action, values in [
        ('review', {}), ('set_status', {}), ('complete', {}),
        ('begin', {'provider': 'google', 'capabilities': ['mail_read', 'calendar_read']}),
        ('begin', {'provider': 'google', 'capabilities': ['calendar_discover']}),
        ('begin', {'provider': 'microsoft', 'capabilities': ['mail_read']}),
        ('save_grant', {'beneficiary_user_id': 'child', 'operations': ['mail_read'],
                        'resources': ['INBOX'], 'recipient_user_ids': ['parent'], 'project_id': None}),
        ('pause', {'connection_id': 'one'}),  # Synthetic provider is not Gmail.
        ('revoke_grant', {'grant_id': 'grant-one', 'expected_revision': 1}),
    ]:
        with pytest.raises(PermissionError):
            server._optional_management(peer, req(action, values))
    with pytest.raises(ValueError, match='identity_payload'):
        server._optional_management(peer, {'version': PROTOCOL_VERSION, 'params': {'action': 'catalog', 'user': 'child'}})
    with pytest.raises(PermissionError, match='peer_denied'):
        server._optional_management(SimpleNamespace(getpeereid=lambda: (os.getuid()+1, 0)), req('catalog'))


def test_owner_outlook_ipc_only_exposes_and_grants_personal_inbox_count(management, monkeypatch):
    import os
    from urllib.parse import urlencode
    from alphonse.agent_v2.integrations.oauth import ClientConfig, OAuthService
    from test_oauth_lifecycle import FakeProvider

    f = management
    f.runtime.integration_store = f.store
    with f.store._connect() as db:
        db.execute("UPDATE v2_integrations SET provider_key='google' WHERE integration_id='one'")
    monkeypatch.setattr(os.path, 'samefile', lambda *_: True)
    redirect = 'http://127.0.0.1:54321/callback'
    oauth = OAuthService(f.service, [ClientConfig('microsoft', 'disposable', redirect)], client_factory=FakeProvider)
    daemon = SimpleNamespace(runtime=f.runtime, experimental_owner_only=True, optional_accounts=lambda session: oauth)
    server = V2DaemonServer(daemon)
    server._accepting_handlers = True
    server._oauth_loopback = SimpleNamespace(start=lambda provider: None)
    peer = SimpleNamespace(getpeereid=lambda: (os.getuid(), 0))

    def request(action, **values):
        return server._optional_management(peer, {'version': PROTOCOL_VERSION, 'params': {'action': action, 'values': values}})

    catalog = request('catalog')
    assert catalog['owner_outlook_inbox_setup']
    assert catalog['providers'] == [{'provider': 'microsoft', 'capabilities': [OUTLOOK_INBOX_OPERATION]}]
    attempt = request('begin', connection_id='outlook', provider='microsoft', display_name='Test Hotmail',
                      capabilities=[OUTLOOK_INBOX_OPERATION], expected_revision=0)
    assert request('oauth_attempt_status', provider='microsoft', attempt_id=attempt['attempt_id']) == {'status': 'pending'}
    callback = redirect + '?' + urlencode({'state': oauth.clients['microsoft'].state, 'code': 'disposable-code'})
    oauth.complete(AccountSession('parent'), callback)
    assert request('oauth_attempt_status', provider='microsoft', attempt_id=attempt['attempt_id']) == {'status': 'connected'}
    account, = [item for item in request('catalog')['connections'] if item['provider'] == 'microsoft']
    assert account['display_name'] == 'Test Hotmail'
    grant = dict(grant_id='outlook-grant', connection_id='outlook', expected_revision=0,
                 expected_connection_revision=account['revision'], beneficiary_user_id='parent',
                 operations=[OUTLOOK_INBOX_OPERATION], resources=[OUTLOOK_INBOX_RESOURCE],
                 recipient_user_ids=['parent'], expires_at=expiry())
    for bad in ({'resources': ['all-mail']}, {'recipient_user_ids': ['child']}, {'project_id': 'project'},
                {'operations': ['mail_read'], 'resources': ['INBOX']}):
        with pytest.raises(PermissionError, match='owner_only_account_scope_unavailable'):
            request('save_grant', **(grant | bad))
    with pytest.raises(ValueError, match='oauth_revision_conflict'):
        request('save_grant', **(grant | {'expected_connection_revision': account['revision'] - 1}))
    with pytest.raises(PermissionError, match='outlook_inbox_grant_requires_desktop'):
        f.management.request(AccountSession('parent'), 'save_grant', grant)
    assert request('save_grant', **grant)['resources'] == [OUTLOOK_INBOX_RESOURCE]
    assert request('revoke_grant', grant_id='outlook-grant', expected_revision=1) == {'revoked': True}


def test_owner_gmail_setup_accepts_only_narrow_self_grant(management, monkeypatch):
    import os
    f = management
    f.runtime.integration_store = f.store
    monkeypatch.setattr(os.path, 'samefile', lambda *_: True)
    f.service.create_connection(AccountSession('parent'), connection_id='gmail', provider_key='google',
                                provider_subject='fixture', display_name='Fixture')
    f.store.install_credentials('gmail', {'token': 'fixture'}, expected_generation=0)
    f.service.save_connection(AccountSession('parent'), connection_id='gmail', expected_revision=1,
                              provider_subject='fixture', capabilities=['mail_read'], granted_scopes=['fixture'])
    daemon = SimpleNamespace(runtime=f.runtime, experimental_owner_only=True,
                             optional_accounts=lambda session: SimpleNamespace())
    server = V2DaemonServer(daemon)
    peer = SimpleNamespace(getpeereid=lambda: (os.getuid(), 0))
    values = dict(grant_id='gmail-owner', connection_id='gmail', expected_revision=0,
                  beneficiary_user_id='parent', operations=['mail_read'], resources=['INBOX'],
                  recipient_user_ids=['parent'], expires_at=expiry(), project_id=None)
    req = lambda value: {'version': PROTOCOL_VERSION, 'params': {'action': 'save_grant', 'values': value}}
    result = server._optional_management(peer, req(values))
    assert result['grantor_user_id'] == result['beneficiary_user_id'] == 'parent'
    with pytest.raises(PermissionError, match='scope_unavailable'):
        server._optional_management(peer, req({**values, 'resources': ['INBOX', 'OTHER']}))


def test_gmail_content_requires_separate_owner_desktop_grant(management, monkeypatch):
    import os
    f = management
    owner = AccountSession('parent')
    f.runtime.integration_store = f.store
    monkeypatch.setattr(os.path, 'samefile', lambda *_: True)
    f.service.create_connection(owner, connection_id='gmail', provider_key='google',
                                provider_subject='fixture', display_name='Fixture')
    f.store.install_credentials('gmail', {'token': 'fixture'}, expected_generation=0)
    f.service.save_connection(owner, connection_id='gmail', expected_revision=1,
                              provider_subject='fixture', capabilities=['mail_read'], granted_scopes=[GMAIL_READ_SCOPE])
    daemon = SimpleNamespace(runtime=f.runtime, experimental_owner_only=True,
                             optional_accounts=lambda session: SimpleNamespace())
    server = V2DaemonServer(daemon)
    peer = SimpleNamespace(getpeereid=lambda: (os.getuid(), 0))
    content = dict(grant_id='content', connection_id='gmail', expected_revision=0,
                   beneficiary_user_id='parent', operations=[GMAIL_CONTENT_OPERATION], resources=['INBOX'],
                   recipient_user_ids=['parent'], expires_at=expiry(), project_id=None)

    def request(action, values):
        return server._optional_management(peer, {'version': PROTOCOL_VERSION,
            'params': {'action': action, 'values': values}})

    with pytest.raises(PermissionError, match='requires_desktop'):
        f.management.request(owner, 'save_grant', content, channel='task:fixture')
    for changes in [{'resources': ['gmail:all']}, {'project_id': 'fixture'},
                    {'recipient_user_ids': ['parent', 'child']}, {'beneficiary_user_id': 'child'},
                    {'operations': ['mail_read', GMAIL_CONTENT_OPERATION]}]:
        with pytest.raises(PermissionError, match='scope_unavailable'):
            request('save_grant', {**content, **changes})
    count = request('save_grant', {**content, 'grant_id': 'count', 'operations': ['mail_read']})
    with pytest.raises(ValueError, match='identity_immutable'):
        request('save_grant', {**content, 'grant_id': count['grant_id'], 'expected_revision': 1})
    grant = request('save_grant', content)
    assert grant['operations'] == [GMAIL_CONTENT_OPERATION]
    assert f.store.get('gmail').connection['capabilities'] == ['mail_read']
    auth = dict(connection_id='gmail', grant_id='content', operation=GMAIL_CONTENT_OPERATION, resources=['INBOX'],
                recipient_user_ids=['parent'], required_scopes=[GMAIL_READ_SCOPE], expected_grant_revision=1)
    with pytest.raises(PermissionError, match='grant_denied'):
        f.service.authorize(owner, **auth)
    for changes in [{'recipient_user_ids': []}, {'grant_id': 'count'}, {'required_scopes': []}, {'resources': ['gmail:all']}]:
        with pytest.raises(PermissionError, match='grant_denied'):
            f.service.authorize(owner, **{**auth, **changes}, owner_gmail_content=True)
    assert f.service.authorize(owner, **auth, owner_gmail_content=True)[1] == grant
    revised = request('save_grant', {**content, 'expected_revision': 1})
    assert revised['revision'] == 2
    with pytest.raises(PermissionError, match='grant_denied'):
        f.service.authorize(owner, **auth, owner_gmail_content=True)
    assert request('revoke_grant', {'grant_id': 'content', 'expected_revision': 2}) == {'revoked': True}
    with pytest.raises(PermissionError, match='grant_denied'):
        f.service.authorize(owner, **{**auth, 'expected_grant_revision': 3}, owner_gmail_content=True)


def test_owner_gmail_label_list_and_grant_are_bound_to_exact_account_and_label(management, monkeypatch):
    import os

    f = management
    owner = AccountSession('parent')
    f.runtime.integration_store = f.store
    monkeypatch.setattr(os.path, 'samefile', lambda *_: True)
    f.service.create_connection(owner, connection_id='gmail', provider_key='google',
                                provider_subject='fixture', display_name='Fixture')
    f.store.install_credentials('gmail', {'token': 'fixture'}, expected_generation=0)
    f.service.save_connection(owner, connection_id='gmail', expected_revision=1,
                              provider_subject='fixture', capabilities=['mail_read'], granted_scopes=[GMAIL_READ_SCOPE])
    label = {'id': 'Label_17', 'name': 'Projects/Planning'}
    list_values = {'connection_id': 'gmail', 'expected_revision': 2}
    calls = []

    def list_labels(session, **values):
        calls.append((session, values))
        return {'labels': [dict(label), {'id': 'SENT', 'name': 'SENT'}]}

    daemon = SimpleNamespace(runtime=f.runtime, experimental_owner_only=True,
                             optional_accounts=lambda session: SimpleNamespace(list_gmail_labels=list_labels))
    server = V2DaemonServer(daemon)
    peer = SimpleNamespace(getpeereid=lambda: (os.getuid(), 0))

    def request(action, values, *, request_peer=peer):
        return server._optional_management(request_peer, {'version': PROTOCOL_VERSION,
            'params': {'action': action, 'values': values}})

    with pytest.raises(PermissionError, match='requires_desktop'):
        f.management.request(owner, 'list_gmail_labels', list_values, channel='task:fixture')
    with pytest.raises(PermissionError, match='peer_denied'):
        request('list_gmail_labels', list_values,
                request_peer=SimpleNamespace(getpeereid=lambda: (os.getuid() + 1, 0)))
    with pytest.raises(ValueError, match='gmail_label_list_request_invalid'):
        request('list_gmail_labels', {**list_values, 'actor_user_id': 'child'})
    assert calls == []
    assert request('list_gmail_labels', list_values)['labels'][0] == label
    assert calls == [(owner, list_values)]

    content = dict(grant_id='label-content', connection_id='gmail', expected_revision=0,
                   beneficiary_user_id='parent', operations=[GMAIL_CONTENT_OPERATION], resources=[label['id']],
                   recipient_user_ids=['parent'], expires_at=None, gmail_label_name=label['name'])
    for changes in [{'resources': [label['id'], label['id']]}, {'resources': ['Label_*']},
                    {'operations': ['mail_read']}, {'recipient_user_ids': ['parent', 'child']}]:
        with pytest.raises(PermissionError, match='scope_unavailable'):
            request('save_grant', {**content, **changes})
    for changes in [{'gmail_label_name': 'Renamed'}, {'resources': ['Label_18']}, {'gmail_label_name': None}]:
        with pytest.raises(ValueError, match='gmail_label_selection_stale'):
            request('save_grant', {**content, **changes})

    grant = request('save_grant', content)
    assert grant['gmail_label_name'] == label['name'] and grant['expires_at'] is None
    with pytest.raises(ValueError, match='gmail_label_grant_exists'):
        request('save_grant', {**content, 'grant_id': 'another-label-content'})
    auth = dict(connection_id='gmail', grant_id=grant['grant_id'], operation=GMAIL_CONTENT_OPERATION,
                resources=[label['id']], recipient_user_ids=['parent'], required_scopes=[GMAIL_READ_SCOPE],
                owner_gmail_content=True)
    assert f.service.authorize(owner, **auth)[1] == grant
    for resources in (['INBOX'], [label['id'], label['id']], ['gmail:all']):
        with pytest.raises(PermissionError, match='account_grant_denied'):
            f.service.authorize(owner, **{**auth, 'resources': resources})
    with pytest.raises(ValueError, match='identity_immutable'):
        request('save_grant', {**content, 'expected_revision': 1,
                               'resources': ['SENT'], 'gmail_label_name': 'SENT'})
    with pytest.raises(ValueError, match='account_grant_invalid'):
        request('save_grant', {**content, 'grant_id': 'count', 'operations': ['mail_read'],
                               'resources': ['INBOX'], 'gmail_label_name': None})
    assert request('revoke_grant', {'grant_id': grant['grant_id'], 'expected_revision': 1}) == {'revoked': True}
    with pytest.raises(PermissionError, match='account_grant_denied'):
        f.service.authorize(owner, **auth)


def test_owner_calendar_setup_accepts_only_one_exact_calendar_for_the_owner(management, monkeypatch):
    import os
    f = management
    f.runtime.integration_store = f.store
    monkeypatch.setattr(os.path, 'samefile', lambda *_: True)
    owner = AccountSession('parent')
    f.service.create_connection(owner, connection_id='calendar', provider_key='google',
                                provider_subject='fixture', display_name='Fixture')
    f.store.install_credentials('calendar', {'token': 'fixture'}, expected_generation=0)
    f.service.save_connection(owner, connection_id='calendar', expected_revision=1,
                              provider_subject='fixture', capabilities=['calendar_read'], granted_scopes=['fixture'])
    daemon = SimpleNamespace(runtime=f.runtime, experimental_owner_only=True,
                             optional_accounts=lambda session: SimpleNamespace())
    server = V2DaemonServer(daemon)
    peer = SimpleNamespace(getpeereid=lambda: (os.getuid(), 0))
    values = dict(grant_id='calendar-owner', connection_id='calendar', expected_revision=0,
                  beneficiary_user_id='parent', operations=['calendar_read'], resources=['fixture@calendar.example'],
                  recipient_user_ids=['parent'], expires_at=expiry(), project_id=None)
    def save(changes):
        return server._optional_management(peer, {'version': PROTOCOL_VERSION,
            'params': {'action': 'save_grant', 'values': {**values, **changes}}})
    for changes in [{'resources': ['primary']}, {'resources': ['*']}, {'resources': ['one', 'two']},
                    {'recipient_user_ids': ['parent', 'child']}, {'project_id': 'other'}]:
        with pytest.raises(PermissionError, match='scope_unavailable'):
            save(changes)
    grant = save({})
    assert grant['resources'] == values['resources'] and grant['recipient_user_ids'] == ['parent']


def test_owner_gmail_setup_rejects_split_owner_database(management):
    import os
    f = management
    f.runtime.integration_store = f.store
    daemon = SimpleNamespace(runtime=f.runtime, experimental_owner_only=True,
                             optional_accounts=lambda session: None)
    server = V2DaemonServer(daemon)
    with pytest.raises(PermissionError, match='owner_database_unavailable'):
        server._optional_management(SimpleNamespace(getpeereid=lambda: (os.getuid(), 0)),
                                    {'version': PROTOCOL_VERSION, 'params': {'action': 'catalog'}})


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
