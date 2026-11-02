"""Real Graph adapter through durable project-check pipeline; synthetic I/O only."""
from datetime import datetime, timedelta, timezone
from dataclasses import replace
import json
import time

import pytest

from test_account_grants import f, expiry
from test_project_checks import checks
from test_project_check_scheduling import scheduled
from test_graph_mail import Transport, page, message, response, link
from test_oauth_lifecycle import FakeProvider
from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.automations import EventAutomationStore
from alphonse.agent_v2.integrations.graph_mail import GraphMailAdapter
from alphonse.agent_v2.integrations.oauth import OAuthService, ClientConfig, SCOPES
from alphonse.agent_v2.services.source_acquisition import SourceAcquisition
from alphonse.agent_v2.services.project_check_worker import CheckContext


@pytest.fixture
def graph_checks(checks):
    f = checks
    f.service.create_connection(AccountSession('parent'), connection_id='ms', provider_key='microsoft',
        provider_subject='disposable-ms', display_name='Selected school mail')
    f.store.install_credentials('ms', dict(access_token='disposable', refresh_token='disposable-refresh',
        expires_at=str(time.time() + 3600), token_type='Bearer', scope=SCOPES['microsoft']['mail_read']), expected_generation=0)
    f.service.save_connection(AccountSession('parent'), connection_id='ms', expected_revision=1,
        provider_subject='disposable-ms', capabilities=['mail_read'], granted_scopes=[SCOPES['microsoft']['mail_read']])
    f.service.save_grant(AccountSession('parent'), grant_id='ms-grant', connection_id='ms', expected_revision=0,
        beneficiary_user_id='child', operations=['mail_read'], resources=['folder-a', 'folder-b'],
        recipient_user_ids=['parent'], expires_at=expiry(), project_id=f.project.project_id)
    f.bindings.configure(actor_id='parent', owner_id='child', binding_id='ms-binding', skill_id='skill:fixture',
        kind='account', target='ms', grant_id='ms-grant', capabilities=('mail_read',),
        required_scopes=(SCOPES['microsoft']['mail_read'],))
    f.bindings.select(actor_id='child', skill_id='skill:fixture', name='mail', binding_id='ms-binding')
    f.definition['source_bindings'] = [dict(skill_id='skill:fixture', role='mail', binding_id='ms-binding',
        connection_id='ms', grant_id='ms-grant', operation='mail_read', resources=['folder-a', 'folder-b'])]
    f.oauth = OAuthService(f.service, [ClientConfig('microsoft', 'disposable-app', 'http://127.0.0.1:54321/callback')], client_factory=FakeProvider)
    return f


@pytest.fixture
def graph_run(graph_checks, request):
    f = request.getfixturevalue('scheduled')
    f.events = EventAutomationStore(f.tmp / 'events.db')
    f.transport = Transport()
    f.adapter = GraphMailAdapter(f.oauth, transport=f.transport, clock=lambda: f.now)
    f.acquisition = SourceAcquisition(f.dispatcher, f.events, adapters={'microsoft': f.adapter}, clock=lambda: f.now)
    f.process = lambda: f.worker(f.acquisition, on_results=f.acquisition.publish_ready).run_once()
    f.run_id = f.dispatch()['check_run_id']
    return f


def test_folder_state_and_provenance_survive_restart_then_delta_replay(graph_run):
    f = graph_run
    f.transport.pages = [page(next=True, token='opaque-page'), page(), page(message('folder-b'), folder='folder-b')]
    first = f.process()
    assert first['state'] == 'pending' and first['pages'] == 1
    with f.schedules._connect() as db:
        saved = json.loads(db.execute('SELECT definition FROM v2_source_streams').fetchone()[0])
    assert json.loads(saved['cursor'])['links']['folder-a'] == link(token='opaque-page', kind='skip')
    f.acquisition = SourceAcquisition(f.dispatcher, f.events, adapters={'microsoft': f.adapter}, clock=lambda: f.now)
    f.process()
    result = f.process()
    assert result['outcome'] == 'complete'
    coverage = next(iter(result['coverage']['sources'].values()))
    assert coverage['resources'] == ['folder-a', 'folder-b'] and coverage['coverage_complete']
    assert coverage['resource_coverage'] == {'folder-a': 'complete', 'folder-b': 'complete'}
    evidence = f.acquisition.evidence(AccountSession('child'), f.run_id)
    assert evidence[0]['connection_id'] == 'ms' and evidence[0]['resource'] == 'folder-b'
    assert evidence[0]['source_id'] == 'immutable-1' and evidence[0]['owner_user_id'] == 'parent'
    f.now += timedelta(days=7)
    f.scheduler.run_once(now=f.now)
    f.transport.pages = [page(token='new-delta'), page(folder='folder-b', token='new-rules')]
    f.process()
    assert f.transport.calls[-1][0] == link()
    assert f.process()['outcome'] == 'no_match'


def test_model_failure_after_graph_retrieval_keeps_evidence_and_processing_independent(graph_run):
    f = graph_run
    f.transport.pages = [page(message(), next=True, token='next'), page(), page(folder='folder-b')]
    f.process()
    saved = f.acquisition.evidence(AccountSession('child'), f.run_id)
    def failing_model(ctx):
        return ctx.perform('analysis', lambda: (_ for _ in ()).throw(RuntimeError('private model failure')))
    assert f.worker(failing_model).run_once()['state'] == 'retry_wait'
    assert len(f.transport.calls) == 1
    assert f.acquisition.processing(AccountSession('child'), f.run_id) == {}
    f.now += timedelta(seconds=5)
    f.process()
    assert f.transport.calls[-1][0] == link(token='next', kind='skip')
    assert f.process()['outcome'] == 'complete'
    assert f.acquisition.evidence(AccountSession('child'), f.run_id) == saved


def test_no_match_quiet_and_partial_not_no_match(graph_run):
    f = graph_run
    f.transport.pages = [page(), page(folder='folder-b')]
    f.process()
    assert f.process()['outcome'] == 'no_match'
    with f.events._connect() as db:
        assert db.execute('SELECT COUNT(*) FROM v2_event_history').fetchone()[0] == 0
    f.now += timedelta(days=7)
    f.scheduler.run_once(now=f.now)
    f.transport.pages = [page(), response({'error': {'code': 'ErrorAccessDenied'}}, 403)]
    f.process()
    result = f.process()
    assert result['outcome'] == 'partial' and result['coverage']['gaps']
    assert next(iter(result['coverage']['sources'].values()))['resource_coverage'] == {
        'folder-a': 'complete', 'folder-b': 'unavailable'}
    with f.events._connect() as db:
        payload = db.execute('SELECT payload_json FROM v2_event_history').fetchone()[0]
    assert json.loads(payload)['outcome'] == 'partial'
    assert 'deltatoken' not in payload and 'disposable' not in payload


def test_expired_delta_does_bounded_baseline_and_never_reports_no_match(graph_run):
    f = graph_run
    f.transport.pages = [response({'error': {'code': 'SyncStateNotFound'}}, 410), page(), page(folder='folder-b')]
    assert f.process()['state'] == 'pending'
    f.process()
    result = f.process()
    assert result['outcome'] == 'partial'
    assert 'cursor_expired_bounded_resync' in result['coverage']['gaps']
    assert '%24filter=receivedDateTime+ge+' in f.transport.calls[1][0]
    assert (f.now - timedelta(days=30)).isoformat().replace(':', '%3A').replace('+', '%2B') in f.transport.calls[1][0]


@pytest.mark.parametrize('delay', [120, 3600, 999999999999999999999999])
def test_retry_after_persisted_no_early_read_or_outside_window(graph_run, delay):
    f = graph_run
    f.transport.pages = [response({'error': {'message': 'secret'}}, 429, {'Retry-After': str(delay)}), page()]
    result = f.process()
    assert result['pages'] == 1 and result['acquisition_checkpoint'] == {}
    if delay >= 3600:
        assert result['outcome'] == 'partial' and result['error_code'] == 'retry_outside_window'
        return
    assert result['state'] == 'retry_wait'
    assert datetime.fromisoformat(result['next_attempt_at']) == f.now + timedelta(seconds=delay)
    f.now += timedelta(seconds=delay - 1)
    assert f.process() is None and len(f.transport.calls) == 1
    f.now += timedelta(seconds=1)
    f.process()
    assert len(f.transport.calls) == 2


def test_revocation_between_pages_prevents_next_graph_request(graph_run):
    f = graph_run
    f.transport.pages = [page(message(), next=True)]
    f.process()
    f.service.revoke_grant(AccountSession('parent'), grant_id='ms-grant', expected_revision=1)
    assert f.process()['outcome'] == 'blocked' and len(f.transport.calls) == 1
    with pytest.raises(PermissionError): f.acquisition.evidence(AccountSession('child'), f.run_id)


def test_crash_after_acceptance_recovers_without_duplicate_graph_read(graph_run, monkeypatch):
    f = graph_run
    f.transport.pages = [page(message(), next=True, token='persisted'), page(), page(folder='folder-b')]
    accept = f.acquisition._accept
    def crash(*a, **kw):
        accept(*a, **kw)
        raise SystemExit('process death')
    monkeypatch.setattr(f.acquisition, '_accept', crash)
    with pytest.raises(SystemExit): f.process()
    assert len(f.acquisition.evidence(AccountSession('child'), f.run_id)) == 1
    monkeypatch.setattr(f.acquisition, '_accept', accept)
    f.now += timedelta(seconds=31)
    f.process()
    assert f.transport.calls[-1][0] == link(token='persisted', kind='skip')
    assert f.process()['outcome'] == 'complete'
    assert len(f.acquisition.evidence(AccountSession('child'), f.run_id)) == 1


def test_attachment_charges_transport_bytes_not_just_decoded_content(graph_run):
    f = graph_run
    data = dict(id='a', contentType='text/plain', contentBytes='aGk=', **{'@odata.type': '#microsoft.graph.fileAttachment'})
    raw = response(data)
    f.transport.pages = [raw]
    run = f.dispatcher.claim('read', f.now, 30)
    ctx = CheckContext(f.worker(), run)
    result = f.acquisition.read_operation(ctx, source_index=0, operation='attachment', resource='folder-a',
        source_id='m', attachment_id='a', max_bytes=100, content_type='text/plain')
    assert result == b'hi' and ctx.run['bytes_read'] == len(raw[2])
