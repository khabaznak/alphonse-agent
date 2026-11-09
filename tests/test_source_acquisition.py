"""Disposable adapter conformance; no real account or model access."""
from dataclasses import replace
from datetime import timedelta
import json

import pytest

from test_account_grants import f
from test_project_checks import checks
from test_project_check_scheduling import scheduled
from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.automations import EventAutomationStore
from alphonse.agent_v2.integrations.acquisition import SourcePage, SourceRecord, CursorExpired
from alphonse.agent_v2.integrations.content import normalize_mime, safe_link
from alphonse.agent_v2.services.source_acquisition import SourceAcquisition, WORKER, EVENT
from alphonse.agent_v2.services.project_check_worker import CheckStep, CheckContext


class Adapter:
    version = '1'
    def __init__(self, pages):
        self.pages, self.calls = list(pages), []

    def search(self, context, **kw):
        self.calls.append(('search', context, kw))
        result = self.pages.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def changes(self, context, **kw):
        self.calls.append(('changes', context, kw))
        result = self.pages.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def attachment(self, context, **kw):
        self.calls.append(('attachment', context, kw))
        return b'bounded attachment'


def record(f, **kw):
    return replace(SourceRecord('folder-a', 'message-1', 'version-1', f.now.isoformat(),
        b'Content-Type: text/plain; charset=utf-8\r\n\r\nAssignment 30%\n> quoted old deadline',
        'https://provider.invalid/item'), **kw)


@pytest.fixture
def acquisition(scheduled):
    f = scheduled
    f.events = EventAutomationStore(f.tmp / 'events.db')
    f.adapter = Adapter([])
    f.acquisition = SourceAcquisition(f.dispatcher, f.events,
        adapters={'synthetic': f.adapter}, clock=lambda: f.now)
    f.process = lambda: f.worker(f.acquisition, on_results=f.acquisition.publish_ready).run_once()
    f.run_id = f.dispatch()['check_run_id']
    return f


def test_bounded_context_durable_page_and_independent_model_failure(acquisition):
    f = acquisition
    f.adapter.pages = [SourcePage((record(f),), 'opaque-page', False), SourcePage((), 'opaque-history', True)]
    first = f.process()
    ctx = f.adapter.calls[0][1]
    assert ctx.session.actor_user_id == 'child'
    assert ctx.grant.grant_id == 'school-grant' and ctx.grant.resources == ('folder-a',)
    assert ctx.limits.pages == 5 and ctx.limits.items == 100
    assert ctx.deadline <= f.now + timedelta(seconds=60)
    assert first['acquisition_checkpoint']['progress']['cursor'] == 'opaque-page'
    saved = f.acquisition.evidence(AccountSession('child'), f.run_id)
    assert saved[0]['source_id'] == 'message-1' and saved[0]['version'] == 'version-1'
    assert 'quoted old deadline' in saved[0]['content']['parts'][0]['text']
    # Simulated classification failure after evidence acceptance. No provider
    # work is rolled back and no processing checkpoint is manufactured.
    def fail(ctx):
        return ctx.perform('analysis', lambda: (_ for _ in ()).throw(RuntimeError('private model error')))
    result = f.worker(fail).run_once()
    assert result['analysis_checkpoint'] == {} and result['state'] == 'retry_wait'
    f.now += timedelta(seconds=5)
    assert f.process()['outcome'] == 'complete'
    assert f.adapter.calls[-1][2]['cursor'] == 'opaque-page'
    f.acquisition.checkpoint_processing(AccountSession('child'), f.run_id,
        stage='classification', evidence_ids=[saved[0]['evidence_id']])
    assert f.dispatcher.get(f.run_id)['acquisition_checkpoint'] == {'source_index': 1}
    assert f.acquisition.processing(AccountSession('child'), f.run_id) == {'classification': [saved[0]['evidence_id']]}


def test_quiet_no_match_and_partial_is_never_no_match(acquisition):
    f = acquisition
    f.adapter.pages = [SourcePage(complete=True)]
    result = f.process()
    assert result['outcome'] == 'no_match' and not result['coverage']['gaps']
    with f.events._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM v2_event_history').fetchone()[0] == 0


@pytest.mark.parametrize('kind', ['coverage', 'expired'])
def test_incomplete_and_expired_cursor_bounded_resync(acquisition, kind):
    f = acquisition
    if kind == 'expired':
        f.adapter.pages = [CursorExpired(), SourcePage(complete=True, cursor='new-history')]
        assert f.process()['state'] == 'pending'
        assert f.process()['outcome'] == 'partial'
        assert f.adapter.calls[1][0] == 'search'
        assert f.adapter.calls[1][2]['since'] == f.now - timedelta(days=30)
    else:
        f.adapter.pages = [SourcePage(complete=True, coverage_complete=False)]
        assert f.process()['outcome'] == 'partial'
    assert f.dispatcher.get(f.run_id)['coverage']['gaps']


def test_restart_after_atomic_accept_never_skips_or_replays_page(acquisition, monkeypatch):
    f = acquisition
    f.adapter.pages = [SourcePage((record(f),), 'next', False), SourcePage(complete=True)]
    original = f.acquisition._accept
    def crash(*a, **kw):
        original(*a, **kw)
        raise SystemExit('simulated process death')
    monkeypatch.setattr(f.acquisition, '_accept', crash)
    with pytest.raises(SystemExit):
        f.process()
    assert len(f.acquisition.evidence(AccountSession('child'), f.run_id)) == 1
    f.now += timedelta(seconds=31)
    f.acquisition = SourceAcquisition(f.dispatcher, f.events, adapters={'synthetic': f.adapter}, clock=lambda: f.now)
    assert f.process()['outcome'] == 'complete'
    assert f.adapter.calls[-1][2]['cursor'] == 'next'
    assert len(f.acquisition.evidence(AccountSession('child'), f.run_id)) == 1


def test_transaction_rolls_back_evidence_and_cursor_together(acquisition):
    f = acquisition
    with f.schedules._connect() as conn:
        conn.execute("CREATE TRIGGER refuse_source BEFORE INSERT ON v2_source_evidence BEGIN SELECT RAISE(ABORT, 'crash'); END")
    f.adapter.pages = [SourcePage((record(f),), 'next')]
    result = f.process()
    assert result['state'] == 'retry_wait' and result['acquisition_checkpoint'] == {}
    with f.schedules._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM v2_source_pages').fetchone()[0] == 0
        assert conn.execute('SELECT COUNT(*) FROM v2_source_streams').fetchone()[0] == 0


def test_publication_crash_replay_one_scoped_event(acquisition, monkeypatch):
    f = acquisition
    f.adapter.pages = [SourcePage((record(f),), 'history', True)]
    publish = f.events.publish_managed
    def crash(**kw):
        result = publish(**kw)
        assert result['accepted']
        raise SystemExit('after journal, before receipt')
    monkeypatch.setattr(f.events, 'publish_managed', crash)
    with pytest.raises(SystemExit):
        f.process()
    monkeypatch.setattr(f.events, 'publish_managed', publish)
    f.acquisition.publish_ready()
    f.acquisition.publish_ready()
    with f.events._connect() as conn:
        rows = conn.execute('SELECT payload_json,envelope_json FROM v2_event_history').fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0]['envelope_json'])['actor_user_id'] == 'child'
    assert 'Assignment' not in rows[0]['payload_json']
    assert f.dispatcher.get(f.run_id)['publication_id'] == 'source-result:' + f.run_id
    assert not f.events.publish(worker_id=WORKER, event_type=EVENT, event_version='1',
        event_id='spoof', occurred_at=f.now.isoformat(), payload={})['accepted']


@pytest.mark.parametrize('violation', ['resource', 'bytes', 'items', 'stalled'])
def test_adapter_scope_and_budget_violations_are_not_success(acquisition, violation):
    f = acquisition
    r = record(f)
    if violation == 'resource': r = replace(r, resource='ungranted-folder')
    page = SourcePage((r,), 'next', False)
    if violation == 'bytes': page = replace(page, bytes_read=1000001)
    if violation == 'items': page = replace(page, records=(r,) * 101)
    if violation == 'stalled': page = replace(page, cursor=None)
    f.adapter.pages = [page]
    result = f.process()
    assert result['outcome'] != 'complete' and result['outcome'] != 'no_match'
    assert f.acquisition.evidence(AccountSession('child'), f.run_id) == []


def test_revocation_prevents_next_page_and_private_expansion(acquisition):
    f = acquisition
    f.adapter.pages = [SourcePage((record(f),), 'next', False)]
    f.process()
    with pytest.raises(PermissionError):
        f.acquisition.evidence(AccountSession('parent'), f.run_id)
    f.service.revoke_grant(AccountSession('parent'), grant_id='school-grant', expected_revision=1)
    assert f.process()['outcome'] == 'blocked'
    assert len(f.adapter.calls) == 1
    with pytest.raises(PermissionError):
        f.acquisition.evidence(AccountSession('child'), f.run_id)


def test_attachment_limits_window_and_grant_enforced(acquisition):
    f = acquisition
    run = f.dispatcher.claim('read', f.now, 30)
    ctx = CheckContext(f.worker(), run)
    assert f.acquisition.read_operation(ctx, source_index=0, operation='attachment',
        resource='folder-a', source_id='m', attachment_id='a', max_bytes=20, content_type='text/plain') == b'bounded attachment'
    assert ctx.run['bytes_read'] == 18
    with pytest.raises(PermissionError):
        f.acquisition.read_operation(ctx, source_index=0, operation='attachment', resource='other', max_bytes=20, content_type='text/plain')
    f.now += timedelta(hours=1)
    with pytest.raises(Exception):
        f.acquisition.read_operation(ctx, source_index=0, operation='attachment', resource='folder-a', max_bytes=20, content_type='text/plain')
    assert len(f.adapter.calls) == 1


def test_inert_html_and_quoted_evidence():
    value = normalize_mime(b'Content-Type: text/html\r\n\r\n<p>Due Friday</p><blockquote>Old Monday</blockquote>'
        b'<script>steal()</script><img src="https://attacker.invalid/tracker"><a href="javascript:steal()">notice</a>', max_bytes=1000)
    text = value['parts'][0]['text']
    assert 'Due Friday' in text and 'Old Monday' in text
    assert 'steal()' not in text and 'attacker' not in text
    assert safe_link('https://user:secret@host/path') == ''
    assert safe_link('javascript:steal()') == ''
    with pytest.raises(ValueError): normalize_mime(b'too big', max_bytes=1)


def test_processing_outside_window_cannot_manufacture_completion(acquisition):
    f = acquisition
    f.adapter.pages = [SourcePage((record(f),), 'history', True)]
    f.process()
    f.now += timedelta(hours=1)
    with pytest.raises(PermissionError, match='outside_window'):
        f.acquisition.checkpoint_processing(AccountSession('child'), f.run_id, stage='capd', evidence_ids=[])


def test_query_and_folder_revision_start_new_baseline(acquisition):
    f = acquisition
    f.adapter.pages = [SourcePage((record(f),), 'old-query-history', True), SourcePage(complete=True)]
    f.process()
    f.definition['query'] = {'text': 'different authorized query'}
    f.checks.save(AccountSession('child'), definition=f.definition, expected_revision=1,
        policy_id=f.policy['policy_id'], activate=True, authorization_reference='explicit query revision')
    f.now += timedelta(days=7)
    f.dispatcher.sync(now=f.now)
    task = [t for t in f.schedules.list_tasks() if t.payload['policy_revision'] == 2][0]
    f.schedules._save_task(replace(task, next_run_at=f.now.isoformat()))
    f.scheduler.run_once(now=f.now)
    assert f.process()['outcome'] == 'no_match'
    assert f.adapter.calls[-1][0] == 'search' and f.adapter.calls[-1][2]['cursor'] is None


def test_crash_after_final_page_recovers_without_another_read(acquisition, monkeypatch):
    f = acquisition
    f.adapter.pages = [SourcePage((record(f),), 'history', True)]
    original = f.acquisition._accept
    def crash(*a, **kw):
        original(*a, **kw)
        raise SystemExit()
    monkeypatch.setattr(f.acquisition, '_accept', crash)
    with pytest.raises(SystemExit): f.process()
    f.now += timedelta(seconds=31)
    assert f.process()['outcome'] == 'complete'
    assert len(f.adapter.calls) == 1


def test_incremental_cursor_then_expiry_and_adapter_version_baseline(acquisition):
    f = acquisition
    f.adapter.pages = [SourcePage((record(f),), 'opaque-history', True), CursorExpired(), SourcePage(complete=True)]
    f.process()
    f.now += timedelta(days=7)
    f.scheduler.run_once(now=f.now)
    assert f.process()['state'] == 'pending'
    assert f.adapter.calls[-1][0] == 'changes'
    assert f.adapter.calls[-1][2]['cursor'] == 'opaque-history'
    assert f.process()['outcome'] == 'partial'
    f.now += timedelta(days=7)
    f.scheduler.run_once(now=f.now)
    f.adapter.version = '2'
    f.adapter.pages = [SourcePage(complete=True)]
    assert f.process()['outcome'] == 'no_match'
    assert f.adapter.calls[-1][0] == 'search' and f.adapter.calls[-1][2]['cursor'] is None


@pytest.mark.parametrize('cause', ['expiry', 'revocation', 'deletion'])
def test_content_removal_leaves_only_provenance(acquisition, cause):
    f = acquisition
    f.adapter.pages = [SourcePage((record(f),), 'next', False)]
    f.process()
    if cause == 'expiry':
        f.now += timedelta(days=7)
        f.acquisition.purge_content()
    elif cause == 'revocation':
        f.service.revoke_grant(AccountSession('parent'), grant_id='school-grant', expected_revision=1)
        f.acquisition.purge_content()
    else:
        f.adapter.pages = [SourcePage((record(f, version='deleted-version', deleted=True),), 'history', True)]
        f.process()
    with f.schedules._connect() as conn:
        rows = conn.execute('SELECT definition FROM v2_source_evidence').fetchall()
    assert all(json.loads(row[0])['content'] == {} for row in rows)
    assert all(json.loads(row[0])['source_id'] == 'message-1' for row in rows)
    assert all(json.loads(row[0])['state'] != 'available' for row in rows)


def test_expiry_does_not_overwrite_a_concurrent_deletion_tombstone(acquisition):
    f = acquisition
    f.adapter.pages = [SourcePage((record(f),), 'next', False)]
    f.process()
    updated = []
    def clock_with_concurrent_deletion():
        if not updated:
            with f.schedules._connect() as conn:
                row = conn.execute('SELECT evidence_id,definition FROM v2_source_evidence').fetchone()
                value = json.loads(row['definition'])
                value.update(state='deleted', deleted=True, content={})
                conn.execute('UPDATE v2_source_evidence SET definition=? WHERE evidence_id=?',
                             (json.dumps(value), row['evidence_id']))
            updated.append(True)
        return f.now + timedelta(days=8)
    f.acquisition.clock = clock_with_concurrent_deletion
    f.acquisition.purge_content()
    with f.schedules._connect() as conn:
        value = json.loads(conn.execute('SELECT definition FROM v2_source_evidence').fetchone()[0])
    assert value['state'] == 'deleted' and value['deleted'] is True and value['content'] == {}


@pytest.mark.parametrize('state', ['expired', 'revoked', 'deleted', 'unknown'])
def test_cleanup_scrubs_bytes_from_nonavailable_evidence(acquisition, state):
    f = acquisition
    f.adapter.pages = [SourcePage((record(f),), 'next', False)]
    f.process()
    with f.schedules._connect() as conn:
        row = conn.execute('SELECT evidence_id,definition FROM v2_source_evidence').fetchone()
        value = json.loads(row['definition'])
        assert value['content']
        value['state'] = state
        conn.execute('UPDATE v2_source_evidence SET definition=? WHERE evidence_id=?',
                     (json.dumps(value), row['evidence_id']))
    f.acquisition.purge_content()
    with f.schedules._connect() as conn:
        value = json.loads(conn.execute('SELECT definition FROM v2_source_evidence').fetchone()[0])
    assert value['content'] == {}
    assert value['state'] == ('revoked' if state == 'unknown' else state)


def test_later_stale_version_cannot_resurrect_deleted_source(acquisition):
    f = acquisition
    f.adapter.pages = [
        SourcePage((record(f),), 'page-one', False),
        SourcePage((record(f, version='deleted-version', deleted=True),), 'page-two', False),
        SourcePage((record(f, version='stale-unseen-version'),), 'history', True),
    ]
    f.process()
    f.process()
    result = f.process()
    with f.schedules._connect() as conn:
        values = [json.loads(row[0]) for row in conn.execute('SELECT definition FROM v2_source_evidence')]
    assert len(values) == 3
    assert all(value['state'] == 'deleted' and value['content'] == {} for value in values)
    assert result['outcome'] == 'partial'
    assert 'source_deletion_reconciliation_required' in result['coverage']['gaps']


def test_recovery_marker_blocks_replayed_source_absent_from_old_snapshot(acquisition):
    f = acquisition
    with f.schedules._connect() as conn:
        conn.execute('''CREATE TABLE v2_recovery_source_tombstones (
            connection_id TEXT NOT NULL, resource TEXT NOT NULL, source_id TEXT NOT NULL,
            PRIMARY KEY(connection_id,resource,source_id)) STRICT''')
        conn.execute("INSERT INTO v2_recovery_source_tombstones VALUES ('one','folder-a','message-1')")
    f.adapter.pages = [SourcePage((record(f),), 'history', True)]
    result = f.process()
    with f.schedules._connect() as conn:
        value = json.loads(conn.execute('SELECT definition FROM v2_source_evidence').fetchone()[0])
    assert value['state'] == 'deleted' and value['content'] == {}
    assert result['outcome'] == 'partial'
    assert 'source_deletion_reconciliation_required' in result['coverage']['gaps']


def test_folder_change_uses_new_bounded_baseline(acquisition):
    from test_account_grants import expiry
    f = acquisition
    f.adapter.pages = [SourcePage((record(f),), 'old-folder-history', True), SourcePage(complete=True)]
    f.process()
    f.service.save_grant(AccountSession('parent'), grant_id='school-grant', connection_id='one', expected_revision=1,
        beneficiary_user_id='child', operations=['read'], resources=['folder-a', 'folder-b'],
        recipient_user_ids=['parent'], expires_at=expiry(), project_id=f.project.project_id)
    f.definition['source_bindings'][0]['resources'] = ['folder-b']
    f.checks.save(AccountSession('child'), definition=f.definition, expected_revision=1,
        policy_id=f.policy['policy_id'], activate=True, authorization_reference='explicit folder change')
    f.now += timedelta(days=7)
    f.dispatcher.sync(now=f.now)
    task = [t for t in f.schedules.list_tasks() if t.payload['policy_revision'] == 2][0]
    f.schedules._save_task(replace(task, next_run_at=f.now.isoformat()))
    f.scheduler.run_once(now=f.now)
    assert f.process()['outcome'] == 'no_match'
    assert f.adapter.calls[-1][0] == 'search' and f.adapter.calls[-1][2]['cursor'] is None
    assert f.adapter.calls[-1][1].grant.resources == ('folder-b',)


def test_adapter_version_change_between_pages_never_reuses_old_cursor(acquisition):
    f = acquisition
    f.adapter.pages = [SourcePage((record(f),), 'old-version-page'), SourcePage(complete=True)]
    f.process()
    f.adapter.version = 'new-contract'
    assert f.process()['outcome'] == 'complete'
    assert f.adapter.calls[-1][0] == 'search'
    assert f.adapter.calls[-1][2]['cursor'] is None


def test_failed_acquisition_publishes_failure_without_private_error(acquisition):
    f = acquisition
    f.adapter.pages = [RuntimeError('secret from provider')]
    result = f.worker(f.acquisition, on_results=f.acquisition.publish_ready, max_attempts=1).run_once()
    assert result['outcome'] == 'failed'
    with f.events._connect() as conn:
        payload = json.loads(conn.execute('SELECT payload_json FROM v2_event_history').fetchone()[0])
    assert payload['outcome'] == 'failed' and 'secret' not in json.dumps(payload)


def test_subscription_before_first_result_routes_existing_event_execution(acquisition):
    f = acquisition
    automation = f.events.create_event_automation(owner_user_id='child', project_id=f.project.project_id,
        name='Interpret acquired evidence', prompt='Interpret the authorized check result',
        event_type=EVENT, event_version='1', source_scope=f.acquisition.resolve(WORKER, f.run_id))
    f.adapter.pages = [SourcePage((record(f),), 'history', True)]
    f.now += timedelta(seconds=1)
    assert f.process()['outcome'] == 'complete'
    f.acquisition.publish_ready()
    with f.events._connect() as conn:
        rows = conn.execute('SELECT automation_id FROM v2_automation_executions').fetchall()
    assert [r[0] for r in rows] == [automation.automation_id]
    assert f.events.visible_automations(actor_user_id='child')[0].automation_id == automation.automation_id


def test_cursor_and_page_retention_preserves_active_and_recent_recovery(acquisition):
    f = acquisition
    f.adapter.pages = [SourcePage((record(f),), 'opaque-history', True)]
    assert f.process()['outcome'] == 'complete'
    with f.schedules._connect() as db:
        scope = db.execute('SELECT scope_key FROM v2_source_streams').fetchone()[0]
        assert db.execute('SELECT COUNT(*) FROM v2_source_pages').fetchone()[0] == 1
    f.now += timedelta(days=31)
    # A still-active run may use this cursor before persisting its checkpoint.
    with f.schedules._connect() as db:
        db.execute("UPDATE v2_project_check_runs SET state='pending' WHERE check_run_id=?", (f.run_id,))
    f.acquisition.purge_source_metadata()
    with f.schedules._connect() as db:
        assert db.execute('SELECT COUNT(*) FROM v2_source_streams').fetchone()[0] == 1
        assert db.execute('SELECT COUNT(*) FROM v2_source_pages').fetchone()[0] == 1
        db.execute("UPDATE v2_project_check_runs SET state='terminal' WHERE check_run_id=?", (f.run_id,))
    f.acquisition.purge_source_metadata()
    with f.schedules._connect() as db:
        assert db.execute('SELECT COUNT(*) FROM v2_source_streams').fetchone()[0] == 0
        assert db.execute('SELECT scope_key FROM v2_source_expired_streams').fetchone()[0] == scope
        assert db.execute('SELECT COUNT(*) FROM v2_source_pages').fetchone()[0] == 0
        assert db.execute('SELECT COUNT(*) FROM v2_source_evidence').fetchone()[0] == 1
        # A cursor without age evidence is retained rather than incorrectly expired.
        db.execute('INSERT INTO v2_source_streams VALUES(?, ?)', ('unknown-age', '{}'))
    f.acquisition.purge_source_metadata()
    with f.schedules._connect() as db:
        assert db.execute('SELECT scope_key FROM v2_source_streams').fetchone()[0] == 'unknown-age'


def test_cursor_retention_uses_most_recent_page(acquisition):
    f = acquisition
    f.adapter.pages = [SourcePage((record(f),), 'opaque-history', True)]
    f.process()
    with f.schedules._connect() as db:
        scope = db.execute('SELECT scope_key FROM v2_source_streams').fetchone()[0]
        db.execute('INSERT INTO v2_source_pages VALUES (?, ?, ?, ?)',
                   (f.run_id, scope, 'newer-page', json.dumps({'observed_at': (f.now + timedelta(days=20)).isoformat()})))
    f.now += timedelta(days=31)
    f.acquisition.purge_source_metadata()
    with f.schedules._connect() as db:
        assert db.execute('SELECT COUNT(*) FROM v2_source_streams').fetchone()[0] == 1
        assert db.execute('SELECT COUNT(*) FROM v2_source_pages').fetchone()[0] == 1


def test_expired_cursor_forces_bounded_resync_and_reports_gap(acquisition):
    f = acquisition
    f.adapter.pages = [SourcePage((record(f),), 'old-history', True), SourcePage((), 'new-history', True)]
    assert f.process()['outcome'] == 'complete'
    with f.schedules._connect() as db:
        row = db.execute('SELECT scope_key, request_key, definition FROM v2_source_pages').fetchone()
        page = json.loads(row['definition'])
        page['observed_at'] = (f.now - timedelta(days=31)).isoformat()
        db.execute('UPDATE v2_source_pages SET definition=? WHERE scope_key=? AND request_key=?',
                   (json.dumps(page), row['scope_key'], row['request_key']))
    f.acquisition.purge_source_metadata()
    f.now += timedelta(days=7)
    f.dispatcher.sync(now=f.now)
    task = f.schedules.list_tasks()[0]
    f.schedules._save_task(replace(task, next_run_at=f.now.isoformat()))
    f.scheduler.run_once(now=f.now)
    result = f.process()
    assert f.adapter.calls[-1][0] == 'search'
    assert f.adapter.calls[-1][2]['cursor'] is None
    assert result['outcome'] == 'partial'
    assert 'cursor_expired_bounded_resync' in result['coverage']['gaps']
    with f.schedules._connect() as db:
        assert db.execute('SELECT COUNT(*) FROM v2_source_expired_streams').fetchone()[0] == 0


def test_old_processing_expires_only_with_shared_idle_recovery_store(acquisition):
    f = acquisition
    f.adapter.pages = [SourcePage(complete=True)]
    assert f.process()['outcome'] == 'no_match'
    with f.schedules._connect() as db:
        db.execute('INSERT INTO v2_source_processing VALUES (?, ?, ?)', (f.run_id, 'capd', '["synthetic-ref"]'))
    f.now += timedelta(days=31)
    # Separate event storage cannot establish that downstream work is idle.
    f.acquisition.purge_source_metadata()
    with f.schedules._connect() as db:
        assert db.execute('SELECT COUNT(*) FROM v2_source_processing').fetchone()[0] == 1
    f.acquisition.events = EventAutomationStore(f.schedules.db_path)
    with f.schedules._connect() as db:
        db.execute('CREATE TABLE v2_project_triage (check_run_id TEXT NOT NULL, evidence_id TEXT NOT NULL, definition TEXT NOT NULL, PRIMARY KEY(check_run_id,evidence_id))')
        db.execute('INSERT INTO v2_project_triage VALUES (?, ?, ?)', (f.run_id, 'synthetic-evidence', '{}'))
        db.execute("INSERT INTO v2_automation_executions(execution_id,automation_id,event_key,status,queued_message_id,created_at,updated_at) VALUES('active','a','e','processing','','2026-01-01','2026-01-01')")
    f.acquisition.purge_source_metadata()
    with f.schedules._connect() as db:
        assert db.execute('SELECT COUNT(*) FROM v2_source_processing').fetchone()[0] == 1
        db.execute("UPDATE v2_automation_executions SET status='completed' WHERE execution_id='active'")
    f.acquisition.purge_source_metadata()
    with f.schedules._connect() as db:
        assert db.execute('SELECT COUNT(*) FROM v2_source_processing').fetchone()[0] == 0
        assert db.execute('SELECT COUNT(*) FROM v2_project_triage').fetchone()[0] == 0
        receipt = json.loads(db.execute('SELECT receipt FROM v2_project_check_runs WHERE check_run_id=?', (f.run_id,)).fetchone()[0])
        assert receipt['processing_details_expired'] is True
        assert receipt['outcome'] == 'no_match'


def test_old_processing_with_unverified_publication_is_preserved(acquisition):
    f = acquisition
    f.adapter.pages = [SourcePage((record(f),), 'history', True)]
    assert f.process()['outcome'] == 'complete'
    with f.schedules._connect() as db:
        db.execute('INSERT INTO v2_source_processing VALUES (?, ?, ?)', (f.run_id, 'capd', '["synthetic-ref"]'))
    f.acquisition.events = EventAutomationStore(f.schedules.db_path)
    f.now += timedelta(days=31)
    f.acquisition.purge_source_metadata()
    with f.schedules._connect() as db:
        assert db.execute('SELECT COUNT(*) FROM v2_source_processing').fetchone()[0] == 1
        receipt = json.loads(db.execute('SELECT receipt FROM v2_project_check_runs WHERE check_run_id=?', (f.run_id,)).fetchone()[0])
        assert 'processing_details_expired' not in receipt


def test_published_processing_expires_without_manufacturing_run_success(acquisition):
    f = acquisition
    f.events = EventAutomationStore(f.schedules.db_path)
    f.acquisition = SourceAcquisition(f.dispatcher, f.events, adapters={'synthetic': f.adapter}, clock=lambda: f.now)
    f.adapter.pages = [SourcePage((record(f),), 'history', True)]
    assert f.process()['outcome'] == 'complete'
    evidence_id = f.acquisition.evidence(AccountSession('child'), f.run_id)[0]['evidence_id']
    f.acquisition.checkpoint_processing(AccountSession('child'), f.run_id, stage='capd', evidence_ids=[evidence_id])
    with f.schedules._connect() as db:
        assert db.execute('SELECT COUNT(*) FROM v2_event_receipts').fetchone()[0] == 1
        assert db.execute('SELECT COUNT(*) FROM v2_source_publications').fetchone()[0] == 1
    f.now += timedelta(days=31)
    f.acquisition.purge_source_metadata()
    with f.schedules._connect() as db:
        assert db.execute('SELECT COUNT(*) FROM v2_source_processing').fetchone()[0] == 0
        assert db.execute('SELECT COUNT(*) FROM v2_event_receipts').fetchone()[0] == 1
        receipt = json.loads(db.execute('SELECT receipt FROM v2_project_check_runs WHERE check_run_id=?', (f.run_id,)).fetchone()[0])
        assert receipt['processing_details_expired'] is True and receipt['outcome'] == 'complete'


def test_expired_run_diagnostics_drop_checkpoints_without_losing_outcome(acquisition):
    f = acquisition
    f.adapter.pages = [SourcePage(complete=True)]
    assert f.process()['outcome'] == 'no_match'
    f.acquisition.events = EventAutomationStore(f.schedules.db_path)
    with f.schedules._connect() as db:
        row = db.execute('SELECT receipt FROM v2_project_check_runs WHERE check_run_id=?', (f.run_id,)).fetchone()
        receipt = json.loads(row['receipt'])
        receipt['acquisition_checkpoint'] = {'cursor': 'synthetic-sensitive-cursor'}
        receipt['analysis_checkpoint'] = {'input': 'synthetic-sensitive-manifest'}
        receipt['coverage']['sources'] = {'synthetic-source': {'resources': ['synthetic-folder']}}
        db.execute('UPDATE v2_project_check_runs SET receipt=? WHERE check_run_id=?', (json.dumps(receipt), f.run_id))
    f.now += timedelta(days=31)
    f.acquisition.purge_source_metadata()
    run = f.dispatcher.get(f.run_id)
    assert run['outcome'] == 'no_match' and run['diagnostics_expired'] is True
    assert run['coverage']['details_expired'] is True
    assert run['acquisition_checkpoint'] == run['analysis_checkpoint'] == {}
    assert 'synthetic-sensitive' not in json.dumps(run) and 'synthetic-folder' not in json.dumps(run)
    assert run['policy_id'] == f.policy['policy_id'] and run['executing_user_id'] == 'child'
