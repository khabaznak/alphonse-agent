"""Run and finding projections over disposable existing acquisition fixtures."""
import json
from types import SimpleNamespace

from test_account_grants import f, expiry
from test_project_check_scheduling import scheduled
from test_source_acquisition import acquisition
from test_findings import checks, recipient_ids, source_count, findings
from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.integration_management import IntegrationManagement


def management(f):
    runtime = SimpleNamespace(project_checks=f.checks, user_store=f.users, skill_store=f.skills)
    return IntegrationManagement(runtime, dispatcher=f.dispatcher, findings=f.findings)


def test_operational_projection_retains_pending_partial_and_hides_checkpoints(findings):
    f = findings
    service = management(f)
    with f.schedules._connect() as conn:
        row = conn.execute('SELECT receipt FROM v2_project_check_runs WHERE check_run_id=?', (f.run_id,)).fetchone()
        receipt = json.loads(row[0])
        receipt.update(outcome='partial', error_code='missed_window', acquisition_checkpoint={'opaque_cursor': 'PRIVATE_CURSOR'},
                       analysis_checkpoint={'private': 'PRIVATE_MODEL_PAYLOAD'})
        conn.execute('UPDATE v2_project_check_runs SET receipt=? WHERE check_run_id=?', (json.dumps(receipt), f.run_id))
    result = service.project(AccountSession('child'), f.project.project_id)
    run = result['checks'][0]['runs'][0]
    assert run['outcome'] == 'partial' and run['error_code'] == 'missed_window'
    assert run['pending_analysis'] is True
    assert run['acquisition_state'] == 'checkpoint_saved'
    serialized = json.dumps(result)
    assert 'PRIVATE_CURSOR' not in serialized and 'PRIVATE_MODEL_PAYLOAD' not in serialized
    assert 'body' not in serialized and 'disposable-' not in serialized
    f.save()
    assert service.project(AccountSession('child'), f.project.project_id)['checks'][0]['runs'][0]['pending_analysis'] is False


def test_recipient_inbox_and_actual_delivery_outcomes_without_project_access(findings):
    f = findings
    service = management(f)
    ref = f.save()
    assert service.finding_views(AccountSession('child')) == []  # operator != recipient
    assert service.finding_views(AccountSession('other')) == []
    result = service.finding_views(AccountSession('parent'))
    assert result[0]['summary'] == f.value['summary']
    assert result[0]['deliveries'] == {}
    f.findings.notify(AccountSession('child'), ref['finding_id'])
    statuses = service.finding_views(AccountSession('parent'))[0]['deliveries']
    assert statuses['parent']['state'] == 'pending'  # queued does not mean delivered
    assert 'source_refs' not in result[0]
    f.service.revoke_grant(AccountSession('parent'), grant_id='school-grant', expected_revision=1)
    assert service.finding_views(AccountSession('parent')) == []


def test_stale_offline_run_and_reconnect_policy_are_visible(findings):
    from datetime import datetime, timedelta, timezone
    f = findings
    service = management(f)
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    with f.schedules._connect() as conn:
        row = conn.execute('SELECT receipt FROM v2_project_check_runs WHERE check_run_id=?', (f.run_id,)).fetchone()
        receipt = json.loads(row[0])
        receipt.update(window_end=past, outcome='', error_code='')
        conn.execute("UPDATE v2_project_check_runs SET state='leased',lease_until=?,receipt=? WHERE check_run_id=?", (past, json.dumps(receipt), f.run_id))
    run = service.project(AccountSession('child'), f.project.project_id)['checks'][0]['runs'][0]
    assert run['stale'] is True and run['offline'] is True
    old = f.store.get('one').connection
    f.service.save_connection(AccountSession('parent'), connection_id='one', expected_revision=old['revision'],
        provider_subject=old['provider_subject'], capabilities=old['capabilities'], granted_scopes=old['granted_scopes'],
        status='reauth_required', health_code='consent_required')
    result = service.project(AccountSession('child'), f.project.project_id)
    assert result['checks'][0]['policy']['status'] == 'needs_review'
    assert result['checks'][0]['next_run_at'] is None
    assert service.catalog(AccountSession('parent'))['connections'][0]['status'] == 'reauth_required'


def test_irrelevant_items_do_not_remain_pending_analysis(findings):
    f = findings
    with f.schedules._connect() as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS v2_project_triage(check_run_id TEXT,evidence_id TEXT,definition TEXT)')
        conn.execute('INSERT INTO v2_project_triage VALUES(?,?,?)', (f.run_id, f.eid, json.dumps({'status': 'irrelevant'})))
    result = management(f).project(AccountSession('child'), f.project.project_id)
    assert result['checks'][0]['runs'][0]['pending_analysis'] is False


def test_expired_processing_details_are_not_reported_as_pending_or_complete(findings):
    f = findings
    with f.schedules._connect() as conn:
        row = conn.execute('SELECT receipt FROM v2_project_check_runs WHERE check_run_id=?', (f.run_id,)).fetchone()
        receipt = json.loads(row[0])
        receipt['processing_details_expired'] = True
        conn.execute('UPDATE v2_project_check_runs SET receipt=? WHERE check_run_id=?', (json.dumps(receipt), f.run_id))
    result = management(f).project(AccountSession('child'), f.project.project_id)
    run = result['checks'][0]['runs'][0]
    assert run['processing_details_expired'] is True
    assert run['pending_analysis'] is None
