"""Source-backed commitments and delivery using disposable household fixtures."""
from dataclasses import replace
from datetime import timedelta
import json
import sqlite3
from types import SimpleNamespace

import pytest

from test_account_grants import f, expiry
from test_project_checks import checks as base_checks
from test_project_check_scheduling import scheduled
from test_source_acquisition import acquisition, record
from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.core.io.channels import ChannelAddress
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.integrations.acquisition import SourcePage
from alphonse.agent_v2.services.findings import FindingService

ACTOR = AccountSession('child')
RECIPIENT = AccountSession('parent')


@pytest.fixture
def recipient_ids(request):
    return getattr(request, 'param', ['parent'])


@pytest.fixture
def source_count(request):
    return getattr(request, 'param', 1)


@pytest.fixture
def checks(f, recipient_ids, source_count):
    fixture = base_checks.__wrapped__(f)
    if recipient_ids != ['parent']:
        fixture.service.save_grant(AccountSession('parent'), grant_id='school-grant', connection_id='one',
            expected_revision=1, beneficiary_user_id='child', operations=['read'], resources=['folder-a'],
            recipient_user_ids=recipient_ids, expires_at=expiry(), project_id=fixture.project.project_id)
        fixture.definition['recipient_user_ids'] = recipient_ids
    if source_count == 2:
        fixture.service.save_grant(AccountSession('child'), grant_id='school-two', connection_id='two',
            expected_revision=0, beneficiary_user_id='child', operations=['read'], resources=['folder-a'],
            recipient_user_ids=recipient_ids, expires_at=expiry(), project_id=fixture.project.project_id)
        fixture.bindings.attach_grant(AccountSession('child'), binding_id='binding-two', grant_id='school-two',
            expected_revision=1, required_scopes=('scope.read',), capabilities=('read',))
        fixture.bindings.select(actor_id='child', skill_id='skill:fixture', name='calendar', binding_id='binding-two')
        fixture.definition['source_bindings'].append(dict(skill_id='skill:fixture', role='calendar',
            binding_id='binding-two', connection_id='two', grant_id='school-two', operation='read', resources=['folder-a']))
    return fixture


@pytest.fixture
def findings(acquisition, source_count):
    f = acquisition
    f.outbox = SQLiteOutboundStore(f.tmp / 'outbox.db')
    f.resolver = SimpleNamespace(resolve_outbound_address=lambda **kw: SimpleNamespace(
        resolved=True, address=ChannelAddress(integration_id='desktop', provider_key='desktop',
            channel_target=kw['alphonse_user_id'], alphonse_user_id=kw['alphonse_user_id'])))
    f.findings = FindingService(f.acquisition, f.outbox, f.resolver)
    f.acquisition.findings = f.findings
    f.adapter.pages = [SourcePage((record(f),), complete=True)]
    if source_count == 2:
        f.adapter.pages.append(SourcePage((record(f, source_id='calendar-event'),), complete=True))
        f.process()
    f.process()
    f.eid = f.acquisition.evidence(ACTOR, f.run_id)[0]['evidence_id']
    f.value = dict(kind='assignment', subject_id='child-a', commitment_id='course-a:assignment-7',
        summary='Prepare the school assignment', facts=[dict(predicate='grade', value=30,
        unit='points', basis='source_observed')], due_at='2026-11-10', timezone='America/Mexico_City',
        uncertainty=['Trimester percentage is unknown.'])
    f.save = lambda **kw: f.findings.record(ACTOR, f.run_id, evidence_ids=kw.pop('evidence_ids', [f.eid]),
        audience_user_ids=kw.pop('audience_user_ids', ['parent']), interpretation=dict(f.value, **kw))
    return f


def evidence(f, *, source_id='message-2', timestamp=None, **kw):
    """Minimal accepted evidence fixture; no provider or private account access."""
    with f.schedules._connect() as conn:
        row = conn.execute('SELECT * FROM v2_source_evidence WHERE evidence_id=?', (f.eid,)).fetchone()
        value = json.loads(row['definition'])
        value.update(evidence_id='evidence_' + source_id, source_id=source_id,
                     source_timestamp=(timestamp or f.now + timedelta(minutes=1)).isoformat(), **kw)
        conn.execute('INSERT INTO v2_source_evidence VALUES(?,?,?,?)',
            (value['evidence_id'], f.run_id, row['scope_key'], json.dumps(value)))
    return value['evidence_id']


def test_persistence_scope_provenance_and_reference_only_checkpoint(findings):
    f = findings
    ref = f.save()
    assert set(ref) == {'finding_id', 'version'}
    restored = FindingService(f.acquisition, f.outbox, f.resolver)
    finding = restored.view(RECIPIENT, ref['finding_id'])
    assert finding['due_at'] == '2026-11-10' and finding['timezone'] == 'America/Mexico_City'
    assert finding['source_refs'][0]['version'] == 'version-1'
    assert finding['state'] == 'open' and finding['facts'][0]['unit'] == 'points'
    assert f.acquisition.processing(ACTOR, f.run_id)['capd'] == [f.eid]
    with pytest.raises(PermissionError):
        restored.view(AccountSession('other'), ref['finding_id'])
    with pytest.raises(PermissionError):
        f.save(audience_user_ids=['other'])
    with pytest.raises(PermissionError):
        restored.notify(RECIPIENT, ref['finding_id'])


@pytest.mark.parametrize('predicate', ['unpaid_balance', 'assignment_completed', 'payment_verified', 'submission_completed'])
def test_reported_sources_cannot_manufacture_verified_outcomes(findings, predicate):
    with pytest.raises(ValueError):
        findings.save(facts=[dict(predicate=predicate, value=True, basis='source_observed', unit=None)])


@pytest.mark.parametrize('predicate', ['overdue_notice', 'payment_report', 'submission_report'])
def test_sender_statements_remain_reports(findings, predicate):
    f = findings
    with pytest.raises(ValueError, match='not_verified'):
        f.save(facts=[dict(predicate=predicate, value=True, basis='source_observed', unit=None)])
    ref = f.save(facts=[dict(predicate=predicate, value=True, basis='sender_reported', unit=None)])
    assert f.findings.view(RECIPIENT, ref['finding_id'])['state'] == 'open'


def test_missing_due_date_timezone_and_grade_units(findings):
    f = findings
    with pytest.raises(ValueError, match='unit_required'):
        f.save(facts=[dict(predicate='grade', value=30, basis='source_observed', unit=None)])
    ref = f.save(due_at=None, timezone=None)
    finding = f.findings.view(RECIPIENT, ref['finding_id'])
    assert finding['due_at'] is finding['timezone'] is None
    f.findings.notify(ACTOR, ref['finding_id'])
    assert '30 (points;' in f.outbox.list()[0].message
    assert 'percentage is unknown' in f.outbox.list()[0].message


def test_correction_duplicate_and_old_quote_create_one_alert_per_version(findings):
    f = findings
    original = f.save()
    first = f.findings.notify(ACTOR, original['finding_id'])['parent']['outbox_message_id']
    correction = evidence(f)
    corrected = f.save(evidence_ids=[correction], due_at='2026-11-12')
    assert corrected['version'] == 2 and corrected['finding_id'] == original['finding_id']
    finding = f.findings.view(RECIPIENT, corrected['finding_id'])
    assert finding['correction_of'] == original
    assert f.findings.view(RECIPIENT, original['finding_id'], version=1)['due_at'] == '2026-11-10'
    duplicate = evidence(f, source_id='calendar-copy', timestamp=f.now + timedelta(minutes=2))
    assert f.save(evidence_ids=[duplicate], due_at='2026-11-12') == corrected
    old = evidence(f, source_id='old-notice', timestamp=f.now - timedelta(days=2))
    assert f.save(evidence_ids=[old], due_at='2026-11-09') == corrected
    # A later email quoting an old notice remains a separate unresolved item.
    quoted = evidence(f, source_id='quoted-notice', timestamp=f.now + timedelta(minutes=3))
    uncertain = f.save(evidence_ids=[quoted], quoted=True, due_at='2026-11-08')
    assert uncertain['finding_id'] != corrected['finding_id']
    assert f.findings.view(RECIPIENT, uncertain['finding_id'])['state'] == 'unresolved'
    assert f.findings.notify(ACTOR, uncertain['finding_id'])['parent']['state'] == 'pending_identity_review'
    for _ in range(3):
        result = f.findings.notify(ACTOR, corrected['finding_id'])
    assert len(f.outbox.list()) == 2
    # Superseded pending version cannot be delivered; current version survives.
    current = f.outbox.claim_next()
    assert current.outbox_message_id == result['parent']['outbox_message_id']
    assert current.outbox_message_id != first and '2026-11-12' in current.message
    assert 'calendar-copy' not in current.message  # accepted correction provenance


@pytest.mark.parametrize('change', [dict(subject_id='child-b'), dict(commitment_id='assignment-8'),
    dict(kind='invoice'), dict(subject_id=''), dict(identity_uncertain=True)])
def test_unrelated_and_ambiguous_identity_never_merge(findings, change):
    f = findings
    first = f.save()
    second = f.save(**change)
    assert first['finding_id'] != second['finding_id']
    if not change.get('subject_id', 'present') or change.get('identity_uncertain'):
        assert f.findings.view(RECIPIENT, second['finding_id'])['state'] == 'unresolved'


def test_replay_after_checkpoint_crash_recovers_without_new_version(findings, monkeypatch):
    f = findings
    checkpoint = f.acquisition.checkpoint_processing
    monkeypatch.setattr(f.acquisition, 'checkpoint_processing', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('crash')))
    with pytest.raises(RuntimeError):
        f.save()
    monkeypatch.setattr(f.acquisition, 'checkpoint_processing', checkpoint)
    ref = f.save()
    assert ref['version'] == 1
    assert f.acquisition.processing(ACTOR, f.run_id)['capd'] == [f.eid]
    f.findings.notify(ACTOR, ref['finding_id'])
    restored = FindingService(f.acquisition, SQLiteOutboundStore(f.outbox.db_path), f.resolver)
    assert restored.notify(ACTOR, ref['finding_id'])['parent']['state'] == 'pending'
    assert len(f.outbox.list()) == 1


@pytest.mark.parametrize('outcome', ['exception', 'expired_lease'])
def test_ambiguous_send_is_durable_and_never_automatically_retried(findings, outcome):
    f = findings
    ref = f.save()
    f.findings.notify(ACTOR, ref['finding_id'])
    message = f.outbox.claim_next()
    if outcome == 'exception':
        f.outbox.mark_failed(message.outbox_message_id, error='provider timeout containing private text')
    else:
        with f.outbox._connect() as conn:
            conn.execute("UPDATE v2_outbox SET lease_expires_at='2000-01-01T00:00:00+00:00'")
        f.outbox.reclaim_expired()
    restored = FindingService(f.acquisition, SQLiteOutboundStore(f.outbox.db_path), f.resolver)
    assert restored.notify(ACTOR, ref['finding_id'])['parent']['state'] == 'ambiguous'
    assert restored.outbox.claim_next() is None
    assert 'private text' not in restored.outbox.get(message.outbox_message_id).last_error


@pytest.mark.parametrize('change', ['revoke', 'delete', 'source_revoke', 'unknown_state', 'finding_expire'])
def test_revocation_deletion_propagates_all_versions_and_local_delivery(findings, change):
    f = findings
    ref = f.save()
    delivery = f.findings.notify(ACTOR, ref['finding_id'])['parent']['outbox_message_id']
    if change == 'revoke':
        f.service.revoke_grant(AccountSession('parent'), grant_id='school-grant', expected_revision=1)
    elif change == 'finding_expire':
        with f.schedules._connect() as conn:
            value = f.findings._load(ref['finding_id'])
            value['retention_deadline'] = (f.now - timedelta(seconds=1)).isoformat()
            conn.execute('UPDATE v2_findings SET definition=? WHERE finding_id=?',
                         (json.dumps(value), ref['finding_id']))
    else:
        with f.schedules._connect() as conn:
            row = conn.execute('SELECT definition FROM v2_source_evidence WHERE evidence_id=?', (f.eid,)).fetchone()
            value = json.loads(row[0])
            value.update(state={'delete': 'deleted', 'source_revoke': 'revoked',
                                'unknown_state': 'future_state'}[change], content={})
            conn.execute('UPDATE v2_source_evidence SET definition=? WHERE evidence_id=?', (json.dumps(value), f.eid))
    with pytest.raises(PermissionError):
        f.findings.view(RECIPIENT, ref['finding_id'])
    # References in UI/memory/checkpoints must resolve through the same gate.
    with pytest.raises(PermissionError):
        f.findings.view(RECIPIENT, ref['finding_id'], version=ref['version'])
    with pytest.raises(PermissionError):
        f.findings.delivery_status(RECIPIENT, ref['finding_id'])
    with pytest.raises(PermissionError):
        f.findings.notify(ACTOR, ref['finding_id'])
    if change == 'revoke':
        with pytest.raises(PermissionError):
            f.acquisition.processing(ACTOR, f.run_id)
    assert f.outbox.claim_next() is None  # delivery gate works before cleanup tick
    f.acquisition.purge_content()
    with f.schedules._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM v2_finding_versions').fetchone()[0] == 0
        persisted = conn.execute('SELECT definition FROM v2_findings').fetchone()[0]
    assert 'Prepare the school assignment' not in persisted and '2026-11-10' not in persisted
    assert f.outbox.get(delivery).message == '[Finding unavailable]'
    assert f.outbox.get(delivery).metadata['finding_delivery_state'] == 'revoked'


def test_historical_version_revalidates_its_own_retention(findings):
    f = findings
    ref = f.save()
    # Simulate an older version with a shorter retention deadline than current.
    with f.schedules._connect() as conn:
        historical = json.loads(conn.execute(
            'SELECT definition FROM v2_finding_versions WHERE finding_id=? AND version=1',
            (ref['finding_id'],)).fetchone()[0])
        historical['retention_deadline'] = (f.now - timedelta(seconds=1)).isoformat()
        conn.execute('UPDATE v2_finding_versions SET definition=? WHERE finding_id=? AND version=1',
                     (json.dumps(historical), ref['finding_id']))
    assert f.findings.view(RECIPIENT, ref['finding_id'])['state'] == 'open'
    with pytest.raises(PermissionError, match='finding_retention_expired'):
        f.findings.view(RECIPIENT, ref['finding_id'], version=1)


def test_restored_evidence_cannot_override_current_revocation(findings):
    f = findings
    ref = f.save()
    delivery = f.findings.notify(ACTOR, ref['finding_id'])['parent']['outbox_message_id']
    # Checkpoints/UI/exportable references carry no copied interpretation.
    locator = json.loads(json.dumps(ref))
    assert set(locator) == {'finding_id', 'version'}
    snapshot = f.tmp / 'synthetic-offline-backup.sqlite'
    with f.schedules._connect() as live, sqlite3.connect(snapshot) as backup:
        live.backup(backup)
    f.service.revoke_grant(AccountSession('parent'), grant_id='school-grant', expected_revision=1)
    f.acquisition.purge_content()
    # Deliberately restore stale evidence only. Current account authority is
    # independent and must fence reads even before the cleanup tick runs again.
    with sqlite3.connect(snapshot) as backup, f.schedules._connect() as live:
        backup.backup(live)
    restored = FindingService(f.acquisition, f.outbox, f.resolver)
    f.acquisition.findings = restored
    for view in (lambda: restored.view(RECIPIENT, locator['finding_id']),
                 lambda: restored.view(RECIPIENT, **locator),
                 lambda: restored.delivery_status(RECIPIENT, locator['finding_id']),
                 lambda: f.acquisition.evidence(ACTOR, f.run_id),
                 lambda: f.acquisition.processing(ACTOR, f.run_id)):
        with pytest.raises(PermissionError):
            view()
    assert f.outbox.claim_next() is None
    f.acquisition.purge_content()
    assert f.outbox.get(delivery).message == '[Finding unavailable]'
    with f.schedules._connect() as conn:
        definitions = [row[0] for row in conn.execute('SELECT definition FROM v2_source_evidence')]
        assert all(json.loads(value)['content'] == {} for value in definitions)
        assert conn.execute('SELECT COUNT(*) FROM v2_finding_versions').fetchone()[0] == 0


def test_unwired_outbox_consumer_fails_closed(findings):
    f = findings
    ref = f.save()
    f.findings.notify(ACTOR, ref['finding_id'])
    assert SQLiteOutboundStore(f.outbox.db_path).claim_next() is None
    assert f.outbox.list()[0].status == 'pending'
    assert f.outbox.claim_next() is not None


def test_partial_coverage_remains_explicit(findings):
    f = findings
    with f.schedules._connect() as conn:
        row = conn.execute('SELECT receipt FROM v2_project_check_runs WHERE check_run_id=?', (f.run_id,)).fetchone()
        receipt = json.loads(row[0]); receipt['coverage']['gaps'] = ['missing-folder']
        conn.execute('UPDATE v2_project_check_runs SET receipt=? WHERE check_run_id=?', (json.dumps(receipt), f.run_id))
    ref = f.save()
    f.findings.notify(ACTOR, ref['finding_id'])
    assert 'Coverage is partial' in f.outbox.list()[0].message


@pytest.mark.parametrize('recipient_ids', [['parent', 'other']], indirect=True)
def test_recipient_delivery_and_ambiguity_are_independent(findings):
    f = findings
    ref = f.save(audience_user_ids=['parent', 'other'])
    outcomes = f.findings.notify(ACTOR, ref['finding_id'])
    assert set(outcomes) == {'parent', 'other'}
    first = f.outbox.claim_next()
    f.outbox.mark_delivered(first.outbox_message_id, provider_message_id='receipt-1')
    second = f.outbox.claim_next()
    f.outbox.mark_failed(second.outbox_message_id, error='timeout')
    assert first.audience_user_id != second.audience_user_id
    assert set(f.findings.delivery_status(ACTOR, ref['finding_id'])) == {'parent', 'other'}
    assert set(f.findings.delivery_status(RECIPIENT, ref['finding_id'])) == {'parent'}
    outcomes = f.findings.notify(ACTOR, ref['finding_id'])
    assert outcomes[first.audience_user_id]['state'] == 'delivered'
    assert outcomes[second.audience_user_id]['state'] == 'ambiguous'
    assert f.outbox.claim_next() is None
    assert len(f.outbox.list()) == 2


@pytest.mark.parametrize('change', ['cache_expire', 'pause'])
def test_retained_findings_preserve_history_after_cache_expiry_or_pause(findings, change):
    f = findings
    ref = f.save()
    if change == 'pause':
        f.checks.set_status(ACTOR, f.policy['policy_id'], expected_revision=1, status='paused')
    else:
        with f.schedules._connect() as conn:
            value = json.loads(conn.execute('SELECT definition FROM v2_source_evidence WHERE evidence_id=?', (f.eid,)).fetchone()[0])
            value['retention_deadline'] = (f.now - timedelta(seconds=1)).isoformat()
            conn.execute('UPDATE v2_source_evidence SET definition=? WHERE evidence_id=?', (json.dumps(value), f.eid))
    f.acquisition.purge_content()
    assert f.findings.view(RECIPIENT, ref['finding_id'])['summary'] == 'Prepare the school assignment'
    if change == 'pause':
        with pytest.raises(PermissionError):
            f.findings.notify(ACTOR, ref['finding_id'])


@pytest.mark.parametrize('source_count', [2], indirect=True)
def test_independent_accounts_reconcile_one_finding_with_separate_provenance(findings):
    f = findings
    sources = f.acquisition.evidence(ACTOR, f.run_id)
    assert {source['connection_id'] for source in sources} == {'one', 'two'}
    first = f.save(evidence_ids=[sources[0]['evidence_id']])
    assert f.save(evidence_ids=[sources[1]['evidence_id']]) == first
    finding = f.findings.view(RECIPIENT, first['finding_id'])
    assert len(finding['source_refs']) == 2
    assert finding['version'] == 1
    f.findings.notify(ACTOR, first['finding_id'])
    assert len(f.outbox.list()) == 1


def test_forged_evidence_and_cross_actor_cannot_record(findings):
    f = findings
    with pytest.raises(PermissionError):
        f.save(evidence_ids=['evidence-forged'])
    with pytest.raises(PermissionError):
        f.findings.record(RECIPIENT, f.run_id, evidence_ids=[f.eid], interpretation=f.value,
                          audience_user_ids=['parent'])
