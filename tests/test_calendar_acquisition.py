"""Disposable calendar wire fixtures; no provider credentials or writes."""
from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace

import pytest

from test_gmail_adapter import OAuth, Wire, NOW
from alphonse.agent_v2.accounts import AccountHandle, AccountSession
from alphonse.agent_v2.integrations.acquisition import CursorExpired, ReadContext, ReadLimits
from alphonse.agent_v2.integrations.content import normalize_mime
from alphonse.agent_v2.integrations.google_calendar import CalendarAdapter, CalendarHTTPError
from alphonse.agent_v2.integrations.oauth import SCOPES
from alphonse.agent_v2.services.project_check_worker import CheckStopped

QUERY = {'text': 'assignment', 'time_min': '2026-10-25T00:00:00-04:00', 'time_max': '2026-11-10T00:00:00-05:00'}


def event(id='instance1', **kw):
    return dict(id=id, etag='"v1"', updated='2026-11-02T14:00:00Z', iCalUID='series@example.invalid',
        recurringEventId='series', originalStartTime={'dateTime': '2026-11-02T09:00:00-05:00', 'timeZone': 'America/New_York'},
        start={'dateTime': '2026-11-02T10:00:00-05:00', 'timeZone': 'America/New_York'},
        end={'dateTime': '2026-11-02T11:00:00-05:00', 'timeZone': 'America/New_York'},
        summary='Assignment meeting', **kw)


@pytest.fixture
def calendar():
    oauth, wire = OAuth(), Wire()
    context = ReadContext(AccountSession('child'), AccountHandle('google-1', 'binding', 'grant',
        1, 2, 'google', 'calendar_read', ('exact@calendar',), ('child',)), NOW + timedelta(minutes=1),
        ReadLimits(5, 100, 1_000_000), project_id='project',
        calendar_window=(QUERY['time_min'], QUERY['time_max']))
    return SimpleNamespace(oauth=oauth, wire=wire, context=context,
        adapter=CalendarAdapter(oauth, transport=wire, clock=lambda: NOW))


def search(f, *items, **kw):
    f.wire.responses = [dict(items=list(items), timeZone='America/New_York', nextSyncToken='sync1', **kw)]
    return f.adapter.search(f.context, query=QUERY, since=NOW)


def content(record):
    return json.loads(normalize_mime(record.mime, max_bytes=100000)['parts'][0]['text'])


def test_recurrence_exception_identity_dst_and_all_day_dates(calendar):
    f = calendar
    before = event('before')
    before.update(start={'dateTime': '2026-10-26T09:00:00-04:00', 'timeZone': 'America/New_York'},
                  end={'dateTime': '2026-10-26T10:00:00-04:00', 'timeZone': 'America/New_York'})
    all_day = event('date')
    all_day.update(start={'date': '2026-11-01'}, end={'date': '2026-11-02'})
    page = search(f, before, event(), all_day)
    assert page.complete and page.coverage_complete
    assert {r.source_id for r in page.records} == {'before', 'instance1', 'date'}
    exception = content(page.records[1])
    assert exception['recurringEventId'] == 'series'
    assert exception['originalStartTime']['dateTime'] == '2026-11-02T09:00:00-05:00'
    assert exception['start']['dateTime'] == '2026-11-02T10:00:00-05:00'
    assert content(page.records[0])['start']['dateTime'].endswith('-04:00')
    assert content(page.records[2])['end'] == {'date': '2026-11-02'}
    assert exception['assignment_completion_verified'] is False
    params = f.wire.calls[0][1]['params']
    assert params['singleEvents'] == 'true' and params['showDeleted'] == 'true'
    assert params['timeMin'] == QUERY['time_min'] and 'q' not in params
    assert all(a['operation'] == 'calendar_read' and a['required_scopes'] == [SCOPES['google']['calendar_read']]
               and a['project_id'] == 'project' and a['expected_grant_revision'] == 2 for a in f.oauth.requests)


def test_paginated_baseline_and_delta_anchor_cancellation_and_move(calendar):
    f = calendar
    first = search(f, event(), nextPageToken='page2')
    assert not first.complete
    f.wire.responses = [dict(items=[], nextSyncToken='sync2')]
    last = f.adapter.search(f.context, query=QUERY, since=NOW, cursor=first.cursor)
    assert last.complete and f.wire.calls[-1][1]['params']['pageToken'] == 'page2'
    cancelled = {'id': 'cancelled', 'status': 'cancelled', 'recurringEventId': 'series',
                 'originalStartTime': {'date': '2026-11-03'}}
    f.wire.responses = [dict(items=[cancelled], nextPageToken='delta2')]
    delta = f.adapter.changes(f.context, query=QUERY, cursor=last.cursor)
    assert delta.records[0].deleted and delta.records[0].source_id == 'cancelled'
    moved = event('instance1')
    moved.update(start={'date': '2027-01-01'}, end={'date': '2027-01-02'}, description='outside-window private text')
    f.wire.responses = [dict(items=[moved], nextSyncToken='sync3')]
    final = f.adapter.changes(f.context, query=QUERY, cursor=delta.cursor)
    assert final.complete and final.records[0].deleted
    assert b'outside-window' not in final.records[0].mime
    for _, kw in f.wire.calls[-2:]:
        assert kw['params']['syncToken'] == 'sync2'
        assert 'timeMin' not in kw['params'] and 'timeMax' not in kw['params']
    assert f.wire.calls[-1][1]['params']['pageToken'] == 'delta2'


def test_expiry_and_changed_scope_require_bounded_resync(calendar):
    f = calendar
    page = search(f)
    f.wire.responses = [CalendarHTTPError(410)]
    with pytest.raises(CursorExpired):
        f.adapter.changes(f.context, query=QUERY, cursor=page.cursor)
    for context, query in [(replace(f.context, project_id='other'), QUERY),
                           (replace(f.context, grant=replace(f.context.grant, resources=('other',))), QUERY),
                           (replace(f.context, calendar_window=(QUERY['time_min'], '2026-11-11T00:00:00-05:00')), dict(QUERY, time_max='2026-11-11T00:00:00-05:00'))]:
        with pytest.raises(CursorExpired):
            f.adapter.changes(context, query=query, cursor=page.cursor)
    assert len(f.wire.calls) == 2


@pytest.mark.parametrize('status,outcome', [(401, 'blocked'), (403, 'blocked'), (404, 'blocked'), (429, 'partial'), (503, 'partial')])
def test_revoked_calendar_and_provider_failure_never_absence(calendar, status, outcome):
    f = calendar
    f.wire.responses = [CalendarHTTPError(status)]
    with pytest.raises(CheckStopped) as exc:
        f.adapter.search(f.context, query=QUERY, since=NOW)
    assert exc.value.outcome == outcome and len(f.wire.calls) == 1


@pytest.mark.parametrize('when', ['before', 'during'])
def test_grant_revocation_fences_results(calendar, when):
    f = calendar
    f.wire.responses = [dict(items=[], nextSyncToken='sync1')]
    if when == 'before':
        f.oauth.revoked = True
    else:
        f.wire.after = lambda: setattr(f.oauth, 'revoked', True)
    with pytest.raises(PermissionError):
        f.adapter.search(f.context, query=QUERY, since=NOW)
    assert len(f.wire.calls) == (0 if when == 'before' else 1)


def test_partial_redacted_events_and_missing_token_are_not_empty_success(calendar):
    f = calendar
    page = search(f, {'id': 'busy', 'summary': 'Busy'})
    assert not page.coverage_complete and page.records == ()
    f.wire.responses = [{'items': []}]
    with pytest.raises(CheckStopped, match='sync_token_missing'):
        f.adapter.search(f.context, query=QUERY, since=NOW)


def test_discovery_consent_is_independent_of_mail_and_event_reads(calendar):
    f = calendar
    with pytest.raises(PermissionError):
        f.adapter.discover(f.context)
    mail = replace(f.context, grant=replace(f.context.grant, operation='mail_read'))
    with pytest.raises(PermissionError):
        f.adapter.search(mail, query=QUERY, since=NOW)
    assert not f.wire.calls
    discover = replace(f.context, grant=replace(f.context.grant, operation='calendar_discover', resources=('calendar:list',)))
    f.wire.responses = [dict(items=[dict(id='candidate', summary='Calendar', accessRole='reader', secret='ignored')], nextPageToken='next')]
    result = f.adapter.discover(discover)
    assert result['calendars'] == [{'id': 'candidate', 'summary': 'Calendar', 'accessRole': 'reader'}]
    assert f.oauth.requests[-1]['required_scopes'] == [SCOPES['google']['calendar_discover']]
    f.wire.responses = [{'items': []}]
    assert f.adapter.discover(discover, cursor=result['cursor'])['cursor'] is None
    with pytest.raises(PermissionError):
        f.adapter.read(discover, resource='candidate', source_id='event')


def test_direct_read_exact_resource_identity_and_window(calendar):
    f = calendar
    with pytest.raises(PermissionError):
        f.adapter.read(f.context, resource='other', source_id='instance1')
    f.wire.responses = [event()]
    assert f.adapter.read(f.context, resource='exact@calendar', source_id='instance1').source_id == 'instance1'
    f.wire.responses = [event('other')]
    with pytest.raises(ValueError, match='identity_mismatch'):
        f.adapter.read(f.context, resource='exact@calendar', source_id='instance1')
    outside = event()
    outside.update(start={'date': '2027-01-01'}, end={'date': '2027-01-02'})
    f.wire.responses = [outside, {'timeZone': 'America/New_York'}]
    with pytest.raises(PermissionError, match='outside_window'):
        f.adapter.read(f.context, resource='exact@calendar', source_id='instance1')
    with pytest.raises(PermissionError):
        f.adapter.read(replace(f.context, calendar_window=None), resource='exact@calendar', source_id='instance1')
    assert '%40' in f.wire.calls[0][0]
    assert not any(hasattr(f.adapter, name) for name in ('create', 'update', 'delete', 'invite', 'watch'))


@pytest.mark.parametrize('limit', ['items', 'pages', 'bytes', 'deadline'])
def test_budgets_before_io(calendar, limit):
    f = calendar
    ctx = replace(f.context, deadline=NOW) if limit == 'deadline' else replace(f.context, limits=replace(f.context.limits, **{limit: 0}))
    with pytest.raises(CheckStopped):
        f.adapter.search(ctx, query=QUERY, since=NOW)
    assert not f.wire.calls


def test_multiple_selected_calendars_keep_separate_tokens(calendar):
    f = calendar
    f.context = replace(f.context, grant=replace(f.context.grant, resources=('one', 'two')))
    first = search(f)
    assert not first.complete
    f.wire.responses = [dict(items=[], nextSyncToken='two-token')]
    last = f.adapter.search(f.context, query=QUERY, since=NOW, cursor=first.cursor)
    assert last.complete and f.wire.calls[-1][0] == 'calendars/two/events'
    f.wire.responses = [dict(items=[], nextSyncToken='one-new'), dict(items=[], nextSyncToken='two-new')]
    delta = f.adapter.changes(f.context, query=QUERY, cursor=last.cursor)
    final = f.adapter.changes(f.context, query=QUERY, cursor=delta.cursor)
    assert final.complete
    assert f.wire.calls[-2][1]['params']['syncToken'] == 'sync1'
    assert f.wire.calls[-1][1]['params']['syncToken'] == 'two-token'


@pytest.mark.parametrize('case', ['ok', 'oversize', 'redirect', 'timeout'])
def test_fixed_transport_bounds_and_no_credential_forwarding(monkeypatch, case):
    import sys
    from datetime import datetime, timezone
    from alphonse.agent_v2.integrations.google_calendar import CalendarTransport
    calls = []
    class RequestFailure(Exception):
        pass
    class Response:
        status_code = 302 if case == 'redirect' else 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_content(self, **kw): yield b'{"ok":true}'
    def get(url, **kw):
        calls.append((url, kw))
        if case == 'timeout': raise RequestFailure('private token detail')
        return Response()
    monkeypatch.setitem(sys.modules, 'requests', SimpleNamespace(get=get, RequestException=RequestFailure))
    kwargs = dict(params={}, token='synthetic', deadline=datetime.now(timezone.utc) + timedelta(seconds=10),
                  max_bytes=2 if case == 'oversize' else 100)
    if case == 'ok':
        assert CalendarTransport().get('calendars/exact/events', **kwargs) == ({'ok': True}, 11)
    else:
        with pytest.raises((CalendarHTTPError, CheckStopped)) as exc:
            CalendarTransport().get('calendars/exact/events', **kwargs)
        assert 'private' not in str(exc.value)
    assert len(calls) == 1 and calls[0][0] == 'https://www.googleapis.com/calendar/v3/calendars/exact/events'
    assert calls[0][1]['allow_redirects'] is False and calls[0][1]['stream'] is True


from test_account_grants import f, expiry
from test_project_checks import checks as base_checks, active
from test_project_check_scheduling import scheduled
from test_source_acquisition import acquisition


@pytest.fixture
def mixed_sources(request):
    return getattr(request, 'param', False)


@pytest.fixture
def checks(f, mixed_sources):
    f = base_checks.__wrapped__(f)
    f.service.create_connection(AccountSession('child'), connection_id='google', provider_key='google',
                                provider_subject='synthetic-subject', display_name='Calendar')
    f.store.install_credentials('google', {'token': 'disposable'}, expected_generation=0)
    f.service.save_connection(AccountSession('child'), connection_id='google', expected_revision=1,
        provider_subject='synthetic-subject', capabilities=['calendar_read'],
        granted_scopes=[SCOPES['google']['calendar_read']])
    f.service.save_grant(AccountSession('child'), grant_id='calendar-grant', connection_id='google',
        expected_revision=0, beneficiary_user_id='child', operations=['calendar_read'], resources=['exact@calendar'],
        recipient_user_ids=['parent'], expires_at=expiry(), project_id=f.project.project_id)
    f.bindings.configure(actor_id='child', owner_id='child', binding_id='calendar-binding', skill_id='skill:fixture',
        kind='account', target='google', grant_id='calendar-grant', capabilities=('calendar_read',),
        required_scopes=(SCOPES['google']['calendar_read'],))
    f.bindings.select(actor_id='child', skill_id='skill:fixture', name='calendar', binding_id='calendar-binding')
    mail = f.definition['source_bindings'][0]
    f.definition['source_bindings'] = [dict(skill_id='skill:fixture', role='calendar', binding_id='calendar-binding',
        connection_id='google', grant_id='calendar-grant', operation='calendar_read', resources=['exact@calendar'])]
    if mixed_sources:
        f.definition['source_bindings'].append(mail)
    f.definition['query']['calendar_window'] = {k: v for k, v in QUERY.items() if k != 'text'}
    return f


def worker_adapter(f, *responses):
    wire = Wire(*responses)
    # Real account/binding authority, synthetic OAuth token and HTTP boundary.
    oauth = OAuth()
    oauth.accounts = f.service
    adapter = CalendarAdapter(oauth, transport=wire, clock=lambda: f.now)
    f.acquisition.adapters[('google', 'calendar_read')] = adapter
    f.acquisition.adapters['google'] = SimpleNamespace(search=lambda *a, **k: pytest.fail('mail adapter selected'))
    return wire


def test_calendar_worker_persists_page_before_cursor_and_recovers(acquisition, monkeypatch):
    f = acquisition
    wire = worker_adapter(f, dict(items=[event()], nextPageToken='next'), dict(items=[], nextSyncToken='sync'))
    accept = f.acquisition._accept
    def crash(*args, **kw):
        accept(*args, **kw)
        raise SystemExit('after acceptance')
    monkeypatch.setattr(f.acquisition, '_accept', crash)
    with pytest.raises(SystemExit): f.process()
    saved = f.acquisition.evidence(AccountSession('child'), f.run_id)
    assert saved[0]['connection_id'] == 'google' and saved[0]['source_id'] == 'instance1'
    monkeypatch.setattr(f.acquisition, '_accept', accept)
    f.now += timedelta(seconds=31)
    assert f.process()['outcome'] == 'complete'
    assert wire.calls[-1][1]['params']['pageToken'] == 'next'
    assert len(f.acquisition.evidence(AccountSession('child'), f.run_id)) == 1


@pytest.mark.parametrize('failure', ['expired', 'redacted', 'revoked'])
def test_calendar_worker_failure_coverage_is_never_no_match(acquisition, failure):
    f = acquisition
    responses = {'expired': [CalendarHTTPError(410), dict(items=[], nextSyncToken='fresh')],
                 'redacted': [dict(items=[{'id': 'busy'}], nextSyncToken='sync')],
                 'revoked': [CalendarHTTPError(404)]}[failure]
    wire = worker_adapter(f, *responses)
    result = f.process()
    if failure == 'expired':
        assert result['state'] == 'pending'
        result = f.process()
        assert wire.calls[-1][1]['params']['timeMin'] == QUERY['time_min']
        assert 'syncToken' not in wire.calls[-1][1]['params']
    assert result['outcome'] == ('blocked' if failure == 'revoked' else 'partial')
    assert result['outcome'] != 'no_match'


@pytest.mark.parametrize('mixed_sources', [True], indirect=True)
def test_calendar_and_mail_references_reconcile_without_completing_assignment(acquisition):
    from alphonse.agent_v2.integrations.acquisition import SourcePage
    from alphonse.agent_v2.services.findings import FindingService
    from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
    from test_source_acquisition import record
    f = acquisition
    wire = worker_adapter(f, dict(items=[event()], nextSyncToken='sync'))
    f.adapter.pages = [SourcePage((record(f),), complete=True)]
    assert f.process()['state'] == 'pending'
    assert f.process()['outcome'] == 'complete'
    assert len(wire.calls) == 1
    assert f.adapter.calls[0][2]['query'] == {'text': 'school'}
    sources = f.acquisition.evidence(AccountSession('child'), f.run_id)
    assert {s['connection_id'] for s in sources} == {'one', 'google'}
    findings = FindingService(f.acquisition, SQLiteOutboundStore(f.tmp / 'outbox.db'), SimpleNamespace())
    interpretation = dict(kind='assignment', subject_id='child', commitment_id='course:assignment7',
        summary='School assignment', facts=[dict(predicate='calendar_presence', value=True, basis='source_observed', unit=None)],
        uncertainty=['No submission or completion evidence.'])
    ref = findings.record(AccountSession('child'), f.run_id, evidence_ids=[s['evidence_id'] for s in sources],
        interpretation=interpretation, audience_user_ids=['parent'])
    result = findings.view(AccountSession('parent'), ref['finding_id'])
    assert len(result['source_refs']) == 2 and result['state'] == 'open'
    for predicate in ('assignment_completed', 'submission_completed'):
        with pytest.raises(ValueError):
            findings.record(AccountSession('child'), f.run_id, evidence_ids=[s['evidence_id'] for s in sources],
                interpretation=dict(interpretation, facts=[dict(predicate=predicate, value=True, basis='source_observed', unit=None)]),
                audience_user_ids=['parent'])
    # Revoked calendar access removes authority; mail alone cannot fill the gap.
    f.service.revoke_grant(AccountSession('child'), grant_id='calendar-grant', expected_revision=1)
    with pytest.raises(PermissionError): findings.view(AccountSession('parent'), ref['finding_id'])


def test_all_day_exclusive_end_uses_calendar_zone_at_window_boundary(calendar):
    f = calendar
    # NY Nov 1 ends at 05:00 UTC on Nov 2, not UTC midnight.
    query = dict(QUERY, time_min='2026-11-02T04:30:00Z', time_max='2026-11-02T05:30:00Z')
    ctx = replace(f.context, calendar_window=(query['time_min'], query['time_max']))
    value = event('all-day')
    value.update(start={'date': '2026-11-01'}, end={'date': '2026-11-02'})
    f.wire.responses = [dict(items=[value], timeZone='America/New_York', nextSyncToken='s')]
    assert len(f.adapter.search(ctx, query=query, since=NOW).records) == 1
    f.wire.responses = [value, {'timeZone': 'America/New_York'}]
    assert content(f.adapter.read(ctx, resource='exact@calendar', source_id='all-day'))['calendar_timezone'] == 'America/New_York'


def test_cancellation_and_window_mismatch_prevent_io(calendar):
    f = calendar
    def cancelled(): raise CheckStopped('cancelled', 'cancelled')
    with pytest.raises(CheckStopped):
        f.adapter.search(replace(f.context, guard=cancelled), query=QUERY, since=NOW)
    with pytest.raises(PermissionError, match='window_mismatch'):
        f.adapter.search(f.context, query=dict(QUERY, time_max='2026-11-11T00:00:00-05:00'), since=NOW)
    with pytest.raises(PermissionError, match='exact_id_required'):
        f.adapter.search(replace(f.context, grant=replace(f.context.grant, resources=('primary',))), query=QUERY, since=NOW)
    assert not f.wire.calls


@pytest.mark.parametrize('access', ['freeBusyReader', 'redacted-private'])
def test_busy_only_visibility_keeps_coverage_partial(calendar, access):
    f = calendar
    value = event()
    if access == 'redacted-private':
        value.pop('summary')
        value['visibility'] = 'private'
    page = search(f, value, accessRole=access if access == 'freeBusyReader' else 'reader')
    assert not page.coverage_complete


def test_incremental_page_size_remains_stable_as_budget_shrinks(calendar):
    f = calendar
    baseline = search(f)
    f.wire.responses = [dict(items=[event()], nextPageToken='p2')]
    delta = f.adapter.changes(f.context, query=QUERY, cursor=baseline.cursor)
    f.wire.responses = [dict(items=[], nextSyncToken='sync2')]
    reduced = replace(f.context, limits=replace(f.context.limits, items=99))
    assert f.adapter.changes(reduced, query=QUERY, cursor=delta.cursor).complete
    assert [call[1]['params']['maxResults'] for call in f.wire.calls] == [100, 100, 100]
