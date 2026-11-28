"""Disposable calendar wire fixtures; no provider credentials or writes."""
from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace

import pytest

from test_gmail_adapter import OAuth, NOW
from alphonse.agent_v2.accounts import AccountHandle, AccountSession
from alphonse.agent_v2.integrations.acquisition import CursorExpired, ReadContext, ReadLimits
from alphonse.agent_v2.integrations.content import normalize_mime
from alphonse.agent_v2.integrations.google_calendar import CalendarAdapter
from alphonse.agent_v2.integrations.google_mcp import GoogleMcpError
from alphonse.agent_v2.integrations.oauth import SCOPES
from alphonse.agent_v2.services.project_check_worker import CheckStopped

class Wire:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []
        self.after = lambda: None

    def call(self, product, tool, arguments, **kwargs):
        assert product == 'calendar'
        self.calls.append((tool, dict(params=arguments, **kwargs)))
        self.after()
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response, len(json.dumps(response).encode())


QUERY = {'text': 'assignment', 'time_min': '2026-10-25T00:00:00-04:00', 'time_max': '2026-11-10T00:00:00-05:00'}


def event(id='instance1', **kw):
    return dict(id=id, updated='2026-11-02T14:00:00Z',
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


def mcp_page(*events, **fields):
    return {'events': list(events), 'timeZone': 'America/New_York', 'accessRole': 'owner', **fields}


def search(f, *items, **kw):
    f.wire.responses = [mcp_page(*items, **kw)]
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
    assert f.wire.calls[0][0] == 'list_events'
    assert params['calendarId'] == 'exact@calendar' and params['orderBy'] == 'startTime'
    assert set(params['eventType']) == {'DEFAULT', 'OUT_OF_OFFICE', 'FOCUS_TIME', 'WORKING_LOCATION', 'BIRTHDAY', 'FROM_GMAIL'}
    assert params['startTime'] == QUERY['time_min'] and 'fullText' not in params
    assert all(a['operation'] == 'calendar_read' and a['required_scopes'] == [SCOPES['google']['calendar_read']]
               and a['project_id'] == 'project' and a['expected_grant_revision'] == 2 for a in f.oauth.requests)


def test_paginated_snapshot_never_manufactures_a_change_cursor_or_tombstone(calendar):
    f = calendar
    first = search(f, event(), nextPageToken='page2')
    assert not first.complete and first.cursor
    f.wire.responses = [mcp_page()]
    last = f.adapter.search(f.context, query=QUERY, since=NOW, cursor=first.cursor)
    assert last.complete and last.cursor is None
    assert f.wire.calls[-1][1]['params']['pageToken'] == 'page2'
    moved = event('instance1')
    moved.update(start={'date': '2027-01-01'}, end={'date': '2027-01-02'}, description='outside-window private text')
    second = search(f, moved)
    assert second.complete and not second.coverage_complete and not second.records
    empty = search(f)
    assert empty.complete and empty.coverage_complete and not empty.records
    assert f.wire.calls[-1][1]['params']['startTime'] == QUERY['time_min']
    assert 'syncToken' not in f.wire.calls[-1][1]['params']


def test_expiry_and_changed_scope_require_bounded_resync(calendar):
    f = calendar
    page = search(f, nextPageToken='next')
    f.wire.responses = [GoogleMcpError(410)]
    with pytest.raises(CursorExpired):
        f.adapter.search(f.context, query=QUERY, since=NOW, cursor=page.cursor)
    for context, query in [(replace(f.context, project_id='other'), QUERY),
                           (replace(f.context, grant=replace(f.context.grant, resources=('other',))), QUERY),
                           (replace(f.context, calendar_window=(QUERY['time_min'], '2026-11-11T00:00:00-05:00')),
                            dict(QUERY, time_max='2026-11-11T00:00:00-05:00'))]:
        with pytest.raises(CursorExpired):
            f.adapter.search(context, query=query, since=NOW, cursor=page.cursor)
    assert len(f.wire.calls) == 2


@pytest.mark.parametrize('status,outcome', [(401, 'blocked'), (403, 'blocked'), (404, 'blocked'), (429, 'partial'), (503, 'partial')])
def test_revoked_calendar_and_provider_failure_never_absence(calendar, status, outcome):
    f = calendar
    f.wire.responses = [GoogleMcpError(status)]
    with pytest.raises(CheckStopped) as exc:
        f.adapter.search(f.context, query=QUERY, since=NOW)
    assert exc.value.outcome == outcome and len(f.wire.calls) == 1


@pytest.mark.parametrize('when', ['before', 'during'])
def test_grant_revocation_fences_results(calendar, when):
    f = calendar
    f.wire.responses = [mcp_page()]
    if when == 'before':
        f.oauth.revoked = True
    else:
        f.wire.after = lambda: setattr(f.oauth, 'revoked', True)
    with pytest.raises(PermissionError):
        f.adapter.search(f.context, query=QUERY, since=NOW)
    assert len(f.wire.calls) == (0 if when == 'before' else 1)


def test_partial_redacted_events_are_not_empty_success(calendar):
    f = calendar
    page = search(f, {'id': 'busy', 'summary': 'Busy'})
    assert not page.coverage_complete and page.records == ()


@pytest.mark.parametrize('fields,complete,covered', [
    ({'accessRole': 'owner'}, True, True),
    ({'accessRole': 'reader'}, True, False),
    ({'accessRole': 'owner', 'nextPageToken': 'more'}, False, False),
    ({'accessRole': 'owner', 'omittedContent': {'count': 1}}, True, False),
])
def test_omitted_event_array_requires_unredacted_terminal_page(calendar, fields, complete, covered):
    f = calendar
    f.wire.responses = [{'timeZone': 'America/New_York', **fields}]
    page = f.adapter.search(f.context, query=QUERY, since=NOW)
    assert page.complete is complete and page.coverage_complete is covered and not page.records


def test_discovery_consent_is_independent_of_mail_and_event_reads(calendar):
    f = calendar
    with pytest.raises(PermissionError):
        f.adapter.discover(f.context)
    mail = replace(f.context, grant=replace(f.context.grant, operation='mail_read'))
    with pytest.raises(PermissionError):
        f.adapter.search(mail, query=QUERY, since=NOW)
    assert not f.wire.calls
    discover = replace(f.context, grant=replace(f.context.grant, operation='calendar_discover', resources=('calendar:list',)))
    f.wire.responses = [dict(calendars=[dict(id='candidate', summary='Calendar', accessRole='reader', secret='ignored')], nextPageToken='next')]
    result = f.adapter.discover(discover)
    assert result['calendars'] == [{'id': 'candidate', 'summary': 'Calendar', 'accessRole': 'reader'}]
    assert f.oauth.requests[-1]['required_scopes'] == [SCOPES['google']['calendar_discover']]
    f.wire.responses = [{'calendars': []}]
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
    f.wire.responses = [outside, mcp_page()]
    with pytest.raises(PermissionError, match='outside_window'):
        f.adapter.read(f.context, resource='exact@calendar', source_id='instance1')
    with pytest.raises(PermissionError):
        f.adapter.read(replace(f.context, calendar_window=None), resource='exact@calendar', source_id='instance1')
    assert f.wire.calls[0][0] == 'get_event'
    assert f.wire.calls[0][1]['params'] == {'calendarId': 'exact@calendar', 'eventId': 'instance1'}
    assert not any(hasattr(f.adapter, name) for name in ('create', 'update', 'delete', 'invite', 'watch'))


def test_private_event_requires_provider_role_that_can_read_details(calendar):
    f = calendar
    item = event(visibility='private', description='private fixture')
    page = search(f, item, accessRole='reader')
    assert not page.coverage_complete and not page.records
    f.wire.responses = [item, mcp_page(accessRole='reader')]
    with pytest.raises(PermissionError, match='details_unavailable'):
        f.adapter.read(f.context, resource='exact@calendar', source_id='instance1')


@pytest.mark.parametrize('limit', ['items', 'pages', 'bytes', 'deadline'])
def test_budgets_before_io(calendar, limit):
    f = calendar
    ctx = replace(f.context, deadline=NOW) if limit == 'deadline' else replace(f.context, limits=replace(f.context.limits, **{limit: 0}))
    with pytest.raises(CheckStopped):
        f.adapter.search(ctx, query=QUERY, since=NOW)
    assert not f.wire.calls


def test_multiple_selected_calendars_keep_separate_pages_and_coverage(calendar):
    f = calendar
    f.context = replace(f.context, grant=replace(f.context.grant, resources=('one', 'two')))
    first = search(f, {'id': 'redacted'}, nextPageToken='page-one')
    assert first.resource_coverage == {'one': 'incomplete', 'two': 'pending'}
    f.wire.responses = [mcp_page(), mcp_page(event())]
    second = f.adapter.search(f.context, query=QUERY, since=NOW, cursor=first.cursor)
    last = f.adapter.search(f.context, query=QUERY, since=NOW, cursor=second.cursor)
    assert last.complete and last.cursor is None and not last.coverage_complete
    assert last.resource_coverage == {'one': 'incomplete', 'two': 'complete'}
    params = [args['params'] for _, args in f.wire.calls]
    assert [p['calendarId'] for p in params] == ['one', 'one', 'two']
    assert [p.get('pageToken') for p in params] == [None, 'page-one', None]


def test_default_transport_uses_calendar_mcp_and_reports_provider_approval(calendar, monkeypatch):
    from alphonse.agent_v2.integrations import google_calendar

    f = calendar
    monkeypatch.setattr(google_calendar, 'GoogleMcpClient', lambda: f.wire)
    adapter = CalendarAdapter(f.oauth, clock=lambda: NOW)
    f.wire.responses = [GoogleMcpError(409, 'approval_required')]
    with pytest.raises(CheckStopped, match='calendar_google_approval_required') as exc:
        adapter.search(f.context, query=QUERY, since=NOW)
    assert exc.value.outcome == 'blocked'
    assert f.wire.calls[0][0] == 'list_events'
    assert f.wire.calls[0][1]['token'] == 'synthetic-token'


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
    wire = worker_adapter(f, mcp_page(event(), nextPageToken='next'), mcp_page())
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
    responses = {'expired': [mcp_page(nextPageToken='old'), GoogleMcpError(410), mcp_page()],
                 'redacted': [mcp_page({'id': 'busy'})],
                 'revoked': [GoogleMcpError(404)]}[failure]
    wire = worker_adapter(f, *responses)
    result = f.process()
    if failure == 'expired':
        assert result['state'] == 'pending'
        assert f.process()['state'] == 'pending'
        result = f.process()
        assert wire.calls[-1][1]['params']['startTime'] == QUERY['time_min']
        assert 'pageToken' not in wire.calls[-1][1]['params']
    assert result['outcome'] == ('blocked' if failure == 'revoked' else 'partial')
    assert result['outcome'] != 'no_match'


@pytest.mark.parametrize('mixed_sources', [True], indirect=True)
def test_calendar_and_mail_references_reconcile_without_completing_assignment(acquisition):
    from alphonse.agent_v2.integrations.acquisition import SourcePage
    from alphonse.agent_v2.services.findings import FindingService
    from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
    from test_source_acquisition import record
    f = acquisition
    wire = worker_adapter(f, mcp_page(event()))
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
    f.wire.responses = [mcp_page(value)]
    assert len(f.adapter.search(ctx, query=query, since=NOW).records) == 1
    f.wire.responses = [value, mcp_page()]
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


def test_snapshot_page_size_remains_stable_as_budget_shrinks(calendar):
    f = calendar
    first = search(f, event(), nextPageToken='p2')
    f.wire.responses = [mcp_page()]
    reduced = replace(f.context, limits=replace(f.context.limits, items=99))
    assert f.adapter.search(reduced, query=QUERY, since=NOW, cursor=first.cursor).complete
    assert [call[1]['params']['pageSize'] for call in f.wire.calls] == [100, 100]
