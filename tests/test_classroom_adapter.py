"""Disposable Classroom wire fixtures. No real account, provider writes or impersonation."""
from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace

import pytest

from test_gmail_adapter import OAuth, Wire, NOW
from alphonse.agent_v2.accounts import AccountSession, AccountHandle
from alphonse.agent_v2.integrations.acquisition import ReadContext, ReadLimits, CursorExpired
from alphonse.agent_v2.integrations.classroom import ClassroomAdapter, ClassroomHTTPError, KINDS
from alphonse.agent_v2.integrations.content import normalize_mime
from alphonse.agent_v2.integrations.oauth import capability_scopes
from alphonse.agent_v2.services.project_check_worker import CheckStopped


def item(**kw):
    return dict(id='assignment1', courseId='course1', state='PUBLISHED', title='Essay',
                updateTime=NOW.isoformat(), maxPoints=30, **kw)


def role(resource='course1'):
    return [{'id': resource, 'courseState': 'ACTIVE'}, {'courseId': resource, 'userId': 'connected-student'}]


@pytest.fixture
def setup():
    oauth, wire = OAuth(), Wire()
    context = ReadContext(AccountSession('child'), AccountHandle('google1', 'binding', 'grant',
        1, 2, 'google', 'classroom_read', ('course1',), ('child',)),
        NOW + timedelta(minutes=1), ReadLimits(5, 100, 1000000), project_id='school')
    adapter = ClassroomAdapter(oauth, transport=wire, clock=lambda: NOW)
    return SimpleNamespace(oauth=oauth, wire=wire, context=context, adapter=adapter)


def search(f, value=None):
    f.wire.responses = [*role(), {'courseWork': [value or item()]}]
    return f.adapter.search(f.context, query={'text': ''}, since=NOW - timedelta(days=30))


def content(record):
    return json.loads(normalize_mime(record.mime, max_bytes=100000)['parts'][0]['text'])


def test_student_role_checked_before_content_and_unknowns_preserved(setup):
    f = setup
    page = search(f)
    assert [c[0] for c in f.wire.calls] == ['courses/course1', 'courses/course1/students/me', 'courses/course1/courseWork']
    assert page.complete and page.resource_coverage == {'course1': 'complete'}
    record = page.records[0]
    assert record.source_id == 'courseWork:assignment1' and record.resource == 'course1'
    data = content(record)
    assert data['id'] == 'assignment1' and data['courseId'] == 'course1'
    assert data['maxPoints'] == 30
    assert all(data[k] is None for k in ('dueDate', 'dueTime', 'grading_weight', 'submission_state'))
    assert data['source_timestamp_known']
    assert all(r['required_scopes'] == capability_scopes('google', 'classroom_read') for r in f.oauth.requests)
    assert not any('students.readonly' in s or 'guardian' in s for r in f.oauth.requests for s in r['required_scopes'])


@pytest.mark.parametrize('response,code', [
    (ClassroomHTTPError(404), 'classroom_student_role_required'),
    (ClassroomHTTPError(403), 'classroom_permission_denied'),
    ({'courseId': 'course1'}, 'classroom_student_role_required')])
def test_guardian_only_or_school_restricted_role_never_reads_content(setup, response, code):
    f = setup
    f.wire.responses = [role()[0], response]
    with pytest.raises(CheckStopped) as exc:
        f.adapter.search(f.context, query={'text': ''}, since=NOW)
    assert exc.value.outcome == 'blocked' and exc.value.code == code
    assert len(f.wire.calls) == 2
    assert not any('/courseWork' in c[0] or '/guardians' in c[0] for c in f.wire.calls)


@pytest.mark.parametrize('status,code,outcome', [(401, 'classroom_reauth_required', 'blocked'),
    (403, 'classroom_permission_denied', 'blocked'), (404, 'classroom_unavailable', 'blocked'),
    (429, 'classroom_unavailable', 'partial'), (503, 'classroom_unavailable', 'partial')])
def test_school_restriction_revoked_consent_and_unavailable_are_explicit(setup, status, code, outcome):
    f = setup
    f.wire.responses = [*role(), ClassroomHTTPError(status)]
    with pytest.raises(CheckStopped) as exc:
        f.adapter.search(f.context, query={'text': ''}, since=NOW)
    assert (exc.value.outcome, exc.value.code) == (outcome, code)
    assert len(f.wire.calls) == 3


@pytest.mark.parametrize('when', ['before', 'during'])
def test_revocation_and_cancellation_fence_io(setup, when):
    f = setup
    f.wire.responses = role()
    if when == 'before':
        f.oauth.revoked = True
    else:
        f.wire.after = lambda: setattr(f.oauth, 'revoked', True)
    with pytest.raises(PermissionError):
        f.adapter.health(f.context)
    assert len(f.wire.calls) == (0 if when == 'before' else 1)
    f.oauth.revoked = False
    f.wire.after = lambda: None
    ctx = replace(f.context, guard=lambda: (_ for _ in ()).throw(CheckStopped('cancelled', 'cancelled')))
    count = len(f.wire.calls)
    with pytest.raises(CheckStopped, match='cancelled'):
        f.adapter.health(ctx)
    assert len(f.wire.calls) == count


def test_repeated_corrected_and_missing_assignment_fields(setup):
    f = setup
    first = search(f).records[0]
    assert search(f).records[0].version == first.version
    changed = item()
    changed.update(description='Corrected deadline', updateTime=(NOW + timedelta(hours=1)).isoformat(),
                   dueDate={'year': 2026, 'month': 11, 'day': 5})
    second = search(f, changed).records[0]
    assert second.source_id == first.source_id and second.version != first.version
    assert content(second)['dueTime'] is None
    missing = {'id': 'assignment1', 'courseId': 'course1', 'state': 'PUBLISHED'}
    unknown = search(f, missing).records[0]
    assert not content(unknown)['source_timestamp_known'] and content(unknown)['maxPoints'] is None
    assert search(f, missing).records[0].version == unknown.version


def test_announcements_require_explicit_operation_and_scope(setup):
    f = setup
    f.context = replace(f.context, grant=replace(f.context.grant, operation='classroom_announcements_read'))
    f.wire.responses = [*role(), {'announcements': [item(text='School notice')]}]
    page = f.adapter.search(f.context, query={'text': ''}, since=NOW)
    assert page.records[0].source_id == 'announcements:assignment1'
    assert f.wire.calls[-1][0] == 'courses/course1/announcements'
    assert f.wire.calls[-1][1]['params']['announcementStates'] == 'PUBLISHED'
    scopes = f.oauth.requests[-1]['required_scopes']
    assert any('announcements.readonly' in s for s in scopes)
    assert not any('coursework' in s for s in scopes)


def test_pages_scope_cursor_expiry_and_refresh_preserve_original_lower_bound(setup):
    f = setup
    f.wire.responses = [*role(), {'courseWork': [item()], 'nextPageToken': 'opaque'}]
    first = f.adapter.search(f.context, query={'text': ''}, since=NOW - timedelta(days=30))
    assert not first.complete
    f.wire.responses = [*role(), {'courseWork': []}]
    last = f.adapter.changes(f.context, query={'text': ''}, cursor=first.cursor)
    assert last.complete and f.wire.calls[-1][1]['params']['pageToken'] == 'opaque'
    f.wire.responses = [*role(), {'courseWork': [item()]}]
    refreshed = f.adapter.changes(f.context, query={'text': ''}, cursor=last.cursor)
    assert refreshed.complete and refreshed.records[0].source_id == first.records[0].source_id
    assert json.loads(refreshed.cursor)['since'] == json.loads(first.cursor)['since']
    ctx = replace(f.context, grant=replace(f.context.grant, grant_revision=3))
    with pytest.raises(CursorExpired):
        f.adapter.changes(ctx, query={'text': ''}, cursor=first.cursor)
    f.wire.responses = [*role(), ClassroomHTTPError(400)]
    with pytest.raises(CursorExpired):
        f.adapter.changes(f.context, query={'text': ''}, cursor=first.cursor)


def test_multiple_courses_have_independent_truthful_coverage(setup):
    f = setup
    f.context = replace(f.context, grant=replace(f.context.grant, resources=('course1', 'course2')))
    first = search(f)
    assert not first.complete and first.resource_coverage == {'course1': 'complete', 'course2': 'pending'}
    f.wire.responses = [*role('course2'), {}]
    last = f.adapter.changes(f.context, query={'text': ''}, cursor=first.cursor)
    assert last.complete and last.resource_coverage == {'course1': 'complete', 'course2': 'complete'}


@pytest.mark.parametrize('limit', ['pages', 'items', 'bytes', 'deadline'])
def test_limits_before_io(setup, limit):
    f = setup
    ctx = replace(f.context, deadline=NOW) if limit == 'deadline' else replace(f.context,
        limits=replace(f.context.limits, **{limit: 0}))
    with pytest.raises(CheckStopped):
        f.adapter.search(ctx, query={'text': ''}, since=NOW)
    assert not f.wire.calls


def test_read_scope_identity_and_no_write_or_impersonation_surface(setup):
    f = setup
    with pytest.raises(PermissionError):
        f.adapter.read(f.context, resource='other', source_id='courseWork:assignment1')
    assert not f.wire.calls
    f.wire.responses = [*role(), item()]
    record = f.adapter.read(f.context, resource='course1', source_id='courseWork:assignment1')
    assert record.source_id == 'courseWork:assignment1'
    with pytest.raises(PermissionError):
        f.adapter.read(f.context, resource='course1', source_id='announcements:assignment1')
    with pytest.raises(PermissionError, match='attachment_unavailable'):
        f.adapter.attachment(f.context)
    assert not any(hasattr(f.adapter, name) for name in ('submit', 'turnIn', 'patch', 'create', 'delete', 'impersonate'))


@pytest.mark.parametrize('case', ['ok', 'bytes', 'redirect', 'timeout'])
def test_transport_fixed_get_bounded_stream_and_no_redirect(monkeypatch, case):
    import sys
    from datetime import datetime, timezone
    from alphonse.agent_v2.integrations.classroom import ClassroomTransport
    calls = []
    class Failure(Exception): pass
    class Response:
        status_code = 302 if case == 'redirect' else 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_content(self, **kw): yield b'{"ok":true}'
    def get(url, **kw):
        calls.append((url, kw))
        if case == 'timeout': raise Failure('private')
        return Response()
    monkeypatch.setitem(sys.modules, 'requests', SimpleNamespace(get=get, RequestException=Failure))
    args = dict(params={}, token='synthetic', deadline=datetime.now(timezone.utc) + timedelta(seconds=10),
                max_bytes=1 if case == 'bytes' else 100)
    if case == 'ok':
        assert ClassroomTransport().get('courses/course1', **args) == ({'ok': True}, 11)
    else:
        with pytest.raises((CheckStopped, ClassroomHTTPError)):
            ClassroomTransport().get('courses/course1', **args)
    assert len(calls) == 1 and calls[0][0] == 'https://classroom.googleapis.com/v1/courses/course1'
    assert calls[0][1]['stream'] and not calls[0][1]['allow_redirects']

# Use the existing durable worker, event store, account/policy fixtures and DB.
from test_source_acquisition import acquisition
from test_account_grants import f
from test_project_checks import checks
from test_project_check_scheduling import scheduled


def worker_fixture(f, *pages):
    wire, oauth = Wire(*pages), OAuth()
    class FixtureAdapter(ClassroomAdapter):
        def search(self, context, **kw):
            # Only the disposable provider identity is remapped. Real policy,
            # grant, cancellation and durable evidence gates remain in force.
            return super().search(replace(context, grant=replace(context.grant,
                provider_key='google', operation='classroom_read')), **dict(kw, query={'text': ''}))
    adapter = FixtureAdapter(oauth, transport=wire, clock=lambda: f.now)
    f.acquisition.adapters[('synthetic', 'read')] = adapter
    # Selecting by operation must precede the provider fallback.
    return adapter, wire


def worker_page(f, *, title='Assignment', more=False):
    value = item()
    value.update(courseId='folder-a', title=title, updateTime=f.now.isoformat())
    page = {'courseWork': [value]}
    if more: page['nextPageToken'] = 'page2'
    return [*role('folder-a'), page]


def test_durable_crash_resume_and_corrected_assignment_versions(acquisition, monkeypatch):
    f = acquisition
    adapter, wire = worker_fixture(f, *worker_page(f, more=True))
    accept = f.acquisition._accept
    def crash(*args, **kw):
        accept(*args, **kw)
        raise SystemExit('after commit')
    monkeypatch.setattr(f.acquisition, '_accept', crash)
    with pytest.raises(SystemExit): f.process()
    saved = f.acquisition.evidence(AccountSession('child'), f.run_id)
    assert len(saved) == 1 and saved[0]['source_id'] == 'courseWork:assignment1'
    monkeypatch.setattr(f.acquisition, '_accept', accept)
    f.now += timedelta(seconds=31)
    wire.responses = worker_page(f, title='Corrected assignment')
    assert f.process()['outcome'] == 'complete'
    assert wire.calls[-1][1]['params']['pageToken'] == 'page2'
    saved = f.acquisition.evidence(AccountSession('child'), f.run_id)
    assert len(saved) == 2 and saved[0]['version'] != saved[1]['version']
    assert f.dispatcher.get(f.run_id)['acquisition_checkpoint'] == {'source_index': 1}
    f.acquisition.publish_ready()
    with f.events._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM v2_event_history').fetchone()[0] == 1


def test_durable_replay_deduplicates_same_assignment_version(acquisition):
    f = acquisition
    pages = worker_page(f, more=True) + worker_page(f)
    adapter, wire = worker_fixture(f, *pages)
    assert f.process()['state'] == 'pending'
    assert f.process()['outcome'] == 'complete'
    assert len(f.acquisition.evidence(AccountSession('child'), f.run_id)) == 1


def test_permission_denied_worker_never_reports_no_match(acquisition):
    f = acquisition
    adapter, wire = worker_fixture(f, ClassroomHTTPError(403))
    assert f.process()['outcome'] == 'blocked'
    assert f.acquisition.evidence(AccountSession('child'), f.run_id) == []
