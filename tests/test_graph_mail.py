"""Graph adapter conformance with disposable OAuth and transport; no network."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import base64
import json

import pytest

from test_oauth_lifecycle import f, connect, revision
from alphonse.agent_v2.accounts import AccountHandle
from alphonse.agent_v2.integrations.acquisition import ReadContext, ReadLimits, CursorExpired, ReadRetryAfter
from alphonse.agent_v2.integrations.content import normalize_mime
from alphonse.agent_v2.integrations.graph_mail import GraphMailAdapter, GraphReadError, ROOT, GraphHTTP
from alphonse.agent_v2.integrations.oauth import SCOPES
from alphonse.agent_v2.services.project_check_worker import CheckStopped


class Transport:
    def __init__(self, *pages):
        self.pages, self.calls = list(pages), []
        self.after = lambda: None

    def get(self, url, **kw):
        self.calls.append((url, kw))
        result = self.pages.pop(0)
        self.after()
        if isinstance(result, Exception):
            raise result
        return result


def response(data, status=200, headers=None):
    return status, headers or {}, json.dumps(data).encode()


def link(folder='folder-a', token='delta', kind='delta'):
    return ROOT + folder + '/messages/delta?$' + kind + 'token=' + token


def page(*items, folder='folder-a', token='delta', next=False):
    return response({'value': list(items), '@odata.nextLink' if next else '@odata.deltaLink':
                     link(folder, token, 'skip' if next else 'delta')})


def message(folder='folder-a', **kw):
    return dict(id='immutable-1', parentFolderId=folder, changeKey='v1',
        lastModifiedDateTime='2026-11-02T12:00:00Z', receivedDateTime='2026-11-01T10:00:00Z',
        subject='School notice', body={'contentType': 'html', 'content':
        '<p>Assignment due Friday</p><blockquote>Earlier notice</blockquote><script>steal()</script><img src="https://evil.invalid/t">'},
        **kw)


@pytest.fixture
def graph(f):
    connect(f, 'microsoft')
    f.accounts.save_grant(f.owner, grant_id='mail', connection_id='one', expected_revision=0,
        beneficiary_user_id='owner', operations=['mail_read'], resources=['folder-a', 'folder-b'],
        recipient_user_ids=[], expires_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat())
    f.context = ReadContext(f.owner, AccountHandle('one', 'binding', 'mail', revision(f), 1,
        'microsoft', 'mail_read', ('folder-a', 'folder-b'), ()), datetime.now(timezone.utc) + timedelta(minutes=1),
        ReadLimits(5, 100, 1000000))
    f.transport = Transport()
    f.adapter = GraphMailAdapter(f.service, transport=f.transport)
    f.since = datetime.now(timezone.utc) - timedelta(days=30)
    f.fetch = lambda cursor=None: f.adapter.search(f.context, query={'text': 'School'}, since=f.since, cursor=cursor)
    return f


def test_folder_pagination_delta_opaque_links_immutable_moved_school_mail(graph):
    f = graph
    f.transport.pages = [page(next=True, token='opaque%2Fskip'), page(token='inbox-delta'),
        page(message('folder-b'), folder='folder-b', token='rules-delta'), page(token='new-inbox'),
        page(message('folder-b'), folder='folder-b', token='new-rules')]
    a = f.fetch()
    assert not a.complete
    b = f.fetch(a.cursor)
    assert not b.complete
    c = f.fetch(b.cursor)
    assert c.complete and c.coverage_complete and c.records[0].resource == 'folder-b'
    assert c.records[0].source_id == 'immutable-1'
    state = json.loads(c.cursor)
    assert state['links'] == {'folder-a': link(token='inbox-delta'), 'folder-b': link('folder-b', 'rules-delta')}
    assert f.transport.calls[1][0] == link(token='opaque%2Fskip', kind='skip')
    d = f.adapter.changes(f.context, query={'text': 'School'}, cursor=c.cursor)
    assert f.transport.calls[-1][0] == link(token='inbox-delta')
    e = f.adapter.changes(f.context, query={'text': 'School'}, cursor=d.cursor)
    assert e.complete and e.records[0].source_id == c.records[0].source_id
    assert all('IdType="ImmutableId"' in call[1]['headers']['Prefer'] for call in f.transport.calls)
    assert all(call[1]['deadline'] == f.context.deadline for call in f.transport.calls)
    text = normalize_mime(c.records[0].mime, max_bytes=100000)['parts'][0]['text']
    assert 'Friday' in text and 'Earlier notice' in text and 'steal' not in text and 'evil.invalid' not in text
    assert set(f.service.clients['microsoft'].scopes) == {'openid', 'offline_access', SCOPES['microsoft']['mail_read']}
    assert f.adapter.capabilities(f.context) == ('mail_read',)


def test_inbox_only_does_not_enumerate_or_claim_other_folders(graph):
    f = graph
    f.context = replace(f.context, grant=replace(f.context.grant, resources=('folder-a',)))
    f.transport.pages = [page()]
    result = f.fetch()
    assert result.complete and list(json.loads(result.cursor)['links']) == ['folder-a']
    assert len(f.transport.calls) == 1
    assert f.adapter.health(f.context)['coverage'] == 'selected_folders_only'


def test_partial_folder_access_keeps_successful_evidence_and_retries_denied_folder(graph):
    f = graph
    f.transport.pages = [response({'error': {'code': 'ErrorAccessDenied'}}, 403),
                         page(message('folder-b'), folder='folder-b'), page()]
    a = f.fetch()
    b = f.fetch(a.cursor)
    assert b.complete and not b.coverage_complete and b.records
    assert json.loads(b.cursor)['links']['folder-a'] is None
    f.adapter.changes(f.context, query={'text': 'School'}, cursor=b.cursor)
    assert '$filter=' not in f.transport.calls[-1][0]  # encoded filter is a new bounded baseline
    assert '%24filter=' in f.transport.calls[-1][0]


@pytest.mark.parametrize('status,code', [(410, 'Gone'), (400, 'SyncStateNotFound'), (400, 'ErrorInvalidSyncStateData')])
def test_expiry_requests_shared_bounded_resync(graph, status, code):
    graph.transport.pages = [response({'error': {'code': code}}, status)]
    with pytest.raises(CursorExpired): graph.fetch()


@pytest.mark.parametrize('status,retry,delay', [(429, '120', 120), (503, '5', 5), (429, 'invalid', 5)])
def test_throttle_has_no_inline_retry_and_no_private_error(graph, status, retry, delay):
    graph.transport.pages = [response({'error': {'message': 'private provider detail'}}, status, {'Retry-After': retry})]
    with pytest.raises(ReadRetryAfter) as exc: graph.fetch()
    assert exc.value.seconds == delay and str(exc.value) == 'source_retry_after'
    assert len(graph.transport.calls) == 1


def test_deleted_record_and_denied_attachment(graph):
    f = graph
    f.transport.pages = [page({'id': 'immutable-1', '@removed': {'reason': 'deleted'}}),
        response({'error': {'code': 'ErrorItemNotFound'}}, 404)]
    result = f.fetch()
    assert result.records[0].deleted and result.records[0].mime == b''
    with pytest.raises(CheckStopped, match='graph_attachment_unavailable'):
        f.adapter.attachment(f.context, resource='folder-a', source_id='immutable-1', attachment_id='a', max_bytes=100)


def test_read_attachment_and_transport_bytes(graph):
    f = graph
    f.transport.pages = [response(message()), response({'@odata.type': '#microsoft.graph.fileAttachment',
        'id': 'a', 'contentType': 'text/plain', 'contentBytes': base64.b64encode(b'notice').decode()})]
    sizes = []
    f.context = replace(f.context, on_bytes=sizes.append)
    record = f.adapter.read(f.context, resource='folder-a', source_id='immutable-1')
    assert record.source_id == 'immutable-1' and sizes[0] > 0
    result = f.adapter.attachment(f.context, resource='folder-a', source_id='immutable-1', attachment_id='a', max_bytes=100)
    assert result == b'notice' and sizes[-1] > len(result)


@pytest.mark.parametrize('data', [
    {'@odata.type': '#microsoft.graph.referenceAttachment', 'sourceUrl': 'https://evil.invalid'},
    {'@odata.type': '#microsoft.graph.itemAttachment'},
    {'@odata.type': '#microsoft.graph.fileAttachment', 'id': 'a', 'contentType': 'text/plain', 'contentBytes': '!!!!'},
])
def test_unsupported_attachment_is_honest_and_never_follows_links(graph, data):
    graph.transport.pages = [response(data)]
    with pytest.raises((CheckStopped, GraphReadError)):
        graph.adapter.attachment(graph.context, resource='folder-a', source_id='m', attachment_id='a', max_bytes=100)
    assert len(graph.transport.calls) == 1


@pytest.mark.parametrize('mutation', ['query', 'actor', 'connection', 'folder', 'revision', 'project'])
def test_cursor_cannot_cross_scope(graph, mutation):
    f = graph
    f.transport.pages = [page()]
    cursor = f.fetch().cursor
    context, query = f.context, {'text': 'School'}
    if mutation == 'query': query = {'text': 'new query'}
    if mutation == 'actor': context = replace(context, session=f.other)
    if mutation == 'connection': context = replace(context, grant=replace(context.grant, connection_id='two'))
    if mutation == 'folder': context = replace(context, grant=replace(context.grant, resources=('folder-b',)))
    if mutation == 'revision': context = replace(context, grant=replace(context.grant, connection_revision=999))
    if mutation == 'project': context = replace(context, project_id='other-project')
    with pytest.raises((GraphReadError, PermissionError)):
        f.adapter.changes(context, query=query, cursor=cursor)
    assert len(f.transport.calls) == 1


@pytest.mark.parametrize('url', ['https://evil.invalid/token', 'http://graph.microsoft.com/v1.0/me/mailFolders/folder-a/messages/delta?$deltatoken=a',
    link('folder-b'), 'https://graph.microsoft.com@evil.invalid/x',
    'https://graph.microsoft.com/v1.0/users/other/messages/delta?$deltatoken=a'])
def test_foreign_links_never_receive_credentials(graph, url):
    graph.transport.pages = [response({'value': [], '@odata.nextLink': url})]
    with pytest.raises(GraphReadError, match='graph_link_invalid'): graph.fetch()
    assert len(graph.transport.calls) == 1


def test_revocation_during_transport_cannot_release_evidence(graph):
    f = graph
    f.transport.pages = [page(message())]
    f.transport.after = lambda: f.accounts.revoke_grant(f.owner, grant_id='mail', expected_revision=1)
    with pytest.raises(PermissionError): f.fetch()
    with pytest.raises(PermissionError): f.fetch()
    assert len(f.transport.calls) == 1


@pytest.mark.parametrize('violation', ['deadline', 'bytes', 'items', 'folder', 'traversal'])
def test_deadline_budget_folder_and_path_guards(graph, violation):
    f = graph
    f.transport.pages = [page(message('folder-b') if violation == 'folder' else message())]
    if violation == 'deadline': f.context = replace(f.context, deadline=datetime.now(timezone.utc) - timedelta(seconds=1))
    if violation == 'bytes': f.context = replace(f.context, limits=ReadLimits(1, 1, 1))
    if violation == 'items': f.context = replace(f.context, limits=ReadLimits(1, 0, 100000))
    with pytest.raises((CheckStopped, PermissionError, GraphReadError)):
        if violation == 'traversal':
            f.adapter.read(f.context, resource='folder-a', source_id='..')
        else: f.fetch()


@pytest.mark.parametrize('identity', ['personal-consumer', 'work-school-tenant'])
def test_personal_and_work_school_delegated_auth_scenarios(f, identity):
    f.service.clients['microsoft'].identity = identity
    health = connect(f, 'microsoft')
    assert health['status'] == 'connected' and health['capabilities'] == ['mail_read']
    assert f.store.get('one').connection['provider_subject'] == identity
    assert not any('Write' in scope or 'Send' in scope for scope in health['granted_scopes'])


def test_real_http_boundary_is_get_only_bounded_no_redirects(monkeypatch, caplog):
    import requests
    calls = []
    class Response:
        status_code = 302
        headers = {'Location': 'https://evil.invalid'}
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def iter_content(self, **kw): yield b'{}'
    class Session:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, **kw):
            import logging
            logging.getLogger('urllib3.connectionpool').debug('GET %s', url)
            calls.append((url, kw, self.trust_env))
            return Response()
    monkeypatch.setattr(requests, 'Session', Session)
    import logging
    logging.getLogger('urllib3.connectionpool').disabled = False
    caplog.set_level(logging.DEBUG)
    result = GraphHTTP().get(link(), token='disposable', deadline=datetime.now(timezone.utc) + timedelta(seconds=10),
        max_bytes=100, headers={})
    assert 'deltatoken' not in caplog.text
    assert result[0] == 302 and len(calls) == 1 and calls[0][1]['allow_redirects'] is False
    assert calls[0][2] is False and calls[0][1]['headers']['Accept-Encoding'] == 'identity'
    with pytest.raises(CheckStopped):
        GraphHTTP().get(link(), token='disposable', deadline=datetime.now(timezone.utc) + timedelta(seconds=10), max_bytes=1, headers={})


def test_remote_revocation_marks_connection_reauth_required(graph):
    graph.transport.pages = [response({'error': {'code': 'InvalidAuthenticationToken'}}, 401)]
    with pytest.raises(CheckStopped, match='graph_reauth_required'): graph.fetch()
    assert graph.service.health(graph.owner, 'one')['status'] == 'reauth_required'
    with pytest.raises(PermissionError): graph.fetch()
    assert len(graph.transport.calls) == 1


def test_filtered_delta_limit_uncertainty_survives_later_rounds(graph):
    f = graph
    f.transport.pages = [page(next=True, token='next'), page(), page(folder='folder-b'), page()]
    state = json.loads(f.fetch().cursor)
    state['counts']['folder-a'] = 5000
    result = f.fetch(json.dumps(state))
    assert not result.coverage_complete
    result = f.fetch(result.cursor)
    assert result.complete and not result.coverage_complete
    result = f.adapter.changes(f.context, query={'text': 'School'}, cursor=result.cursor)
    assert not result.coverage_complete and result.resource_coverage['folder-a'] == 'incomplete'


def test_http_date_retry_after(graph):
    from email.utils import format_datetime
    f = graph
    now = datetime.now(timezone.utc).replace(microsecond=0)
    f.adapter.clock = lambda: now
    f.transport.pages = [response({}, 429, {'Retry-After': format_datetime(now + timedelta(seconds=90), usegmt=True)})]
    with pytest.raises(ReadRetryAfter) as exc: f.fetch()
    assert 90 <= exc.value.seconds <= 91


def test_cancellation_guard_fences_each_transport_use(graph):
    f = graph
    f.transport.pages = [page()]
    def stopped():
        raise CheckStopped('cancelled', 'cancelled')
    f.context = replace(f.context, guard=stopped)
    with pytest.raises(CheckStopped, match='cancelled'): f.fetch()
    assert not f.transport.calls
