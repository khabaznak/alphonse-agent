"""Optional student-only Classroom GETs. No guardian fallback or impersonation.

Classroom has no change feed for coursework: changes are bounded list refreshes,
not a deletion detector or an assertion of a student's submission outcome.
"""
from datetime import datetime, timezone
from email.message import EmailMessage
import hashlib
import json
import re

from alphonse.agent_v2.integrations.acquisition import CursorExpired, SourcePage, SourceRecord
from alphonse.agent_v2.integrations.oauth import capability_scopes
from alphonse.agent_v2.services.project_check_worker import CheckStopped

KINDS = {'classroom_read': 'courseWork', 'classroom_announcements_read': 'announcements'}


def _id(value):
    # Canonical provider IDs only; aliases, user IDs and URLs are not inputs.
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,256}', value):
        raise ValueError('classroom_id_invalid')
    return value


class ClassroomHTTPError(Exception):
    def __init__(self, status):
        self.status = status
        super().__init__('classroom_http_error')


class ClassroomTransport:
    def get(self, path, *, params, token, deadline, max_bytes):
        import requests
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            raise CheckStopped('partial', 'classroom_deadline')
        try:
            with requests.get('https://classroom.googleapis.com/v1/' + path,
                    params=params, headers={'Authorization': 'Bearer ' + token},
                    timeout=min(20, remaining), stream=True, allow_redirects=False) as response:
                if response.status_code != 200:
                    raise ClassroomHTTPError(response.status_code)
                body = bytearray()
                for chunk in response.iter_content(chunk_size=8192):
                    if datetime.now(timezone.utc) >= deadline:
                        raise CheckStopped('partial', 'classroom_deadline')
                    if len(body) + len(chunk) > max_bytes:
                        raise CheckStopped('partial', 'classroom_bytes_budget')
                    body.extend(chunk)
                return json.loads(body), len(body)
        except requests.RequestException:
            raise CheckStopped('partial', 'classroom_unavailable') from None
        except (ValueError, UnicodeError):
            raise CheckStopped('partial', 'classroom_response_invalid') from None


class ClassroomAdapter:
    version = 'classroom-student-readonly-1'

    def __init__(self, oauth, *, transport=None, clock=None):
        self.oauth = oauth
        self.transport = transport or ClassroomTransport()
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def _authorize(self, context):
        context.guard()
        if self.clock() >= context.deadline:
            raise CheckStopped('partial', 'classroom_deadline')
        h = context.grant
        if h.provider_key != 'google' or h.operation not in KINDS or not h.resources:
            raise PermissionError('classroom_scope_denied')
        for resource in h.resources:
            _id(resource)
        oauth = self.oauth(context.session) if callable(self.oauth) else self.oauth
        args = dict(connection_id=h.connection_id, grant_id=h.grant_id,
            operation=h.operation, resources=list(h.resources), recipient_user_ids=list(h.recipient_user_ids),
            required_scopes=capability_scopes('google', h.operation), project_id=context.project_id,
            expected_grant_revision=h.grant_revision)
        oauth.accounts.authorize(context.session, **args)
        return oauth, args

    def _get(self, context, path, params, budget, *, continuation=False):
        oauth, args = self._authorize(context)
        if budget[0] <= 0:
            raise CheckStopped('partial', 'classroom_bytes_budget')
        token = oauth.token(context.session, authorization=args)
        try:
            value, size = self.transport.get(path, params=params, token=token,
                deadline=context.deadline, max_bytes=budget[0])
        except ClassroomHTTPError as exc:
            if exc.status == 400 and continuation:
                raise CursorExpired() from None
            code = {401: 'classroom_reauth_required', 403: 'classroom_permission_denied',
                    404: 'classroom_unavailable'}.get(exc.status, 'classroom_unavailable')
            raise CheckStopped('blocked' if exc.status in {401, 403, 404} else 'partial', code) from None
        if type(size) is not int or not 0 <= size <= budget[0] or not isinstance(value, dict):
            raise CheckStopped('partial', 'classroom_response_budget')
        budget[0] -= size
        context.on_bytes(size)
        self._authorize(context)  # Revoke, pause and cancellation fence returned evidence.
        return value

    def _course(self, context, resource, budget):
        if resource not in context.grant.resources:
            raise PermissionError('classroom_resource_denied')
        path = 'courses/' + _id(resource)
        course = self._get(context, path, {'fields': 'id,courseState'}, budget)
        if course.get('id') != resource or course.get('courseState') != 'ACTIVE':
            raise CheckStopped('blocked', 'classroom_course_unavailable')
        try:
            student = self._get(context, path + '/students/me', {'fields': 'courseId,userId'}, budget)
        except CheckStopped as exc:
            if exc.code == 'classroom_unavailable':
                raise CheckStopped('blocked', 'classroom_student_role_required') from None
            raise
        if student.get('courseId') != resource or not student.get('userId'):
            raise CheckStopped('blocked', 'classroom_student_role_required')
        return path

    def capabilities(self, context):
        self._authorize(context)
        return ('search', 'read', 'changes')

    def health(self, context):
        self._authorize(context)
        budget = [context.limits.bytes]
        for resource in context.grant.resources:
            self._course(context, resource, budget)
        return {'provider': 'google', 'status': 'healthy', 'role': 'student'}

    def _key(self, context, query):
        if query != {'text': ''}:
            raise ValueError('classroom_query_invalid')
        h = context.grant
        return hashlib.sha256(json.dumps([self.version, h.connection_id, h.grant_id,
            h.grant_revision, h.resources, h.operation, query], sort_keys=True).encode()).hexdigest()

    def _cursor(self, context, query, cursor):
        try:
            state = json.loads(cursor)
            if (state['key'] != self._key(context, query) or type(state['index']) is not int
                    or not 0 <= state['index'] < len(context.grant.resources)
                    or type(state['done']) is not bool or not 1 <= state['size'] <= 100
                    or (state['page'] is not None and not isinstance(state['page'], str))):
                raise ValueError()
            if datetime.fromisoformat(state['since']).tzinfo is None:
                raise ValueError()
            return state
        except (ValueError, TypeError, KeyError):
            raise CursorExpired() from None

    def _record(self, context, resource, value):
        kind = KINDS[context.grant.operation]
        if value.get('courseId') != resource or value.get('state') != 'PUBLISHED':
            raise CheckStopped('partial', 'classroom_content_outside_scope')
        provider_id = _id(value.get('id'))
        # Explicit nulls keep missing due dates/state/weight unknown. Points are points.
        data = {k: value.get(k) for k in ('id', 'courseId', 'title', 'description', 'text',
            'creationTime', 'updateTime', 'dueDate', 'dueTime', 'maxPoints', 'workType')}
        data.update(provider='google_classroom', kind=kind, submission_state=None,
                    grading_weight=None, source_timestamp_known=bool(value.get('updateTime') or value.get('creationTime')))
        raw = json.dumps(data, sort_keys=True, ensure_ascii=False)
        version = hashlib.sha256(raw.encode()).hexdigest()
        timestamp = value.get('updateTime') or value.get('creationTime') or self.clock().isoformat()
        msg = EmailMessage()
        msg.set_content(raw)
        # URL is a convenience pointer only, never fetched or used as provenance.
        return SourceRecord(resource, kind + ':' + provider_id, version, timestamp, msg.as_bytes())

    def search(self, context, *, query, since, cursor=None):
        self._authorize(context)
        if context.limits.pages <= 0 or context.limits.items <= 0:
            raise CheckStopped('partial', 'classroom_page_budget')
        key = self._key(context, query)
        state = self._cursor(context, query, cursor) if cursor else dict(key=key, index=0,
            since=since.isoformat(), page=None, done=False, size=min(50, context.limits.items))
        if state['done']:
            raise CursorExpired()
        # A continuation must keep the provider request identical despite remaining limits.
        if context.limits.items < state['size']:
            raise CheckStopped('partial', 'classroom_item_budget')
        budget = [context.limits.bytes]
        resource = context.grant.resources[state['index']]
        path = self._course(context, resource, budget)
        kind = KINDS[context.grant.operation]
        params = {'pageSize': state['size'],
                  'courseWorkStates' if kind == 'courseWork' else 'announcementStates': 'PUBLISHED'}
        if state['page']:
            params['pageToken'] = state['page']
        value = self._get(context, path + '/' + kind, params, budget, continuation=bool(state['page']))
        items = value.get(kind, [])
        if not isinstance(items, list) or len(items) > state['size']:
            raise CheckStopped('partial', 'classroom_item_budget')
        records = []
        lower = datetime.fromisoformat(state['since'])
        for item in items:
            record = self._record(context, resource, item)
            if datetime.fromisoformat(record.source_timestamp.replace('Z', '+00:00')) >= lower:
                records.append(record)
        state['page'] = value.get('nextPageToken') or None
        if state['page'] is not None and not isinstance(state['page'], str):
            raise CheckStopped('partial', 'classroom_response_invalid')
        if not state['page']:
            state['index'] += 1
        complete = state['index'] == len(context.grant.resources)
        coverage = {r: ('complete' if i < state['index'] else 'pending')
                    for i, r in enumerate(context.grant.resources)}
        if complete:
            state.update(done=True, index=0)
        return SourcePage(tuple(records), json.dumps(state, sort_keys=True), complete, True,
                          context.limits.bytes - budget[0], coverage)

    def changes(self, context, *, query, cursor):
        self._authorize(context)
        state = self._cursor(context, query, cursor)
        if state['done']:
            state.update(done=False, index=0, page=None)
        return self.search(context, query=query, since=datetime.fromisoformat(state['since']),
                           cursor=json.dumps(state))

    def read(self, context, *, resource, source_id):
        self._authorize(context)
        kind = KINDS[context.grant.operation]
        prefix, sep, provider_id = source_id.partition(':')
        if not sep or prefix != kind:
            raise PermissionError('classroom_source_denied')
        _id(provider_id)
        budget = [context.limits.bytes]
        path = self._course(context, resource, budget)
        value = self._get(context, path + '/' + kind + '/' + provider_id, {}, budget)
        if value.get('id') != provider_id:
            raise PermissionError('classroom_identity_mismatch')
        return self._record(context, resource, value)

    def attachment(self, context, **kwargs):
        self._authorize(context)
        raise PermissionError('classroom_attachment_unavailable')
