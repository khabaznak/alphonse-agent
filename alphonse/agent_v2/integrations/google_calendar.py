"""Managed read/sync support for the shipped family-calendar skill.

Only fixed Google GET endpoints. No writes, invitations, watch registrations or
implicit account selection. Provider payloads are untrusted source evidence.
"""
from __future__ import annotations

from datetime import datetime, timezone
from email.message import EmailMessage
import hashlib
import json
from urllib.parse import quote
from zoneinfo import ZoneInfo

from alphonse.agent_v2.integrations.acquisition import CursorExpired, SourcePage, SourceRecord
from alphonse.agent_v2.integrations.oauth import SCOPES
from alphonse.agent_v2.services.project_check_worker import CheckStopped


class CalendarHTTPError(Exception):
    def __init__(self, status):
        self.status = status
        super().__init__('calendar_http_error')


class CalendarTransport:
    def get(self, path, *, params, token, deadline, max_bytes):
        import requests
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            raise CheckStopped('partial', 'calendar_deadline')
        try:
            with requests.get('https://www.googleapis.com/calendar/v3/' + path,
                    params=params, headers={'Authorization': 'Bearer ' + token},
                    timeout=min(20, remaining), stream=True, allow_redirects=False) as response:
                if response.status_code != 200:
                    raise CalendarHTTPError(response.status_code)
                body = bytearray()
                for chunk in response.iter_content(chunk_size=8192):
                    if datetime.now(timezone.utc) >= deadline or len(body) + len(chunk) > max_bytes:
                        raise CheckStopped('partial', 'calendar_transport_budget')
                    body.extend(chunk)
                return json.loads(body), len(body)
        except requests.RequestException:
            raise CheckStopped('partial', 'calendar_transport_unavailable') from None
        except (ValueError, UnicodeError):
            raise CheckStopped('partial', 'calendar_response_invalid') from None


def _id(value):
    if not isinstance(value, str) or not value or value in {'.', '..'} or len(value) > 4096 or any(ord(c) < 32 for c in value):
        raise ValueError('calendar_id_invalid')
    # Never interpret the mutable "primary" alias as an exact calendar selection.
    if value == 'primary':
        raise PermissionError('calendar_exact_id_required')
    return quote(value, safe='')


def _instant(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('calendar_offset_required')
    return dt


class CalendarAdapter:
    version = 'family-calendar-readonly-1'

    def __init__(self, oauth, *, transport=None, clock=None):
        self.oauth = oauth
        self.transport = transport or CalendarTransport()
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def _authorize(self, context, operation='calendar_read'):
        context.guard()
        if self.clock() >= context.deadline:
            raise CheckStopped('partial', 'calendar_deadline')
        h = context.grant
        if h.provider_key != 'google' or h.operation != operation or not h.resources:
            raise PermissionError('calendar_scope_denied')
        for resource in h.resources:
            _id(resource)
        if operation == 'calendar_discover' and h.resources != ('calendar:list',):
            raise PermissionError('calendar_discovery_scope_denied')
        oauth = self.oauth(context.session) if callable(self.oauth) else self.oauth
        authorization = dict(connection_id=h.connection_id, grant_id=h.grant_id,
            operation=operation, resources=list(h.resources), recipient_user_ids=list(h.recipient_user_ids),
            required_scopes=[SCOPES['google'][operation]], project_id=context.project_id,
            expected_grant_revision=h.grant_revision)
        oauth.accounts.authorize(context.session, **authorization)
        return oauth, authorization

    def _get(self, context, path, params, budget, operation='calendar_read'):
        oauth, authorization = self._authorize(context, operation)
        if budget[0] <= 0:
            raise CheckStopped('partial', 'calendar_bytes_budget')
        token = oauth.token(context.session, authorization=authorization)
        try:
            value, size = self.transport.get(path, params=params, token=token,
                deadline=context.deadline, max_bytes=budget[0])
        except CalendarHTTPError as exc:
            if exc.status in {401, 403, 404}:
                raise CheckStopped('blocked', 'calendar_unavailable_or_revoked') from None
            if exc.status == 410:
                raise CursorExpired() from None
            if exc.status == 429 or exc.status >= 500:
                raise CheckStopped('partial', 'calendar_throttled_or_unavailable') from None
            raise CheckStopped('partial', 'calendar_request_failed') from None
        if type(size) is not int or not 0 <= size <= budget[0] or not isinstance(value, dict):
            raise CheckStopped('partial', 'calendar_response_budget')
        budget[0] -= size
        context.on_bytes(size)
        self._authorize(context, operation)  # Fence revocation/cancellation during I/O.
        return value

    def health(self, context):
        # Check each exact calendar, never an account-wide discovery as a shortcut.
        budget = [context.limits.bytes]
        for resource in context.grant.resources:
            self._get(context, 'calendars/' + _id(resource) + '/events',
                      {'maxResults': 1, 'timeMin': self.clock().isoformat(),
                       'timeMax': context.deadline.isoformat()}, budget)
        return {'provider': 'google', 'status': 'healthy'}

    def capabilities(self, context):
        self._authorize(context)
        return ('search', 'read', 'changes')

    def discover(self, context, *, cursor=None):
        """Explicit setup only; its independent list grant never authorizes events."""
        self._authorize(context, 'calendar_discover')
        self._limits(context)
        params = {'maxResults': min(250, context.limits.items)}
        if cursor:
            params['pageToken'] = self._state(context, {}, cursor)['page']
        value = self._get(context, 'users/me/calendarList', params, [context.limits.bytes], 'calendar_discover')
        items = value.get('items', [])
        if not isinstance(items, list) or len(items) > context.limits.items:
            raise CheckStopped('partial', 'calendar_item_budget')
        # Discovery is never a binding or automatic selection, including primary.
        return {'calendars': [{k: item[k] for k in ('id', 'summary', 'timeZone', 'accessRole') if k in item}
                              for item in items],
                'cursor': json.dumps({'key': self._key(context, {}), 'page': value['nextPageToken']})
                          if value.get('nextPageToken') else None}

    def _key(self, context, query):
        h = context.grant
        return hashlib.sha256(json.dumps([self.version, context.session.actor_user_id, context.project_id,
            h.connection_id, h.binding_id, h.grant_id, h.grant_revision, h.operation, h.resources, query],
            sort_keys=True).encode()).hexdigest()

    def _state(self, context, query, cursor):
        try:
            state = json.loads(cursor)
            if state['key'] != self._key(context, query):
                raise ValueError()
            return state
        except (ValueError, KeyError, TypeError):
            raise CursorExpired() from None

    def _limits(self, context):
        if context.limits.pages <= 0 or context.limits.items <= 0:
            raise CheckStopped('partial', 'calendar_page_budget')

    def _query(self, query):
        if (set(query) != {'text', 'time_min', 'time_max'} or not isinstance(query['text'], str)
                or len(query['text']) > 4096
                or not 0 < (_instant(query['time_max']) - _instant(query['time_min'])).total_seconds() <= 366 * 86400):
            raise ValueError('calendar_window_required')

    def _record(self, resource, event, *, zone='UTC'):
        source_id = event['id']
        _id(source_id)
        deleted = event.get('status') == 'cancelled'
        # Sparse cancelled instances may omit ETag/updated. Hash the tombstone,
        # retaining originalStartTime and recurringEventId in its version identity.
        version = event.get('etag') or hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()
        timestamp = event.get('updated') or self.clock().isoformat()
        _instant(timestamp)
        message = EmailMessage()
        fields = ('id', 'iCalUID', 'summary', 'description', 'location', 'start', 'end', 'recurrence',
                  'recurringEventId', 'originalStartTime', 'status', 'visibility', 'sequence', 'updated', 'htmlLink')
        data = {k: event[k] for k in fields if k in event}
        data.update(calendar_id=resource, calendar_timezone=zone,
                    evidence_kind='calendar_presence', assignment_completion_verified=False)
        message.set_content(json.dumps(data, ensure_ascii=False, sort_keys=True))
        return SourceRecord(resource, source_id, version, timestamp, message.as_bytes(), deleted=deleted)

    def _in_window(self, event, query, zone):
        def point(value):
            if 'dateTime' in value:
                return _instant(value['dateTime'])
            # All-day end is an exclusive local DATE, never a UTC midnight.
            return datetime.fromisoformat(value['date']).replace(tzinfo=ZoneInfo(zone))
        return point(event['end']) > _instant(query['time_min']) and point(event['start']) < _instant(query['time_max'])

    def _page(self, context, query, cursor, *, changes):
        self._authorize(context)
        self._limits(context)
        self._query(query)
        if context.calendar_window != (query['time_min'], query['time_max']):
            raise PermissionError('calendar_window_mismatch')
        state = self._state(context, query, cursor) if cursor else dict(
            key=self._key(context, query), index=0, page=None, tokens={}, mode='scan',
            page_size=min(250, context.limits.items))
        resource = context.grant.resources[state['index']]
        params = dict(maxResults=state['page_size'], singleEvents='true', showDeleted='true')
        delta = changes and state['mode'] == 'sync'
        if delta:
            params['syncToken'] = state['tokens'][resource]
        else:
            params.update(timeMin=query['time_min'], timeMax=query['time_max'])
        # q is deliberately not sent: a renamed event must still invalidate old
        # evidence. Text relevance is judged downstream over bounded event data.
        if state['page']:
            params['pageToken'] = state['page']
        budget = [context.limits.bytes]
        value = self._get(context, 'calendars/' + _id(resource) + '/events', params, budget)
        items = value.get('items', [])
        if not isinstance(items, list) or len(items) > context.limits.items:
            raise CheckStopped('partial', 'calendar_item_budget')
        zone = value.get('timeZone', 'UTC')
        records, covered = [], value.get('accessRole') != 'freeBusyReader'
        for event in items:
            if event.get('status') == 'cancelled':
                records.append(self._record(resource, event, zone=zone))
                continue
            if not event.get('start') or not event.get('end'):
                covered = False  # Free/busy-only or otherwise redacted event.
                continue
            if event.get('visibility') == 'private' and not event.get('summary'):
                covered = False
            if self._in_window(event, query, zone):
                records.append(self._record(resource, event, zone=zone))
            elif delta:
                # Moved out of the authorized event window: redact stale cache,
                # but do not retain its new out-of-window contents.
                records.append(self._record(resource, {k: v for k, v in dict(event, status='cancelled').items()
                    if k in {'id', 'etag', 'updated', 'status', 'recurringEventId', 'originalStartTime'}}, zone=zone))
        state['page'] = value.get('nextPageToken')
        if not state['page']:
            token = value.get('nextSyncToken')
            if not isinstance(token, str) or not token:
                raise CheckStopped('partial', 'calendar_sync_token_missing')
            state['tokens'][resource] = token
            state['index'] += 1
        complete = state['index'] == len(context.grant.resources)
        if complete:
            state.update(index=0, mode='sync')
        return SourcePage(tuple(records), json.dumps(state, sort_keys=True), complete, covered,
                          context.limits.bytes - budget[0])

    def search(self, context, *, query, since, cursor=None):
        # Calendar event time is the explicitly authorized window, not updated
        # time: an old recurring series can have a future occurrence.
        return self._page(context, query, cursor, changes=False)

    def changes(self, context, *, query, cursor):
        return self._page(context, query, cursor, changes=True)

    def read(self, context, *, resource, source_id):
        if resource not in context.grant.resources:
            raise PermissionError('calendar_resource_denied')
        if context.calendar_window is None:
            raise PermissionError('calendar_window_required')
        budget = [context.limits.bytes]
        value = self._get(context, 'calendars/' + _id(resource) + '/events/' + _id(source_id), {}, budget)
        if value.get('id') != source_id:
            raise ValueError('calendar_identity_mismatch')
        query = dict(zip(('time_min', 'time_max'), context.calendar_window))
        zone = value.get('start', {}).get('timeZone', 'UTC')
        if 'date' in value.get('start', {}):
            metadata = self._get(context, 'calendars/' + _id(resource) + '/events',
                dict(timeMin=query['time_min'], timeMax=query['time_max'], maxResults=1), budget)
            zone = metadata['timeZone']
        if value.get('status') != 'cancelled' and not self._in_window(value, query, zone):
            raise PermissionError('calendar_event_outside_window')
        return self._record(resource, value, zone=zone)

    def attachment(self, context, **kwargs):
        self._authorize(context)
        raise PermissionError('calendar_attachment_unsupported')
