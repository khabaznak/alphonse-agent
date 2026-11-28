"""Managed Calendar reads through Google’s official MCP server."""
from __future__ import annotations

from datetime import date, datetime, timezone
from email.message import EmailMessage
import hashlib
import json
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from alphonse.agent_v2.accounts import exact_calendar_id
from alphonse.agent_v2.integrations.acquisition import CursorExpired, SourcePage, SourceRecord
from alphonse.agent_v2.integrations.google_mcp import GoogleMcpClient, GoogleMcpError
from alphonse.agent_v2.integrations.oauth import SCOPES
from alphonse.agent_v2.services.project_check_worker import CheckStopped


_READABLE_ROLES = {'reader', 'writerWithoutPrivateAccess', 'writer', 'owner'}
_PRIVATE_DETAIL_ROLES = {'writer', 'owner'}
_EVENT_TYPES = ('DEFAULT', 'OUT_OF_OFFICE', 'FOCUS_TIME', 'WORKING_LOCATION', 'BIRTHDAY', 'FROM_GMAIL')
_OMITTED_FIELDS = ('omittedContent', 'omittedEvents', 'incomplete')
_EVIDENCE_FIELDS = ('id', 'iCalUID', 'summary', 'description', 'location', 'start', 'end', 'recurrence',
                    'recurringEventId', 'originalStartTime', 'status', 'visibility', 'sequence', 'updated', 'htmlLink')
_SUMMARY_FIELDS = ('summary', 'start', 'end', 'status', 'endTimeUnspecified')


def _instant(value):
    if not isinstance(value, str):
        raise ValueError('calendar_offset_required')
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('calendar_offset_required')
    return dt


def _event_id(value):
    if (not isinstance(value, str) or not 0 < len(value) <= 4096 or value != value.strip()
            or not value.isprintable() or value in {'.', '..'}):
        raise ValueError('calendar_event_id_invalid')
    return value


def _zone(value):
    try:
        return ZoneInfo(value)
    except (TypeError, ValueError, ZoneInfoNotFoundError):
        raise CheckStopped('partial', 'calendar_timezone_unavailable') from None


def _next_page(value):
    cursor = value.get('nextPageToken')
    if 'nextPageToken' in value and (not isinstance(cursor, str) or not 0 < len(cursor) <= 8192):
        raise CheckStopped('partial', 'calendar_page_token_invalid')
    return cursor


def _point(value, zone):
    if 'dateTime' in value and 'date' not in value:
        return _instant(value['dateTime'])
    if 'date' in value and 'dateTime' not in value:
        # All-day end is an exclusive local date.
        return datetime.combine(date.fromisoformat(value['date']), datetime.min.time(), zone)
    raise ValueError('calendar_event_time_invalid')


class CalendarAdapter:
    version = 'family-calendar-mcp-snapshot-1'

    def __init__(self, oauth, *, transport=None, clock=None):
        self.oauth = oauth
        self.transport = transport or GoogleMcpClient()
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def _authorize(self, context, operation='calendar_read'):
        context.guard()
        if self.clock() >= context.deadline:
            raise CheckStopped('partial', 'calendar_deadline')
        h = context.grant
        if h.provider_key != 'google' or h.operation != operation or not h.resources:
            raise PermissionError('calendar_scope_denied')
        if operation == 'calendar_discover':
            if h.resources != ('calendar:list',):
                raise PermissionError('calendar_discovery_scope_denied')
        elif not all(exact_calendar_id(resource) for resource in h.resources):
            raise PermissionError('calendar_exact_id_required')
        oauth = self.oauth(context.session) if callable(self.oauth) else self.oauth
        authorization = dict(connection_id=h.connection_id, grant_id=h.grant_id,
            operation=operation, resources=list(h.resources), recipient_user_ids=list(h.recipient_user_ids),
            required_scopes=[SCOPES['google'][operation]], project_id=context.project_id,
            expected_grant_revision=h.grant_revision)
        oauth.accounts.authorize(context.session, **authorization)
        return oauth, authorization

    def _call(self, context, tool, arguments, budget, operation='calendar_read'):
        oauth, authorization = self._authorize(context, operation)
        if budget[0] <= 0:
            raise CheckStopped('partial', 'calendar_bytes_budget')
        token = oauth.token(context.session, authorization=authorization)
        context.guard()
        try:
            value, size = self.transport.call('calendar', tool, arguments, token=token,
                deadline=context.deadline, max_bytes=budget[0], guard=context.guard)
        except GoogleMcpError as exc:
            if exc.reason == 'approval_required':
                raise CheckStopped('blocked', 'calendar_google_approval_required') from None
            if exc.reason == 'service_disabled':
                raise CheckStopped('blocked', 'calendar_api_disabled') from None
            if exc.status in {401, 403, 404}:
                raise CheckStopped('blocked', 'calendar_unavailable_or_revoked') from None
            if exc.status == 410 and arguments.get('pageToken'):
                raise CursorExpired() from None
            if exc.reason in {'deadline', 'bytes_budget'}:
                raise CheckStopped('partial', 'calendar_transport_budget') from None
            if exc.status == 429 or exc.status >= 500:
                raise CheckStopped('partial', 'calendar_throttled_or_unavailable') from None
            raise CheckStopped('partial', 'calendar_request_failed') from None
        if type(size) is not int or not 0 <= size <= budget[0] or not isinstance(value, dict):
            raise CheckStopped('partial', 'calendar_response_budget')
        budget[0] -= size
        context.on_bytes(size)
        self._authorize(context, operation)
        return value

    def _list(self, context, resource, window, budget, *, page_size, cursor=None):
        if resource not in context.grant.resources or not exact_calendar_id(resource):
            raise PermissionError('calendar_resource_denied')
        args = {'calendarId': resource, 'startTime': window[0], 'endTime': window[1],
                'pageSize': page_size, 'eventType': list(_EVENT_TYPES), 'orderBy': 'startTime'}
        if cursor is not None:
            args['pageToken'] = cursor
        return self._call(context, 'list_events', args, budget)

    def health(self, context):
        self._limits(context)
        if context.calendar_window is None:
            raise PermissionError('calendar_window_required')
        self._window(context.calendar_window)
        budget = [context.limits.bytes]
        for resource in context.grant.resources:
            value = self._list(context, resource, context.calendar_window, budget, page_size=1)
            if value.get('accessRole') not in _READABLE_ROLES:
                raise CheckStopped('blocked', 'calendar_unavailable_or_revoked')
        return {'provider': 'google', 'status': 'healthy'}

    def capabilities(self, context):
        self._authorize(context)
        return ('search', 'read')

    def list_summary(self, context):
        """Return a bounded projection for the private owner agenda."""
        self._limits(context)
        if len(context.grant.resources) != 1 or context.calendar_window is None:
            raise PermissionError('calendar_summary_scope_denied')
        window = self._window(context.calendar_window)
        value = self._list(context, context.grant.resources[0], context.calendar_window, [context.limits.bytes],
                           page_size=min(250, context.limits.items))
        zone, events, covered = self._visible_page(value, window, context.limits.items)
        token = _next_page(value)
        result = {'accessRole': value.get('accessRole'), 'timeZone': zone.key, 'coverage_complete': covered,
                  'items': [{k: item[k] for k in _SUMMARY_FIELDS if k in item} for item in events]}
        if token:
            result['nextPageToken'] = token
        return result

    def discover(self, context, *, cursor=None):
        """Explicit setup only. Discovery does not select or grant event access."""
        self._authorize(context, 'calendar_discover')
        self._limits(context)
        args = {'pageSize': min(250, context.limits.items)}
        state = self._state(context, {}, cursor) if cursor else None
        if state:
            args.update(pageToken=state['page'], pageSize=state['page_size'])
        value = self._call(context, 'list_calendars', args, [context.limits.bytes], 'calendar_discover')
        items = value.get('calendars', [])
        if not isinstance(items, list) or len(items) > context.limits.items:
            raise CheckStopped('partial', 'calendar_item_budget')
        token = _next_page(value)
        if state is not None and token == state['page']:
            raise CheckStopped('partial', 'calendar_page_token_stalled')
        return {'calendars': [{k: item[k] for k in ('id', 'summary', 'timeZone', 'accessRole') if k in item}
                              for item in items if isinstance(item, dict) and exact_calendar_id(item.get('id'))],
                'cursor': json.dumps({'key': self._key(context, {}), 'page': token, 'index': 0,
                                      'page_size': args['pageSize'], 'incomplete': []}) if token else None}

    def _key(self, context, query):
        h = context.grant
        return hashlib.sha256(json.dumps([self.version, context.session.actor_user_id, context.project_id,
            h.connection_id, h.connection_revision, h.binding_id, h.grant_id, h.grant_revision, h.operation,
            h.resources, query], sort_keys=True).encode()).hexdigest()

    def _state(self, context, query, cursor):
        try:
            state = json.loads(cursor)
            if (state['key'] != self._key(context, query) or type(state['index']) is not int
                    or not 0 <= state['index'] < len(context.grant.resources) or type(state['page_size']) is not int
                    or not 1 <= state['page_size'] <= 250 or not isinstance(state['incomplete'], list)
                    or any(type(index) is not int or not 0 <= index <= state['index'] for index in state['incomplete'])
                    or (state['page'] is not None and (not isinstance(state['page'], str)
                                                      or not 0 < len(state['page']) <= 8192))):
                raise ValueError()
            return state
        except (ValueError, KeyError, TypeError):
            raise CursorExpired() from None

    @staticmethod
    def _limits(context):
        if context.limits.pages <= 0 or context.limits.items <= 0:
            raise CheckStopped('partial', 'calendar_page_budget')

    @staticmethod
    def _window(bounds):
        start, end = map(_instant, bounds)
        if not 0 < (end - start).total_seconds() <= 366 * 86400:
            raise ValueError('calendar_window_required')
        return start, end

    @staticmethod
    def _visible_page(value, window, limit):
        events = value.get('events', [])
        if not isinstance(events, list) or len(events) > limit:
            raise CheckStopped('partial', 'calendar_item_budget')
        access = value.get('accessRole')
        if access == 'none':
            raise CheckStopped('blocked', 'calendar_unavailable_or_revoked')
        if access not in _READABLE_ROLES and access != 'freeBusyReader':
            raise CheckStopped('partial', 'calendar_access_role_unavailable')
        zone = _zone(value.get('timeZone'))
        # Google omits the repeated events field for an empty result. Accept that
        # default only for a terminal page with unredacted calendar access.
        event_page = 'events' in value or (access in _PRIVATE_DETAIL_ROLES and 'nextPageToken' not in value)
        covered = access in _READABLE_ROLES and event_page and not any(value.get(key) for key in _OMITTED_FIELDS)
        visible = []
        if access == 'freeBusyReader':
            return zone, visible, False
        for event in events:
            try:
                if (not isinstance(event, dict) or not isinstance(event.get('summary', ''), str)
                        or event.get('status', 'confirmed') not in ('confirmed', 'tentative', 'cancelled')
                        or event.get('recurrence') and not event.get('recurringEventId')
                        or event.get('visibility') == 'private'
                        and (access not in _PRIVATE_DETAIL_ROLES or not event.get('summary'))):
                    raise ValueError()
                start, end = (_point(event[k], zone) for k in ('start', 'end'))
                if start >= end or end <= window[0] or start >= window[1]:
                    raise ValueError()
                visible.append(event)
            except (AttributeError, KeyError, TypeError, ValueError):
                covered = False
        return zone, visible, covered

    def _record(self, resource, event, *, zone):
        source_id = _event_id(event.get('id'))
        deleted = event.get('status') == 'cancelled'
        timestamp = event.get('updated') or self.clock().isoformat()
        _instant(timestamp)
        message = EmailMessage()
        data = {k: event[k] for k in _EVIDENCE_FIELDS if k in event}
        data.update(calendar_id=resource, calendar_timezone=zone,
                    evidence_kind='calendar_presence', assignment_completion_verified=False)
        content = json.dumps(data, ensure_ascii=False, sort_keys=True)
        version = hashlib.sha256(content.encode()).hexdigest()
        message.set_content(content)
        return SourceRecord(resource, source_id, version, timestamp, message.as_bytes(), deleted=deleted)

    def search(self, context, *, query, since, cursor=None):
        self._authorize(context)
        self._limits(context)
        if set(query) != {'text', 'time_min', 'time_max'} or not isinstance(query['text'], str) or len(query['text']) > 4096:
            raise ValueError('calendar_window_required')
        bounds = (query['time_min'], query['time_max'])
        if context.calendar_window != bounds:
            raise PermissionError('calendar_window_mismatch')
        window = self._window(bounds)
        state = self._state(context, query, cursor) if cursor else dict(key=self._key(context, query),
            index=0, page=None, page_size=min(250, context.limits.items), incomplete=[])
        index = state['index']
        resource = context.grant.resources[index]
        budget = [context.limits.bytes]
        # Calendar evidence is driven by the authorized occurrence window rather
        # than a provider search filter or last-modified time.
        value = self._list(context, resource, bounds, budget, page_size=state['page_size'], cursor=state['page'])
        zone, events, covered = self._visible_page(value, window, context.limits.items)
        records = []
        for event in events:
            try:
                records.append(self._record(resource, event, zone=zone.key))
            except (TypeError, ValueError):
                covered = False
        if not covered and index not in state['incomplete']:
            state['incomplete'].append(index)
        token = _next_page(value)
        if token and token == state['page']:
            raise CheckStopped('partial', 'calendar_page_token_stalled')
        state.update(page=token, index=index if token else index + 1)
        complete = state['index'] == len(context.grant.resources)
        resource_coverage = {name: ('incomplete' if i in state['incomplete'] else
                                    'complete' if i < state['index'] else
                                    'in_progress' if i == index and token else 'pending')
                             for i, name in enumerate(context.grant.resources)}
        # The next occurrence starts a fresh bounded snapshot. An absent event
        # is not a deletion signal and no pseudo-delta cursor survives a scan.
        return SourcePage(tuple(records), None if complete else json.dumps(state, sort_keys=True),
                          complete, not state['incomplete'], context.limits.bytes - budget[0], resource_coverage)

    def changes(self, context, *, query, cursor):
        # MCP supplies no cross-occurrence change cursor.
        self._authorize(context)
        raise CursorExpired()

    def read(self, context, *, resource, source_id):
        if resource not in context.grant.resources or not exact_calendar_id(resource):
            raise PermissionError('calendar_resource_denied')
        if context.calendar_window is None:
            raise PermissionError('calendar_window_required')
        window = self._window(context.calendar_window)
        budget = [context.limits.bytes]
        value = self._call(context, 'get_event', {'calendarId': resource, 'eventId': _event_id(source_id)}, budget)
        if value.get('id') != source_id:
            raise ValueError('calendar_identity_mismatch')
        private = value.get('visibility') == 'private'
        if private and not value.get('summary'):
            raise PermissionError('calendar_event_details_unavailable')
        start_value, end_value = value.get('start'), value.get('end')
        if not isinstance(start_value, dict) or not isinstance(end_value, dict):
            raise PermissionError('calendar_event_time_unavailable')
        zone_name = start_value.get('timeZone', 'UTC')
        if 'date' in start_value or private:
            metadata = self._list(context, resource, context.calendar_window, budget, page_size=1)
            role = metadata.get('accessRole')
            if role not in _READABLE_ROLES:
                raise CheckStopped('blocked', 'calendar_unavailable_or_revoked')
            if private and role not in _PRIVATE_DETAIL_ROLES:
                raise PermissionError('calendar_event_details_unavailable')
            if 'date' in start_value:
                zone_name = metadata.get('timeZone')
        zone = _zone(zone_name)
        start, end = _point(start_value, zone), _point(end_value, zone)
        if start >= end or end <= window[0] or start >= window[1]:
            raise PermissionError('calendar_event_outside_window')
        return self._record(resource, value, zone=zone.key)

    def attachment(self, context, **kwargs):
        self._authorize(context)
        raise PermissionError('calendar_attachment_unsupported')
