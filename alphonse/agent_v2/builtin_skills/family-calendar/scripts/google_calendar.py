"""Reusable Google Calendar client. No catalog, household state, or credentials here."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import date, datetime, timezone
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

API = 'https://www.googleapis.com/calendar/v3'
SCOPE = 'https://www.googleapis.com/auth/'
FIELDS = {'summary', 'description', 'location', 'start', 'end', 'recurrence'}


class CalendarError(Exception):
    """Static, non-secret error safe for command evidence."""


class ProviderError(CalendarError):
    def __init__(self, status):
        self.status = status
        super().__init__({401: 'account_revoked_reconnect', 403: 'scope_or_calendar_access_denied',
                          412: 'event_changed_reconcile'}.get(status, 'provider_request_failed'))


def digest(request):
    """Exact approved effect, including stable operation ID and notification policy."""
    return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def transport(method, url, body=None, headers=None):
    """Fixed Google endpoints only; never log HTTP exceptions or credentials."""
    data = None if body is None else json.dumps(body).encode()
    try:
        with urlopen(Request(url, data=data, method=method, headers=headers or {}), timeout=25) as response:
            return json.load(response)
    except HTTPError as exc:
        raise ProviderError(exc.code) from None
    except Exception:
        raise CalendarError('provider_unavailable_reconcile_before_retry') from None


def refresh(credentials):
    try:
        data = urlencode({key: credentials[key] for key in ('client_id', 'client_secret', 'refresh_token')}
                         | {'grant_type': 'refresh_token'}).encode()
        request = Request('https://oauth2.googleapis.com/token', data=data,
                          headers={'Content-Type': 'application/x-www-form-urlencoded'})
        with urlopen(request, timeout=25) as response:
            result = json.load(response)
        if not result.get('access_token'):
            raise ValueError()
        return result
    except HTTPError as exc:
        raise CalendarError('account_revoked_reconnect' if exc.code in (400, 401) else 'oauth_unavailable') from None
    except Exception:
        raise CalendarError('oauth_unavailable') from None


def instant(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)', value):
        raise CalendarError('explicit_rfc3339_offset_required')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return result


def endpoint(value):
    if not isinstance(value, dict):
        raise CalendarError('event_time_invalid')
    if set(value) == {'date'}:
        return date.fromisoformat(value['date'])
    if set(value) != {'dateTime', 'timeZone'}:
        raise CalendarError('event_time_requires_offset_and_iana_zone')
    dt = instant(value['dateTime'])
    local = dt.astimezone(ZoneInfo(value['timeZone']))
    if local.replace(tzinfo=None) != dt.replace(tzinfo=None) or local.utcoffset() != dt.utcoffset():
        raise CalendarError('time_zone_offset_mismatch')
    return dt


def validate_event(event):
    if not isinstance(event, dict) or not event or set(event) - FIELDS:
        raise CalendarError('unsupported_event_fields')
    # Both endpoints are mandatory, including for updates: no implicit date arithmetic.
    if 'start' not in event or 'end' not in event:
        raise CalendarError('explicit_start_and_end_required')
    start, end = endpoint(event['start']), endpoint(event['end'])
    if type(start) is not type(end) or end <= start:
        raise CalendarError('event_interval_invalid')
    if 'summary' in event and (not isinstance(event['summary'], str) or not event['summary'].strip()):
        raise CalendarError('summary_invalid')
    if 'recurrence' in event:
        rules = event['recurrence']
        if not isinstance(rules, list) or len(rules) != 1 or not isinstance(rules[0], str):
            raise CalendarError('recurrence_unsupported')
        # Deliberately bounded recurrence subset; complex RRULEs require another reviewed route.
        if not re.fullmatch(r'RRULE:FREQ=(DAILY|WEEKLY|MONTHLY);COUNT=([1-9]|[1-9]\d|[1-9]\d\d)', rules[0]):
            raise CalendarError('recurrence_requires_bounded_count')
        if 'dateTime' in event['start'] and event['start']['timeZone'] != event['end']['timeZone']:
            raise CalendarError('recurrence_zone_mismatch')


def matches(saved, expected):
    for key, value in expected.items():
        actual = saved.get(key)
        if key in {'start', 'end'}:
            if actual is None or endpoint(actual) != endpoint(value):
                return False
            if value.get('timeZone') and actual.get('timeZone') != value['timeZone']:
                return False
        elif actual != value:
            return False
    return saved.get('status') != 'cancelled'


class Calendar:
    def __init__(self, bindings, *, send=transport, renew=refresh, now=None):
        self.bindings, self.send, self.renew = bindings, send, renew
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.token = None

    def call(self, method, path, body=None, params=None, headers=None):
        if self.token is None:
            account = self.bindings['calendar']
            token = self.renew(account['credentials'])
            # Prefer fresh OAuth scope evidence; older refresh responses may omit it.
            granted = token.get('scope', account['configuration'].get('granted_scopes', ''))
            scopes = set(granted.split())
            required = ({'calendar', 'calendar.events'} if self.writing else
                        {'calendar', 'calendar.readonly', 'calendar.freebusy', 'calendar.events.freebusy'}
                        if self.op == 'availability' else
                        {'calendar', 'calendar.events', 'calendar.readonly', 'calendar.events.readonly'})
            if not scopes.intersection(SCOPE + name for name in required):
                raise CalendarError('missing_scopes_reconnect')
            self.token = token['access_token']
        return self.send(method, API + path + ('?' + urlencode(params) if params else ''), body,
                         {'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json', **(headers or {})})

    def run(self, request):
        if not isinstance(request, dict):
            raise CalendarError('request_object_required')
        op = request.get('operation')
        self.op, self.token = op, None
        self.writing = op in {'create', 'update'}
        allowed_keys = {'operation', 'calendar_id'} | ({'event', 'operation_id', 'send_updates', 'event_id', 'etag', 'recurrence_scope'} if self.writing else {'time_min', 'time_max', 'event_id'})
        if set(request) - allowed_keys or op not in {'list', 'availability', 'get', 'create', 'update'}:
            raise CalendarError('request_invalid')
        account = self.bindings.get('calendar', {})
        if account.get('kind') != 'account' or account.get('provider') != 'google-calendar':
            raise CalendarError('explicit_google_calendar_binding_required')
        policy = account.get('configuration', {}).get('calendars', {})
        calendar_id = request.get('calendar_id')
        if not isinstance(calendar_id, str) or calendar_id not in policy or op not in policy[calendar_id]:
            raise CalendarError('calendar_operation_denied')
        if self.writing:
            grant = self.bindings.get('authorization', {})
            config = grant.get('configuration', {})
            if grant.get('kind') != 'configuration' or config.get('request_sha256') != digest(request):
                raise CalendarError('exact_effect_authorization_required')
            if instant(config.get('expires_at')) <= self.now():
                raise CalendarError('effect_authorization_expired')
            if not isinstance(request.get('operation_id'), str) or not re.fullmatch(r'[A-Za-z0-9_-]{16,128}', request['operation_id']):
                raise CalendarError('stable_operation_id_required')
            if request.get('send_updates') != 'none':
                raise CalendarError('invitations_not_supported')
            validate_event(request.get('event'))
        path = '/calendars/' + quote(calendar_id, safe='') + '/events'
        if op in {'list', 'availability'}:
            lo, hi = instant(request.get('time_min')), instant(request.get('time_max'))
            if hi <= lo:
                raise CalendarError('query_interval_invalid')
            if op == 'availability':
                result = self.call('POST', '/freeBusy', {'timeMin': request['time_min'], 'timeMax': request['time_max'], 'items': [{'id': calendar_id}]})
                calendars = result.get('calendars', {})
                if (calendar_id not in calendars or calendars[calendar_id].get('errors')
                        or not isinstance(calendars[calendar_id].get('busy'), list)):
                    raise CalendarError('availability_unknown')
                return calendars[calendar_id]
            events, page, seen = [], None, set()
            while True:
                params = {'timeMin': request['time_min'], 'timeMax': request['time_max'], 'singleEvents': 'true', 'orderBy': 'startTime', 'maxResults': 250}
                if page:
                    params['pageToken'] = page
                result = self.call('GET', path, params=params)
                events.extend(result.get('items', []))
                page = result.get('nextPageToken')
                if not page:
                    return {'items': events}
                if page in seen or len(seen) >= 100:
                    raise CalendarError('pagination_incomplete')
                seen.add(page)
        if op == 'get':
            return self.call('GET', path + '/' + self.event_id(request))
        if op == 'create':
            if set(request) & {'event_id', 'etag', 'recurrence_scope'}:
                raise CalendarError('create_request_invalid')
            if not request['event'].get('summary'):
                raise CalendarError('summary_required')
            event_id = hashlib.sha256(request['operation_id'].encode()).hexdigest()
            target = path + '/' + event_id
            try:
                saved = self.call('GET', target)
            except ProviderError as exc:
                if exc.status != 404:
                    raise
                try:
                    self.call('POST', path, {**request['event'], 'id': event_id}, {'sendUpdates': 'none'})
                except ProviderError as conflict:
                    if conflict.status != 409:
                        raise
                saved = self.call('GET', target)
        else:
            event_id = self.event_id(request)
            target = path + '/' + event_id
            saved = self.call('GET', target)
            scope = request.get('recurrence_scope')
            if saved.get('attendees'):
                raise CalendarError('attendee_event_updates_not_supported')
            expected_scope = 'instance' if saved.get('recurringEventId') else 'series' if saved.get('recurrence') else 'event'
            if scope != expected_scope or ('recurrence' in request['event'] and scope != 'series'):
                raise CalendarError('explicit_recurrence_scope_required')
            if not matches(saved, request['event']):
                if not request.get('etag') or request['etag'] != saved.get('etag'):
                    raise CalendarError('event_changed_reconcile')
                self.call('PATCH', target, request['event'], {'sendUpdates': 'none'}, {'If-Match': request['etag']})
                saved = self.call('GET', target)
        if saved.get('id') != event_id or not matches(saved, request['event']):
            raise CalendarError('readback_mismatch_reconcile')
        return {'verified': True, 'event': saved}

    @staticmethod
    def event_id(request):
        value = request.get('event_id')
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            raise CalendarError('explicit_event_id_required')
        return value


def main():
    try:
        bindings = json.loads(os.environ['ALPHONSE_SKILL_BINDINGS'])
        request = json.load(sys.stdin)
        print(json.dumps(Calendar(bindings).run(request), ensure_ascii=False))
        return 0
    except CalendarError as exc:
        print(json.dumps({'error': str(exc)}), file=sys.stderr)
    except Exception:
        print('{"error":"calendar_input_or_configuration_invalid"}', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
