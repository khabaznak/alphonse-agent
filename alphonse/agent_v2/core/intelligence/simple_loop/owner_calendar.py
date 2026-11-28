"""One private Calendar agenda read for the authenticated local Desktop owner."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import os
import re
from zoneinfo import ZoneInfo

from alphonse.agent_v2.accounts import AccountHandle, AccountSession, exact_calendar_id
from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.tools.registry import ToolDefinition
from alphonse.agent_v2.integrations.acquisition import CursorExpired, ReadContext, ReadLimits
from alphonse.agent_v2.integrations.google_calendar import CalendarAdapter
from alphonse.agent_v2.integrations.oauth import OAuthError, SCOPES
from alphonse.agent_v2.services.project_check_worker import CheckStopped

OWNER_CALENDAR_TOOL_ID = 'native.owner_calendar_agenda'
SCOPE = SCOPES['google']['calendar_read']
MAX_EVENTS = 40
_AGENDA_LIMITS = ReadLimits(1, 200, 128 * 1024)
_INSTANT = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?(?:Z|[+-][0-9]{2}:[0-5][0-9])')
_SCHEMA = {'type': 'object', 'properties': {
    'time_min': {'type': 'string', 'description': 'Start of the requested window in RFC3339 with an explicit offset.'},
    'time_max': {'type': 'string', 'description': 'End of the requested window in RFC3339 with an explicit offset.'},
    'day_offset': {'type': 'integer', 'minimum': -366, 'maximum': 366,
                   'description': 'First local calendar day relative to today in the owner’s timezone. Today is 0 and tomorrow is 1.'},
    'days': {'type': 'integer', 'minimum': 1, 'maximum': 31, 'default': 1,
             'description': 'Number of consecutive local days for day_offset, starting at the first local day.'},
}, 'oneOf': [
    {'required': ['time_min', 'time_max'], 'not': {'anyOf': [{'required': ['day_offset']}, {'required': ['days']}]}},
    {'required': ['day_offset'], 'not': {'anyOf': [{'required': ['time_min']}, {'required': ['time_max']}]}},
],
   'additionalProperties': False}
_FAILURES = {
    'calendar_access_required': 'Connect Google Calendar in Settings → Integrations and grant access to the exact calendar you want to use.',
    'calendar_choice_required': 'More than one Google calendar is authorized. Keep access enabled for the one calendar you want to use.',
    'reconnect_required': 'Reconnect Google Calendar in Settings → Integrations and try again.',
    'credential_busy': 'Google Calendar is busy. Try again shortly. If it stays busy, reconnect Google Calendar in Settings → Integrations.',
    'api_disabled': ('Google Calendar rejected the read because Calendar MCP is disabled. In Google Cloud, enable '
                     'Calendar MCP in the project configured for Alphonse, then try again.'),
    'approval_required': 'Google Calendar requires an additional approval for this read.',
    'read_unavailable': 'Google Calendar could not verify events for this time window. Check the calendar and try again.',
}


def _instant(value):
    if not isinstance(value, str) or _INSTANT.fullmatch(value) is None:
        raise ValueError('owner_calendar_offset_required')
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def calendar_window(arguments, *, at=None, timezone_name=None):
    if isinstance(arguments, dict) and 'day_offset' in arguments:
        offset, days = arguments['day_offset'], arguments.get('days', 1)
        if (set(arguments) - {'day_offset', 'days'} or type(offset) is not int or not -366 <= offset <= 366
                or type(days) is not int or not 1 <= days <= 31 or at is None or at.utcoffset() is None):
            raise ValueError('owner_calendar_relative_window_invalid')
        zone = ZoneInfo(timezone_name)
        first_day = at.astimezone(zone).date() + timedelta(days=offset)
        return tuple(datetime.combine(day, datetime.min.time(), zone)
                     for day in (first_day, first_day + timedelta(days=days)))
    if not isinstance(arguments, dict) or set(arguments) != {'time_min', 'time_max'}:
        raise ValueError('owner_calendar_window_required')
    start, end = (_instant(arguments[key]) for key in ('time_min', 'time_max'))
    if start.microsecond or end.microsecond or not timedelta(0) < end - start <= timedelta(days=32):
        raise ValueError('owner_calendar_window_invalid')
    return start, end


def _binding(conn, actor):
    """Find exactly one eligible account and calendar using the current snapshot."""
    matches = {}
    now = datetime.now(timezone.utc)
    for row in conn.execute('''SELECT g.grant_id, g.connection_id, g.revision, g.definition,
            i.connection_json, i.credential_generation FROM v2_account_grants g
            JOIN v2_integrations i ON i.integration_id = g.connection_id
            JOIN v2_users u ON u.user_id = ? AND u.is_active = 1
            WHERE i.provider_key = 'google' AND i.enabled = 1 AND i.credential_ref IS NOT NULL
              AND i.credential_ref != '' AND i.credential_generation > 0
              AND i.revocation_pending = 0 AND i.credential_deletion_pending = 0
            ORDER BY g.grant_id''', (actor,)):
        try:
            grant, connection = json.loads(row['definition']), json.loads(row['connection_json'])
            if (grant['beneficiary_user_id'] != actor or grant['grantor_user_id'] != actor
                    or grant['recipient_user_ids'] != [actor] or grant['project_id'] is not None
                    or grant['schema_version'] != 1 or grant['grant_id'] != row['grant_id']
                    or grant['connection_id'] != row['connection_id'] or grant['revision'] != row['revision']
                    or grant['revoked_at'] or datetime.fromisoformat(grant['expires_at'].replace('Z', '+00:00')) <= now
                    or grant['operations'] != ['calendar_read'] or not isinstance(grant['resources'], list)
                    or len(grant['resources']) != 1 or not exact_calendar_id(grant['resources'][0])
                    or connection['owner_user_id'] != actor or connection['schema_version'] != 1
                    or connection['status'] != 'connected' or connection['health_code'] != 'healthy'
                    or 'calendar_read' not in connection['capabilities'] or SCOPE not in connection['granted_scopes']):
                continue
            resource = grant['resources'][0]
            matches.setdefault((row['connection_id'], resource), {
                'connection_id': row['connection_id'], 'connection_revision': connection['revision'],
                'credential_generation': row['credential_generation'], 'grant_id': row['grant_id'],
                'grant_revision': row['revision'], 'resource': resource,
            })
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    if not matches:
        return None, 'calendar_access_required'
    if len(matches) != 1:
        return None, 'calendar_choice_required'
    return next(iter(matches.values())), None


def _event_summary(event, window, zone):
    start_value, end_value = event['start'], event['end']
    all_day = 'date' in start_value and 'dateTime' not in start_value
    if all_day:
        start, end = (date.fromisoformat(item['date']) for item in (start_value, end_value))
        starts_at, ends_at = (datetime.combine(item, datetime.min.time(), zone) for item in (start, end))
    else:
        start, end = (_instant(item['dateTime']) for item in (start_value, end_value))
        starts_at, ends_at = start, end
    if starts_at >= ends_at or ends_at <= window[0] or starts_at >= window[1]:
        raise ValueError('owner_calendar_event_outside_window')
    title = event.get('summary', '')
    if not isinstance(title, str):
        raise ValueError('owner_calendar_title_invalid')
    # Keep titles single-line and strip provider-supplied control characters.
    title = ' '.join(''.join(ch if ch.isprintable() else ' ' for ch in title).split())
    if len(title) > 160:
        title = title[:159] + '…'
    return {'title': title, 'start': start.isoformat(), 'end': end.isoformat(), 'all_day': all_day,
            'end_unspecified': event.get('endTimeUnspecified') is True}


class _CurrentTokenOnly:
    def __init__(self, oauth):
        self.oauth = oauth
        self.accounts = oauth.accounts

    def token(self, session, *, authorization):
        return self.oauth.token_without_refresh(session, authorization=authorization)


class OwnerCalendarService:
    def __init__(self, accounts, oauth_factory, *, transport=None):
        self.accounts = accounts
        self.oauth_factory = oauth_factory
        self.adapter = CalendarAdapter(lambda session: _CurrentTokenOnly(oauth_factory(session)), transport=transport)

    def binding(self, actor):
        with self.accounts.integrations._connect() as conn:
            return _binding(conn, actor)

    def observe(self, actor, arguments, *, guard, timezone_name=None):
        started = datetime.now(timezone.utc)
        window = calendar_window(arguments, at=started, timezone_name=timezone_name)
        return self._observe_window(actor, window, started=started, guard=guard)

    def observe_agenda(self, actor, start_day, end_day, *, zone, guard):
        """Read one page for the local Agenda's visible range, with managed renewal if needed."""
        if type(start_day) is not date or type(end_day) is not date or not 0 < (end_day - start_day).days <= 42:
            raise ValueError('owner_calendar_agenda_range_invalid')
        window = tuple(datetime.combine(day, datetime.min.time(), zone).astimezone(timezone.utc)
                       for day in (start_day, end_day))
        if window[1] <= window[0]:
            raise ValueError('owner_calendar_agenda_range_invalid')
        started = datetime.now(timezone.utc)
        guard()
        binding, failure = self.binding(actor)
        if binding is None:
            return {'status': 'unavailable', 'reason': failure}

        def check_grant():
            guard()
            current, _ = self.binding(actor)
            if current is None or any(current[key] != value for key, value in binding.items()
                                      if key != 'credential_generation'):
                raise CheckStopped('blocked', 'owner_calendar_authority_changed')

        result = self._observe_window(actor, window, started=started, guard=check_grant, limits=_AGENDA_LIMITS)
        if result.get('reason') != 'reconnect_required':
            return result

        # This settings-authorized Desktop fetch can renew credentials using the
        # existing OAuth lease. The model's read-only Calendar tool cannot do so.
        check_grant()
        session = AccountSession(actor)
        authorization = dict(connection_id=binding['connection_id'], grant_id=binding['grant_id'],
                             operation='calendar_read', resources=[binding['resource']], recipient_user_ids=[actor],
                             required_scopes=[SCOPE], project_id=None, expected_grant_revision=binding['grant_revision'])
        try:
            self.oauth_factory(session).token(session, authorization=authorization,
                                             expected_connection_revision=binding['connection_revision'])
        except OAuthError as exc:
            if exc.args in (('oauth_reauth_required',), ('oauth_refresh_not_persisted',)):
                guard()
                return {'status': 'unavailable', 'reason': 'reconnect_required'}
            if exc.args == ('oauth_operation_in_progress',):
                check_grant()
                return {'status': 'unavailable', 'reason': 'credential_busy'}
            return {'status': 'unavailable', 'reason': 'read_unavailable'}
        check_grant()
        return self._observe_window(actor, window, started=datetime.now(timezone.utc), guard=check_grant,
                                    limits=_AGENDA_LIMITS)

    def _observe_window(self, actor, window, *, started, guard, limits=None):
        time_min, time_max = (value.isoformat() for value in window)
        guard()
        binding, failure = self.binding(actor)
        guard()
        if binding is None:
            return {'status': 'unavailable', 'reason': failure}

        def check():
            guard()
            if self.binding(actor)[0] != binding:
                raise CheckStopped('blocked', 'owner_calendar_authority_changed')

        resources = (binding['resource'],)
        handle = AccountHandle(binding['connection_id'], '', binding['grant_id'], binding['connection_revision'],
                               binding['grant_revision'], 'google', 'calendar_read', resources, (actor,))
        limits = limits or ReadLimits(1, MAX_EVENTS, 48 * 1024)
        context = ReadContext(AccountSession(actor), handle, datetime.now(timezone.utc) + timedelta(seconds=18),
                              limits, guard=check, calendar_window=(time_min, time_max))
        try:
            value = self.adapter.list_summary(context)
            zone = ZoneInfo(value['timeZone'])
            items = value.get('items', [])
            if (value['accessRole'] not in ('reader', 'writerWithoutPrivateAccess', 'writer', 'owner')
                    or not isinstance(items, list) or len(items) > limits.items):
                raise ValueError('owner_calendar_response_invalid')
            events = [_event_summary(event, window, zone) for event in items if event.get('status') != 'cancelled']
            if 'nextPageToken' in value and (not isinstance(value['nextPageToken'], str) or not value['nextPageToken']):
                raise ValueError('owner_calendar_continuation_invalid')
        except OAuthError as exc:
            check()
            reason = 'read_unavailable'
            if exc.args == ('oauth_refresh_required',):
                reason = 'reconnect_required'
            elif exc.args == ('oauth_operation_in_progress',):
                reason = 'credential_busy'
            return {'status': 'unavailable', 'reason': reason}
        except CheckStopped as exc:
            check()
            reason = {'calendar_api_disabled': 'api_disabled',
                      'calendar_google_approval_required': 'approval_required'}.get(exc.code, 'read_unavailable')
            return {'status': 'unavailable', 'reason': reason}
        except (AttributeError, CursorExpired, KeyError, OSError, TypeError, ValueError):
            check()  # Revocations and cancellation are not ordinary provider failures.
            return {'status': 'unavailable', 'reason': 'read_unavailable'}
        check()
        partial = 'nextPageToken' in value or not value['coverage_complete']
        return {'status': 'partial' if partial else 'complete', 'events': events,
                'window': [time_min, time_max], 'timezone': zone.key, 'authority': binding,
                'observed_at': started.isoformat()}


def _same_database(context, service):
    return (isinstance(service, OwnerCalendarService) and type(context.messages) is SQLiteMessageQueue
            and os.path.samefile(context.messages.db_path, service.accounts.integrations.db_path)
            and os.path.samefile(context.messages.db_path, service.accounts.users.db_path))


def _authorized(actor, context):
    return (bool(actor) and actor == context.task.user and context.cancellation_checker() is False
            and _same_database(context, context.owner_calendar_service)
            and context.messages.verify_trusted_local_owner_claim(context.claim_receipt, context.task))


def execute_owner_calendar(arguments, *, context):
    actor = context.task.user

    def guard():
        if not _authorized(actor, context):
            raise CheckStopped('cancelled', 'owner_calendar_claim_unavailable')

    guard()
    timezone_name = context.user_timezone_provider(actor) if 'day_offset' in arguments else None
    return context.owner_calendar_service.observe(actor, arguments, guard=guard, timezone_name=timezone_name)


def _failure_message(payload):
    if (isinstance(payload, dict) and set(payload) == {'status', 'reason'} and payload['status'] == 'unavailable'
            and isinstance(payload['reason'], str)):
        return _FAILURES.get(payload['reason'])
    return None


def _valid_payload(arguments, payload, *, timezone_name=None, presented_at=None):
    if _failure_message(payload):
        calendar_window(arguments, at=datetime.now(timezone.utc), timezone_name=timezone_name)
        return True
    if (not isinstance(payload, dict) or set(payload) != {'status', 'events', 'window', 'timezone', 'authority', 'observed_at'}
            or payload['status'] not in ('complete', 'partial') or not isinstance(payload['events'], list)
            or len(payload['events']) > MAX_EVENTS):
        return False
    zone = ZoneInfo(payload['timezone'])
    observed = datetime.fromisoformat(payload['observed_at'])
    if observed.tzinfo is None or (observed > presented_at if presented_at is not None
            else not timedelta(0) <= datetime.now(timezone.utc) - observed <= timedelta(minutes=5)):
        return False
    window = calendar_window(arguments, at=observed, timezone_name=timezone_name)
    if payload['window'] != [value.isoformat() for value in window]:
        return False
    for event in payload['events']:
        if (not isinstance(event, dict) or set(event) != {'title', 'start', 'end', 'all_day', 'end_unspecified'}
                or type(event['all_day']) is not bool or type(event['end_unspecified']) is not bool):
            return False
        key = 'date' if event['all_day'] else 'dateTime'
        if _event_summary({'summary': event['title'], 'start': {key: event['start']}, 'end': {key: event['end']},
                           'endTimeUnspecified': event['end_unspecified']}, window, zone) != event:
            return False
    return True


def valid_owner_calendar(actor, arguments, payload, *, context):
    try:
        if not _authorized(actor, context) or not _valid_payload(arguments, payload,
                timezone_name=context.user_timezone_provider(actor) if 'day_offset' in arguments else None):
            return False
        if _failure_message(payload):
            return True
        service = context.owner_calendar_service
        binding, _ = service.binding(actor)
        if binding != payload['authority']:
            return False
        service.accounts.authorize(AccountSession(actor), connection_id=binding['connection_id'], grant_id=binding['grant_id'],
            operation='calendar_read', resources=[binding['resource']], recipient_user_ids=[actor], required_scopes=[SCOPE],
            expected_grant_revision=binding['grant_revision'])
        return _authorized(actor, context)
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        return False


def valid_owner_calendar_locked(conn, actor, arguments, payload, *, context):
    try:
        return (_same_database(context, context.owner_calendar_service) and _valid_payload(arguments, payload,
                    timezone_name=context.user_timezone_provider(actor) if 'day_offset' in arguments else None)
                and (_failure_message(payload) is not None or _binding(conn, actor)[0] == payload['authority']))
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        return False


def valid_calendar_card_history(conn, actor, arguments, payload, *, presented_at):
    """Recheck today's grant for a snapshot already delivered privately."""
    binding, _ = _binding(conn, actor)
    return (binding is not None and binding == payload.get('authority')
            and _valid_payload(arguments, payload, presented_at=presented_at,
                timezone_name=payload.get('timezone') if 'day_offset' in arguments else None))


def format_calendar_observation(payload, *, private_delivery=False):
    failure = _failure_message(payload)
    if failure is not None:
        return failure
    if not private_delivery:
        return 'Google Calendar results were displayed privately in Desktop.'
    start, end = (value.replace('T', ' ') for value in payload['window'])
    lines = [f'Google Calendar events from {start} to {end}.']
    if payload['status'] == 'partial':
        lines.append('Google Calendar returned limited results. Some events may be missing from this window.')
    if not payload['events'] and payload['status'] == 'complete':
        lines.append('No events were returned in this window.')
    for event in payload['events']:
        title = re.sub(r'([\\`*_{}\[\]()#+.!|<>~-])', r'\\\1', event['title']) or 'Title unavailable'
        if event['all_day']:
            start = date.fromisoformat(event['start'])
            last_day = date.fromisoformat(event['end']) - timedelta(days=1)
            when = f'{start} (all day)' if start == last_day else f'{start} to {last_day} (all day)'
        else:
            when = event['start'].replace('T', ' ')
            if not event['end_unspecified']:
                when += ' to ' + event['end'].replace('T', ' ')
        lines.append(f'- {title} · {when}')
    return '\n\n'.join(lines)


def build_owner_calendar_tool_definition():
    descriptor = ToolDescriptor(OWNER_CALENDAR_TOOL_ID, 'owner_calendar_agenda', ToolKind.NATIVE,
        'Read an agenda from one authorized Google calendar for the owner in Desktop. For relative whole-day requests, '
        'use day_offset and let the host resolve local dates. Use explicit timestamps for absolute dates or intraday ranges. '
        'For an intraday range relative to today, use owner_clock to resolve the dates in the owner’s time zone. '
        'Event titles and times are delivered privately to Desktop. After the read, return an answer so the '
        'host can deliver its verified result. Writes, other recipients and provider account selection are unavailable.',
        _SCHEMA, ('calendar_read',), ('native', 'calendar'), read_only=True, metadata={'sensitive_content': True})
    return ToolDefinition(descriptor, execute_owner_calendar, _SCHEMA, accepts_context=True)
