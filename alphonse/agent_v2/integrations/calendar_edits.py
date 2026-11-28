"""Owner-confirmed Calendar updates. Google MCP has no atomic version check."""
from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from email.errors import HeaderParseError
from email.headerregistry import Address
import hashlib
import json
import re
import secrets
import sqlite3
import threading
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from alphonse.agent_v2.accounts import (
    AccountSession, CALENDAR_WRITE_OPERATION, CALENDAR_WRITE_SCOPE, exact_calendar_id,
)
from alphonse.agent_v2.integrations.google_mcp import GoogleMcpClient, GoogleMcpError
from alphonse.agent_v2.integrations.oauth import OAuthError

_EVENT_ID = re.compile(r'[A-Za-z0-9_-]{1,1024}\Z')
_REVIEW_TOKEN = re.compile(r'[A-Za-z0-9_-]{43}\Z')
_DATE = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}\Z')
_TIMESTAMP = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:Z|[+-][0-9]{2}:[0-9]{2})\Z')
_TEXT_LIMITS = {'summary': 256, 'description': 4096, 'location': 512}
_TIME_KEYS = {'start', 'end', 'time_zone'}
_INVITE_FIELD = 'invite_guests'
_WARNING = ('Apply this edit here or edit the event in Google Calendar. Avoid doing both at the same time. '
            'A last-second change in Google Calendar can still be overwritten.')
_GUEST_WARNING = 'Simultaneous guest-list edits can also affect who receives email.'
_READ_LIMIT = 128 * 1024
_OMITTED_FIELDS = ('omittedContent', 'omittedEvents', 'incomplete')
_DEFAULT_REVIEW_MINUTES = 5
_MAX_REVIEW_MINUTES = 30


def _identifier(value):
    return isinstance(value, str) and 0 < len(value) <= 512 and value == value.strip() and value.isprintable()


def _event_id(value, calendar_id):
    if isinstance(value, str) and _EVENT_ID.fullmatch(value):
        return value
    try:
        if not isinstance(value, str) or len(value) > 3072 or value != value.strip():
            raise ValueError()
        parsed = urlsplit(value)
        if parsed.scheme != 'https' or parsed.netloc != 'calendar.google.com' or parsed.fragment:
            raise ValueError()
        if parsed.path in {'/calendar/event', '/calendar/u/0/event'}:
            query = parse_qs(parsed.query, max_num_fields=10)
            if len(query.get('eid', ())) != 1:
                raise ValueError()
            encoded = query['eid'][0]
        else:
            match = re.fullmatch(r'/calendar(?:/u/[0-9]+)?/r/eventedit/([A-Za-z0-9_=-]+)', parsed.path)
            if match is None:
                raise ValueError()
            encoded = match[1]
        decoded = base64.b64decode(encoded + '=' * (-len(encoded) % 4), altchars=b'-_', validate=True).decode()
        identifier, calendar = decoded.split(' ', 1)
        if calendar != calendar_id or _EVENT_ID.fullmatch(identifier) is None:
            raise ValueError()
        return identifier
    except (ValueError, TypeError, UnicodeError):
        raise ValueError('calendar_edit_event_id_invalid_or_wrong_calendar') from None


def _instant(value):
    if not isinstance(value, str) or not _TIMESTAMP.fullmatch(value):
        raise ValueError('calendar_edit_explicit_time_required')
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def _time_pair(fields):
    start, end = fields.get('start'), fields.get('end')
    try:
        if not isinstance(start, str) or not isinstance(end, str):
            raise ValueError()
        if _DATE.fullmatch(start) and _DATE.fullmatch(end):
            if 'time_zone' in fields or date.fromisoformat(end) <= date.fromisoformat(start):
                raise ValueError()
            return {'startTime': start, 'endTime': end, 'allDay': True}
        zone = ZoneInfo(fields['time_zone'])
        lo, hi = _instant(start), _instant(end)
        if (hi <= lo or lo.astimezone(zone).utcoffset() != lo.utcoffset()
                or hi.astimezone(zone).utcoffset() != hi.utcoffset()):
            raise ValueError()
        # Offsets disambiguate DST folds. The explicit IANA zone preserves the
        # intended Calendar display and future recurrence behavior.
        return {'startTime': start, 'endTime': end, 'timeZone': fields['time_zone'], 'allDay': False}
    except (KeyError, TypeError, ValueError, ZoneInfoNotFoundError):
        raise ValueError('calendar_edit_time_invalid') from None


def _email_address(value):
    if (not isinstance(value, str) or not 0 < len(value) <= 254
            or not value.isascii() or not value.isprintable() or value != value.strip()):
        return False
    try:
        address = Address(addr_spec=value)
    except (ValueError, HeaderParseError):
        return False
    return bool(address.domain and 0 < len(address.username) <= 64 and address.addr_spec == value)


def _invite_addresses(values):
    if (not isinstance(values, list) or not values or not all(_email_address(value) for value in values)
            or len({value.lower() for value in values}) != len(values)):
        raise ValueError('calendar_edit_invite_addresses_invalid')
    return [{'email': value} for value in values]


def _guests(event):
    attendees = event.get('attendees', [])
    if event.get('attendeesOmitted') or not isinstance(attendees, list):
        raise ValueError('calendar_edit_guest_list_unavailable')
    permissions = event.get('guestPermissions')
    organizer = event.get('organizer')
    if (isinstance(permissions, dict) and permissions.get('guestsCanSeeGuests') is False
            and (not isinstance(organizer, dict) or organizer.get('self') is not True)):
        raise ValueError('calendar_edit_guest_list_unavailable')
    guests = {}
    for attendee in attendees:
        if (not isinstance(attendee, dict) or not _email_address(attendee.get('email'))
                or attendee.get('additionalGuests', 0) != 0 or attendee.get('asyncOperation')):
            raise ValueError('calendar_edit_guest_list_unavailable')
        address = attendee['email']
        key = address.lower()
        if key in guests:
            raise ValueError('calendar_edit_guest_list_unavailable')
        guest = {'email': address}
        name = attendee.get('displayName')
        if _identifier(name):
            guest['name'] = name
        guests[key] = guest
    return guests


def _patch(fields):
    if not isinstance(fields, dict) or not fields or set(fields) - (_TEXT_LIMITS.keys() | _TIME_KEYS | {_INVITE_FIELD}):
        raise ValueError('calendar_edit_fields_invalid')
    patch = {}
    for key, limit in _TEXT_LIMITS.items():
        if key not in fields:
            continue
        value = fields[key]
        if (not isinstance(value, str) or len(value) > limit or (key == 'summary' and not value.strip())
                or not all(c.isprintable() or (key == 'description' and c in '\n\t') for c in value)):
            raise ValueError('calendar_edit_fields_invalid')
        patch[key] = value
    if _TIME_KEYS.intersection(fields):
        patch.update(_time_pair(fields))
    if _INVITE_FIELD in fields:
        patch['addedAttendees'] = _invite_addresses(fields[_INVITE_FIELD])
    return patch


def _scope(event, event_id, *, inviting=False):
    if (not isinstance(event, dict) or event.get('id') != event_id
            or any(event.get(key) for key in _OMITTED_FIELDS)
            or event.get('status') not in (None, 'confirmed', 'CONFIRMED')
            or event.get('eventType') not in (None, 'default', 'DEFAULT')
            or (not inviting and (event.get('attendees') or event.get('attendeesOmitted')))
            or event.get('conferenceData') or event.get('conferenceUrl')
            or not isinstance(event.get('start'), dict) or not isinstance(event.get('end'), dict)):
        raise ValueError('calendar_edit_event_unavailable_or_unsupported')
    if event.get('recurringEventId'):
        return 'instance'
    return 'series' if event.get('recurrence') else 'event'


def _digest(event):
    return hashlib.sha256(json.dumps(event, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _effect(event, fields):
    visible = {k: event.get(k, '') for k in _TEXT_LIMITS if k in fields}
    if _TIME_KEYS.intersection(fields):
        for key in ('start', 'end'):
            point = event[key]
            visible[key] = point.get('date', point.get('dateTime'))
        visible['time_zone'] = event['start'].get('timeZone')
    return visible


def _matches(actual, requested):
    for key, value in requested.items():
        if key in {'start', 'end'} and isinstance(value, str) and _TIMESTAMP.fullmatch(value):
            try:
                if _instant(actual[key]) != _instant(value):
                    return False
            except (KeyError, TypeError, ValueError):
                return False
        elif actual.get(key) != value:
            return False
    return True


@dataclass
class _Review:
    actor: str
    connection_revision: int
    grant_revision: int
    calendar_id: str
    event_id: str
    connection_id: str
    grant_id: str
    snapshot: str
    patch: dict
    public: dict
    expires: datetime
    origin: str | None
    state: str = 'pending'


class CalendarEditService:
    """Place a read-only proposal into an owner review queue; commit via Desktop."""

    def __init__(self, accounts, oauth, *, mcp=None, clock=None):
        self.accounts = accounts
        self.oauth = oauth
        self.mcp = mcp or GoogleMcpClient()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()
        self._reviews = {}
        self._busy_events = set()

    def _authority(self, session, connection_id, grant_id, calendar_id, *, review=None):
        if not isinstance(session, AccountSession):
            raise PermissionError('calendar_edit_owner_required')
        authorization = dict(connection_id=connection_id, grant_id=grant_id,
            operation=CALENDAR_WRITE_OPERATION, resources=[calendar_id], recipient_user_ids=[session.actor_user_id],
            required_scopes=[CALENDAR_WRITE_SCOPE], owner_calendar_write=True,
            expected_grant_revision=review.grant_revision if review is not None else None)
        record, grant = self.accounts.authorize(session, **authorization)
        if review is not None and record.connection['revision'] != review.connection_revision:
            raise PermissionError('calendar_edit_authority_changed')
        return record.connection['revision'], grant['revision'], authorization, grant['expires_at']

    def _read(self, calendar_id, event_id, token, deadline, guard):
        event, _ = self.mcp.call('calendar', 'get_event', {'calendarId': calendar_id, 'eventId': event_id},
                                  token=token, deadline=deadline, max_bytes=_READ_LIMIT, guard=guard)
        guard()
        return event

    def _prune(self):
        now = self.clock()
        self._reviews = {token: review for token, review in self._reviews.items()
                         if review.state != 'pending' or review.expires > now}

    def propose(self, session, *, connection_id, grant_id, calendar_id, event_id, fields,
                recurrence_scope=None, guard=lambda: None, origin=None, expected_summary=None,
                review_minutes=_DEFAULT_REVIEW_MINUTES):
        if not (_identifier(connection_id) and _identifier(grant_id) and exact_calendar_id(calendar_id)):
            raise ValueError('calendar_edit_exact_grant_required')
        if recurrence_scope not in (None, 'event', 'instance', 'series'):
            raise ValueError('calendar_edit_recurrence_scope_invalid')
        if type(review_minutes) is not int or not _DEFAULT_REVIEW_MINUTES <= review_minutes <= _MAX_REVIEW_MINUTES:
            raise ValueError('calendar_edit_review_duration_invalid')
        event_id = _event_id(event_id, calendar_id)
        patch = _patch(fields)
        holder = self._authority(session, connection_id, grant_id, calendar_id)
        revision, grant_revision, authorization, grant_expiry = holder
        deadline = self.clock() + timedelta(seconds=18)

        def check():
            guard()
            current = self._authority(session, connection_id, grant_id, calendar_id)
            if current[:2] != holder[:2] or self.clock() >= deadline:
                raise PermissionError('calendar_edit_authority_changed_or_expired')

        check()
        oauth = self.oauth(session)
        if oauth is None:
            raise ValueError('oauth_setup_unavailable')
        try:
            token = oauth.token(session, authorization=authorization, expected_connection_revision=revision)
            check()
            event = self._read(calendar_id, event_id, token, deadline, check)
        except GoogleMcpError as exc:
            return self._blocked(exc)
        except OAuthError:
            return {'status': 'unavailable', 'reason': 'calendar_edit_reconnect_required'}
        invitations = patch.get('addedAttendees')
        scope = _scope(event, event_id, inviting=invitations is not None)
        if scope != (recurrence_scope or 'event'):
            raise ValueError('calendar_edit_explicit_recurrence_scope_required')
        title = event.get('summary') or ''
        if not isinstance(title, str):
            raise ValueError('calendar_edit_event_unavailable_or_unsupported')
        if expected_summary is not None and title.casefold() != expected_summary.casefold():
            return {'status': 'unavailable', 'reason': 'calendar_edit_exact_event_required'}
        requested = dict(fields)
        requested.pop(_INVITE_FIELD, None)
        before = _effect(event, requested)
        if invitations is None and _matches(before, requested):
            return {'status': 'unavailable', 'reason': 'calendar_edit_no_change'}
        guests = None
        if invitations is not None:
            current_guests = _guests(event)
            if any(guest['email'].lower() in current_guests for guest in invitations):
                raise ValueError('calendar_edit_invite_already_attending')
            guests = dict(current=list(current_guests.values()), added=deepcopy(invitations), notification_level='ALL')
        expires = min(self.clock() + timedelta(minutes=review_minutes), datetime.fromisoformat(grant_expiry))
        review_token = secrets.token_urlsafe(32)
        public = dict(status='review_required', review_token=review_token, expires_at=expires.isoformat(),
                      calendar_id=calendar_id, event_id=event_id, recurrence_scope=scope,
                      event_title=' '.join(title.split())[:256],
                      event_start=event['start'].get('date', event['start'].get('dateTime')),
                      before=before, after=requested,
                      concurrent_edit_warning=_WARNING)
        if guests is not None:
            public['guests'] = guests
            public['concurrent_edit_warning'] += ' ' + _GUEST_WARNING
        check()
        with self._lock:
            self._prune()
            if len(self._reviews) >= 64:
                raise ValueError('calendar_edit_reviews_busy')
            if origin and any(r.actor == session.actor_user_id and r.origin == origin
                              for r in self._reviews.values()):
                raise ValueError('calendar_edit_review_already_pending')
            self._reviews[review_token] = _Review(session.actor_user_id, revision, grant_revision, calendar_id,
                event_id, connection_id, grant_id, _digest(event), patch, public, expires, origin)
        return deepcopy(public)

    def propose_for_conversation(self, session, *, target, fields, recurrence_scope=None, origin, guard):
        """Resolve a target within the sole explicitly granted write calendar."""
        if not _identifier(origin) or not isinstance(target, dict):
            raise ValueError('calendar_edit_target_invalid')
        _patch(fields)
        use_id = set(target) == {'event_id'}
        if not use_id and set(target) != {'summary', 'time_min', 'time_max'}:
            raise ValueError('calendar_edit_target_invalid')
        guard()
        with self.accounts.integrations._connect() as conn:
            rows = conn.execute('''SELECT definition FROM v2_account_grants
                WHERE json_extract(definition,'$.grantor_user_id')=?
                 AND json_extract(definition,'$.operations')=json_array(?)
                 AND json_extract(definition,'$.revoked_at')='' LIMIT 65''',
                (session.actor_user_id, CALENDAR_WRITE_OPERATION)).fetchall()
        if len(rows) > 64:
            return {'status': 'unavailable', 'reason': 'calendar_edit_select_one_write_calendar'}
        bindings = []
        for row in rows:
            grant = json.loads(row['definition'])
            if len(grant['resources']) != 1:
                continue
            scope = dict(connection_id=grant['connection_id'], grant_id=grant['grant_id'], calendar_id=grant['resources'][0])
            try:
                self._authority(session, **scope)
            except (PermissionError, ValueError):
                continue
            bindings.append(scope)
        if len(bindings) != 1:
            return {'status': 'unavailable', 'reason': ('calendar_edit_access_required' if not bindings
                                                        else 'calendar_edit_select_one_write_calendar')}
        scope = bindings[0]
        event_id = target['event_id'] if use_id else self._find_event(session, target, scope, guard)
        if isinstance(event_id, dict):
            return event_id
        result = self.propose(session, **scope, event_id=event_id, fields=fields, recurrence_scope=recurrence_scope,
                              origin=origin, guard=guard, expected_summary=target.get('summary'))
        if result['status'] == 'review_required':
            # The model sees an outcome. Only the authenticated local review
            # surface can retrieve the token or calendar details.
            return {'status': 'review_required'}
        return result

    def _find_event(self, session, target, scope, guard):
        summary = target['summary']
        try:
            lo, hi = (_instant(target[k]) for k in ('time_min', 'time_max'))
            if not _identifier(summary) or not timedelta(0) < hi - lo <= timedelta(days=32):
                raise ValueError()
        except (TypeError, ValueError):
            raise ValueError('calendar_edit_lookup_window_invalid') from None
        pin = self._authority(session, **scope)
        deadline = self.clock() + timedelta(seconds=18)

        def check():
            guard()
            if self._authority(session, **scope)[:2] != pin[:2] or self.clock() >= deadline:
                raise PermissionError('calendar_edit_authority_changed_or_expired')

        oauth = self.oauth(session)
        if oauth is None:
            return {'status': 'unavailable', 'reason': 'calendar_edit_access_required'}
        try:
            token = oauth.token(session, authorization=pin[2], expected_connection_revision=pin[0])
            check()
            result, _ = self.mcp.call('calendar', 'list_events', {'calendarId': scope['calendar_id'],
                'startTime': target['time_min'], 'endTime': target['time_max'], 'pageSize': 50, 'eventType': ['DEFAULT']},
                token=token, deadline=deadline, max_bytes=_READ_LIMIT, guard=check)
            check()
        except GoogleMcpError as exc:
            return self._blocked(exc)
        except OAuthError:
            return {'status': 'unavailable', 'reason': 'calendar_edit_reconnect_required'}
        events = result.get('events')
        matches = []
        if isinstance(events, list):
            matches = [value.get('id') for value in events if isinstance(value, dict)
                       and isinstance(value.get('summary'), str) and value['summary'].casefold() == summary.casefold()]
        if (result.get('nextPageToken') or any(result.get(key) for key in _OMITTED_FIELDS)
                or result.get('accessRole') not in ('owner', 'writer')
                or len(matches) != 1 or not isinstance(matches[0], str) or not _EVENT_ID.fullmatch(matches[0])):
            return {'status': 'unavailable', 'reason': 'calendar_edit_exact_event_required'}
        return matches[0]

    def reviews(self, session):
        if not isinstance(session, AccountSession):
            raise PermissionError('calendar_edit_owner_required')
        with self._lock:
            self._prune()
            pending = [review for review in self._reviews.values()
                       if review.actor == session.actor_user_id and review.state == 'pending']
        result = []
        for review in pending:
            try:
                self._authority(session, review.connection_id, review.grant_id, review.calendar_id, review=review)
            except (PermissionError, ValueError):
                continue
            with self._lock:
                if review.state == 'pending' and self.clock() < review.expires:
                    result.append(deepcopy(review.public))
        return {'reviews': result}

    def receipts(self, session, *, local_date):
        owner = self.accounts.users.admin_user()
        if (not isinstance(session, AccountSession) or owner is None
                or owner.user_id != session.actor_user_id or not owner.is_active
                or owner.role not in ('admin', 'owner') or not self.accounts.users.is_admin(owner.user_id)):
            raise PermissionError('calendar_edit_owner_required')
        if not isinstance(local_date, str) or not _DATE.fullmatch(local_date):
            raise ValueError('calendar_edit_receipt_date_invalid')
        try:
            day = date.fromisoformat(local_date)
            zone = ZoneInfo(self.accounts.users.timezone(owner.user_id))
            start = datetime.combine(day, time.min, zone).astimezone(timezone.utc)
            end = datetime.combine(day + timedelta(days=1), time.min, zone).astimezone(timezone.utc)
        except (ValueError, OverflowError):
            raise ValueError('calendar_edit_receipt_date_invalid') from None
        records = self.accounts.integrations.calendar_edit_receipts(owner.user_id, start.isoformat(timespec='microseconds'),
                                                                    end.isoformat(timespec='microseconds'))
        return {'receipts': records[:100][::-1], 'truncated': len(records) > 100}

    def discard_pending_conversation_review(self, actor, origin):
        if not origin:
            return
        with self._lock:
            self._reviews = {token: review for token, review in self._reviews.items()
                             if (review.actor, review.origin, review.state) != (actor, origin, 'pending')}

    def _lookup(self, session, token):
        if not isinstance(session, AccountSession) or not isinstance(token, str) or not _REVIEW_TOKEN.fullmatch(token):
            raise PermissionError('calendar_edit_review_denied')
        self._prune()
        review = self._reviews.get(token)
        if review is None or review.actor != session.actor_user_id or review.expires <= self.clock():
            raise PermissionError('calendar_edit_review_expired_or_denied')
        return review

    def cancel(self, session, *, review_token):
        with self._lock:
            review = self._lookup(session, review_token)
            if review.state == 'dispatching':
                return {'status': 'unknown', 'reason': 'calendar_edit_already_dispatched'}
            if review.state not in {'pending', 'checking'}:
                raise PermissionError('calendar_edit_review_expired_or_denied')
            if review.state == 'pending':
                del self._reviews[review_token]
            review.state = 'cancelled'
        return {'status': 'cancelled'}

    @staticmethod
    def _blocked(exc):
        reason = {'approval_required': 'calendar_edit_google_approval_required',
                  'service_disabled': 'calendar_edit_api_disabled'}.get(exc.reason, 'calendar_edit_provider_unavailable')
        return {'status': 'unavailable', 'reason': reason}

    def commit(self, session, *, review_token, notification_level=None, guard=lambda: None):
        with self._lock:
            review = self._lookup(session, review_token)
            if review.state != 'pending':
                raise PermissionError('calendar_edit_review_expired_or_denied')
            guests = review.public.get('guests')
            expected_notification_level = guests['notification_level'] if guests is not None else None
            if notification_level != expected_notification_level:
                return {'status': 'unavailable', 'reason': 'calendar_edit_notification_approval_required'}
            key = (review.calendar_id, review.event_id)
            if key in self._busy_events:
                return {'status': 'unavailable', 'reason': 'calendar_edit_busy'}
            approved_at = self.clock()
            review.state = 'checking'
            self._busy_events.add(key)
        try:
            return self._commit_claimed(session, review, guard, approved_at)
        finally:
            with self._lock:
                self._reviews.pop(review_token, None)
                self._busy_events.remove(key)

    def _commit_claimed(self, session, review, guard, approved_at):
        deadline = min(review.expires, self.clock() + timedelta(seconds=20))

        def check():
            guard()
            with self._lock:
                if review.state == 'cancelled' or self.clock() >= deadline:
                    raise PermissionError('calendar_edit_cancelled_or_expired')
            return self._authority(session, review.connection_id, review.grant_id, review.calendar_id, review=review)

        try:
            _, _, authorization, _ = check()
            oauth = self.oauth(session)
            if oauth is None:
                raise OAuthError('oauth_setup_unavailable')
            token = oauth.token(session, authorization=authorization, expected_connection_revision=review.connection_revision)
            check()
            current = self._read(review.calendar_id, review.event_id, token, deadline, check)
        except GoogleMcpError as exc:
            return self._blocked(exc)
        except (OAuthError, PermissionError):
            return {'status': 'unavailable', 'reason': 'calendar_edit_authority_unavailable'}
        if _digest(current) != review.snapshot:
            return {'status': 'conflict', 'reason': 'calendar_edit_changed_since_review'}
        try:
            check()
            with self._lock:
                if review.state != 'checking' or self.clock() >= deadline:
                    return {'status': 'unavailable', 'reason': 'calendar_edit_cancelled_or_expired'}
                review.state = 'dispatching'
        except PermissionError:
            return {'status': 'unavailable', 'reason': 'calendar_edit_authority_unavailable'}

        # Never retry a write. An unconfirmed request may already have changed Google.
        guests = review.public.get('guests')
        notification_level = guests['notification_level'] if guests is not None else 'NONE'
        result = None
        try:
            result, _ = self.mcp.call('calendar', 'update_event', {'calendarId': review.calendar_id,
                'eventId': review.event_id, 'notificationLevel': notification_level, **review.patch},
                token=token, deadline=deadline, max_bytes=_READ_LIMIT, guard=check)
        except GoogleMcpError as exc:
            if exc.reason in {'approval_required', 'service_disabled'}:
                return self._blocked(exc)
        except PermissionError:
            pass
        try:
            saved = self._read(review.calendar_id, review.event_id, token, deadline, check)
            same_scope = _scope(saved, review.event_id, inviting=guests is not None) == review.public['recurrence_scope']
            updated = _effect(saved, review.public['after'])
            matching_guests = (guests is None or _guests(saved).keys() == {
                guest['email'].lower() for group in (guests['current'], guests['added']) for guest in group
            })
            verified = (isinstance(result, dict) and result.get('id') == review.event_id
                        and same_scope and matching_guests and _matches(updated, review.public['after']))
        except (GoogleMcpError, PermissionError, ValueError, KeyError, TypeError):
            verified = False
        if not verified:
            return {'status': 'unknown', 'reason': 'calendar_edit_requires_reconciliation'}
        outcome = dict(status='updated', calendar_id=review.calendar_id, event_id=review.event_id,
                       recurrence_scope=review.public['recurrence_scope'], after=updated)
        start, end = saved['start'], saved['end']
        title = saved.get('summary')
        receipt = dict(id=str(uuid4()), approved_at=approved_at.astimezone(timezone.utc).isoformat(timespec='microseconds'),
                       event_title=' '.join(title.split())[:256] if isinstance(title, str) else review.public['event_title'],
                       event_start=start.get('date', start.get('dateTime')),
                       event_end=None if saved.get('endTimeUnspecified') else end.get('date', end.get('dateTime')),
                       recurrence_scope=review.public['recurrence_scope'], changes=updated)
        if guests is not None:
            outcome.update(guests_verified=True, notification_level=notification_level)
            verified_guests = _guests(saved)
            receipt['invited_guests'] = [verified_guests[guest['email'].lower()] for guest in guests['added']]
            receipt['notification_level'] = notification_level
        outcome['receipt'] = receipt
        try:
            self.accounts.integrations.save_calendar_edit_receipt(review.actor, receipt)
        except (OSError, sqlite3.Error):
            # Google already confirmed the effect. Keep the receipt in the IPC
            # response and never retry the provider mutation for a local failure.
            outcome['receipt_recorded'] = False
        return outcome
