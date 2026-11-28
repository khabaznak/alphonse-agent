"""Calendar-edit Dojo fixtures using a scripted model and disposable MCP data."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
import sqlite3
from types import SimpleNamespace
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import pytest

from alphonse.agent_v2.accounts import CALENDAR_WRITE_OPERATION, CALENDAR_WRITE_SCOPE
from alphonse.agent_v2.core.inference.models import InferenceResult
from alphonse.agent_v2.core.intelligence.simple_loop.owner_calendar_edit import OWNER_CALENDAR_EDIT_TOOL_ID
from alphonse.agent_v2.integrations.calendar_edits import CalendarEditService
from alphonse.agent_v2.integrations.google_mcp import GoogleMcpError
from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
from alphonse.agent_v2.ipc import PROTOCOL_VERSION
from test_calendar_acquisition import Wire
from test_owner_calendar import CALENDAR_ID, PAGE, WINDOW, calendar_worker, worker_status
from test_owner_gmail import f


EVENT_ID = 'pottery_handoff1'
EVENT = {**PAGE['events'][0], 'id': EVENT_ID, 'status': 'confirmed', 'location': 'Workshop A',
         'updated': '2026-11-25T12:00:00Z'}
CURRENT_GUEST = {'email': 'riley.chen@example.com', 'displayName': 'Riley Chen'}
NEW_GUEST = 'maya.santos@example.com'


@pytest.fixture
def edit_service(f):
    f.service.begin(f.owner, connection_id='write', provider='google', capabilities=[CALENDAR_WRITE_OPERATION],
                    expected_revision=0)
    client = f.service.clients['google']
    assert CALENDAR_WRITE_SCOPE in client.scopes
    f.service.complete(f.owner, client.config.redirect_uri + '?' + urlencode({'state': client.state, 'code': 'disposable-code'}))
    f.accounts.save_grant(f.owner, grant_id='write-grant', connection_id='write', expected_revision=0,
        beneficiary_user_id='owner', operations=[CALENDAR_WRITE_OPERATION], resources=[CALENDAR_ID],
        recipient_user_ids=['owner'], expires_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat())
    backend = CalendarWire()
    return CalendarEditService(f.accounts, lambda _: f.service, mcp=backend), backend


class CalendarWire:
    def __init__(self):
        self.event = deepcopy(EVENT)
        self.calls = []
        self.update_error = None
        self.lose_response = False
        self.before_read = None
        self.readback = None
        self.list_metadata = {}

    def call(self, product, tool, arguments, *, token, deadline, max_bytes, guard):
        guard()
        assert product == 'calendar' and arguments['calendarId'] == CALENDAR_ID
        assert token == 'disposable-access'
        self.calls.append((tool, arguments))
        if tool == 'list_events':
            return {**self.list_metadata, 'accessRole': 'owner', 'events': [deepcopy(self.event)]}, 100
        assert arguments['eventId'] == self.event['id']
        if tool == 'get_event':
            if self.before_read is not None:
                self.before_read()
            snapshot = deepcopy(self.event)
            if self.readback is not None and any(item[0] == 'update_event' for item in self.calls):
                self.readback(snapshot)
            return snapshot, 100
        assert tool == 'update_event'
        if self.update_error:
            raise self.update_error
        for key in ('summary', 'description', 'location'):
            if key in arguments:
                self.event[key] = arguments[key]
        if 'startTime' in arguments:
            for side in ('start', 'end'):
                self.event[side] = {'date': arguments[f'{side}Time']} if arguments['allDay'] else {
                    'dateTime': arguments[f'{side}Time'], 'timeZone': arguments['timeZone'],
                }
        if 'addedAttendees' in arguments:
            self.event.setdefault('attendees', []).extend(deepcopy(arguments['addedAttendees']))
        self.event['updated'] = '2026-11-27T12:00:00Z'
        if self.lose_response:
            raise GoogleMcpError(504, 'unavailable')
        return deepcopy(self.event), 100


def propose(service, session, **kwargs):
    fields = kwargs.pop('fields', {'location': 'Workshop B'})
    return service.propose(session, connection_id='write', grant_id='write-grant', calendar_id=CALENDAR_ID,
                           event_id=EVENT_ID, fields=fields, **kwargs)


def test_owner_review_approval_updates_and_verifies_once(f, edit_service):
    service, wire = edit_service
    review = propose(service, f.owner)
    assert review['before'] == {'location': 'Workshop A'} and review['after'] == {'location': 'Workshop B'}
    assert review['event_start'] == '2026-12-01T09:15:00-06:00'
    token = review['review_token']
    assert service.reviews(f.owner) == {'reviews': [review]}
    assert service.reviews(f.other) == {'reviews': []}
    with pytest.raises(PermissionError):
        service.commit(f.other, review_token=token)
    assert [tool for tool, _ in wire.calls] == ['get_event']
    assert service.commit(f.owner, review_token=token)['status'] == 'updated'
    assert wire.calls == [('get_event', {'calendarId': CALENDAR_ID, 'eventId': EVENT_ID})] * 2 + [
        ('update_event', {'calendarId': CALENDAR_ID, 'eventId': EVENT_ID,
                          'notificationLevel': 'NONE', 'location': 'Workshop B'}),
        ('get_event', {'calendarId': CALENDAR_ID, 'eventId': EVENT_ID}),
    ]
    with pytest.raises(PermissionError):
        service.commit(f.owner, review_token=token)


def test_owner_review_duration_is_bounded_by_the_exact_write_grant(f, edit_service):
    service, wire = edit_service
    now = datetime.now(timezone.utc)
    service.clock = lambda: now
    grant_expiry = now + timedelta(minutes=12)
    f.accounts.save_grant(f.owner, grant_id='write-grant', connection_id='write', expected_revision=1,
        beneficiary_user_id='owner', operations=[CALENDAR_WRITE_OPERATION], resources=[CALENDAR_ID],
        recipient_user_ids=['owner'], expires_at=grant_expiry.isoformat())

    for invalid in (4, 31, True, '30'):
        with pytest.raises(ValueError, match='calendar_edit_review_duration_invalid'):
            propose(service, f.owner, review_minutes=invalid)
    assert wire.calls == []

    ordinary = propose(service, f.owner)
    assert datetime.fromisoformat(ordinary['expires_at']) == now + timedelta(minutes=5)
    service.cancel(f.owner, review_token=ordinary['review_token'])

    extended = propose(service, f.owner, review_minutes=30)
    assert datetime.fromisoformat(extended['expires_at']) == grant_expiry
    service.clock = lambda: now + timedelta(minutes=6)
    assert service.reviews(f.owner)['reviews'] == [extended]
    service.clock = lambda: grant_expiry
    with pytest.raises(PermissionError, match='calendar_edit_review_expired_or_denied'):
        service.commit(f.owner, review_token=extended['review_token'])
    assert [tool for tool, _ in wire.calls] == ['get_event', 'get_event']


def test_review_lists_every_invitation_recipient_and_only_approval_dispatches(f, edit_service):
    service, wire = edit_service
    wire.event['attendees'] = [CURRENT_GUEST]
    fields = {'description': 'Meet by the shelving.', 'invite_guests': [NEW_GUEST]}
    review = propose(service, f.owner, fields=fields)
    expected_guests = {'current': [{'email': CURRENT_GUEST['email'], 'name': 'Riley Chen'}],
                       'added': [{'email': NEW_GUEST}], 'notification_level': 'ALL'}
    assert review['guests'] == expected_guests
    assert review['before'] == {'description': EVENT['description']}
    assert review['after'] == {'description': fields['description']}
    assert 'who receives email' in review['concurrent_edit_warning']
    assert [tool for tool, _ in wire.calls] == ['get_event']
    assert service.reviews(f.other) == {'reviews': []}

    # The public snapshot is a copy and cannot change the pending effect.
    review['guests']['added'][0]['email'] = 'unexpected@example.com'
    assert service.reviews(f.owner)['reviews'][0]['guests'] == expected_guests
    with pytest.raises(PermissionError):
        service.commit(f.other, review_token=review['review_token'], notification_level='ALL')
    result = service.commit(f.owner, review_token=review['review_token'], notification_level='ALL')
    receipt = result.pop('receipt')
    assert result == {'status': 'updated', 'calendar_id': CALENDAR_ID, 'event_id': EVENT_ID,
        'recurrence_scope': 'event', 'after': {'description': fields['description']},
        'guests_verified': True, 'notification_level': 'ALL'}
    assert receipt['invited_guests'] == [{'email': NEW_GUEST}]
    assert receipt['changes'] == {'description': fields['description']}
    assert receipt['notification_level'] == 'ALL'
    assert [tool for tool, _ in wire.calls] == ['get_event', 'get_event', 'update_event', 'get_event']
    assert wire.calls[-2][1] == {'calendarId': CALENDAR_ID, 'eventId': EVENT_ID,
        'notificationLevel': 'ALL', 'description': fields['description'], 'addedAttendees': [{'email': NEW_GUEST}]}
    assert [guest['email'] for guest in wire.event['attendees']] == [CURRENT_GUEST['email'], NEW_GUEST]
    with pytest.raises(PermissionError):
        service.commit(f.owner, review_token=review['review_token'], notification_level='ALL')


def test_verified_receipts_are_durable_and_scoped_to_owner_approval_day(f, edit_service):
    service, wire = edit_service
    f.accounts.users.set_timezone('America/Mexico_City', f.owner.actor_user_id)
    approved_at = datetime.now(timezone.utc).replace(hour=2, minute=30)
    service.clock = lambda: approved_at
    approval_day = (approved_at - timedelta(days=1)).date().isoformat()
    next_day = approved_at.date().isoformat()
    review = propose(service, f.owner, fields={'summary': 'New event title'})
    assert service.receipts(f.owner, local_date=approval_day) == {'receipts': [], 'truncated': False}

    receipt = service.commit(f.owner, review_token=review['review_token'])['receipt']
    assert receipt['event_title'] == 'New event title'
    assert receipt['event_start'] == EVENT['start']['dateTime']
    assert receipt['event_end'] == EVENT['end']['dateTime']
    assert receipt['changes'] == {'summary': 'New event title'}
    assert receipt['approved_at'] == approved_at.isoformat(timespec='microseconds')
    assert review['review_token'] not in json.dumps(receipt)
    assert CALENDAR_ID not in json.dumps(receipt)

    reopened = SQLiteIntegrationStore(f.store.db_path)
    assert reopened.calendar_edit_receipts(f.owner.actor_user_id,
        (approved_at - timedelta(seconds=1)).isoformat(), (approved_at + timedelta(seconds=1)).isoformat()) == [receipt]
    assert service.receipts(f.owner, local_date=approval_day) == {'receipts': [receipt], 'truncated': False}
    assert service.receipts(f.owner, local_date=next_day) == {'receipts': [], 'truncated': False}
    with pytest.raises(PermissionError):
        service.receipts(f.other, local_date=approval_day)
    with pytest.raises(ValueError, match='calendar_edit_receipt_date_invalid'):
        service.receipts(f.owner, local_date='2026-02-30')
    assert [tool for tool, _ in wire.calls].count('update_event') == 1


def test_local_receipt_storage_failure_does_not_hide_or_retry_verified_google_effect(f, edit_service, monkeypatch):
    service, wire = edit_service
    review = propose(service, f.owner)

    def storage_failure(*args):
        raise sqlite3.OperationalError('disposable storage failure')

    monkeypatch.setattr(f.store, 'save_calendar_edit_receipt', storage_failure)
    result = service.commit(f.owner, review_token=review['review_token'])
    assert result['status'] == 'updated' and result['receipt_recorded'] is False
    assert result['receipt']['changes'] == {'location': 'Workshop B'}
    assert [tool for tool, _ in wire.calls].count('update_event') == 1


@pytest.mark.parametrize('requested', [
    [], 'maya.santos@example.com', ['Maya <maya.santos@example.com>'], [None], ['maya@@example.com'],
    ['maya.santos@example.com', 'MAYA.SANTOS@example.com'], ['a@example.com,b@example.com'],
    ['a@example.com\n'], ['maya@example.com (Maya)'],
])
def test_invalid_invitation_recipients_fail_before_reading_or_writing(f, edit_service, requested):
    service, wire = edit_service
    with pytest.raises(ValueError, match='calendar_edit_invite_addresses_invalid'):
        propose(service, f.owner, fields={'invite_guests': requested})
    assert wire.calls == []


@pytest.mark.parametrize('attendee_fields', [
    {'attendees': None}, {'attendees': [{'displayName': 'Redacted guest'}]},
    {'attendees': [{'email': 'bad address'}]}, {'attendees': [{'email': 'riley.chen@example.com', 'additionalGuests': 1}]},
    {'attendees': [CURRENT_GUEST, {**CURRENT_GUEST, 'email': CURRENT_GUEST['email'].upper()}]},
    {'attendees': [CURRENT_GUEST], 'attendeesOmitted': True},
    {'attendees': [CURRENT_GUEST], 'organizer': {'self': False}, 'guestPermissions': {'guestsCanSeeGuests': False}},
])
def test_incomplete_recipient_evidence_prevents_a_review(f, edit_service, attendee_fields):
    service, wire = edit_service
    wire.event.update(attendee_fields)
    with pytest.raises(ValueError, match='calendar_edit_guest_list_unavailable'):
        propose(service, f.owner, fields={'invite_guests': [NEW_GUEST]})
    assert [tool for tool, _ in wire.calls] == ['get_event']
    assert service.reviews(f.owner) == {'reviews': []}


def test_do_not_reinvite_existing_guests_or_silently_enable_other_guest_edits(f, edit_service):
    service, wire = edit_service
    wire.event['attendees'] = [CURRENT_GUEST]
    with pytest.raises(ValueError, match='calendar_edit_invite_already_attending'):
        propose(service, f.owner, fields={'invite_guests': [NEW_GUEST, CURRENT_GUEST['email'].upper()]})
    with pytest.raises(ValueError, match='calendar_edit_event_unavailable_or_unsupported'):
        propose(service, f.owner)
    assert [tool for tool, _ in wire.calls] == ['get_event', 'get_event']


def test_invite_only_review_preserves_event_fields_and_requires_recurrence_scope(f, edit_service):
    service, wire = edit_service
    wire.event['recurringEventId'] = 'pottery_series'
    address = 'maya.santos+pottery@example.com'
    fields = {'invite_guests': [address]}
    with pytest.raises(ValueError, match='calendar_edit_explicit_recurrence_scope_required'):
        propose(service, f.owner, fields=fields)
    review = propose(service, f.owner, fields=fields, recurrence_scope='instance')
    assert review['recurrence_scope'] == 'instance' and review['before'] == review['after'] == {}
    assert review['guests'] == {'current': [], 'added': [{'email': address}], 'notification_level': 'ALL'}
    result = service.commit(f.owner, review_token=review['review_token'], notification_level='ALL')
    assert result['status'] == 'updated' and result['guests_verified']
    assert wire.calls[-2][1] == {'calendarId': CALENDAR_ID, 'eventId': EVENT_ID,
        'notificationLevel': 'ALL', 'addedAttendees': [{'email': address}]}
    assert wire.event['location'] == EVENT['location']


def test_guest_list_changed_after_review_never_dispatches(f, edit_service):
    service, wire = edit_service
    review = propose(service, f.owner, fields={'invite_guests': [NEW_GUEST]})
    wire.event['attendees'] = [CURRENT_GUEST]
    assert service.commit(f.owner, review_token=review['review_token'], notification_level='ALL')['status'] == 'conflict'
    assert [tool for tool, _ in wire.calls] == ['get_event', 'get_event']


@pytest.mark.parametrize('case', ['missing_new', 'missing_existing', 'extra', 'omitted', 'lost_response'])
def test_invitations_without_exact_post_update_evidence_are_unknown_and_not_retried(f, edit_service, case):
    service, wire = edit_service
    wire.event['attendees'] = [CURRENT_GUEST]
    review = propose(service, f.owner, fields={'invite_guests': [NEW_GUEST]})

    def readback(snapshot):
        if case == 'missing_new':
            snapshot['attendees'].pop()
        elif case == 'missing_existing':
            snapshot['attendees'].pop(0)
        elif case == 'extra':
            snapshot['attendees'].append({'email': 'unexpected@example.com'})
        elif case == 'omitted':
            snapshot['attendeesOmitted'] = True

    wire.readback = readback
    wire.lose_response = case == 'lost_response'
    assert service.commit(f.owner, review_token=review['review_token'], notification_level='ALL') == {
        'status': 'unknown', 'reason': 'calendar_edit_requires_reconciliation',
    }
    zone = ZoneInfo(f.accounts.users.timezone(f.owner.actor_user_id))
    approval_day = service.clock().astimezone(zone).date().isoformat()
    assert service.receipts(f.owner, local_date=approval_day)['receipts'] == []
    with pytest.raises(PermissionError):
        service.commit(f.owner, review_token=review['review_token'], notification_level='ALL')
    assert sum(tool == 'update_event' for tool, _ in wire.calls) == 1


@pytest.mark.parametrize('case, status', [('newer', 'conflict'), ('google_approval', 'unavailable'),
    ('lost_response', 'unknown'), ('revoked', 'unavailable'), ('cancelled_during_read', 'unavailable')])
def test_observed_conflicts_cancel_and_uncertain_updates_fail_closed(f, edit_service, case, status):
    service, wire = edit_service
    token = propose(service, f.owner)['review_token']
    if case == 'newer':
        wire.event['description'] = 'Google edit after review'
    elif case == 'google_approval':
        wire.update_error = GoogleMcpError(409, 'approval_required')
    elif case == 'lost_response':
        wire.lose_response = True
    elif case == 'revoked':
        f.accounts.revoke_grant(f.owner, grant_id='write-grant', expected_revision=1)
    elif case == 'cancelled_during_read':
        wire.before_read = lambda: service.cancel(f.owner, review_token=token)
    assert service.commit(f.owner, review_token=token)['status'] == status
    with pytest.raises(PermissionError):
        service.commit(f.owner, review_token=token)
    assert sum(tool == 'update_event' for tool, _ in wire.calls) == (case in {'lost_response', 'google_approval'})
    assert wire.event['location'] == ('Workshop B' if case == 'lost_response' else 'Workshop A')


def test_write_authorization_cannot_be_used_by_generic_bindings(f, edit_service):
    with pytest.raises(PermissionError):
        f.accounts.authorize(f.owner, connection_id='write', grant_id='write-grant', operation=CALENDAR_WRITE_OPERATION,
            resources=[CALENDAR_ID], recipient_user_ids=['owner'], required_scopes=[CALENDAR_WRITE_SCOPE])


def test_title_lookup_refuses_incomplete_results(f, edit_service):
    service, wire = edit_service
    wire.list_metadata['omittedContent'] = ['event']
    result = service.propose_for_conversation(f.owner, target={'summary': EVENT['summary'], **WINDOW},
        fields={'location': 'Workshop B'}, origin='disposable-message', guard=lambda: None)
    assert result == {'status': 'unavailable', 'reason': 'calendar_edit_exact_event_required'}
    assert [tool for tool, _ in wire.calls] == ['list_events'] and service.reviews(f.owner) == {'reviews': []}


def test_moving_occurrence_requires_explicit_scope_and_reviews_exact_timing(f, edit_service):
    service, wire = edit_service
    wire.event['recurringEventId'] = 'pottery_series'
    args = dict(connection_id='write', grant_id='write-grant', calendar_id=CALENDAR_ID, event_id=EVENT_ID,
                fields={'start': '2026-12-01T11:00:00-06:00', 'end': '2026-12-01T12:00:00-06:00',
                        'time_zone': 'America/Mexico_City'})
    with pytest.raises(ValueError, match='calendar_edit_explicit_recurrence_scope_required'):
        service.propose(f.owner, **args)
    review = service.propose(f.owner, recurrence_scope='instance', **args)
    assert review['before']['start'] == EVENT['start']['dateTime'] and review['after'] == args['fields']
    result = service.commit(f.owner, review_token=review['review_token'])
    assert result['status'] == 'updated' and result['after'] == args['fields']
    assert wire.calls[-2][1] == {'calendarId': CALENDAR_ID, 'eventId': EVENT_ID, 'notificationLevel': 'NONE',
        'startTime': args['fields']['start'], 'endTime': args['fields']['end'], 'timeZone': 'America/Mexico_City',
        'allDay': False}


class ScriptedCalendarEdit:
    supports_image_input = False

    def __init__(self, fields):
        self.fields = fields
        self.requests = []

    def generate_json(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            action = {'kind': 'tool', 'tool_id': OWNER_CALENDAR_EDIT_TOOL_ID, 'text': None, 'arguments': {
                'target': {'summary': EVENT['summary'], **WINDOW}, 'fields': self.fields,
            }}
        else:
            action = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': (
                'I already changed the event, updated the location, and verified the result. '
                'There is no further approval required to apply this change in Calendar.')}
        return InferenceResult(json_value=action, raw_response=json.dumps(action),
                               usage={'input_tokens': 100, 'output_tokens': 20})


@pytest.mark.parametrize('inviting', [False, True])
def test_local_conversation_stages_review_and_only_desktop_ipc_can_commit(f, edit_service, tmp_path, inviting):
    service, wire = edit_service
    def configure(runtime):
        runtime.integration_management.calendar_edits = service

    fields = {'location': 'Workshop B'}
    if inviting:
        fields['invite_guests'] = [NEW_GUEST]
        wire.event['attendees'] = [CURRENT_GUEST]
    provider = ScriptedCalendarEdit(fields)
    with calendar_worker(f, tmp_path, Wire(PAGE), provider, configure_runtime=configure) as (daemon, queue, _, _):
        daemon.ipc._accepting_handlers = True
        peer = SimpleNamespace(getpeereid=lambda: (os.getuid(), 0))

        def request(action, **values):
            return daemon.ipc._optional_management(peer, {'version': PROTOCOL_VERSION,
                'params': {'action': action, 'values': values}})

        text = f'Mueve {EVENT["summary"]} al Workshop B el 1 de diciembre de 2026.'
        if inviting:
            text += f' Invita a {NEW_GUEST}.'
        message = queue.enqueue_trusted_local_owner(prompt=text, peer_uid=os.getuid())
        assert worker_status(lambda: daemon.owner_message_status(message)) == 'ready'
        assert [tool for tool, _ in wire.calls] == ['list_events', 'get_event']
        [review] = request('calendar_edit_reviews')['reviews']
        prompt = '\n'.join(request.prompt for request in provider.requests)
        assert review['review_token'] not in prompt and EVENT['description'] not in prompt
        assert CURRENT_GUEST['email'] not in prompt
        approval = {'notification_level': 'ALL'} if inviting else {}
        wrong_approvals = ({}, {'notification_level': 'NONE'}) if inviting else ({'notification_level': 'ALL'},)
        for wrong in wrong_approvals:
            assert request('calendar_edit_commit', review_token=review['review_token'], **wrong) == {
                'status': 'unavailable', 'reason': 'calendar_edit_notification_approval_required',
            }
            assert [tool for tool, _ in wire.calls] == ['list_events', 'get_event']
            assert request('calendar_edit_reviews') == {'reviews': [review]}
        result = request('calendar_edit_commit', review_token=review['review_token'], **approval)
        assert result['status'] == 'updated' and wire.event['location'] == 'Workshop B'
        zone = ZoneInfo(f.accounts.users.timezone(f.owner.actor_user_id))
        approval_day = datetime.fromisoformat(result['receipt']['approved_at']).astimezone(zone).date().isoformat()
        assert request('calendar_edit_receipts', local_date=approval_day)['receipts'] == [result['receipt']]
        if inviting:
            assert result['guests_verified'] and wire.calls[-2][1]['notificationLevel'] == 'ALL'
        with pytest.raises(PermissionError):
            request('calendar_edit_commit', review_token=review['review_token'], **approval)
        assert request('calendar_edit_reviews') == {'reviews': []}
        # The owner can review and approve before picking up the chat reply.
        assert 'prepared your Calendar edit for review' in daemon.take_owner_desktop_draft(message)['message']
