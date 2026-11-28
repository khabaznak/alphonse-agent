"""The conversation tool can stage a review; only authenticated Desktop can commit."""
from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.tools.registry import ToolDefinition

OWNER_CALENDAR_EDIT_TOOL_ID = 'native.owner_calendar_edit_proposal'

_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['target', 'fields'],
    'properties': {
        'target': {'oneOf': [
            {'type': 'object', 'additionalProperties': False, 'required': ['event_id'], 'properties': {
                'event_id': {'type': 'string', 'description': 'An exact event ID or the Google Calendar event link supplied by the owner.'},
            }},
            {'type': 'object', 'additionalProperties': False, 'required': ['summary', 'time_min', 'time_max'], 'properties': {
                'summary': {'type': 'string', 'description': 'The exact current event title from the owner.'},
                'time_min': {'type': 'string', 'description': 'Beginning of a narrow lookup window, RFC3339 with explicit offset.'},
                'time_max': {'type': 'string', 'description': 'Exclusive end of the lookup window, RFC3339 with explicit offset.'},
            }},
        ]},
        'fields': {'type': 'object', 'additionalProperties': False, 'minProperties': 1, 'properties': {
            'summary': {'type': 'string'}, 'description': {'type': 'string'}, 'location': {'type': 'string'},
            'invite_guests': {'type': 'array', 'minItems': 1, 'uniqueItems': True, 'items': {'type': 'string'},
                'description': 'Exact email addresses explicitly supplied or confirmed by the owner for new guests. '
                               'The review shows every attendee Google is asked to notify, including existing guests.'},
            'start': {'type': 'string', 'description': 'New start, RFC3339 with offset or YYYY-MM-DD for all-day.'},
            'end': {'type': 'string', 'description': 'New end, RFC3339 with offset or exclusive YYYY-MM-DD for all-day.'},
            'time_zone': {'type': 'string', 'description': 'Explicit IANA timezone for a timed change; omit for an all-day change.'},
        }},
        'recurrence_scope': {'enum': ['event', 'instance', 'series'], 'description': 'Explicit owner choice when changing a recurring event.'},
    },
}

_FAILURES = frozenset({
    'calendar_edit_access_required', 'calendar_edit_select_one_write_calendar',
    'calendar_edit_exact_event_required', 'calendar_edit_reconnect_required',
    'calendar_edit_google_approval_required', 'calendar_edit_api_disabled', 'calendar_edit_provider_unavailable',
    'calendar_edit_no_change', 'calendar_edit_event_id_invalid_or_wrong_calendar',
    'calendar_edit_time_invalid', 'calendar_edit_explicit_recurrence_scope_required',
    'calendar_edit_event_unavailable_or_unsupported', 'calendar_edit_target_invalid',
    'calendar_edit_lookup_window_invalid', 'calendar_edit_fields_invalid',
    'calendar_edit_invite_addresses_invalid', 'calendar_edit_invite_already_attending',
    'calendar_edit_guest_list_unavailable',
    'calendar_edit_reviews_busy', 'calendar_edit_review_already_pending',
})

_REVIEW_READY = ('I prepared your Calendar edit for review. While the review is available above the message composer, '
                 'you can approve it there to send the update to Google Calendar.')
_FAILURE_TEXT = {
    'calendar_edit_access_required': 'Connect Google Calendar editing and grant access to the specific calendar in Settings.',
    'calendar_edit_select_one_write_calendar': 'Select a single calendar for editing in Settings before trying again.',
    'calendar_edit_exact_event_required': 'Send the exact Google Calendar event link so I can prepare a review for the right event.',
    'calendar_edit_reconnect_required': 'Reconnect Google Calendar editing in Settings and ask for the edit again.',
    'calendar_edit_google_approval_required': 'Google requested additional approval before I could prepare the Calendar review.',
    'calendar_edit_api_disabled': 'Enable the Calendar MCP service for Alphonse’s Google Cloud project and ask for the edit again.',
    'calendar_edit_no_change': 'The requested values already match the Calendar event.',
    'calendar_edit_explicit_recurrence_scope_required': 'Please specify whether to change this occurrence or the entire series.',
    'calendar_edit_time_invalid': 'Please provide the new start and end, including a time zone for a timed event.',
    'calendar_edit_invite_addresses_invalid': 'Please provide a distinct email address for each person you want to invite.',
    'calendar_edit_invite_already_attending': 'At least one requested person is already on the guest list. Please confirm which new guests to invite.',
    'calendar_edit_guest_list_unavailable': 'I could not confirm the full guest list for this event. Check the guest list in Google Calendar.',
}


def _trusted(context):
    return (type(context.messages) is SQLiteMessageQueue and isinstance(context.task.message_id, str)
            and bool(context.task.message_id) and context.owner_calendar_edit_service is not None
            and callable(context.cancellation_checker) and context.cancellation_checker() is False
            and context.messages.verify_trusted_local_owner_claim(context.claim_receipt, context.task))


def valid_calendar_edit_observation(payload):
    return (payload == {'status': 'review_required'} or isinstance(payload, dict)
            and set(payload) == {'status', 'reason'} and payload['status'] == 'unavailable'
            and isinstance(payload['reason'], str) and payload['reason'] in _FAILURES)


def calendar_edit_response(payload):
    if payload['status'] == 'review_required':
        return _REVIEW_READY
    return _FAILURE_TEXT.get(payload['reason'], 'I could not prepare the Calendar edit. Review the target and try again.')


def execute_calendar_edit_proposal(arguments, *, context):
    def guard():
        if not _trusted(context):
            raise PermissionError('calendar_edit_local_owner_claim_required')

    guard()
    try:
        return context.owner_calendar_edit_service.propose_for_conversation(AccountSession(context.task.user),
            **arguments, origin=context.task.message_id, guard=guard)
    except ValueError as exc:
        if len(exc.args) == 1 and exc.args[0] in _FAILURES:
            return {'status': 'unavailable', 'reason': exc.args[0]}
        raise


def build_owner_calendar_edit_tool_definition():
    descriptor = ToolDescriptor(OWNER_CALENDAR_EDIT_TOOL_ID, 'owner_calendar_edit_proposal', ToolKind.NATIVE,
        'Prepare a private Desktop review for an explicitly requested edit to an existing Google Calendar event. '
        'Use this tool when the owner asks to change an event. Resolve the intended event with the supplied exact ID '
        'or link, or provide its exact title and a narrow offset-bearing time window. Ask if a material detail is '
        'ambiguous. For a time change, supply both times and the IANA zone for a timed event. A recurring edit '
        'requires the owner to choose instance or series. The tool reads Calendar and stages a local review only. '
        'For an invitation, use only exact guest email addresses supplied or confirmed by the owner. '
        'A review requires Desktop approval before any Calendar mutation.',
        _SCHEMA, ('calendar',), ('native', 'calendar'), read_only=True,
        metadata={'sensitive_content': True})
    return ToolDefinition(descriptor, execute_calendar_edit_proposal, _SCHEMA, accepts_context=True)
