"""Claim-scoped, short-lived observation of the host's configured clock.

This clock reports the host system clock; it does not assert clock accuracy.
"""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition

OWNER_CLOCK_TOOL_ID = 'native.owner_clock'
MAX_CLOCK_AGE = timedelta(minutes=5)


def execute_owner_clock(arguments, *, context):
    if arguments != {} or context.cancellation_checker() is not False:
        raise ValueError('owner_clock_unavailable')
    actor = context.task.user
    zone = context.user_timezone_provider(actor)
    local_zone = ZoneInfo(zone)
    instant = datetime.now(timezone.utc)
    if context.cancellation_checker() is not False:
        raise ValueError('owner_clock_unavailable')
    return {'observed_at_utc': instant.isoformat(), 'timezone': zone,
            'local_time': instant.astimezone(local_zone).isoformat()}


def valid_clock_observation(actor, arguments, observation, *, context, allow_expired=False):
    try:
        if (context.task.user != actor or arguments != {} or context.cancellation_checker() is not False
                or not isinstance(observation, dict)
                or set(observation) != {'observed_at_utc', 'timezone', 'local_time'}
                or not all(isinstance(value, str) for value in observation.values())
                or context.user_timezone_provider(actor) != observation['timezone']):
            return False
        instant = datetime.fromisoformat(observation['observed_at_utc'])
        if instant.tzinfo != timezone.utc or instant.isoformat() != observation['observed_at_utc']:
            return False
        local = instant.astimezone(ZoneInfo(observation['timezone'])).isoformat()
        age = datetime.now(timezone.utc) - instant
        return (type(allow_expired) is bool and observation['local_time'] == local and timedelta(0) <= age
                and (allow_expired or age <= MAX_CLOCK_AGE)
                and context.cancellation_checker() is False)
    except Exception:
        return False


def build_owner_clock_tool_definition():
    schema = {'type': 'object', 'properties': {}, 'additionalProperties': False}
    return ToolDefinition(descriptor=ToolDescriptor(tool_id=OWNER_CLOCK_TOOL_ID, name='owner_clock',
        kind=ToolKind.NATIVE, description='For current date or time, read the host system clock in the owner configured timezone. Describe it as of the observed instant; it is not the future delivery time.',
        argument_schema=schema, capabilities=('clock',), read_only=True), callable=execute_owner_clock,
        argument_schema=schema, accepts_context=True)


def display_response(response, bindings, *, private_desktop=False, timezone_name=None):
    from .owner_calendar import OWNER_CALENDAR_TOOL_ID, format_calendar_observation
    from .owner_schedules import OWNER_SCHEDULES_TOOL_ID
    from .owner_gmail import OWNER_GMAIL_TOOL_ID, owner_gmail_failure_message
    from .owner_outlook import OWNER_OUTLOOK_TOOL_ID, format_outlook_observation
    from .owner_gmail_search import OWNER_GMAIL_SEARCH_TOOL_ID, final_gmail_search_result, format_gmail_search_observation
    if any(binding.get('tool_id') == OWNER_SCHEDULES_TOOL_ID for binding in bindings.values()):
        response += '\n\nEarlier scheduled work is paused. New Desktop reminders appear only while Desktop is open.'
    gmail = [binding['gmail_observation'] for binding in bindings.values()
             if binding.get('tool_id') == OWNER_GMAIL_TOOL_ID]
    provider_reads = []
    if gmail:
        # A model draft can contradict a provider count. For this narrow slice,
        # deliver only the verified observation and explicitly limit its scope.
        item = gmail[-1]
        failure = owner_gmail_failure_message(item)
        if failure is None:
            unit = 'thread' if item['unit'] == 'threads' else 'message'
            if item['count'] != 1:
                unit += 's'
            provider_reads.append(f"Gmail INBOX unread {unit}: {item['count']} (as of {item['observed_at']}). "
                                      'This response covers the inbox count only.')
        else:
            provider_reads.append(failure)
    for binding in bindings.values():
        if binding.get('tool_id') == OWNER_CALENDAR_TOOL_ID:
            provider_reads.append(format_calendar_observation(binding['calendar_observation'], private_delivery=private_desktop))
        if binding.get('tool_id') == OWNER_OUTLOOK_TOOL_ID:
            provider_reads.append(format_outlook_observation(binding['outlook_observation'],
                private_delivery=private_desktop, timezone_name=timezone_name))
    searches = [binding['gmail_search_observation'] for binding in bindings.values()
                if binding.get('tool_id') == OWNER_GMAIL_SEARCH_TOOL_ID]
    if searches:
        provider_reads.append(format_gmail_search_observation(final_gmail_search_result(searches), response,
            private_delivery=private_desktop, searches=searches))
    if provider_reads:
        return '\n\n'.join(provider_reads)
    clocks = [binding['clock_observation'] for binding in bindings.values()
              if binding.get('tool_id') == OWNER_CLOCK_TOOL_ID]
    if not clocks:
        return response
    notes = [f"{item['local_time']} ({item['timezone']}; UTC {item['observed_at_utc']})" for item in clocks]
    return response + '\n\nHost clock observed as of: ' + '; '.join(notes)
