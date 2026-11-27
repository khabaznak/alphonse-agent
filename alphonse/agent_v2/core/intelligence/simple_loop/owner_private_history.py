"""Re-render acknowledged provider snapshots for the authenticated owner Desktop."""
from datetime import datetime
import re

from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.interfaces.private_provider_a2ui import calendar_agenda_surface, gmail_summary_surface
from .owner_calendar import OWNER_CALENDAR_TOOL_ID, valid_calendar_card_history
from .owner_gmail_search import (
    OWNER_GMAIL_SEARCH_TOOL_ID, final_gmail_search_result, gmail_source_title, gmail_summary_text,
    valid_gmail_card_history,
)


def render_private_card(message_id, response, observations, timezone_name):
    searches = [item['gmail_search_observation'] for item in observations.values()
                if item.get('tool_id') == OWNER_GMAIL_SEARCH_TOOL_ID]
    mail = final_gmail_search_result(searches)
    if mail is not None and mail['status'] in {'complete', 'partial'}:
        return 'gmail', gmail_summary_surface(message_id, mail, source=gmail_source_title(mail),
            summary=gmail_summary_text(response), searches=searches, timezone_name=timezone_name)
    for item in observations.values():
        if item.get('tool_id') == OWNER_CALENDAR_TOOL_ID:
            calendar = item['calendar_observation']
            if calendar['status'] in {'complete', 'partial'}:
                return 'calendar', calendar_agenda_surface(message_id, calendar, timezone_name=timezone_name)
    return None


def private_cards_for_desktop_history(queue, *, owner, events, timezone_name):
    """Recheck the finalized reply and current exact grant in one read transaction."""
    from .owner_atomic_checkpoint import _finalized_owner_reply

    if type(queue) is not SQLiteMessageQueue or not queue.experimental_claim_fencing:
        return {}
    receipts = {
        'Gmail Inbox matches were displayed privately in Desktop.',
        'A Gmail label search result was displayed privately in Desktop.',
        'Google Calendar results were displayed privately in Desktop.',
    }
    eligible = [event for event in events if event.owner_user_id == owner and event.role == 'assistant'
                and event.source == 'desktop' and event.visibility == 'private'
                and any(item in receipts for item in event.content.split('\n\n'))
                and re.fullmatch(r'outbound:owner:[0-9a-f]{32}', event.source_message_id)]
    if not eligible:
        return {}
    cards = {}
    with queue._connect() as conn:
        conn.execute('BEGIN')
        for event in eligible:
            try:
                message_id = event.source_message_id.removeprefix('outbound:owner:')
                row = conn.execute("""SELECT created_at FROM v2_conversation_events
                    WHERE event_id=? AND owner_user_id=? AND source_message_id=?
                      AND source='desktop' AND visibility='private' AND shared_with='[]'""",
                    (event.event_id, owner, event.source_message_id)).fetchone()
                if row is None:
                    continue
                finalized = _finalized_owner_reply(conn, message_id, owner=owner)
                if finalized is None:
                    continue
                state, observations = finalized
                if state.status != 'completed':
                    continue
                presented_at = datetime.fromisoformat(row['created_at'])
                for binding in observations.values():
                    if binding.get('tool_id') == OWNER_GMAIL_SEARCH_TOOL_ID:
                        payload = binding['gmail_search_observation']
                        if payload['status'] in {'complete', 'partial'} and not valid_gmail_card_history(
                                conn, owner, binding['arguments'], payload, presented_at=presented_at):
                            break
                    elif binding.get('tool_id') == OWNER_CALENDAR_TOOL_ID:
                        payload = binding['calendar_observation']
                        if not valid_calendar_card_history(conn, owner, binding['arguments'], payload,
                                                           presented_at=presented_at):
                            break
                else:
                    rendered = render_private_card(message_id, state.response, observations, timezone_name)
                    if rendered is not None:
                        provider, envelopes = rendered
                        cards[event.event_id] = {'provider': provider, 'envelopes': envelopes}
            except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
                # A pruned, invalid, or revoked snapshot leaves its original receipt.
                continue
    return cards
