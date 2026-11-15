"""Narrow local reminder action: authority is the direct owner's current request."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition

OWNER_REMINDER_CREATE_TOOL_ID = 'native.owner_reminder_create'
_SCHEMA = {'type': 'object', 'additionalProperties': False,
    'properties': {'message': {'type': 'string', 'minLength': 1, 'maxLength': 500},
                   'due_at': {'type': 'string', 'description': 'One ISO 8601 datetime with explicit offset in the owner timezone.'},
                   'request_quote': {'type': 'string', 'description': 'The entire direct-owner request, verbatim.'}},
    'required': ['message', 'due_at', 'request_quote']}
_EXPLICIT = re.compile(r'(?i)(?<!\w)(?:remind\s+me|set\s+(?:me\s+)?(?:a\s+)?reminder|recu[eé]rdame)(?!\w)')
_NEGATED = re.compile(r"(?is)(?<!\w)(?:do\s+not|don['’]t|never|no)\b.{0,80}?(?:set\s+(?:a\s+)?reminder|remind\s+me|recu[eé]rdame)(?!\w)")
_AMBIGUOUS_SCOPE = re.compile(
    r'(?i)(?<!\w)(?:if|unless|whether|should|would|could|might|maybe|perhaps|'
    r'every|each|daily|weekly|monthly|yearly|hourly|recurr(?:ing|ence)|repeat(?:ed|ing)?|'
    r'after|before|once|until|only|later|tomorrow|today|tonight|next|'
    r'si|cuando|despu[eé]s|antes|hasta|solo|s[oó]lo|ma[ñn]ana|hoy|quiz[aá]s?|tal\s+vez|cada|diari[oa]s?|semanal(?:mente)?|'
    r'mensual(?:mente)?|anual(?:mente)?|repetir|recurrente)(?!\w)')
_CALENDAR_TIME = re.compile(
    r'(?i)(?<!\w)(?:(today|tomorrow)\s+at|(hoy|ma[ñn]ana)\s+a\s+las?)\s+'
    r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s*[.!]?\s*$')


def is_direct_reminder_request(goal):
    # This conservative publication guard intentionally covers more phrasing
    # than the separate, narrow mutation validator. A list query such as
    # "what reminders do I have" does not match an instruction to remind.
    return isinstance(goal, str) and bool(_EXPLICIT.search(goal))


def validate_reminder_request(task, arguments, timezone_name, *, observed_at):
    """Validate explicit direct request and exact one-time, owner-local scope.

    Language interpretation is probabilistic; these narrow checks intentionally
    refuse unsupported or negated phrasings rather than granting authority from
    earlier retrieved content or model-composed identifiers.
    """
    if not isinstance(arguments, dict) or set(arguments) != {'message', 'due_at', 'request_quote'}:
        raise ValueError('owner_reminder_request_invalid')
    text, due_text, quote = (arguments[k] for k in ('message', 'due_at', 'request_quote'))
    goal = getattr(task, 'goal', None)
    calendar = _CALENDAR_TIME.search(goal) if isinstance(goal, str) else None
    # Only the terminal, supported time expression is removed from the
    # qualifier check; conditions elsewhere in the request remain forbidden.
    scope = goal[:calendar.start()] if calendar else goal
    if (not isinstance(goal, str) or not isinstance(quote, str) or not quote.strip()
            or quote != goal or not _EXPLICIT.search(quote) or _NEGATED.search(goal)
            or '?' in goal or '¿' in goal or _AMBIGUOUS_SCOPE.search(scope)
            or not isinstance(text, str) or not text.strip() or len(text) > 500
            or text.strip().casefold() not in goal.casefold()
            or not isinstance(due_text, str) or not isinstance(timezone_name, str)):
        raise ValueError('owner_reminder_request_invalid')
    try:
        due = datetime.fromisoformat(due_text)
        if due.tzinfo is None:
            raise ValueError
        zone = ZoneInfo(timezone_name)
        local = due.astimezone(zone)
        if due.utcoffset() != local.utcoffset() or due.replace(tzinfo=None) != local.replace(tzinfo=None):
            raise ValueError
        # Reject repeated wall times instead of choosing an offset for the owner.
        wall = local.replace(tzinfo=None)
        a, b = wall.replace(tzinfo=zone, fold=0), wall.replace(tzinfo=zone, fold=1)
        if a.utcoffset() != b.utcoffset():
            raise ValueError
        if calendar:
            if not isinstance(observed_at, str):
                raise ValueError
            clock = datetime.fromisoformat(observed_at)
            if clock.tzinfo is None:
                raise ValueError
            hour, minute = int(calendar.group(3)), int(calendar.group(4) or 0)
            if not 1 <= hour <= 12 or minute > 59:
                raise ValueError
            hour = hour % 12 + (12 if calendar.group(5).lower() == 'pm' else 0)
            day_word = (calendar.group(1) or calendar.group(2)).lower()
            requested_day = clock.astimezone(zone).date() + timedelta(days=day_word in ('tomorrow', 'mañana', 'manana'))
            if (local.date() != requested_day or local.hour != hour or local.minute != minute
                    or local.second != 0 or local.microsecond != 0):
                raise ValueError
            time_pattern = re.escape(calendar.group(0).strip().rstrip('.!'))
        elif due_text in goal:
            time_pattern = r'(?:at|a\s+las?)\s+' + re.escape(due_text)
        else:
            # Support an explicit numeric relative duration without trusting
            # the model to invent an unmentioned date. The host clock is the
            # reference; unsupported natural-language times must be clarified.
            matches = list(re.finditer(r'(?i)\b(?:in|en)\s+(\d+)\s*'
                r'(minutes?|mins?|minutos?|hours?|hrs?|horas?)\b', goal))
            if len(matches) != 1 or not isinstance(observed_at, str):
                raise ValueError
            value = int(matches[0].group(1))
            if not 1 <= value <= 24 * 365:
                raise ValueError
            unit = matches[0].group(2).lower()
            delta = timedelta(hours=value) if unit.startswith(('hour', 'hr', 'hora')) else timedelta(minutes=value)
            clock = datetime.fromisoformat(observed_at)
            if clock.tzinfo is None or abs((due.astimezone(timezone.utc) - (clock + delta)).total_seconds()) > 30:
                raise ValueError
            time_pattern = re.escape(matches[0].group(0))
        # Bind content and time to one complete imperative.  The deterministic
        # grammar deliberately refuses multi-clause requests it cannot scope.
        match = re.fullmatch(
            r'(?is)\s*(?:please\s+|por\s+favor\s+)?(?:remind\s+me\s+(?:to\s+)?|recu[eé]rdame\s+)'
            r'(?P<content>[^;,.!?\n]+?)\s+' + time_pattern + r'\s*[.!]?\s*', goal)
        if match is None or match.group('content').strip().casefold() != text.strip().casefold():
            raise ValueError
    except Exception:
        raise ValueError('owner_reminder_time_invalid') from None
    return {'message': text.strip(), 'due_at': due.astimezone(timezone.utc).isoformat()}


def build_owner_reminder_create_tool_definition():
    def unavailable(*_args, **_kwargs):
        raise PermissionError('owner_reminder_atomic_host_required')
    descriptor = ToolDescriptor(OWNER_REMINDER_CREATE_TOOL_ID, 'owner_reminder_create', ToolKind.NATIVE,
        'Create exactly one one-time reminder for the direct owner on this Desktop only. '
        'First call owner_clock; use the observed owner timezone and current time. '
        'Supported times include an explicit ISO timestamp, a numeric duration, or today/tomorrow at an explicit am/pm time. '
        'Use only when the owner explicitly asks to create it; quote their exact request. '
        'Ask a question if time, timezone, or reminder content is unclear. '
        'The reminder appears only while Desktop is open; no external delivery or recurrence.',
        _SCHEMA, ('reminders',), ('native',), read_only=False)
    return ToolDefinition(descriptor, unavailable, _SCHEMA, accepts_context=True)


def valid_reminder_locked(conn, actor, arguments, observation, *, task):
    """Recheck the exact created owner reminder inside a caller's transaction."""
    try:
        if (not isinstance(observation, dict) or set(observation) != {'id', 'message', 'due_at', 'state'}
                or observation['state'] != 'pending' or not isinstance(arguments, dict)
                or not isinstance(observation['id'], str) or not isinstance(actor, str)):
            return False
        row = conn.execute('''SELECT id,owner_id,request_id,message,due_at,state FROM owner_desktop_reminders
            WHERE id=? AND owner_id=? AND request_id=?''',
            (observation['id'], actor, task.message_id)).fetchone()
        return bool(row is not None and row['state'] in {'pending', 'offered', 'acknowledged'}
                    and row['message'] == observation['message'] and row['due_at'] == observation['due_at']
                    and isinstance(arguments.get('message'), str)
                    and arguments['message'].strip() == row['message']
                    and datetime.fromisoformat(arguments['due_at']).astimezone(timezone.utc).isoformat() == row['due_at'])
    except Exception:
        return False


def valid_reminder_observation(actor, arguments, observation, *, context):
    """Point-in-time read; the pickup transaction must check it again."""
    try:
        queue = context.messages
        if context.task.user != actor or context.cancellation_checker() is not False:
            return False
        with queue._connect() as conn:
            conn.execute('BEGIN')
            return valid_reminder_locked(conn, actor, arguments, observation, task=context.task)
    except Exception:
        return False
