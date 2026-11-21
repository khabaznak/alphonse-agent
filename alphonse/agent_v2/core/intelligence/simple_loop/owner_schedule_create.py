"""Bounded owner scheduling arguments and independent stored-effect checks.

Interpretation is delegated to the model; authority comes from the fenced
owner request. The scheduled work remains the original request verbatim.
"""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import json

from alphonse.agent_v2.core.tools.registry.native.scheduled_task import SCHEDULED_TASK_TOOL_ID

SCHEMA = {'type': 'object', 'additionalProperties': False, 'properties': {
    'operation': {'type': 'string', 'enum': ['create', 'cancel'], 'description': 'Create one future task or cancel one existing owner task. Defaults to create.'},
    'run_at': {'type': 'string', 'description': 'Requested one-time run time as ISO 8601 with an explicit UTC offset.'},
    'scheduled_task_id': {'type': 'string', 'description': 'For cancellation, the exact ID returned by owner_schedules.'},
    'request_quote': {'type': 'string', 'description': 'The complete current owner request, verbatim.'},
}, 'required': ['request_quote']}


def validate_schedule_request(task, arguments, timezone_name, *, observed_at, now):
    if (not isinstance(arguments, dict) or set(arguments) not in ({'run_at', 'request_quote'}, {'operation', 'run_at', 'request_quote'})
            or arguments.get('operation', 'create') != 'create'
            or not isinstance(task.goal, str) or not task.goal.strip()
            or arguments['request_quote'] != task.goal or not isinstance(arguments['run_at'], str)):
        raise ValueError('owner_schedule_request_invalid')
    try:
        zone = ZoneInfo(timezone_name)
        due = datetime.fromisoformat(arguments['run_at'])
        clock = datetime.fromisoformat(observed_at)
        if due.tzinfo is None or clock.tzinfo is None or now.tzinfo is None:
            raise ValueError
        due.astimezone(zone)  # Resolve the owner's zone; preserve the explicit instant.
        if (not now < due <= now + timedelta(days=366)
                or abs((now - clock.astimezone(timezone.utc)).total_seconds()) > 120):
            raise ValueError
        return due.astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError, OverflowError):
        raise ValueError('owner_schedule_time_invalid') from None


def valid_schedule_locked(conn, actor, arguments, observation, *, task):
    try:
        from .owner_schedule_origin import verified_schedule_origin_locked
        if (not isinstance(observation, dict) or set(observation) != {'id', 'run_at', 'prompt'}
                or observation['prompt'] != task.goal or arguments.get('request_quote') != task.goal):
            return False
        row = conn.execute('''SELECT owner_user_id,prompt,schedule_json,status,idempotency_key,delivery_mode
            FROM v2_scheduled_tasks WHERE scheduled_task_id=?''', (observation['id'],)).fetchone()
        schedule = json.loads(row['schedule_json']) if row else None
        origin = verified_schedule_origin_locked(conn, observation['id'])
        return bool(row and origin and origin['request_message_id'] == task.message_id
                    and row['owner_user_id'] == actor and row['prompt'] == task.goal
                    and row['status'] in {'active', 'completed'} and row['delivery_mode'] == 'pdca'
                    and row['idempotency_key'] == 'owner-schedule:' + task.message_id
                    and schedule.get('kind') == 'once'
                    and datetime.fromisoformat(schedule['run_at']).astimezone(timezone.utc).isoformat()
                    == observation['run_at'])
    except Exception:
        return False


def valid_schedule_observation(actor, arguments, observation, *, context):
    try:
        if context.task.user != actor or context.cancellation_checker() is not False:
            return False
        with context.messages._connect() as conn:
            conn.execute('BEGIN')
            return valid_schedule_locked(conn, actor, arguments, observation, task=context.task)
    except Exception:
        return False
