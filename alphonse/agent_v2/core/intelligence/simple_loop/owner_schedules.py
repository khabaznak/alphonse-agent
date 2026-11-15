"""Owner-scoped, read-only view of existing schedules while execution is paused."""
from __future__ import annotations

from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore, schedule_summary
from alphonse.agent_v2.core.tools.registry import ToolDefinition

OWNER_SCHEDULES_TOOL_ID = 'native.owner_schedules'
MAX_SCHEDULES = 100


def execute_owner_schedules(arguments, *, context):
    """Read existing records; never create, run, or deliver scheduled work."""
    if (arguments != {} or not isinstance(context.schedule_store, ScheduledTaskStore)
            or not context.task.user or context.cancellation_checker() is not False):
        raise ValueError('owner_schedules_unavailable')
    owner = context.task.user
    rows = (context.schedule_store.list_tasks(owner_user_id=owner, status='active', limit=MAX_SCHEDULES + 1)
            + context.schedule_store.list_tasks(owner_user_id=owner, status='paused', limit=MAX_SCHEDULES + 1))
    if context.task.user != owner or context.cancellation_checker() is not False:
        raise ValueError('owner_schedules_unavailable')
    with context.schedule_store._connect() as conn:
        table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='owner_desktop_reminders'").fetchone()
        reminders = conn.execute("SELECT id,message,due_at,state FROM owner_desktop_reminders WHERE owner_id=? AND state IN ('pending','offered') ORDER BY due_at,id LIMIT 101", (owner,)).fetchall() if table else []
    if context.task.user != owner or context.cancellation_checker() is not False:
        raise ValueError('owner_schedules_unavailable')
    if len(rows) + len(reminders) > MAX_SCHEDULES:
        return {'status': 'too_many', 'legacy_scheduler_running': False, 'desktop_requires_open': True,
                'schedules': [], 'desktop_reminders': []}
    if any(row.owner_user_id != owner for row in rows):
        raise ValueError('owner_schedules_unavailable')
    rows.sort(key=lambda row: (row.next_run_at is None, row.next_run_at or '', row.scheduled_task_id))
    return {'status': 'complete', 'legacy_scheduler_running': False, 'desktop_requires_open': True,
        'desktop_reminders': [{'id': row[0], 'message': row[1], 'due_at': row[2], 'state': row[3]}
                              for row in reminders], 'schedules': [
        {'id': row.scheduled_task_id, 'name': row.name, 'description': row.description,
         'prompt': row.prompt, 'project_id': row.project_id, 'schedule': schedule_summary(row.schedule),
         'timezone': row.timezone, 'next_run_at': row.next_run_at, 'status': row.status}
        for row in rows]}


def valid_schedules_observation(actor, arguments, observation, *, context):
    """Re-read actor-scoped records before projection, recovery and delivery."""
    try:
        return (isinstance(actor, str) and bool(actor) and context.task.user == actor
                and arguments == {} and isinstance(observation, dict)
                and observation == execute_owner_schedules(arguments, context=context))
    except Exception:
        return False


def build_owner_schedules_tool_definition():
    schema = {'type': 'object', 'properties': {}, 'additionalProperties': False}
    return ToolDefinition(descriptor=ToolDescriptor(tool_id=OWNER_SCHEDULES_TOOL_ID, name='owner_schedules',
        kind=ToolKind.NATIVE, description='List the owner’s active or paused older schedules and new Desktop reminders. '
        'Older schedules are paused; new Desktop reminders appear only while Desktop is open. More than 100 records are unavailable; '
        'this tool cannot create or change schedules.', argument_schema=schema,
        capabilities=('scheduling',), read_only=True), callable=execute_owner_schedules,
        argument_schema=schema, accepts_context=True)
