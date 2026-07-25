"""v2-native ScheduledTask tool."""

from __future__ import annotations

from typing import Any

from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.core import ToolExecutionContext
from alphonse.agent_v2.core.core import ToolKind
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.core.scheduled_tasks import schedule_summary
from alphonse.agent_v2.core.tools.registry import ToolDefinition

SCHEDULED_TASK_TOOL_ID = "native.scheduled_task"
SCHEDULED_TASK_TOOL_NAME = "scheduled_task"

SCHEDULED_TASK_ARGUMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {
            "type": "string",
            "description": "Short user-visible name for the scheduled task.",
        },
        "description": {
            "type": "string",
            "description": "Optional description of why this task exists.",
        },
        "prompt": {
            "type": "string",
            "description": "Detailed prompt to re-enter into CAPD when the schedule fires.",
        },
        "schedule_kind": {
            "type": "string",
            "enum": ["once", "rrule"],
            "description": "Schedule type.",
        },
        "run_at": {
            "type": "string",
            "description": "ISO datetime for one-off scheduled tasks.",
        },
        "rrule": {
            "type": "string",
            "description": "RFC 5545 RRULE string for recurring scheduled tasks.",
        },
        "dtstart": {
            "type": "string",
            "description": "Optional ISO datetime start for RRULE schedules.",
        },
        "timezone": {
            "type": "string",
            "description": "IANA timezone name; defaults to UTC.",
        },
        "enabled": {
            "type": "boolean",
            "description": "Whether the scheduled task starts active.",
        },
    },
    "required": ["name", "prompt", "schedule_kind"],
}


def build_scheduled_task_tool_definition() -> ToolDefinition:
    """Build the native ScheduledTask tool definition."""
    descriptor = ToolDescriptor(
        tool_id=SCHEDULED_TASK_TOOL_ID,
        name=SCHEDULED_TASK_TOOL_NAME,
        kind=ToolKind.NATIVE,
        description=(
            "Schedule a future or recurring task by storing a prompt that will be "
            "queued back into Alphonse's v2 CAPD loop when due."
        ),
        argument_schema=dict(SCHEDULED_TASK_ARGUMENT_SCHEMA),
        capabilities=("scheduling", "reminders", "recurring_tasks", "future_work"),
        tags=("native", "scheduling"),
    )
    return ToolDefinition(
        descriptor=descriptor,
        callable=execute_scheduled_task,
        argument_schema=dict(SCHEDULED_TASK_ARGUMENT_SCHEMA),
        enabled=True,
        accepts_context=True,
    )


def execute_scheduled_task(
    arguments: dict[str, Any],
    *,
    context: ToolExecutionContext | None = None,
) -> dict[str, Any]:
    """Create a scheduled v2 prompt task."""
    if context is None:
        raise ValueError("scheduled_task_context_required")
    store = context.schedule_store if context.schedule_store is not None else ScheduledTaskStore.default()
    if not hasattr(store, "create_task"):
        raise TypeError("scheduled_task_store_invalid")

    task = context.task
    owner_user_id = str(getattr(task, "user", "") or "").strip()
    if not owner_user_id:
        raise ValueError("scheduled_task_owner_required")
    default_timezone = context.user_timezone_provider(owner_user_id) if callable(context.user_timezone_provider) else "UTC"
    record = store.create_task(
        owner_user_id=owner_user_id,
        project_id=str(getattr(task, "project_id", "") or "").strip(),
        name=str(arguments.get("name") or "").strip(),
        description=str(arguments.get("description") or "").strip(),
        prompt=str(arguments.get("prompt") or "").strip(),
        schedule_kind=str(arguments.get("schedule_kind") or "").strip(),  # type: ignore[arg-type]
        run_at=_optional_text(arguments.get("run_at")),
        rrule=_optional_text(arguments.get("rrule")),
        dtstart=_optional_text(arguments.get("dtstart")),
        origin_channel=dict(task.metadata.get("channel") or {})
        if isinstance(task.metadata.get("channel"), dict)
        else {},
        timezone_name=str(arguments.get("timezone") or "").strip() or str(default_timezone or "UTC"),
        enabled=bool(arguments.get("enabled", True)),
    )
    return {
        "scheduled_task_id": record.scheduled_task_id,
        "name": record.name,
        "status": record.status,
        "next_run_at": record.next_run_at,
        "schedule_summary": schedule_summary(record.schedule),
    }


def _optional_text(value: Any) -> str | None:
    rendered = str(value or "").strip()
    return rendered or None
