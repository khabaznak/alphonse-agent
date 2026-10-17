"""Native tools for finding reminders and adding a delivery destination."""

from __future__ import annotations

from typing import Any

from alphonse.agent_v2.core.core import ToolDescriptor, ToolExecutionContext, ToolKind
from alphonse.agent_v2.core.scheduled_tasks import schedule_summary
from alphonse.agent_v2.core.tools.registry import ToolDefinition

SCHEDULED_TASK_DELIVERY_TOOL_ID = "native.scheduled_task_delivery"
SCHEDULED_TASK_DELIVERY_TOOL_NAME = "scheduled_task_delivery"

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "operation": {"type": "string", "enum": ["list", "add_delivery_channel"]},
        "scheduled_task_id": {"type": "string"},
        "provider_key": {"type": "string", "enum": ["telegram"]},
    },
    "required": ["operation"],
}


def build_scheduled_task_delivery_tool_definition() -> ToolDefinition:
    descriptor = ToolDescriptor(
        tool_id=SCHEDULED_TASK_DELIVERY_TOOL_ID,
        name=SCHEDULED_TASK_DELIVERY_TOOL_NAME,
        kind=ToolKind.NATIVE,
        description=(
            "List the current user's active reminders in this project, or add the user's registered Telegram address "
            "to one pending reminder's delivery destinations. Use this to change where an existing reminder arrives."
        ),
        argument_schema=dict(_SCHEMA),
        capabilities=("scheduling", "reminders", "delivery_routing"),
        tags=("native", "scheduling", "delivery"),
    )
    return ToolDefinition(
        descriptor=descriptor,
        callable=execute_scheduled_task_delivery,
        argument_schema=dict(_SCHEMA),
        enabled=True,
        accepts_context=True,
    )


def execute_scheduled_task_delivery(
    arguments: dict[str, Any],
    *,
    context: ToolExecutionContext | None = None,
) -> dict[str, Any]:
    if context is None or context.schedule_store is None:
        raise ValueError("scheduled_task_delivery_context_required")
    owner = str(context.task.user or "").strip()
    project_id = str(context.task.project_id or "").strip()
    if not owner:
        raise ValueError("scheduled_task_delivery_owner_required")
    operation = str(arguments.get("operation") or "").strip()
    if operation == "list":
        tasks = context.schedule_store.list_tasks(owner_user_id=owner, project_id=project_id, status="active", limit=100)
        return {
            "tasks": [
                {
                    "scheduled_task_id": task.scheduled_task_id,
                    "name": task.name,
                    "description": task.description,
                    "prompt": task.prompt,
                    "schedule_summary": schedule_summary(task.schedule),
                    "delivery_channels": _delivery_channels(task.origin_channel),
                }
                for task in tasks
                if task.delivery_mode == "direct"
            ]
        }
    if operation != "add_delivery_channel":
        raise ValueError("scheduled_task_delivery_operation_invalid")

    task_id = str(arguments.get("scheduled_task_id") or "").strip()
    if not task_id:
        raise ValueError("scheduled_task_id_required")
    scheduled = context.schedule_store.get_task_for_owner(task_id, owner_user_id=owner)
    if scheduled is None or scheduled.project_id != project_id or scheduled.delivery_mode != "direct":
        raise ValueError("scheduled_reminder_not_found")

    resolver = context.identity_resolver
    if resolver is None:
        raise ValueError("scheduled_delivery_identity_unavailable")
    integration = resolver.integration_for(provider_key=str(arguments.get("provider_key") or "telegram"))
    resolved = resolver.resolve_outbound_address(
        alphonse_user_id=owner,
        preferred_integration_id=integration.integration_id,
    )
    address = resolved.address if resolved.resolved else None
    if address is None or address.provider_key != "telegram":
        raise ValueError("scheduled_delivery_channel_not_configured")

    origin = dict(scheduled.origin_channel)
    channels = _delivery_channels(origin)
    channel = address.to_dict()
    if channel not in channels:
        origin["delivery_channels"] = [*channels, channel]
        scheduled = context.schedule_store.update_task(task_id, origin_channel=origin)
    return {
        "scheduled_task_id": scheduled.scheduled_task_id,
        "name": scheduled.name,
        "delivery_channels": _delivery_channels(scheduled.origin_channel),
        "status": "updated",
    }


def _delivery_channels(origin: dict[str, Any]) -> list[dict[str, Any]]:
    channels = origin.get("delivery_channels") if isinstance(origin, dict) else None
    result = [dict(item) for item in channels if isinstance(item, dict)] if isinstance(channels, list) else []
    if origin and all(str(origin.get(key) or "").strip() for key in ("integration_id", "provider_key", "channel_target")):
        primary = {key: origin.get(key, "") for key in (
            "integration_id", "provider_key", "channel_target", "alphonse_user_id", "provider_user_id",
            "provider_message_id", "reply_to_provider_message_id", "thread_id",
        )}
        if primary not in result:
            result.insert(0, primary)
    return result
