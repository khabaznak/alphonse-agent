"""Channel-neutral delivery of a message to another Alphonse user."""

from __future__ import annotations

from typing import Any

from alphonse.agent_v2.core.core import ToolDescriptor, ToolExecutionContext, ToolKind
from alphonse.agent_v2.core.io import channel_address_from_metadata
from alphonse.agent_v2.core.tools.registry import ToolDefinition

DELIVER_MESSAGE_TOOL_ID = "native.deliver_message"
DELIVER_MESSAGE_TOOL_NAME = "deliver_message"


def build_deliver_message_tool_definition() -> ToolDefinition:
    descriptor = ToolDescriptor(
        tool_id=DELIVER_MESSAGE_TOOL_ID,
        name=DELIVER_MESSAGE_TOOL_NAME,
        kind=ToolKind.NATIVE,
        description="Deliver a message to another registered Alphonse user. Recipient resolution and channel routing are deterministic.",
        argument_schema={
            "type": "object", "additionalProperties": False,
            "properties": {
                "recipient": {"type": "string", "description": "Registered person's display name."},
                "message": {"type": "string", "description": "Message to deliver."},
                "expects_reply": {"type": "boolean", "description": "Whether the sender is asking for a reply."},
            },
            "required": ["recipient", "message"],
        },
        capabilities=("communication", "person_to_person"), tags=("communication", "delivery"),
    )
    return ToolDefinition(descriptor=descriptor, callable=execute_deliver_message, argument_schema=dict(descriptor.argument_schema), enabled=True, accepts_context=True)


def execute_deliver_message(arguments: dict[str, Any], *, context: ToolExecutionContext | None = None) -> dict[str, Any]:
    if context is None or context.delivery_sink is None:
        raise ValueError("communication_delivery_unavailable")
    recipient = str(arguments.get("recipient") or "").strip()
    message = str(arguments.get("message") or "").strip()
    if not recipient:
        raise ValueError("recipient_required")
    if not message:
        raise ValueError("message_required")
    origin = channel_address_from_metadata(context.task.metadata)
    if origin is None:
        raise ValueError("origin_address_required")
    result = context.delivery_sink({
        "event_type": "communication.deliver",
        "task": context.task.to_dict(),
        "sender_user_id": context.task.user,
        "origin": origin.to_dict(),
        "recipient": recipient,
        "message": message,
        "expects_reply": bool(arguments.get("expects_reply")),
    })
    return dict(result) if isinstance(result, dict) else {"status": "delivery_failed"}
