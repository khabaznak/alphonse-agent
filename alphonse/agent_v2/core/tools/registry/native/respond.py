"""v2-native Respond tool."""

from __future__ import annotations

from typing import Any

from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.core import ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition

RESPOND_TOOL_ID = "native.respond"
RESPOND_TOOL_NAME = "respond"

RESPOND_ARGUMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "message": {
            "type": "string",
            "description": "User-visible response text to send back to the requester.",
        },
        "tone": {
            "type": "string",
            "description": "Optional tone label for the response.",
        },
    },
    "required": ["message"],
}


def build_respond_tool_definition() -> ToolDefinition:
    """Build the native Respond tool definition."""
    descriptor = ToolDescriptor(
        tool_id=RESPOND_TOOL_ID,
        name=RESPOND_TOOL_NAME,
        kind=ToolKind.NATIVE,
        description=(
            "Return a direct user-visible response, including greetings, conversational answers, "
            "status summaries, or a concise presentation of completed work."
        ),
        argument_schema=dict(RESPOND_ARGUMENT_SCHEMA),
        capabilities=("conversation", "user_response"),
        tags=("native", "conversation"),
    )
    return ToolDefinition(
        descriptor=descriptor,
        callable=execute_respond,
        argument_schema=dict(RESPOND_ARGUMENT_SCHEMA),
        enabled=True,
    )


def execute_respond(arguments: dict[str, Any]) -> dict[str, str]:
    """Return a direct response payload."""
    message = str(arguments.get("message") or "").strip()
    if not message:
        raise ValueError("respond_message_required")
    tone = str(arguments.get("tone") or "").strip() or "neutral"
    return {"message": message, "tone": tone}
