"""Human-level cross-person asset delivery; routing remains deterministic."""
from __future__ import annotations
from typing import Any
from alphonse.agent_v2.core.core import ToolDescriptor, ToolExecutionContext, ToolKind
from alphonse.agent_v2.core.io import channel_address_from_metadata
from alphonse.agent_v2.core.tools.registry import ToolDefinition

SEND_ATTACHMENT_TOOL_ID = "native.send_attachment"

def build_send_attachment_tool_definition(asset_store: Any | None = None) -> ToolDefinition:
    descriptor = ToolDescriptor(SEND_ATTACHMENT_TOOL_ID, "send_attachment", ToolKind.NATIVE, "Send a selected durable attachment to a registered person without choosing their channel.", {"type":"object","additionalProperties":False,"properties":{"recipient":{"type":"string"},"asset_id":{"type":"string"},"message":{"type":"string"}},"required":["recipient","asset_id"]}, ("communication","attachments"), ("native","attachments"))
    def execute(arguments: dict[str, Any], *, context: ToolExecutionContext | None = None) -> dict[str, Any]: return execute_send_attachment(arguments, context=context, asset_store=asset_store)
    return ToolDefinition(descriptor, execute, dict(descriptor.argument_schema), enabled=True, accepts_context=True)

def execute_send_attachment(arguments: dict[str, Any], *, context: ToolExecutionContext | None = None, asset_store: Any | None = None) -> dict[str, Any]:
    if context is None or context.delivery_sink is None: raise ValueError("attachment_delivery_unavailable")
    origin = channel_address_from_metadata(context.task.metadata)
    if origin is None: raise ValueError("origin_address_required")
    recipient, asset_id = str(arguments.get("recipient") or "").strip(), str(arguments.get("asset_id") or "").strip()
    if not recipient or not asset_id: raise ValueError("recipient_and_asset_id_required")
    if asset_store is None or asset_store.get(asset_id, requester_user_id=context.task.user) is None: raise ValueError("attachment_not_found_or_forbidden")
    return dict(context.delivery_sink({"event_type":"communication.deliver","task":context.task.to_dict(),"sender_user_id":context.task.user,"origin":origin.to_dict(),"recipient":recipient,"message":str(arguments.get("message") or "I sent you an attachment."),"asset_ids":[asset_id]}))
