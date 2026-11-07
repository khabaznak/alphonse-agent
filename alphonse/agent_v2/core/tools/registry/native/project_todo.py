"""Explicit, owner-scoped project to-do mutations."""
from __future__ import annotations

import hashlib
from typing import Any

from alphonse.agent_v2.core.core import ToolDescriptor, ToolExecutionContext, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition

TOOL_ID = "native.project_todo"
SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "action": {"type": "string", "enum": ["create"]},
    "title": {"type": "string", "description": "To-do title for create."},
    "request_key": {"type": "string", "description": "Optional stable key distinguishing multiple same-title to-dos in this request."},
}, "required": ["action"]}


def build_project_todo_tool_definition() -> ToolDefinition:
    descriptor = ToolDescriptor(tool_id=TOOL_ID, name="project_todo", kind=ToolKind.NATIVE,
        description="Create an explicit to-do for the requester’s currently associated project when requested. A to-do is not a conversation checkpoint or schedule. Completion must be recorded by the person in the project view.",
        argument_schema=SCHEMA, capabilities=("project_todos",), tags=("native", "projects", "todos"),
        metadata={"side_effect_class": "project_mutation"})
    return ToolDefinition(descriptor=descriptor, callable=execute_project_todo, argument_schema=SCHEMA, enabled=True, accepts_context=True)


def execute_project_todo(arguments: dict[str, Any], *, context: ToolExecutionContext | None = None) -> dict[str, Any]:
    if context is None or context.schedule_store is None or context.project_store is None:
        raise ValueError("project_todo_context_required")
    task = context.task
    actor = str(getattr(task, "user", "") or "").strip()
    project_id = str(getattr(task, "project_id", "") or "").strip()
    resolver = context.identity_resolver
    users = getattr(resolver, "_user_store", None)
    member = users.get_user(actor) if users is not None and actor else None
    if member is None or not member.is_active:
        raise PermissionError("project_todo_actor_forbidden")
    project = context.project_store.get_project(project_id, requester_user_id=actor, requester_is_admin=bool(users.is_admin(actor)))
    if project is None or project.status != "active":
        raise PermissionError("project_todo_project_forbidden")
    if context.cancellation_checker and context.cancellation_checker():
        raise RuntimeError("project_todo_cancelled")
    action = str(arguments.get("action") or "")
    if action == "create":
        title = str(arguments.get("title") or "").strip()
        source = str(getattr(task, "message_id", "") or getattr(task, "task_id", "") or "").strip()
        if not source:
            raise ValueError("project_todo_request_identity_required")
        token = str(arguments.get("request_key") or title).strip()
        key = "agent:" + hashlib.sha256((source + "\0" + token).encode()).hexdigest()
        item = context.schedule_store.create_project_todo(owner_user_id=actor, project_id=project_id, title=title, idempotency_key=key)
    else:
        raise ValueError("project_todo_action_invalid")
    return {"todo_id": item["todo_id"], "project_id": project_id, "title": item["title"], "status": item["status"],
            "verification": {"status": "verified", "source": "project_todo_store_readback"}}


def build_project_todo_list_tool_definition() -> ToolDefinition:
    schema = {"type": "object", "additionalProperties": False, "properties": {
        "offset": {"type": "integer", "minimum": 0, "description": "Page offset; pages contain at most 50 items."}
    }}
    descriptor = ToolDescriptor(tool_id="native.project_todo_list", name="project_todo_list", kind=ToolKind.NATIVE,
        description="List the requester's explicit to-dos in the associated project. Read-only; status changes are controlled in the project view.",
        argument_schema=schema, capabilities=("project_todos",), tags=("native", "projects", "todos"), read_only=True)
    return ToolDefinition(descriptor=descriptor, callable=execute_project_todo_list, argument_schema=schema, enabled=True, accepts_context=True)


def execute_project_todo_list(arguments: dict[str, Any], *, context: ToolExecutionContext | None = None) -> dict[str, Any]:
    if context is None or context.schedule_store is None or context.project_store is None:
        raise ValueError("project_todo_context_required")
    actor = str(getattr(context.task, "user", "") or "").strip()
    project_id = str(getattr(context.task, "project_id", "") or "").strip()
    users = getattr(context.identity_resolver, "_user_store", None)
    member = users.get_user(actor) if users is not None and actor else None
    if member is None or not member.is_active:
        raise PermissionError("project_todo_actor_forbidden")
    project = context.project_store.get_project(project_id, requester_user_id=actor, requester_is_admin=bool(users.is_admin(actor)))
    if project is None:
        raise PermissionError("project_todo_project_forbidden")
    offset = max(0, int(arguments.get("offset") or 0))
    items = context.schedule_store.project_todos_page(owner_user_id=actor, project_id=project_id, limit=51, offset=offset)
    return {"items": items[:50], "has_more": len(items) > 50, "offset": offset, "project_id": project_id}
