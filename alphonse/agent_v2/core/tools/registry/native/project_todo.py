"""Explicit, owner-scoped project to-do mutations."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

from alphonse.agent_v2.core.core import ToolDescriptor, ToolExecutionContext, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition

TOOL_ID = "native.project_todo"
SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "action": {"type": "string", "enum": ["create"]},
    "title": {"type": "string", "description": "To-do title for create."},
    "project_name": {"type": "string", "description": "Optional exact name of the project named by the requester. Ask which project if the target is missing or ambiguous."},
    "project_id": {"type": "string", "description": "Optional project ID explicitly supplied by the requester when names are ambiguous."},
    "request_key": {"type": "string", "description": "Optional stable key distinguishing multiple same-title to-dos in this request."},
}, "required": ["action"]}


def build_project_todo_tool_definition() -> ToolDefinition:
    descriptor = ToolDescriptor(tool_id=TOOL_ID, name="project_todo", kind=ToolKind.NATIVE,
        description="Create an explicitly requested to-do in the associated project, or in an active project named by the requester in the current conversation. Use the exact project name or an explicitly provided project ID. If the intended project is missing, ambiguous, or not matched, ask an open-text question for its exact name or ID rather than guessing. Do not infer a project merely because its name occurs in the to-do title. To-dos are not conversation checkpoints or schedules. Completion is recorded in the project view.",
        argument_schema=SCHEMA, capabilities=("project_todos",), tags=("native", "projects", "todos"),
        metadata={"side_effect_class": "project_mutation"})
    return ToolDefinition(descriptor=descriptor, callable=execute_project_todo, argument_schema=SCHEMA, enabled=True, accepts_context=True)


def execute_project_todo(arguments: dict[str, Any], *, context: ToolExecutionContext | None = None) -> dict[str, Any]:
    if context is None or context.schedule_store is None or context.project_store is None:
        raise ValueError("project_todo_context_required")
    task = context.task
    actor = str(getattr(task, "user", "") or "").strip()
    resolver = context.identity_resolver
    users = getattr(resolver, "_user_store", None)
    member = users.get_user(actor) if users is not None and actor else None
    if member is None or not member.is_active:
        raise PermissionError("project_todo_actor_forbidden")
    project = _resolve_todo_project(task, context, users, arguments)
    project_id = project.project_id
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
    return {"todo_id": item["todo_id"], "project_id": project_id, "project_name": project.name, "title": item["title"], "status": item["status"],
            "verification": {"status": "verified", "source": "project_todo_store_readback"}}


def _normalize_project_reference(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def _resolve_todo_project(task: Any, context: ToolExecutionContext, users: Any, arguments: dict[str, Any]) -> Any:
    """Resolve a user-named project without letting a model grant access or invent a target."""
    actor = str(task.user or "").strip()
    attached = str(getattr(task, "project_id", "") or "").strip()
    supplied_name = _normalize_project_reference(arguments.get("project_name") or "")
    supplied_id = str(arguments.get("project_id") or "").strip()
    if attached:
        project = context.project_store.get_project(attached, requester_user_id=actor, requester_is_admin=bool(users.is_admin(actor)))
        if project is None or project.status != "active":
            raise PermissionError("project_todo_project_forbidden")
        if (supplied_id and supplied_id != attached) or (supplied_name and supplied_name != _normalize_project_reference(project.name)):
            raise ValueError("project_todo_project_mismatch: Ask the requester to clarify the project.")
        return project

    projects = [p for p in context.project_store.list_visible_projects(actor, requester_is_admin=bool(users.is_admin(actor)))
                if p.status == "active" and not p.is_system_home]
    # Only the current task's authenticated requester turns can establish the target;
    # model text, older unrelated conversations and other speakers cannot do so.
    turns = [str(getattr(task, "goal", "") or "")]
    turns.extend(str(row.get("content") or "") for row in getattr(task, "conversation_records", [])
                 if isinstance(row, dict) and str(row.get("speaker") or "").strip() == actor)
    for raw in reversed(turns):
        text = _normalize_project_reference(raw)
        spans = []
        for project in projects:
            for token in {_normalize_project_reference(project.name), _normalize_project_reference(project.project_id)}:
                if token:
                    for match in re.finditer(r"(?<!\w)" + re.escape(token) + r"(?!\w)", text):
                        spans.append((match.start(), match.end(), project))
        # A shorter name embedded inside a longer project name is not a second mention.
        spans = [span for span in spans if not any(other[0] <= span[0] and other[1] >= span[1]
                 and (other[0], other[1]) != (span[0], span[1]) for other in spans)]
        matches = {span[2].project_id: span[2] for span in spans}
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError("project_todo_project_ambiguous: Ask the requester which project to use.")
        project = next(iter(matches.values()))
        if (supplied_id and supplied_id != project.project_id) or (supplied_name and supplied_name != _normalize_project_reference(project.name)):
            raise ValueError("project_todo_project_mismatch: Ask the requester to clarify the project.")
        return project
    raise ValueError("project_todo_project_reference_required: Ask the requester to name the project.")


def build_project_todo_list_tool_definition() -> ToolDefinition:
    schema = {"type": "object", "additionalProperties": False, "properties": {
        "offset": {"type": "integer", "minimum": 0, "description": "Page offset; pages contain at most 50 items."}
    }}
    descriptor = ToolDescriptor(tool_id="native.project_todo_list", name="project_todo_list", kind=ToolKind.NATIVE,
        description="List the requester’s explicit to-dos for the associated or user-named project. If the target is missing or ambiguous, list accessible active project names and IDs so the requester can clarify with an open-text answer. Read-only; status changes are controlled in the project view.",
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
    offset = max(0, int(arguments.get("offset") or 0))
    try:
        if project_id:
            project = context.project_store.get_project(project_id, requester_user_id=actor, requester_is_admin=bool(users.is_admin(actor)))
            if project is None:
                raise PermissionError("project_todo_project_forbidden")
        else:
            project = _resolve_todo_project(context.task, context, users, {})
    except ValueError as exc:
        projects = [p for p in context.project_store.list_visible_projects(actor, requester_is_admin=bool(users.is_admin(actor)))
                    if p.status == "active" and not p.is_system_home]
        return {"project_selection_required": True, "reason": str(exc).split(":", 1)[0],
                "projects": [{"project_id": p.project_id, "name": p.name, "description": p.description} for p in projects[offset:offset + 50]],
                "has_more": len(projects) > offset + 50, "offset": offset}
    items = context.schedule_store.project_todos_page(owner_user_id=actor, project_id=project.project_id, limit=51, offset=offset)
    return {"items": items[:50], "has_more": len(items) > 50, "offset": offset, "project_id": project.project_id, "project_name": project.name}
