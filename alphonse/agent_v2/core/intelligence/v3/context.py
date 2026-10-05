"""Shared context inputs for the V3 conversation and task paths."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.task_state import TaskState


def conversation_context(task: "TaskState", context: "CoreLoopContext", *, limit: int = 30) -> str:
    """Build a bounded cross-project conversation window plus configured context."""
    sections: list[str] = []
    if context.conversation_store is not None and task.user:
        recent = getattr(context.conversation_store, "list_recent", None)
        if callable(recent):
            active_ids = {str(task.project_id or "").strip()}
            list_projects = getattr(context.project_store, "list_visible_projects", None)
            if callable(list_projects):
                try:
                    active_ids.update(str(item.project_id) for item in list_projects(task.user))
                except Exception:
                    pass
            try:
                events = recent(owner_user_id=task.user, limit=limit)
            except Exception:
                events = []
            rows = []
            for event in events:
                role = str(getattr(event, "role", "")).strip().capitalize()
                content = str(getattr(event, "content", "")).strip()
                project = str(getattr(event, "project_id", "")).strip()
                if project and project not in active_ids:
                    continue
                if content:
                    content = content[:600]
                    label = f"{role} [{project}]" if project else role
                    rows.append(f"- {label}: {content}")
            if rows:
                sections.append("Recent household conversation across projects:\n" + "\n".join(rows))

    global_context = _prompt_file(context, "GlobalContext.md")
    if global_context:
        sections.append("Household and global context:\n" + global_context)

    project = None
    if context.project_store is not None and task.project_id:
        getter = getattr(context.project_store, "get_project", None)
        if callable(getter):
            try:
                project = getter(task.project_id, requester_user_id=task.user)
            except TypeError:
                project = getter(task.project_id)
    if project is not None:
        try:
            project_text = Path(project.context_path).read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError, AttributeError):
            project_text = ""
        if project_text:
            sections.append(f"Selected project context ({project.name}):\n{project_text}")
    return "\n\n".join(sections)[:12000]


def _prompt_file(context: "CoreLoopContext", name: str) -> str:
    if context.prompts is None:
        return ""
    try:
        prompt_file = context.prompts.load(name)
    except Exception:
        return ""
    return str(getattr(prompt_file, "content", "") or "").strip()
