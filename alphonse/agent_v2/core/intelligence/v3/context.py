"""Shared context inputs for the V3 conversation and task paths."""

from __future__ import annotations

from pathlib import Path
import re
from typing import TYPE_CHECKING

from alphonse.agent_v2.agent_config import parse_global_context_sections

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.task_state import TaskState


_INVARIANT_GLOBAL_SECTIONS = {
    "family definition",
    "household location and setup",
    "alphonse's household-wide interaction defaults",
    "privacy and sharing boundaries",
}


def prepare_context_curation(task: "TaskState", context: "CoreLoopContext") -> tuple[str, list[dict[str, str]]]:
    """Return always-included context and bounded optional context candidates."""
    sections: list[str] = []
    active_projects = _visible_active_projects(task, context)
    active_ids = {str(getattr(item, "project_id", "") or "").strip() for item in active_projects}
    if task.project_id:
        active_ids.add(str(task.project_id).strip())
    recent_text = _recent_conversation(task, context, active_ids)
    if recent_text:
        sections.append(recent_text)
    global_context = _prompt_file(context, "GlobalContext.md")
    if global_context:
        parsed = parse_global_context_sections(global_context)
        if parsed:
            preamble = _global_preamble(global_context)
            if preamble:
                sections.append("Global context instructions:\n" + preamble)
            for heading, body in parsed.items():
                if not body or heading.casefold() == "please remove":
                    continue
                if heading.casefold() in _INVARIANT_GLOBAL_SECTIONS:
                    sections.append(f"{heading}:\n{body}")
        else:
            # Preserve older free-form documents until they are structured.
            sections.append("Household and global context:\n" + global_context)

    selected_project = _project_candidate(task, context)
    if selected_project is not None:
        sections.append(f"Selected project context ({selected_project['title']}):\n{selected_project['content']}")

    candidates = _global_candidates(global_context)
    for project in active_projects:
        project_id = str(getattr(project, "project_id", "") or "").strip()
        if not project_id or project_id == str(task.project_id or "").strip():
            continue
        candidate = _project_candidate_for_record(project, context)
        if candidate is not None:
            candidates.append(candidate)
    return "\n\n".join(sections)[:12000], candidates[:24]


def conversation_context(
    task: "TaskState",
    context: "CoreLoopContext",
    *,
    selected_context_ids: list[str] | tuple[str, ...] = (),
    include_selected_project: bool = True,
) -> str:
    """Render invariant context plus the optional candidates selected at admission."""
    invariant, candidates = prepare_context_curation(task, context)
    selected = set(str(item) for item in selected_context_ids)
    sections = [invariant] if invariant else []
    for candidate in candidates:
        if candidate["id"] in selected:
            sections.append(f"{candidate['title']}:\n{candidate['content']}")
    if not include_selected_project:
        project = _project_candidate(task, context)
        if project is not None:
            sections = [section for section in sections if not section.startswith(f"Selected project context ({project['title']}):")]
    return "\n\n".join(sections)[:24000]


def recent_conversation_for_curation(task: "TaskState", context: "CoreLoopContext") -> str:
    active_projects = _visible_active_projects(task, context)
    active_ids = {str(getattr(item, "project_id", "") or "").strip() for item in active_projects}
    if task.project_id:
        active_ids.add(str(task.project_id).strip())
    return _recent_conversation(task, context, active_ids)[:5000]


def skill_candidates_for_curation(context: "CoreLoopContext") -> list[dict[str, str]]:
    store = context.skill_store
    if store is None:
        return []
    list_skills = getattr(store, "list_skills", None)
    if not callable(list_skills):
        return []
    try:
        return [skill.candidate() for skill in list_skills()]
    except Exception:
        return []


def selected_skill_guidance(task: "TaskState", context: "CoreLoopContext", *, max_chars: int = 16000) -> tuple[str, list[str]]:
    """Load selected skill instructions within a deterministic prompt budget."""
    selection = task.metadata.get("v3_skill_selection")
    selected_ids = selection.get("selected_skill_ids", []) if isinstance(selection, dict) else []
    store = context.skill_store
    getter = getattr(store, "get", None) if store is not None else None
    if not callable(getter):
        return "", []
    blocks: list[str] = []
    loaded_ids: list[str] = []
    used_chars = 0
    for skill_id in selected_ids:
        try:
            skill = getter(str(skill_id))
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if skill is None:
            continue
        block = (
            f"### {skill.name} (`{skill.skill_id}`)\n"
            f"Description: {skill.description}\n\n{skill.instructions}"
        )
        if used_chars + len(block) > max_chars:
            continue
        blocks.append(block)
        loaded_ids.append(skill.skill_id)
        used_chars += len(block)
    if not blocks:
        return "", []
    return "\n\n".join(blocks), loaded_ids


def _global_candidates(content: str) -> list[dict[str, str]]:
    parsed = parse_global_context_sections(content)
    candidates: list[dict[str, str]] = []
    if not parsed:
        return candidates
    for heading, body in parsed.items():
        if not body or heading.casefold() == "please remove" or heading.casefold() in _INVARIANT_GLOBAL_SECTIONS:
            continue
        candidates.append({
            "id": f"global:{_slug(heading)}",
            "title": f"Global context — {heading}",
            "content": body[:6000],
            "snippet": body[:500],
        })
    return candidates


def _project_candidate(task: "TaskState", context: "CoreLoopContext") -> dict[str, str] | None:
    if context.project_store is None or not task.project_id:
        return None
    getter = getattr(context.project_store, "get_project", None)
    if not callable(getter):
        return None
    try:
        project = getter(task.project_id, requester_user_id=task.user)
    except TypeError:
        project = getter(task.project_id)
    return _project_candidate_for_record(project, context) if project is not None else None


def _project_candidate_for_record(project: Any, context: "CoreLoopContext") -> dict[str, str] | None:
    try:
        content = Path(project.context_path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError, AttributeError):
        content = ""
    if not content:
        return None
    project_id = str(getattr(project, "project_id", "") or "").strip()
    title = str(getattr(project, "name", "") or project_id or "Project").strip()
    description = str(getattr(project, "description", "") or "").strip()
    candidate_content = f"{description}\n\n{content}".strip() if description else content
    return {
        "id": f"project:{project_id}",
        "title": f"Active project — {title}",
        "content": candidate_content[:8000],
        "snippet": candidate_content[:500],
    }


def _visible_active_projects(task: "TaskState", context: "CoreLoopContext") -> list[Any]:
    list_projects = getattr(context.project_store, "list_visible_projects", None)
    if not callable(list_projects) or not task.user:
        return []
    try:
        return list(list_projects(task.user))
    except Exception:
        return []


def _recent_conversation(task: "TaskState", context: "CoreLoopContext", active_project_ids: set[str]) -> str:
    store = context.conversation_store
    recent = getattr(store, "list_recent", None)
    if not callable(recent) or not task.user:
        return ""
    try:
        events = recent(owner_user_id=task.user, limit=30)
    except Exception:
        return ""
    rows = []
    for event in events:
        role = str(getattr(event, "role", "")).strip().capitalize()
        content = str(getattr(event, "content", "")).strip()
        project = str(getattr(event, "project_id", "")).strip()
        if project and project not in active_project_ids:
            continue
        if content:
            label = f"{role} [{project}]" if project else role
            rows.append(f"- {label}: {content[:600]}")
    return "Recent household conversation across projects:\n" + "\n".join(rows) if rows else ""


def _global_preamble(content: str) -> str:
    preamble: list[str] = []
    for line in content.splitlines():
        if line.startswith("## "):
            break
        preamble.append(line)
    return "\n".join(preamble).strip()[:1200]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").casefold()).strip("-")[:80] or "section"


def _prompt_file(context: "CoreLoopContext", name: str) -> str:
    if context.prompts is None:
        return ""
    try:
        prompt_file = context.prompts.load(name)
    except Exception:
        return ""
    return str(getattr(prompt_file, "content", "") or "").strip()
