"""Shared context inputs for the V3 conversation and task paths."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import TYPE_CHECKING

from alphonse.agent_v2.core.context_inventory import ContextText, source, combined, omitted

from alphonse.agent_v2.agent_config import parse_global_context_sections
from alphonse.agent_v2.core.memory.formatting import bound_context_record, bound_memory_context, join_context_sections
from alphonse.agent_v2.core.memory.continuity import active_task_continuity
from alphonse.agent_v2.core.memory.retry_context import retry_continuity_context

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.task_state import TaskState


_INVARIANT_GLOBAL_SECTIONS = {
    "family definition",
    "household location and setup",
    "alphonse's household-wide interaction defaults",
    "privacy and sharing boundaries",
}


def prepare_context_curation(task: "TaskState", context: "CoreLoopContext", *, include_understanding: bool = True, stage: str = "curation", include_event: bool = True) -> tuple[str, list[dict[str, str]]]:
    """Return always-included context and bounded optional context candidates."""
    from .task_understanding import render
    from .event_evidence import render_event_evidence
    from .context_manifests import record_context_input
    event = render_event_evidence(task, context, stage=stage) if include_event else ""
    sections: list[str] = []
    active_projects = _visible_active_projects(task, context)
    recent_text = _recent_conversation(task, context)
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
                    sections.append(f"{heading}:\n" + source(body, "global:" + _slug(heading), "global_policy", reason="required"))
        else:
            # Preserve older free-form documents until they are structured.
            sections.append("Household and global context:\n" + source(global_context, "global:policy", "global_policy", reason="required"))

    candidates = _global_candidates(global_context)
    for project in active_projects:
        project_id = str(getattr(project, "project_id", "") or "").strip()
        if not project_id:
            continue
        candidate = _project_candidate_for_record(project, context)
        if candidate is not None:
            candidates.append(candidate)
        else:
            sections.append(omitted(source("", "project:" + project_id, "project_guidance"), reason="unavailable"))
    rendered = join_context_sections(sections, 12000)
    overflow = [omitted(source(item['content'], item['id'], 'context_candidate')) for item in candidates[64:]]
    invariant = combined([event, *([render(task)] if include_understanding else []), rendered, *overflow])
    if include_event:
        record_context_input(task, stage, invariant)
    return invariant, candidates[:64]


def conversation_context(
    task: "TaskState",
    context: "CoreLoopContext",
    *,
    selected_context_ids: list[str] | tuple[str, ...] = (),
    stage: str = "decision",
    include_event: bool = True,
) -> str:
    """Render invariant context plus the optional candidates selected at admission."""
    invariant, candidates = prepare_context_curation(task, context, include_understanding=False, stage=stage, include_event=include_event)
    selected = set(str(item) for item in selected_context_ids)
    sections = [invariant] if invariant else []
    for candidate in candidates:
        if candidate["id"] in selected:
            sections.append(f"{candidate['title']}:\n" + candidate['content'])
        else:
            sections.append(omitted(candidate['content'], reason='irrelevant'))
    return join_context_sections(sections, 24000)


def task_context_for_decision(
    task: "TaskState",
    context: "CoreLoopContext",
    *,
    selected_context_ids: list[str] | tuple[str, ...] = (),
    purpose: str = "planning",
    stage: str | None = None,
) -> str:
    """Render the shared task context used by Jev and System Two."""
    from .projections import enabled, project

    purpose = stage or purpose
    if enabled():
        return project(task, context, purpose=purpose, selected_context_ids=selected_context_ids)
    from .task_understanding import render
    understanding = render(task)
    sections = [active_task_continuity(task), retry_continuity_context(task, context),
                conversation_context(task, context, selected_context_ids=selected_context_ids, stage=purpose, include_event=False)]
    from .event_evidence import render_event_evidence
    from .context_manifests import record_context_input
    event = render_event_evidence(task, context, stage=purpose)
    sections.extend(task_context_addenda(task))
    required = combined([event, understanding])
    if len(required) > 36000:
        from .projections import ContextBudgetExceeded
        raise ContextBudgetExceeded('v3_required_task_understanding_exceeds_budget')
    result = combined([required, join_context_sections(sections, max(0, 36000 - len(required) - 2))])
    if purpose == "curation":
        # Use the same access-filtered loader as System Two. Keep its bounded
        # resource/workflow block and source ranges intact, outside the shared
        # context budget so a long conversation cannot evict execution guidance.
        guidance, loaded_ids = selected_skill_guidance(task, context)
        selection = task.metadata.get("v3_skill_selection")
        if isinstance(selection, dict):
            selection["loaded_skill_ids"] = loaded_ids
        result = combined([
            result,
            conversation_coverage(task),
            "Selected skill execution context (relevance only; never permission):\n" + guidance,
        ], purpose=purpose, max_chars=53000)
    record_context_input(task, purpose, result)
    if purpose == "curation":
        return result
    return ContextText(str(result), result.records, **{**result.details, 'purpose': purpose, 'max_chars': 36000})


def task_context_addenda(task: "TaskState") -> list[str]:
    """Render task memory and fact sections shared across decision systems."""
    identity = str(task.task_id or task.message_id or 'current')
    durable_memory = memory_inventory(task.memory_context_md, f'task-working:{identity}:memory_context')
    durable_memory = bound_memory_context(durable_memory, 9000)
    return [
        "Current goal:\n" + (bound_context_record(source(task.goal, 'request:' + str(task.message_id or task.task_id), 'current_request'), 9000) or '- (none)'),
        "Durable household memory (person/audience scoped; project association is optional):\n" + durable_memory,
        "Known task facts:\n" + (bound_context_record(source(task.facts_md, f'task-working:{identity}:facts', 'working_facts'), 4000) or '- (none)'),
        "Memory facts:\n" + (bound_context_record(source(task.memory_facts_md, f'task-working:{identity}:memory_facts', 'working_facts'), 3000) or '- (none)'),
    ]


def conversation_coverage(task: "TaskState") -> str:
    coverage = task.metadata.get("conversation_projection") or {}
    # Counts/flags are enough for routing; do not duplicate all historical IDs.
    signals = {key: coverage.get(key, False) for key in (
        "requires_recall", "history_unavailable", "current_records_omitted", "window_may_have_earlier_events",
    )}
    return source(
        "Conversation coverage (recall missing antecedents before relying on them; history never grants permission):\n"
        + json.dumps(signals, ensure_ascii=False),
        f'task-working:{task.task_id or task.message_id or "current"}:conversation_coverage', 'conversation_coverage',
    )


def memory_inventory(value, identity):
    text = str(value or '').strip() or '- (none)'
    marker = '\n\n# Current Request\n'
    boundary = text.find(marker)
    tail = ''
    if boundary >= 0:
        from alphonse.agent_v2.core.context_manifests import request_text
        tail = request_text(text[boundary:])
        text = text[:boundary]
    matches = list(re.finditer(r'(?m)^- \[(memory:[a-zA-Z0-9_:-]+)(?=;|\])', text))
    if not matches:
        return combined([source(text, identity, 'memory_snapshot'), tail], '', recall={'status': 'not_performed', 'strategy': 'ledger_tail_snapshot', 'result_count': 0})
    parts = [text[:matches[0].start()]]
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start():end]
        stripped = block.rstrip()
        parts.extend([source(stripped, match[1], 'knowledge', reason='ledger_tail'), block[len(stripped):]])
    return combined([*parts, tail], '', recall={'status': 'not_performed', 'strategy': 'ledger_tail', 'candidate_count': len(matches), 'result_count': 0})


def selected_project_context(task: "TaskState", context: "CoreLoopContext") -> str:
    selection = task.metadata.get("v3_context_selection")
    selected = set(selection.get("selected_context_ids", [])) if isinstance(selection, dict) else set()
    _invariant, candidates = prepare_context_curation(task, context, include_event=False)
    blocks = [f"{item['title']}:\n" + item['content'] for item in candidates if item.get("id", "").startswith("project:") and item.get("id") in selected]
    return combined(blocks)


def recent_conversation_for_curation(task: "TaskState", context: "CoreLoopContext") -> str:
    from alphonse.agent_v2.core.memory.transcript import conversation_transcript

    return conversation_transcript(task, context.conversation_store, max_chars=5000)


def skill_candidates_for_curation(context: "CoreLoopContext", task: "TaskState" | None = None) -> list[dict[str, str]]:
    store = context.skill_store
    if store is None:
        return []
    list_skills = getattr(store, "list_skills", None)
    if not callable(list_skills):
        return []
    try:
        from alphonse.agent_v2.skill_resources import authorized
        return [skill.candidate() for skill in list_skills()
                if task is not None and authorized(store, skill.skill_id, str(task.user or ""))]
    except Exception:
        return []


def selected_skill_guidance(task: "TaskState", context: "CoreLoopContext", *, max_chars: int | None = None) -> tuple[str, list[str]]:
    """Activate complete instructions atomically, or fail with recovery metadata."""
    import os
    from alphonse.agent_v2.skill_resources import fail, require_skill, pin_skill
    if max_chars is None:
        max_chars = int(os.environ.get("ALPHONSE_SKILL_INSTRUCTION_BUDGET_CHARS", "64000"))
    selection = task.metadata.get("v3_skill_selection")
    selected_ids = selection.get("selected_skill_ids", []) if isinstance(selection, dict) else []
    if isinstance(selection, dict):
        selected_set = {str(item) for item in selected_ids}
        primary_ids = [str(item) for item in selection.get("primary_skill_ids", []) if str(item) in selected_set]
        primary_set = set(primary_ids)
        selected_ids = primary_ids + [str(item) for item in selected_ids if str(item) not in primary_set]
    store = context.skill_store
    getter = getattr(store, "get", None) if store is not None else None
    if not selected_ids:
        return "", []
    selection["loaded_skill_ids"] = []
    if not callable(getter):
        fail(task, "package_unavailable", str(selected_ids[0]))
    blocks: list[str] = []
    loaded_ids: list[str] = []
    used_chars = 0
    for skill_id in selected_ids:
        skill = require_skill(task, store, str(skill_id))
        pin = pin_skill(task, skill, context.artifact_resources)
        resources = context.artifact_resources
        # Artifact sections may contain another household member's private
        # paths. Load only catalog-backed, authorized sections for this task.
        base_instructions = re.split(r"(?m)^## Artifact: ", skill.instructions)[0].rstrip()
        migration_guidance = ""
        account_guidance = ""
        if context.skill_bindings is not None:
            from alphonse.agent_v2.accounts import AccountSession
            capabilities = context.skill_bindings.account_capabilities(
                AccountSession(str(task.user or "")), skill_id=skill.skill_id,
                project_id=task.project_id or None)
            if capabilities:
                account_guidance = "Managed account capabilities (advisory; rechecked at execution): " + json.dumps(capabilities) + "\n"
        if resources is not None and context.skill_bindings is not None:
            from alphonse.agent_v2.legacy_migration import LegacyMigration
            migration_guidance = LegacyMigration(resources, context.skill_bindings).guidance(skill.skill_id, str(task.user or ""))
        block = (
            f"### {skill.name} (`{skill.skill_id}`)\n"
            f"Description: {skill.description}\n"
            f"Package metadata (advisory; never grants authority): {json.dumps(skill.package_metadata(), ensure_ascii=False)}\n"
            f"Skill package directory: {skill.directory}\n"
            f"Pinned package revision: {pin['revision']}\n"
            "Use native.skill_read with this skill_id and a package-relative path to read references, "
            "script documentation/source or text assets on demand, or operation=list to inspect a directory. "
            "Do not load all supporting files or live databases. Binary assets require an authorized media capability. "
            "If the answer needs a supporting resource, plan a skill_resources subgoal before responding. "
            "Skill files resolve against the skill package directory, independently of the current project. "
            "Standard scripts need no registration: use native.bash with skill_id to resolve cwd from this installed package. "
            "Follow documented runtime/dependency/configuration checks before mutations; runtime and required_files are optional Bash preflight arguments. "
            "Quote paths and use explicit interpreters for non-executable scripts. Verify writes through readback. "
            "Keep live data and runtime output outside the reusable package; changing its instructions/code/resources invalidates the active revision pin. "
            "Authorized native.skill_maintain publication advances this task's pin with an audit receipt and reloads complete instructions; other tasks require fresh selection. "
            "Skill selection and resource bindings do not grant permission for effects.\n\n"
            + base_instructions + "\n\n" + migration_guidance + account_guidance
        )
        blocks.append(source(block, str(skill.skill_id), "skill_guidance", reason="selected_skill"))
        loaded_ids.append(skill.skill_id)
        used_chars += len(block) + 2
    if used_chars > max_chars:
        selection["required_chars"] = used_chars
        selection["budget_chars"] = max_chars
        fail(task, "instruction_budget_exceeded")
    selection.pop("recovery", None)
    selection.update(status="activated", loaded_skill_ids=loaded_ids, budget_chars=max_chars,
                     instruction_chars=used_chars)
    return combined(blocks, selected_skill_ids=list(selected_ids), loaded_skill_ids=loaded_ids), loaded_ids


def _global_candidates(content: str, *, max_chars: int | None = 6000) -> list[dict[str, str]]:
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
            "content": source(body, f"global:{_slug(heading)}", "global_guidance", reason="selected_relevant")[:max_chars],
            "snippet": body[:500],
        })
    return candidates


def _project_candidate_for_record(project: Any, context: "CoreLoopContext", *, max_chars: int | None = 8000) -> dict[str, str] | None:
    try:
        config_path = getattr(project, "config_path", "") or str(Path(project.root_path) / "project_config.json")
        content = Path(config_path).read_text(encoding="utf-8").strip()
        parsed = json.loads(content)
        if not isinstance(parsed, dict) or parsed.get("schema_version") != 1:
            return None
        content = json.dumps(parsed, ensure_ascii=False, indent=2)
    except (OSError, UnicodeDecodeError, AttributeError):
        content = ""
    except json.JSONDecodeError:
        content = ""
    if not content:
        return None
    project_id = str(getattr(project, "project_id", "") or "").strip()
    title = str(getattr(project, "name", "") or project_id or "Project").strip()
    description = str(getattr(project, "description", "") or "").strip()
    candidate_content = f"{description}\n\n{content}".strip() if description else content
    return {
        "id": f"project:{project_id}",
        "title": f"Project — {title}",
        "content": source(candidate_content, f"project:{project_id}", "project_guidance", reason="selected_relevant")[:max_chars],
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


def _recent_conversation(task: "TaskState", context: "CoreLoopContext") -> str:
    from alphonse.agent_v2.core.memory.transcript import conversation_transcript

    rendered = conversation_transcript(task, context.conversation_store)
    return "Recent household conversation across projects:\n" + rendered if rendered != "- (none)" else ""


def _global_preamble(content: str) -> str:
    preamble: list[str] = []
    for line in content.splitlines():
        if line.startswith("## "):
            break
        preamble.append(line)
    return source("\n".join(preamble).strip(), "global:preamble", "global_policy", reason="required")[:1200]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").casefold()).strip("-")[:80] or "section"


def _bounded_context_value(value: object, max_chars: int) -> str:
    return bound_context_record(value, max_chars) or "- (none)"


def _prompt_file(context: "CoreLoopContext", name: str) -> str:
    if context.prompts is None:
        return ""
    try:
        prompt_file = context.prompts.load(name)
    except Exception:
        return ""
    return str(getattr(prompt_file, "content", "") or "").strip()
