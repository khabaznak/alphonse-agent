"""Declarative skill resources projected through fixed, read-only Desktop bindings.

No scripts, URLs, arbitrary UI, or tool registration. Relevance is an existing Jev
selection; current source authorization is independently checked on every poll.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from alphonse.agent_v2 import home_widgets, care_widgets
from alphonse.agent_v2.coordination_widgets import valid_binding, snapshot
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.core import ToolExecutionContext
from alphonse.agent_v2.core.tools.registry.native.memory import execute_read_memory_source


def definitions(skill):
    root = Path(skill.directory).resolve()
    path = root / "widgets.json"
    try:
        if not path.resolve(strict=True).is_relative_to(root) or path.stat().st_size > 32_000:
            return []
        spec = json.loads(path.read_text())
        if not isinstance(spec, dict) or set(spec) != {"version", "widgets"} or type(spec["version"]) is not int or spec["version"] != 1 or not isinstance(spec["widgets"], list):
            return []
        result, seen = [], set()
        for item in spec["widgets"][:16]:
            if not isinstance(item, dict) or set(item) - {"key", "title", "binding", "suggestion"}:
                continue
            key, title, binding = item.get("key"), item.get("title"), item.get("binding")
            if not isinstance(key, str) or not re.fullmatch(r"[a-z0-9-]{1,64}", key) or key in seen:
                continue
            if not isinstance(title, str) or not 1 <= len(title) <= 100 or not isinstance(binding, dict):
                continue
            if not home_widgets.valid_binding(binding) and not care_widgets.valid_binding(binding) and not valid_binding(binding) and binding != {"type": "agenda"} and not (set(binding) == {"type", "source_id"} and binding["type"] == "memory" and isinstance(binding["source_id"], str) and 0 < len(binding["source_id"]) <= 256):
                continue
            if not isinstance(item.get("suggestion", "show"), str) or item.get("suggestion", "show") not in {"show", "update", "pin", "remove", "expire"}:
                continue
            seen.add(key)
            result.append(item)
        return result
    except (OSError, ValueError, UnicodeError, RecursionError):
        return []


def project_widgets(runtime, user, task, *, expires_at, requested_ids=()):
    selection = task.metadata.get("v3_skill_selection", {}) if task else {}
    selected = selection.get("selected_skill_ids", []) if isinstance(selection, dict) else []
    selected = set(x for x in selected if isinstance(x, str)) if isinstance(selected, list) else set()
    widgets = []
    # Installed packages are resources, not an authority to read their named sources.
    for skill in runtime.skill_store.list_skills()[:64]:
        for item in definitions(skill):
            binding = item["binding"]
            identity = hashlib.sha256(f'{user}\0{skill.skill_id}\0{item["key"]}'.encode()).hexdigest()[:32]
            widget_id = f"skill-widget:{identity}"
            if skill.skill_id not in selected and widget_id not in requested_ids and not (binding["type"] == "care" and "care" in requested_ids):
                continue
            row = {"id": widget_id, "kind": "skill", "skillId": skill.skill_id,
                   "contextId": task.task_id if task and skill.skill_id in selected else "",
                   "scope": "member", "member": user, "title": item["title"],
                   "binding": binding["type"], "status": "unavailable", "text": "",
                   "suggestion": item.get("suggestion", "show") if skill.skill_id in selected else "update",
                   "relevant": skill.skill_id in selected, "expiresAt": expires_at}
            if binding["type"] == "agenda":
                # A navigation binding carries no schedule records; the workspace authorizes its own reads.
                row.update(status="ready", text="Review scheduled work and its history.")
            elif binding["type"] == "home":
                payload = home_widgets.snapshot(runtime, user, skill.skill_id, binding)
                if payload is not None:
                    row.update(status="ready", home=payload)
            elif binding["type"] == "care":
                payload = care_widgets.snapshot(runtime, user, skill.skill_id)
                if payload is not None:
                    row.update(status="ready", care=payload)
            elif binding["type"] == "coordination":
                payload = snapshot(runtime, user, skill.skill_id, binding)
                if payload is not None:
                    row.update(status="ready", coordination=payload)
            else:
                reader_task = TaskState(task_id="widget-read", message_id="widget-read", user=user)
                result = execute_read_memory_source({"source_id": binding["source_id"], "neighbors": 0, "max_chars": 4000}, context=ToolExecutionContext(
                    task=reader_task, messages=runtime.queue, memory=runtime.core.memory, conversation_store=runtime.conversation_store))
                if result.get("status") == "found" and result.get("items"):
                    row.update(status="ready", text=result["items"][0]["content"], historical=True)
            widgets.append(row)
    return widgets
