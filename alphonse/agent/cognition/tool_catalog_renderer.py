from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from alphonse.agent.tools.registry2 import ToolRegistry
from alphonse.agent.tools.spec import ToolSpec


def pretty_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _group_key(spec: ToolSpec) -> str:
    tags = {str(tag).strip().lower() for tag in (spec.domain_tags or [])}
    if tags.intersection({"communication", "messaging", "delivery", "audio", "output", "transcription"}):
        return "communication_audio"
    if tags.intersection({"automation", "jobs", "control", "time", "reminders", "planning"}):
        return "planning_automation"
    if tags.intersection({"ops", "terminal", "ssh", "automation"}):
        return "terminal_ops"
    if tags.intersection({"identity", "admin", "users", "lookup", "access-control", "onboarding"}):
        return "identity_admin"
    if tags.intersection({"vision", "image", "analysis", "telegram", "files"}):
        return "vision_files"
    if tags.intersection({"context", "settings"}):
        return "context_runtime"
    return "general"


def _group_metadata(group: str) -> dict[str, str]:
    if group == "communication_audio":
        return {
            "title": "Communication and Audio",
            "when_to_choose": "Use for messaging, voice notes, TTS, and transcription tasks.",
            "avoid_when": "Avoid for scheduling, identity admin, or shell execution tasks.",
        }
    if group == "planning_automation":
        return {
            "title": "Planning, Time, and Jobs",
            "when_to_choose": "Use for time checks, reminders, and recurring/scheduled workflows.",
            "avoid_when": "Avoid when you only need immediate chat responses or direct shell access.",
        }
    if group == "terminal_ops":
        return {
            "title": "Terminal and Remote Ops",
            "when_to_choose": "Use for command execution, diagnostics, and constrained remote operations.",
            "avoid_when": "Avoid when a dedicated higher-level capability already solves the task.",
        }
    if group == "identity_admin":
        return {
            "title": "Identity and Admin",
            "when_to_choose": "Use for user lookup, onboarding, and access management actions.",
            "avoid_when": "Avoid for regular conversation or non-user-management requests.",
        }
    if group == "vision_files":
        return {
            "title": "Vision and Files",
            "when_to_choose": "Use for image/file retrieval and visual analysis workflows.",
            "avoid_when": "Avoid for plain text-only requests with no media/file input.",
        }
    if group == "context_runtime":
        return {
            "title": "Runtime Context",
            "when_to_choose": "Use to fetch settings/user context before sensitive decisions.",
            "avoid_when": "Avoid when the needed context is already explicit and fresh.",
        }
    return {
        "title": "General",
        "when_to_choose": "Use when no specialized group better matches the request.",
        "avoid_when": "Avoid when a more specific capability group clearly applies.",
    }


def build_render_context(specs: list[ToolSpec]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for spec in specs:
        group = _group_key(spec)
        if group not in groups:
            groups[group] = {"group": group, **_group_metadata(group), "tools": []}
        groups[group]["tools"].append(
            {
                "key": spec.key,
                "description": spec.description.strip(),
                "when_to_use": spec.when_to_use.strip(),
                "returns": spec.returns.strip(),
                "domain_tags": spec.domain_tags,
                "safety_level": spec.safety_level.value,
                "requires_confirmation": spec.requires_confirmation,
                "input_schema_pretty": pretty_json(spec.input_schema),
                "examples": spec.examples,
            }
        )
    ordered = sorted(groups.values(), key=lambda item: str(item.get("title") or ""))
    for group in ordered:
        tools = group.get("tools")
        if isinstance(tools, list):
            tools.sort(key=lambda item: str(item.get("key") or ""))
    return ordered


def render_tool_catalog(registry: ToolRegistry, template_dir: str | Path) -> str:
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=False)
    template = env.get_template("planning.tools.md.j2")
    specs = registry.specs_for_catalog()
    groups_context = build_render_context(specs)
    return str(template.render(groups=groups_context)).strip()
