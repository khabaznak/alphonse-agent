from __future__ import annotations

import re
from typing import Any

from alphonse.agent.cognition.prompt_templates_runtime import (
    PLANNING_TOOLS_TEMPLATE,
    render_prompt_template,
)

_HEADING_RE = re.compile(r"^###\s+([A-Za-z_][A-Za-z0-9_.-]*)\(([^)]*)\)\s*$")
_INPUT_RE = re.compile(r"^\s*-\s+`([^`]+)`\s+\(([^,]+),\s*(required|optional)\)\s*$", re.IGNORECASE)


def format_available_abilities() -> str:
    tools = planner_tool_catalog_data().get("tools")
    if not isinstance(tools, list):
        return ""
    lines: list[str] = []
    for item in tools:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "").strip()
        if not tool:
            continue
        params = item.get("input_parameters") if isinstance(item.get("input_parameters"), list) else []
        rendered = ", ".join(_render_param_signature(param) for param in params if isinstance(param, dict))
        description = str(item.get("description") or "No description.")
        lines.append(f"- {tool}({rendered}) -> {description}")
    return "\n".join(lines)


def format_available_ability_catalog() -> str:
    return render_prompt_template(PLANNING_TOOLS_TEMPLATE, {}).strip()


def planner_tool_catalog_data() -> dict[str, Any]:
    tools: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    lines = format_available_ability_catalog().splitlines()
    for raw in lines:
        line = raw.rstrip()
        heading = _HEADING_RE.match(line.strip())
        if heading:
            if isinstance(current, dict):
                tools.append(current)
            current = {
                "tool": heading.group(1),
                "description": "",
                "when_to_use": "",
                "returns": "",
                "input_parameters": [],
            }
            continue
        if not isinstance(current, dict):
            continue
        stripped = line.strip()
        if stripped.startswith("- Description:"):
            current["description"] = stripped.replace("- Description:", "", 1).strip()
            continue
        if stripped.startswith("- When to use:"):
            current["when_to_use"] = stripped.replace("- When to use:", "", 1).strip()
            continue
        if stripped.startswith("- Returns:"):
            current["returns"] = stripped.replace("- Returns:", "", 1).strip()
            continue
        input_match = _INPUT_RE.match(line)
        if input_match:
            params = current.get("input_parameters")
            if not isinstance(params, list):
                params = []
                current["input_parameters"] = params
            params.append(
                {
                    "name": input_match.group(1).strip(),
                    "type": input_match.group(2).strip(),
                    "required": input_match.group(3).strip().lower() == "required",
                }
            )
    if isinstance(current, dict):
        tools.append(current)
    return {"tools": tools}


def planner_tool_names() -> list[str]:
    return [
        str(item.get("tool") or "").strip()
        for item in (planner_tool_catalog_data().get("tools") or [])
        if isinstance(item, dict) and str(item.get("tool") or "").strip()
    ]


def _render_param_signature(param: dict[str, Any]) -> str:
    name = str(param.get("name") or "").strip()
    ptype = str(param.get("type") or "string").strip()
    required = bool(param.get("required", False))
    return f"{name}{'' if required else '?'}:{ptype}"
