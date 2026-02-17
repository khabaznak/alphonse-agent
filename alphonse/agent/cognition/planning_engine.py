from __future__ import annotations

from pathlib import Path
from typing import Any

from alphonse.agent.cognition.tool_catalog_renderer import render_tool_catalog
from alphonse.agent.tools.registry2 import build_planner_tool_registry


def format_available_abilities() -> str:
    return format_available_ability_catalog()


def format_available_ability_catalog() -> str:
    template_dir = Path(__file__).resolve().parent / "templates"
    return render_tool_catalog(build_planner_tool_registry(), template_dir)


def planner_tool_catalog_data() -> dict[str, Any]:
    registry = build_planner_tool_registry()
    tools: list[dict[str, Any]] = []
    for spec in registry.specs_for_catalog():
        params: list[dict[str, Any]] = []
        schema = spec.input_schema if isinstance(spec.input_schema, dict) else {}
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required_raw = schema.get("required") if isinstance(schema.get("required"), list) else []
        required = {str(item).strip() for item in required_raw if str(item).strip()}
        for name, definition in properties.items():
            if not isinstance(definition, dict):
                continue
            params.append(
                {
                    "name": str(name).strip(),
                    "type": str(definition.get("type") or "string").strip(),
                    "required": str(name).strip() in required,
                }
            )
        tools.append(
            {
                "tool": spec.key,
                "description": spec.description,
                "when_to_use": spec.when_to_use,
                "returns": spec.returns,
                "input_parameters": params,
            }
        )
    return {"tools": tools}


def planner_tool_names() -> list[str]:
    return [
        str(item.get("tool") or "").strip()
        for item in (planner_tool_catalog_data().get("tools") or [])
        if isinstance(item, dict) and str(item.get("tool") or "").strip()
    ]
