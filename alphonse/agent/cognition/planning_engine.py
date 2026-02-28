from __future__ import annotations

from typing import Any

from alphonse.agent.cognition.tool_schemas import canonical_tool_names
from alphonse.agent.cognition.tool_schemas import llm_tool_schemas
from alphonse.agent.tools.registry import build_default_tool_registry


def format_available_abilities() -> str:
    return format_available_ability_catalog()


def format_available_ability_catalog() -> str:
    payload = planner_tool_catalog_data()
    tools = payload.get("tools") if isinstance(payload, dict) else []
    lines = ["# Available Tools"]
    for item in tools if isinstance(tools, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("tool") or "").strip()
        if not name:
            continue
        desc = str(item.get("description") or "").strip()
        lines.append(f"### `{name}`")
        if desc:
            lines.append(desc)
    return "\n\n".join(lines).strip()


def planner_tool_catalog_data() -> dict[str, Any]:
    registry = build_default_tool_registry()
    schemas = llm_tool_schemas(registry)
    tools: list[dict[str, Any]] = []
    for schema in schemas:
        fn = schema.get("function") if isinstance(schema, dict) else None
        if not isinstance(fn, dict):
            continue
        params: list[dict[str, Any]] = []
        params_schema = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        properties = params_schema.get("properties") if isinstance(params_schema.get("properties"), dict) else {}
        required_raw = params_schema.get("required") if isinstance(params_schema.get("required"), list) else []
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
                "tool": str(fn.get("name") or "").strip(),
                "description": str(fn.get("description") or "").strip(),
                "when_to_use": "",
                "returns": "",
                "input_parameters": params,
            }
        )
    return {"tools": tools}


def planner_tool_names() -> list[str]:
    return canonical_tool_names(build_default_tool_registry())
