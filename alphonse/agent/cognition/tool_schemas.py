from __future__ import annotations

from typing import Any

from alphonse.agent.tools.registry import build_default_tool_registry
from alphonse.agent.tools.registry import planner_canonical_tool_names
from alphonse.agent.tools.registry import planner_tool_descriptions
from alphonse.agent.tools.registry import planner_tool_parameters
from alphonse.agent.tools.registry import planner_tool_schemas


def canonical_tool_names(tool_registry: Any) -> list[str]:
    return planner_canonical_tool_names(tool_registry)


def llm_tool_schemas(tool_registry: Any) -> list[dict[str, Any]]:
    return planner_tool_schemas(tool_registry)


def tool_descriptions(tool_registry: Any) -> dict[str, str]:
    return planner_tool_descriptions(tool_registry)


def tool_parameters(tool_name: str) -> dict[str, Any] | None:
    registry = build_default_tool_registry()
    return planner_tool_parameters(registry, tool_name)


def required_args(tool_name: str) -> list[str]:
    params = tool_parameters(tool_name)
    if not isinstance(params, dict):
        return []
    required = params.get("required")
    if not isinstance(required, list):
        return []
    return [str(item) for item in required if str(item)]
