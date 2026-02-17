from __future__ import annotations

from typing import Any

from alphonse.agent.tools.registry2 import build_planner_tool_registry, planner_tool_schemas_from_specs


def planner_tool_schemas() -> list[dict[str, Any]]:
    registry = build_planner_tool_registry()
    return planner_tool_schemas_from_specs(registry)

