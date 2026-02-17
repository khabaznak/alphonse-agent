from __future__ import annotations

from alphonse.agent.cognition.planning_engine import planner_tool_catalog_data
from alphonse.agent.cognition.tool_schemas import planner_tool_schemas
from alphonse.agent.tools.registry2 import build_planner_tool_registry


def test_planner_tool_schemas_match_registry_specs() -> None:
    registry = build_planner_tool_registry()
    expected = [spec.key for spec in registry.specs_for_schemas()]
    actual = [
        str(item.get("function", {}).get("name") or "")
        for item in planner_tool_schemas()
        if isinstance(item, dict)
    ]
    assert actual == expected


def test_catalog_tools_match_registry_catalog_flags() -> None:
    registry = build_planner_tool_registry()
    expected = {spec.key for spec in registry.specs_for_catalog()}
    payload = planner_tool_catalog_data()
    tools = payload.get("tools") if isinstance(payload, dict) else []
    actual = {
        str(item.get("tool") or "").strip()
        for item in (tools if isinstance(tools, list) else [])
        if isinstance(item, dict)
    }
    assert actual == expected

