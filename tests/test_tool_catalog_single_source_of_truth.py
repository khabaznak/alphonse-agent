from __future__ import annotations

from alphonse.agent.cognition.tool_schemas import canonical_tool_names
from alphonse.agent.cognition.tool_schemas import llm_tool_schemas
from alphonse.agent.cognition.planning_engine import planner_tool_catalog_data
from alphonse.agent.tools.registry import build_default_tool_registry


def test_runtime_llm_tool_schemas_match_canonical_runtime_names() -> None:
    registry = build_default_tool_registry()
    expected = canonical_tool_names(registry)
    actual = [
        str(item.get("function", {}).get("name") or "")
        for item in llm_tool_schemas(registry)
        if isinstance(item, dict)
    ]
    assert actual == expected


def test_catalog_tools_match_runtime_canonical_names() -> None:
    registry = build_default_tool_registry()
    expected = set(canonical_tool_names(registry))
    payload = planner_tool_catalog_data()
    tools = payload.get("tools") if isinstance(payload, dict) else []
    actual = {
        str(item.get("tool") or "").strip()
        for item in (tools if isinstance(tools, list) else [])
        if isinstance(item, dict)
    }
    assert actual == expected
