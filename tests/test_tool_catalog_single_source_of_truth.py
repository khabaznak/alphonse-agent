from __future__ import annotations

from alphonse.agent.cognition.planning_engine import planner_tool_catalog_data
from alphonse.agent.tools.registry import build_default_tool_registry
from alphonse.agent.tools.registry import planner_canonical_tool_names
from alphonse.agent.tools.registry import planner_tool_schemas


def test_runtime_llm_tool_schemas_match_canonical_runtime_names() -> None:
    registry = build_default_tool_registry()
    expected = planner_canonical_tool_names(registry)
    actual = [
        str(item.get("function", {}).get("name") or "")
        for item in planner_tool_schemas(registry)
        if isinstance(item, dict)
    ]
    assert actual == expected


def test_catalog_tools_match_runtime_canonical_names() -> None:
    registry = build_default_tool_registry()
    expected = set(planner_canonical_tool_names(registry))
    payload = planner_tool_catalog_data()
    tools = payload.get("tools") if isinstance(payload, dict) else []
    actual = {
        str(item.get("tool") or "").strip()
        for item in (tools if isinstance(tools, list) else [])
        if isinstance(item, dict)
    }
    assert actual == expected


def test_vision_tools_surface_matches_expected_contract() -> None:
    registry = build_default_tool_registry()
    names = set(planner_canonical_tool_names(registry))
    assert "vision.analyze_image" in names
    assert "vision.extract_text" in names
    assert "analyze_telegram_image" not in names
