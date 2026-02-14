from __future__ import annotations

from alphonse.agent.cognition.planning_engine import (
    format_available_abilities,
    format_available_ability_catalog,
    planner_tool_catalog_data,
)


def test_available_abilities_include_summary_and_optional_marker() -> None:
    rendered = format_available_abilities()
    assert "askQuestion(" in rendered
    assert "one clear question" in rendered
    assert "getTime()" in rendered


def test_available_ability_catalog_is_llm_focused() -> None:
    payload = planner_tool_catalog_data()
    assert isinstance(payload, dict)
    assert "tools" in payload
    assert "io_channels" not in payload
    tools = payload["tools"]
    assert isinstance(tools, list)
    ask = next(
        (item for item in tools if isinstance(item, dict) and item.get("tool") == "askQuestion"),
        None,
    )
    assert isinstance(ask, dict)
    assert "description" in ask
    assert "when_to_use" in ask
    assert "returns" in ask


def test_available_ability_catalog_has_minimal_tools_only() -> None:
    payload = planner_tool_catalog_data()
    tools = payload.get("tools") if isinstance(payload, dict) else []
    names = {
        str(item.get("tool") or "").strip()
        for item in (tools if isinstance(tools, list) else [])
        if isinstance(item, dict)
    }
    assert names == {
        "askQuestion",
        "getTime",
        "createReminder",
        "getMySettings",
        "getUserDetails",
    }


def test_available_ability_catalog_prompt_is_markdown() -> None:
    rendered = format_available_ability_catalog()
    assert rendered.startswith("## Tool Catalog")
    assert "### askQuestion" in rendered
