from __future__ import annotations

from alphonse.agent.tools.registry import build_default_tool_registry
from alphonse.agent.tools.registry import planner_canonical_tool_names
from alphonse.agent.tools.registry import planner_tool_schemas


def test_planner_tool_schemas_are_llm_focused() -> None:
    schemas = planner_tool_schemas(build_default_tool_registry())
    assert isinstance(schemas, list)
    assert schemas
    get_time = next(
        (
            item
            for item in schemas
            if isinstance(item, dict)
            and isinstance(item.get("function"), dict)
            and item["function"].get("name") == "get_time"
        ),
        None,
    )
    assert isinstance(get_time, dict)
    function = get_time["function"]
    assert "description" in function
    assert "parameters" in function


def test_available_tool_catalog_has_minimal_tools_only() -> None:
    names = set(planner_canonical_tool_names(build_default_tool_registry()))
    required = {
        "get_time",
        "create_reminder",
        "jobs.create",
        "jobs.list",
        "jobs.pause",
        "jobs.resume",
        "jobs.delete",
        "jobs.run_now",
        "execution.run_terminal",
        "execution.call_mcp",
        "audio.speak_local",
        "audio.transcribe",
        "get_my_settings",
        "get_user_details",
    }
    assert required.issubset(names)
