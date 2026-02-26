from __future__ import annotations

from alphonse.agent.cognition.planning_engine import (
    format_available_abilities,
    format_available_ability_catalog,
    planner_tool_catalog_data,
)


def test_available_abilities_renders_full_markdown_catalog() -> None:
    rendered = format_available_abilities()
    assert rendered.startswith("# Available Tools")
    assert "### `askQuestion`" in rendered
    assert "one clear question" in rendered
    assert "### `get_time`" in rendered


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
    required = {
        "askQuestion",
        "get_time",
        "create_reminder",
        "job_create",
        "job_list",
        "job_pause",
        "job_resume",
        "job_delete",
        "job_run_now",
        "terminal_sync",
        "mcp_call",
        "terminal_async",
        "terminal_async_command_status",
        "local_audio_output_speak",
        "stt_transcribe",
        "get_my_settings",
        "get_user_details",
    }
    assert required.issubset(names)


def test_available_ability_catalog_prompt_is_markdown() -> None:
    rendered = format_available_ability_catalog()
    assert rendered.startswith("# Available Tools")
    assert "### `askQuestion`" in rendered
