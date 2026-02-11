from __future__ import annotations

import json

from alphonse.agent.cognition.intent_discovery_engine import (
    format_available_abilities,
    format_available_ability_catalog,
)


def test_available_abilities_include_summary_and_optional_marker() -> None:
    rendered = format_available_abilities()
    assert "askQuestion(" in rendered
    assert "missing end-user data" in rendered
    assert "?:" in rendered


def test_available_ability_catalog_includes_io_channels() -> None:
    payload = json.loads(format_available_ability_catalog())
    assert isinstance(payload, dict)
    assert "tools" in payload
    assert "io_channels" in payload
    io_channels = payload["io_channels"]
    assert isinstance(io_channels, dict)
    assert "senses" in io_channels
    assert "extremities" in io_channels


def test_available_ability_catalog_includes_fact_tools() -> None:
    payload = json.loads(format_available_ability_catalog())
    tools = payload.get("tools") if isinstance(payload, dict) else []
    names = {
        str(item.get("tool") or "").strip()
        for item in (tools if isinstance(tools, list) else [])
        if isinstance(item, dict)
    }
    assert "facts.user.get" in names
    assert "facts.agent.get" in names
