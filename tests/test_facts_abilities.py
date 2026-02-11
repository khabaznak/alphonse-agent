from __future__ import annotations

from alphonse.agent.cognition.abilities import json_runtime
from alphonse.agent.cognition.abilities.json_runtime import load_json_abilities
from alphonse.agent.tools.registry import build_default_tool_registry


def _ability(intent_name: str):
    for ability in load_json_abilities():
        if ability.intent_name == intent_name:
            return ability
    raise AssertionError(f"ability not found: {intent_name}")


def test_facts_user_get_returns_fact_updates(monkeypatch) -> None:
    monkeypatch.setattr(
        json_runtime.identity_profile,
        "get_display_name",
        lambda conversation_key: "Alex",
    )
    monkeypatch.setattr(
        json_runtime,
        "_principal_id_from_state",
        lambda state: "principal-1",
    )
    monkeypatch.setattr(
        json_runtime,
        "list_location_profiles",
        lambda principal_id, label=None, active_only=True, limit=10: [
            {"label": "home", "address_text": "123 Main St"}
        ],
    )
    monkeypatch.setattr(
        json_runtime,
        "list_users",
        lambda active_only=True, limit=50: [
            {"display_name": "Nina", "relationship": "wife", "role": "adult", "is_admin": False}
        ],
    )

    ability = _ability("facts.user.get")
    result = ability.execute(
        {
            "conversation_key": "telegram:123",
            "channel_type": "telegram",
            "channel_target": "123",
            "locale": "en-US",
            "timezone": "UTC",
        },
        build_default_tool_registry(),
    )

    facts = result.get("fact_updates")
    assert isinstance(facts, dict)
    assert facts.get("user", {}).get("name") == "Alex"
    assert facts.get("user", {}).get("principal_id") == "principal-1"


def test_facts_agent_get_returns_fact_updates() -> None:
    ability = _ability("facts.agent.get")
    result = ability.execute(
        {
            "channel_type": "telegram",
            "channel_target": "123",
        },
        build_default_tool_registry(),
    )

    facts = result.get("fact_updates")
    assert isinstance(facts, dict)
    assert facts.get("agent", {}).get("name") == "Alphonse"
    assert facts.get("agent", {}).get("status") == "online"
