from __future__ import annotations

from alphonse.agent.cognition.abilities.json_runtime import load_json_abilities
from alphonse.agent.cognition.plans import PlanType
from alphonse.agent.tools.registry import build_default_tool_registry


def test_time_current_ability_loaded_from_json() -> None:
    abilities = load_json_abilities()
    intent_names = {ability.intent_name for ability in abilities}
    assert "time.current" in intent_names


def test_time_current_json_ability_executes_with_clock_tool() -> None:
    ability = next(item for item in load_json_abilities() if item.intent_name == "time.current")
    tools = build_default_tool_registry()
    result = ability.execute(
        {
            "locale": "es-MX",
            "timezone": "America/Mexico_City",
        },
        tools,
    )
    assert isinstance(result.get("response_text"), str)
    assert "America/Mexico_City" in result["response_text"]


def test_meta_gaps_list_json_ability_emits_query_status_plan() -> None:
    ability = next(item for item in load_json_abilities() if item.intent_name == "meta.gaps_list")
    tools = build_default_tool_registry()
    result = ability.execute(
        {
            "channel_type": "webui",
            "channel_target": "webui",
            "chat_id": "webui",
            "locale": "es-MX",
        },
        tools,
    )
    plans = result.get("plans") or []
    assert plans
    plan = plans[0]
    assert plan.get("plan_type") == PlanType.QUERY_STATUS.value
    payload = plan.get("payload") or {}
    assert payload.get("include") == ["gaps_summary"]
    assert payload.get("locale") == "es-MX"
