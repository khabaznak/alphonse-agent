from __future__ import annotations

import sqlite3

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


def test_db_ability_specs_extend_runtime_loader(tmp_path) -> None:
    db_path = tmp_path / "nerve-db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE ability_specs (
              intent_name TEXT PRIMARY KEY,
              kind TEXT NOT NULL,
              tools_json TEXT NOT NULL,
              spec_json TEXT NOT NULL,
              enabled INTEGER NOT NULL DEFAULT 1,
              source TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO ability_specs (
              intent_name, kind, tools_json, spec_json, enabled, source, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 1, 'test', datetime('now'), datetime('now'))
            """,
            (
                "test.intent",
                "plan_emit",
                "[]",
                '{"intent_name":"test.intent","kind":"plan_emit","tools":[],"plan":{"plan_type":"QUERY_STATUS","payload":{"include":["gaps_summary"]}}}',
            ),
        )
    abilities = load_json_abilities(db_path=str(db_path))
    intent_names = {ability.intent_name for ability in abilities}
    assert "test.intent" in intent_names


def test_db_ability_spec_overrides_file_spec(tmp_path) -> None:
    db_path = tmp_path / "nerve-db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE ability_specs (
              intent_name TEXT PRIMARY KEY,
              kind TEXT NOT NULL,
              tools_json TEXT NOT NULL,
              spec_json TEXT NOT NULL,
              enabled INTEGER NOT NULL DEFAULT 1,
              source TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO ability_specs (
              intent_name, kind, tools_json, spec_json, enabled, source, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 1, 'test', datetime('now'), datetime('now'))
            """,
            (
                "time.current",
                "plan_emit",
                "[]",
                '{"intent_name":"time.current","kind":"plan_emit","tools":[],"plan":{"plan_type":"QUERY_STATUS","payload":{"include":["gaps_summary"]}}}',
            ),
        )
    ability = next(
        item for item in load_json_abilities(db_path=str(db_path)) if item.intent_name == "time.current"
    )
    tools = build_default_tool_registry()
    result = ability.execute(
        {
            "channel_type": "webui",
            "channel_target": "webui",
        },
        tools,
    )
    plans = result.get("plans") or []
    assert plans
    assert plans[0].get("plan_type") == PlanType.QUERY_STATUS.value
