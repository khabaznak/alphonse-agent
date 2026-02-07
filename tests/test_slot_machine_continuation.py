from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.cognition.intent_catalog import reset_catalog_service
from alphonse.agent.cognition.plans import PlanType
from alphonse.agent.cortex.graph import invoke_cortex
from alphonse.agent.nervous_system.migrate import apply_schema


class FakeLLM:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        return self._payload


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    reset_catalog_service()


def _base_state() -> dict[str, str]:
    return {
        "chat_id": "8553589429",
        "channel_type": "telegram",
        "channel_target": "8553589429",
        "timezone": "America/Mexico_City",
        "locale": "es-MX",
    }


def _next_state(previous: dict[str, object]) -> dict[str, object]:
    state: dict[str, object] = {**_base_state()}
    state["slots"] = previous.get("slots_collected") or {}
    state["slot_machine"] = previous.get("slot_machine")
    state["catalog_intent"] = previous.get("catalog_intent")
    state["intent"] = previous.get("last_intent")
    state["intent_category"] = previous.get("intent_category")
    state["routing_rationale"] = previous.get("routing_rationale")
    state["routing_needs_clarification"] = previous.get("routing_needs_clarification")
    state["locale"] = previous.get("locale") or _base_state()["locale"]
    return state


def test_create_asks_trigger_then_time_answer_schedules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    first_llm = FakeLLM(
        """
        {
          "language": "es",
          "social": {"is_greeting": false, "is_farewell": false, "is_thanks": false, "text": null},
          "actions": [
            {
              "verb": "recuérdame",
              "object": "bajar por agua",
              "target": null,
              "details": null,
              "priority": "normal"
            }
          ],
          "entities": ["bajar por agua"],
          "constraints": {"times": [], "numbers": [], "locations": []},
          "questions": [],
          "commands": [],
          "raw_intent_hint": "single_action",
          "confidence": "high"
        }
        """
    )
    first = invoke_cortex(
        _base_state(),
        "recuérdame bajar por agua",
        llm_client=first_llm,
    )
    first_state = _next_state(first.cognition_state)
    assert first_state.get("slot_machine")
    assert first.cognition_state.get("slots_collected", {}).get("reminder_text")

    second_llm = FakeLLM(
        """
        {
          "language": "es",
          "social": {"is_greeting": false, "is_farewell": false, "is_thanks": false, "text": null},
          "actions": [],
          "entities": [],
          "constraints": {"times": ["en 5 min"], "numbers": ["5"], "locations": []},
          "questions": [],
          "commands": [],
          "raw_intent_hint": "other",
          "confidence": "medium"
        }
        """
    )
    second = invoke_cortex(
        first_state,
        "en 5 min",
        llm_client=second_llm,
    )
    assert second.plans
    assert second.plans[0].plan_type == PlanType.SCHEDULE_TIMED_SIGNAL
    assert second.cognition_state.get("slot_machine") is None


def test_create_asks_reminder_then_text_answer_schedules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    first_llm = FakeLLM(
        """
        {
          "language": "es",
          "social": {"is_greeting": false, "is_farewell": false, "is_thanks": false, "text": null},
          "actions": [
            {
              "verb": "recuérdame",
              "object": null,
              "target": null,
              "details": null,
              "priority": "normal"
            }
          ],
          "entities": [],
          "constraints": {"times": ["en 5 min"], "numbers": ["5"], "locations": []},
          "questions": [],
          "commands": [],
          "raw_intent_hint": "single_action",
          "confidence": "high"
        }
        """
    )
    first = invoke_cortex(
        _base_state(),
        "recuérdame en 5 min",
        llm_client=first_llm,
    )
    first_state = _next_state(first.cognition_state)
    machine = first_state.get("slot_machine") or {}
    assert machine.get("slot_name") == "reminder_text"
    assert first.cognition_state.get("slots_collected", {}).get("trigger_time")

    second_llm = FakeLLM(
        """
        {
          "language": "es",
          "social": {"is_greeting": false, "is_farewell": false, "is_thanks": false, "text": null},
          "actions": [
            {
              "verb": "bajar",
              "object": "por agua",
              "target": null,
              "details": null,
              "priority": "normal"
            }
          ],
          "entities": ["bajar por agua"],
          "constraints": {"times": [], "numbers": [], "locations": []},
          "questions": [],
          "commands": [],
          "raw_intent_hint": "single_action",
          "confidence": "medium"
        }
        """
    )
    second = invoke_cortex(
        first_state,
        "bajar por agua",
        llm_client=second_llm,
    )
    assert second.plans
    assert second.plans[0].plan_type == PlanType.SCHEDULE_TIMED_SIGNAL
    assert second.cognition_state.get("slot_machine") is None


def test_stale_trigger_time_not_reused_for_new_create_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    llm = FakeLLM(
        """
        {
          "language": "en",
          "social": {"is_greeting": false, "is_farewell": false, "is_thanks": false, "text": null},
          "actions": [
            {
              "verb": "remind",
              "object": "go to the kitchen",
              "target": null,
              "details": null,
              "priority": "normal"
            }
          ],
          "entities": ["go to the kitchen"],
          "constraints": {"times": [], "numbers": [], "locations": []},
          "questions": [],
          "commands": [],
          "raw_intent_hint": "single_action",
          "confidence": "high"
        }
        """
    )
    state = _base_state()
    state["slots"] = {
        "reminder_text": "old reminder",
        "trigger_time": {
            "kind": "trigger_at",
            "trigger_at": "2026-02-06T22:12:18.137791-06:00",
        },
    }
    state["catalog_intent"] = "timed_signals.create"
    state["intent"] = "timed_signals.create"
    state["intent_category"] = "task_plane"
    result = invoke_cortex(state, "remind me to go to the kitchen", llm_client=llm)
    assert not result.plans
    machine = result.cognition_state.get("slot_machine") or {}
    assert machine.get("slot_name") == "trigger_time"
    slots = result.cognition_state.get("slots_collected") or {}
    assert "reminder_text" in slots
    assert "trigger_time" not in slots
