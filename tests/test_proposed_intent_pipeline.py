from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.cognition.intent_catalog import reset_catalog_service
from alphonse.agent.cognition.plans import PlanType
from alphonse.agent.cortex.graph import invoke_cortex
from alphonse.agent.nervous_system.migrate import apply_schema


class SequenceLLM:
    def __init__(self, payloads: list[str]) -> None:
        self._payloads = list(payloads)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        if not self._payloads:
            return "{}"
        return self._payloads.pop(0)


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    reset_catalog_service()


def _base_state() -> dict[str, str]:
    return {
        "chat_id": "webui",
        "channel_type": "webui",
        "channel_target": "webui",
        "timezone": "America/Mexico_City",
        "locale": "es-MX",
    }


def test_proposed_intent_alias_maps_to_known_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    llm = SequenceLLM(
        [
            """
            {
              "language": "es",
              "social": {"is_greeting": false, "is_farewell": false, "is_thanks": false, "text": null},
              "actions": [],
              "entities": [],
              "constraints": {"times": [], "numbers": [], "locations": []},
              "questions": [],
              "commands": [],
              "raw_intent_hint": "other",
              "confidence": "low"
            }
            """,
            '{"intent":"recordatorios","aliases":["reminder list"],"confidence":0.74}',
        ]
    )
    result = invoke_cortex(_base_state(), "Muéstrame pendientes", llm_client=llm)
    assert result.meta.get("intent") == "timed_signals.list"
    assert result.plans
    assert result.plans[0].plan_type == PlanType.QUERY_STATUS


def test_unmapped_proposed_intent_creates_gap_with_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    llm = SequenceLLM(
        [
            """
            {
              "language": "en",
              "social": {"is_greeting": false, "is_farewell": false, "is_thanks": false, "text": null},
              "actions": [],
              "entities": [],
              "constraints": {"times": [], "numbers": [], "locations": []},
              "questions": [],
              "commands": [],
              "raw_intent_hint": "other",
              "confidence": "low"
            }
            """,
            '{"intent":"time query","aliases":["clock check"],"confidence":0.91}',
            "Can you share your city?",
        ]
    )
    result = invoke_cortex(_base_state(), "Can you tell me the time?", llm_client=llm)
    assert result.meta.get("intent") == "unknown"
    assert result.meta.get("proposed_intent") == "time query"
    assert result.plans
    gap_plan = result.plans[0]
    assert gap_plan.plan_type == PlanType.CAPABILITY_GAP
    assert gap_plan.payload.get("reason") == "proposed_intent_unmapped"
    metadata = gap_plan.payload.get("metadata") or {}
    assert metadata.get("proposed_intent") == "time query"
