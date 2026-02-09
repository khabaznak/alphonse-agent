from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.cognition.intent_catalog import reset_catalog_service
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


def test_reminder_request_now_asks_for_clarification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    llm = FakeLLM(
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
    result = invoke_cortex(
        _base_state(),
        "recuérdame bajar por agua",
        llm_client=llm,
    )
    assert result.meta.get("intent") == "unknown"
    assert result.reply_text
    assert not result.plans


def test_follow_up_does_not_schedule_without_ability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    llm = FakeLLM(
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
    result = invoke_cortex(
        _base_state(),
        "en 5 min",
        llm_client=llm,
    )
    assert result.meta.get("intent") == "unknown"
    assert result.reply_text
    assert not result.plans
