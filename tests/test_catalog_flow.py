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
        return self._payload


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    reset_catalog_service()


def test_timed_signals_create_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    llm = FakeLLM(
        """
        {
          "language": "en",
          "social": {"is_greeting": false, "is_farewell": false, "is_thanks": false, "text": null},
          "actions": [
            {
              "verb": "remind",
              "object": "drink water",
              "target": null,
              "details": null,
              "priority": "normal"
            }
          ],
          "entities": ["drink water"],
          "constraints": {
            "times": ["fifteen minutes"],
            "numbers": [],
            "locations": []
          },
          "questions": [],
          "commands": [],
          "raw_intent_hint": "single_action",
          "confidence": "high"
        }
        """
    )
    state = {
        "chat_id": "123",
        "channel_type": "telegram",
        "channel_target": "123",
        "timezone": "UTC",
        "locale": "en-US",
    }
    result = invoke_cortex(
        state,
        "Remind me to drink water in fifteen minutes",
        llm_client=llm,
    )
    assert result.meta.get("intent") == "unknown"
    assert result.reply_text
    assert not result.plans


def test_timed_signals_list_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    llm = FakeLLM(
        """
        {
          "language": "es",
          "social": {"is_greeting": false, "is_farewell": false, "is_thanks": false, "text": null},
          "actions": [],
          "entities": ["recordatorios"],
          "constraints": {"times": [], "numbers": [], "locations": []},
          "questions": ["Qué recordatorios pendientes tengo?"],
          "commands": [],
          "raw_intent_hint": "question_only",
          "confidence": "high"
        }
        """
    )
    state = {
        "chat_id": "123",
        "channel_type": "telegram",
        "channel_target": "123",
        "timezone": "UTC",
        "locale": "es-MX",
    }
    result = invoke_cortex(state, "Qué recordatorios pendientes tengo?", llm_client=llm)
    assert result.meta.get("intent") == "unknown"
    assert result.reply_text
    assert not result.plans


def test_geo_stub_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    llm = FakeLLM(
        """
        {
          "language": "es",
          "social": {"is_greeting": false, "is_farewell": false, "is_thanks": false, "text": null},
          "actions": [
            {
              "verb": "recuérdame",
              "object": "cerrar la puerta",
              "target": null,
              "details": null,
              "priority": "normal"
            }
          ],
          "entities": [],
          "constraints": {"times": [], "numbers": [], "locations": ["al llegar a casa"]},
          "questions": [],
          "commands": [],
          "raw_intent_hint": "single_action",
          "confidence": "medium"
        }
        """
    )
    state = {
        "chat_id": "123",
        "channel_type": "telegram",
        "channel_target": "123",
        "timezone": "UTC",
        "locale": "es-MX",
    }
    result = invoke_cortex(
        state,
        "Recuérdame al llegar a casa cerrar la puerta",
        llm_client=llm,
    )
    assert result.reply_text
    assert not result.plans


def test_mixed_language_acuerdame_routes_to_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    llm = FakeLLM(
        """
        {
          "language": "mixed",
          "social": {"is_greeting": false, "is_farewell": false, "is_thanks": false, "text": null},
          "actions": [
            {
              "verb": "acuérdame",
              "object": "bajar por agua",
              "target": null,
              "details": null,
              "priority": "normal"
            }
          ],
          "entities": ["bajar por agua"],
          "constraints": {"times": ["en 2 min"], "numbers": ["2"], "locations": []},
          "questions": [],
          "commands": [],
          "raw_intent_hint": "single_action",
          "confidence": "high"
        }
        """
    )
    state = {
        "chat_id": "123",
        "channel_type": "telegram",
        "channel_target": "123",
        "timezone": "America/Mexico_City",
        "locale": "es-MX",
    }
    result = invoke_cortex(
        state,
        "please acuérdame de bajar por agua en 2min",
        llm_client=llm,
    )
    assert result.meta.get("intent") == "unknown"
    assert result.reply_text
    assert not result.plans


def test_multi_action_remind_with_pronoun_object_uses_details_as_task(
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
              "object": "me",
              "target": null,
              "details": "to go down to get water",
              "priority": "high"
            },
            {
              "verb": "go",
              "object": "water",
              "target": null,
              "details": "in 2 min",
              "priority": "normal"
            }
          ],
          "entities": ["water"],
          "constraints": {"times": ["2 min"], "numbers": [], "locations": [], "questions": [], "commands": []},
          "raw_intent_hint": "multi_action",
          "confidence": "high"
        }
        """
    )
    state = {
        "chat_id": "123",
        "channel_type": "telegram",
        "channel_target": "123",
        "timezone": "UTC",
        "locale": "en-US",
    }
    result = invoke_cortex(
        state,
        "Remind me to go down to get water in 2 min",
        llm_client=llm,
    )
    assert result.meta.get("intent") == "unknown"
    assert result.reply_text
    assert not result.plans
