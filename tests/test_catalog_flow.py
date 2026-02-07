from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from alphonse.agent.cognition.intent_catalog import reset_catalog_service
from alphonse.agent.cortex.graph import invoke_cortex
from alphonse.agent.cognition.plans import PlanType
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
          "intent_name": "timed_signals.create",
          "confidence": 0.9,
          "slot_guesses": {
            "reminder_text": "drink water",
            "trigger_time": "fifteen minutes"
          },
          "needs_clarification": false
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
    assert result.plans
    plan = result.plans[0]
    assert plan.plan_type == PlanType.SCHEDULE_TIMED_SIGNAL
    trigger_at = plan.payload.get("trigger_at")
    assert trigger_at
    now = datetime.now(tz=timezone.utc)
    scheduled = datetime.fromisoformat(trigger_at)
    assert scheduled > now


def test_timed_signals_list_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    llm = FakeLLM(
        """
        {
          "intent_name": "timed_signals.list",
          "confidence": 0.8,
          "slot_guesses": {},
          "needs_clarification": false
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
    assert result.plans
    assert result.plans[0].plan_type == PlanType.QUERY_STATUS


def test_geo_stub_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    llm = FakeLLM(
        """
        {
          "intent_name": "timed_signals.create",
          "confidence": 0.8,
          "slot_guesses": {
            "reminder_text": "cerrar la puerta",
            "trigger_geo": "al llegar a casa"
          },
          "needs_clarification": false
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
    assert "ubicación" in result.reply_text.lower() or "location" in result.reply_text.lower()
    assert not result.plans
