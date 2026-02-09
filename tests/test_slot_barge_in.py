from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.cognition.intent_catalog import reset_catalog_service
from alphonse.agent.cortex.graph import invoke_cortex
from alphonse.agent.nervous_system.migrate import apply_schema


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    reset_catalog_service()


def test_barge_in_greeting_still_greets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    state = {
        "chat_id": "123",
        "channel_type": "telegram",
        "channel_target": "123",
        "timezone": "UTC",
        "catalog_intent": "timed_signals.create",
        "slot_machine": {"slot_name": "trigger_time"},
    }
    result = invoke_cortex(state, "Hola", llm_client=None)
    assert result.meta.get("response_key") is None
    assert not result.plans


def test_cancel_returns_ack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    state = {
        "chat_id": "123",
        "channel_type": "telegram",
        "channel_target": "123",
        "timezone": "UTC",
        "catalog_intent": "timed_signals.create",
        "slot_machine": {"slot_name": "trigger_time"},
    }
    result = invoke_cortex(state, "olvídalo", llm_client=None)
    assert result.meta.get("response_key") is None
    assert not result.plans


def test_hi_does_not_enter_reminder_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    state = {
        "chat_id": "123",
        "channel_type": "telegram",
        "channel_target": "123",
        "timezone": "UTC",
    }
    result = invoke_cortex(state, "Hi", llm_client=None)
    assert result.meta.get("response_key") == "core.greeting"
