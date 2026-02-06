from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.cognition.slots.slot_fsm import create_machine, serialize_machine
from alphonse.agent.cortex.graph import invoke_cortex
from alphonse.agent.nervous_system.migrate import apply_schema


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)


def test_barge_in_greeting_pauses_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    machine = create_machine(
        "trigger_time",
        "time_expression",
        {"timezone": "UTC"},
    )
    state = {
        "chat_id": "123",
        "channel_type": "telegram",
        "channel_target": "123",
        "timezone": "UTC",
        "catalog_intent": "timed_signals.create",
        "slot_machine": serialize_machine(machine),
    }
    result = invoke_cortex(state, "Hola", llm_client=None)
    assert result.meta.get("response_key") == "core.greeting"
    paused = result.cognition_state.get("slot_machine") or {}
    assert paused.get("paused_at") is not None


def test_cancel_clears_machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    machine = create_machine(
        "trigger_time",
        "time_expression",
        {"timezone": "UTC"},
    )
    state = {
        "chat_id": "123",
        "channel_type": "telegram",
        "channel_target": "123",
        "timezone": "UTC",
        "catalog_intent": "timed_signals.create",
        "slot_machine": serialize_machine(machine),
    }
    result = invoke_cortex(state, "olvídalo", llm_client=None)
    assert result.meta.get("response_key") == "ack.cancelled"
    assert result.cognition_state.get("slot_machine") is None


def test_reminder_text_guard_rejects_greeting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    machine = create_machine(
        "reminder_text",
        "string",
        {"timezone": "UTC"},
    )
    state = {
        "chat_id": "123",
        "channel_type": "telegram",
        "channel_target": "123",
        "timezone": "UTC",
        "catalog_intent": "timed_signals.create",
        "slot_machine": serialize_machine(machine),
    }
    result = invoke_cortex(state, "Hola", llm_client=None)
    assert result.meta.get("response_key") == "core.greeting"
    paused = result.cognition_state.get("slot_machine") or {}
    assert paused.get("paused_at") is not None


def test_resume_flow_asks_pending_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    machine = create_machine(
        "trigger_time",
        "time_expression",
        {"timezone": "UTC"},
    )
    machine.paused_at = "2026-02-06T12:00:00Z"
    state = {
        "chat_id": "123",
        "channel_type": "telegram",
        "channel_target": "123",
        "timezone": "UTC",
        "catalog_intent": "timed_signals.create",
        "slot_machine": serialize_machine(machine),
    }
    result = invoke_cortex(state, "continuar", llm_client=None)
    assert result.meta.get("response_key") == "clarify.trigger_time"


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
