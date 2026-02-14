from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.actions.handle_incoming_message import HandleIncomingMessageAction
from alphonse.agent.cortex.state_store import save_state, load_state
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.senses.bus import Signal


def _send_text(action: HandleIncomingMessageAction, text: str) -> None:
    signal = Signal(
        type="cli.message_received",
        payload={"text": text, "origin": "cli", "chat_id": "cli"},
        source="cli",
    )
    action.execute({"signal": signal, "state": None, "outcome": None, "ctx": None})


def test_pending_name_consumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    save_state(
        "telegram:123",
        {
            "pending_interaction": {
                "type": "SLOT_FILL",
                "key": "user_name",
                "context": {"ability_intent": "core.onboarding.start"},
                "created_at": "now",
            }
        },
    )
    action = HandleIncomingMessageAction()
    signal = Signal(
        type="telegram.message_received",
        payload={"text": "Alex", "origin": "telegram", "chat_id": "123"},
        source="telegram",
    )
    action.execute({"signal": signal, "state": None, "outcome": None, "ctx": None})
    state = load_state("telegram:123")
    assert state is not None
    assert state.get("pending_interaction") is None


def test_pending_non_consumable_empty_text_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    save_state(
        "telegram:123",
        {
            "pending_interaction": {
                "type": "SLOT_FILL",
                "key": "user_name",
                "context": {"ability_intent": "core.onboarding.start"},
                "created_at": "now",
            }
        },
    )
    action = HandleIncomingMessageAction()
    signal = Signal(
        type="telegram.message_received",
        payload={"text": "", "origin": "telegram", "chat_id": "123"},
        source="telegram",
    )
    with pytest.raises(ValueError, match="normalized incoming text must be non-empty"):
        action.execute({"signal": signal, "state": None, "outcome": None, "ctx": None})
    state = load_state("telegram:123")
    assert state is not None
    assert state.get("pending_interaction") is not None
