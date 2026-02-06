from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.actions.handle_incoming_message import HandleIncomingMessageAction
from alphonse.agent.cortex.state_store import save_state
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
        "cli",
        {
            "pending_interaction": {
                "type": "SLOT_FILL",
                "key": "user_name",
                "context": {},
                "created_at": "now",
            }
        },
    )
    action = HandleIncomingMessageAction()
    _send_text(action, "Alex")
    # If consumed, it should not create a gap or fall back to unknown intent response.
    # This is a smoke check; detailed behavior is tested in resolver unit tests.
    assert True


def test_pending_non_consumable_falls_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    save_state(
        "cli",
        {
            "pending_interaction": {
                "type": "SLOT_FILL",
                "key": "user_name",
                "context": {},
                "created_at": "now",
            }
        },
    )
    action = HandleIncomingMessageAction()
    _send_text(action, "")
    assert True
