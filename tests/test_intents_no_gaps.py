from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.actions.handle_incoming_message import HandleIncomingMessageAction
from alphonse.agent.nervous_system.capability_gaps import list_gaps
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.senses.bus import Signal


def _send_text(action: HandleIncomingMessageAction, text: str) -> None:
    signal = Signal(
        type="cli.message_received",
        payload={"text": text, "origin": "cli", "chat_id": "cli"},
        source="cli",
    )
    action.execute({"signal": signal, "state": None, "outcome": None, "ctx": None})


def test_common_intents_do_not_create_gaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    action = HandleIncomingMessageAction()
    samples = [
        "Hi",
        "Good morning",
        "gaps list",
        "What reminders do you have scheduled?",
        "What else can you do?",
        "Buenos días",
        "¿Qué recordatorios tienes?",
    ]
    for text in samples:
        _send_text(action, text)
        assert list_gaps(limit=10, include_all=False) == []
