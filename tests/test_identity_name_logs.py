from __future__ import annotations

import logging
from pathlib import Path

import pytest

from alphonse.agent.actions import handle_incoming_message as him
from alphonse.agent.actions.handle_incoming_message import HandleIncomingMessageAction
from alphonse.agent.cognition.plans import PlanType
from alphonse.agent.cortex.state_store import save_state
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.senses.bus import Signal


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)


def _send_text(
    action: HandleIncomingMessageAction, text: str, *, chat_id: str = "123"
) -> None:
    signal = Signal(
        type="telegram.message_received",
        payload={"text": text, "origin": "telegram", "chat_id": chat_id},
        source="telegram",
    )
    action.execute({"signal": signal, "state": None, "outcome": None, "ctx": None})


def _find_index(messages: list[str], needle: str) -> int:
    for idx, message in enumerate(messages):
        if needle in message:
            return idx
    return -1


def test_pending_and_identity_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _prepare_db(tmp_path, monkeypatch)

    class FakePlanExecutor:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def execute(self, plans, context, exec_context) -> None:
            for plan in plans:
                if plan.plan_type == PlanType.COMMUNICATE:
                    return None

    monkeypatch.setattr(him, "PlanExecutor", FakePlanExecutor)
    monkeypatch.setattr(him, "_build_llm_client", lambda: None)

    caplog.set_level(logging.INFO)
    action = HandleIncomingMessageAction()

    save_state(
        "telegram:123",
        {
            "pending_interaction": {
                "type": "SLOT_FILL",
                "key": "user_name",
                "context": {"ability_intent": "core.identity.query_user_name"},
                "created_at": "now",
            }
        },
    )
    _send_text(action, "Alex")

    messages = [record.getMessage() for record in caplog.records]
    set_idx = _find_index(messages, "identity display_name set key=telegram:123")
    get_idx = _find_index(messages, "identity display_name get key=telegram:123")

    assert set_idx != -1
    assert get_idx != -1
