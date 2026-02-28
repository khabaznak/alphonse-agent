from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.actions import handle_incoming_message as him
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


class _FakeCortexResult:
    def __init__(self, cognition_state: dict) -> None:
        self.reply_text = None
        self.plans = []
        self.cognition_state = cognition_state
        self.meta = {}


class _FakeGraph:
    def invoke(self, state: dict, _packed_input: str, *, llm_client=None):
        _ = llm_client
        cognition_state = {
            "pending_interaction": state.get("pending_interaction"),
            "ability_state": state.get("ability_state"),
            "task_state": state.get("task_state"),
            "planning_context": state.get("planning_context"),
            "locale": state.get("locale"),
        }
        return _FakeCortexResult(cognition_state)


class _FakePlanExecutor:
    def execute(self, plans, context, exec_context) -> None:
        _ = (plans, context, exec_context)
        return None


def test_pending_name_raw_event_keeps_pending_interaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(him, "_CORTEX_GRAPH", _FakeGraph())
    monkeypatch.setattr(him, "PlanExecutor", lambda: _FakePlanExecutor())
    monkeypatch.setattr(him, "build_llm_client", lambda: None)
    action = HandleIncomingMessageAction()
    signal = Signal(
        type="telegram.message_received",
        payload={"text": "Alex", "origin": "telegram", "chat_id": "123"},
        source="telegram",
    )
    action.execute({"signal": signal, "state": None, "outcome": None, "ctx": None})
    state = load_state("telegram:123")
    assert state is not None
    assert state.get("pending_interaction") is not None


def test_pending_non_consumable_empty_text_does_not_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(him, "_CORTEX_GRAPH", _FakeGraph())
    monkeypatch.setattr(him, "PlanExecutor", lambda: _FakePlanExecutor())
    monkeypatch.setattr(him, "build_llm_client", lambda: None)
    action = HandleIncomingMessageAction()
    signal = Signal(
        type="telegram.message_received",
        payload={"text": "", "origin": "telegram", "chat_id": "123"},
        source="telegram",
    )
    action.execute({"signal": signal, "state": None, "outcome": None, "ctx": None})
    state = load_state("telegram:123")
    assert state is not None
