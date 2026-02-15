from __future__ import annotations

from pathlib import Path

from alphonse.agent.actions import handle_incoming_message as him
from alphonse.agent.actions.handle_incoming_message import HandleIncomingMessageAction
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.senses.bus import Signal


class _FakeCortexResult:
    def __init__(self) -> None:
        self.reply_text = None
        self.plans = []
        self.cognition_state = {}
        self.meta = {}


def test_handle_incoming_packs_provider_event_raw_json(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    captured: dict[str, str] = {}

    class _FakeGraph:
        def invoke(self, state: dict, text: str, *, llm_client=None):
            _ = (state, llm_client)
            captured["text"] = text
            return _FakeCortexResult()

    class _FakePlanExecutor:
        def execute(self, plans, context, exec_context) -> None:
            _ = (plans, context, exec_context)
            return None

    monkeypatch.setattr(him, "_CORTEX_GRAPH", _FakeGraph())
    monkeypatch.setattr(him, "PlanExecutor", lambda: _FakePlanExecutor())
    monkeypatch.setattr(him, "build_llm_client", lambda: None)

    action = HandleIncomingMessageAction()
    provider_event = {
        "update_id": 123,
        "message": {
            "message_id": 99,
            "chat": {"id": 8553589429},
            "voice": {"file_id": "voice-file-1", "duration": 4},
            "caption": "voice note",
        },
    }
    signal = Signal(
        type="telegram.message_received",
        payload={
            "text": "",
            "channel": "telegram",
            "chat_id": "8553589429",
            "provider_event": provider_event,
        },
        source="telegram",
    )
    action.execute({"signal": signal, "state": None, "outcome": None, "ctx": None})
    packed = captured.get("text", "")
    assert "Incoming Provider Event" in packed
    assert '"update_id": 123' in packed
    assert '"voice"' in packed
    assert '"file_id": "voice-file-1"' in packed
