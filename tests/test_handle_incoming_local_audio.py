from __future__ import annotations

from pathlib import Path

from alphonse.agent.actions import handle_incoming_message as him
from alphonse.agent.actions.handle_incoming_message import HandleIncomingMessageAction
from alphonse.agent.cognition.plans import CortexPlan
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.senses.bus import Signal


class _FakeCortexResult:
    def __init__(self, reply_text: str) -> None:
        self.reply_text = reply_text
        self.plans = []
        self.cognition_state = {}
        self.meta = {}


def test_handle_incoming_triggers_local_audio_when_requested(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    class _FakeGraph:
        def invoke(self, state: dict, text: str, *, llm_client=None):  # noqa: ANN001
            _ = (state, text, llm_client)
            return _FakeCortexResult("Estoy listo para ayudarte.")

    class _FakePlanExecutor:
        def execute(self, plans, context, exec_context) -> None:  # noqa: ANN001
            _ = (plans, context, exec_context)
            return None

    captured: dict[str, object] = {}

    class _FakeLocalAudioTool:
        def execute(self, *, text: str, voice: str = "default", blocking: bool = False, volume=None):  # noqa: ANN001
            captured["text"] = text
            captured["voice"] = voice
            captured["blocking"] = blocking
            return {"status": "ok"}

    monkeypatch.setattr(him, "_CORTEX_GRAPH", _FakeGraph())
    monkeypatch.setattr(him, "PlanExecutor", lambda: _FakePlanExecutor())
    monkeypatch.setattr(him, "build_llm_client", lambda: None)
    monkeypatch.setattr(him, "LocalAudioOutputSpeakTool", lambda: _FakeLocalAudioTool())

    action = HandleIncomingMessageAction()
    signal = Signal(
        type="api.message_received",
        payload={
            "text": "Estado?",
            "channel": "webui",
            "target": "webui",
            "controls": {"audio_mode": "local_audio"},
        },
        source="api",
    )
    action.execute({"signal": signal, "state": None, "outcome": None, "ctx": None})

    assert captured["text"] == "Estoy listo para ayudarte."
    assert captured["blocking"] is False


def test_handle_incoming_persists_tts_transcript_when_reply_text_is_empty(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    class _FakeCortexResult:
        def __init__(self) -> None:
            self.reply_text = ""
            self.plans = [CortexPlan(tool="local_audio_output.speak", parameters={"text": "Te lo mando por voz."})]
            self.cognition_state = {}
            self.meta = {}

    class _FakeGraph:
        def invoke(self, state: dict, text: str, *, llm_client=None):  # noqa: ANN001
            _ = (state, text, llm_client)
            return _FakeCortexResult()

    class _FakePlanExecutor:
        def execute(self, plans, context, exec_context) -> None:  # noqa: ANN001
            _ = (plans, context, exec_context)
            return None

    captured: dict[str, str] = {}

    def _capture_session_update(*, assistant_message: str, **kwargs):  # noqa: ANN003
        captured["assistant_message"] = assistant_message
        previous = kwargs.get("previous")
        return previous if isinstance(previous, dict) else {}

    monkeypatch.setattr(him, "_CORTEX_GRAPH", _FakeGraph())
    monkeypatch.setattr(him, "PlanExecutor", lambda: _FakePlanExecutor())
    monkeypatch.setattr(him, "build_llm_client", lambda: None)
    monkeypatch.setattr(him, "build_next_session_state", _capture_session_update)
    monkeypatch.setattr(him, "commit_session_state", lambda *_args, **_kwargs: None)

    action = HandleIncomingMessageAction()
    signal = Signal(
        type="api.message_received",
        payload={"text": "mándamelo en audio", "channel": "webui", "target": "webui"},
        source="api",
    )
    action.execute({"signal": signal, "state": None, "outcome": None, "ctx": None})

    assert captured["assistant_message"] == "[TTS transcript] Te lo mando por voz."
