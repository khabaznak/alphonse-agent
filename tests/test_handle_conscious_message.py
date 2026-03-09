from __future__ import annotations

from alphonse.agent.actions.conscious_message_handler import build_incoming_message_envelope
from alphonse.agent.actions.handle_conscious_message import HandleConsciousMessageAction
from alphonse.agent.nervous_system.senses.bus import Bus, Signal


def test_handle_conscious_message_enqueues_pdca_slice(monkeypatch) -> None:
    action = HandleConsciousMessageAction()
    called: dict[str, object] = {}

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_session_user_id",
        lambda **_: "u-1",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_session_timezone",
        lambda *_: "UTC",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_day_session",
        lambda **_: {"session_id": "s-1"},
    )
    def _enqueue(**kwargs):
        called.update(kwargs)
        return "task-123"

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.enqueue_pdca_slice",
        _enqueue,
    )

    envelope = build_incoming_message_envelope(
        message_id="m-1",
        channel_type="telegram",
        channel_target="123",
        provider="telegram",
        text="Hello",
        correlation_id="c-1",
        actor_external_user_id="u-ext",
    )
    result = action.execute({"signal": Signal(type="telegram.message_received", payload=envelope), "ctx": Bus()})
    assert result.intention_key == "NOOP"
    assert result.payload.get("task_id") == "task-123"
    assert called.get("session_user_id") == "u-1"


def test_handle_conscious_message_does_not_parse_control_intent(monkeypatch) -> None:
    action = HandleConsciousMessageAction()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_session_user_id",
        lambda **_: "u-1",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_session_timezone",
        lambda *_: "UTC",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_day_session",
        lambda **_: {"session_id": "s-1"},
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.enqueue_pdca_slice",
        lambda **kwargs: captured.update(kwargs) or "task-cancel-test",
    )

    envelope = build_incoming_message_envelope(
        message_id="m-1",
        channel_type="telegram",
        channel_target="123",
        provider="telegram",
        text="Please cancel task",
        correlation_id="c-1",
    )
    result = action.execute({"signal": Signal(type="telegram.message_received", payload=envelope), "ctx": Bus()})
    assert result.payload.get("task_id") == "task-cancel-test"
    assert captured.get("payload", {}).get("text") == "Please cancel task"
