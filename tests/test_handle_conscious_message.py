from __future__ import annotations

from alphonse.agent.actions.conscious_message_handler import build_incoming_message_envelope
from alphonse.agent.actions.handle_conscious_message import HandleConsciousMessageAction
from alphonse.agent.nervous_system.senses.bus import Bus, Signal


def test_handle_conscious_message_enqueues_pdca_slice(monkeypatch) -> None:
    action = HandleConsciousMessageAction()
    called: dict[str, object] = {}
    presence: dict[str, object] = {}

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
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.is_pdca_slicing_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.emit_presence_phase_changed",
        lambda **kwargs: presence.update(kwargs),
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
    result = action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=envelope), "ctx": Bus()})
    assert result.intention_key == "NOOP"
    assert result.payload.get("task_id") == "task-123"
    assert called.get("session_user_id") == "u-1"
    assert presence.get("phase") == "acknowledged"


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
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.is_pdca_slicing_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.emit_presence_phase_changed",
        lambda **_: None,
    )

    envelope = build_incoming_message_envelope(
        message_id="m-1",
        channel_type="telegram",
        channel_target="123",
        provider="telegram",
        text="Please cancel task",
        correlation_id="c-1",
    )
    result = action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=envelope), "ctx": Bus()})
    assert result.payload.get("task_id") == "task-cancel-test"
    assert captured.get("payload", {}).get("text") == "Please cancel task"


def test_handle_conscious_message_fail_fast_when_slicing_disabled(monkeypatch) -> None:
    action = HandleConsciousMessageAction()
    called: dict[str, object] = {}

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.is_pdca_slicing_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.emit_presence_phase_changed",
        lambda **_: None,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.enqueue_pdca_slice",
        lambda **kwargs: called.update(kwargs) or "unexpected-task",
    )

    envelope = build_incoming_message_envelope(
        message_id="m-2",
        channel_type="telegram",
        channel_target="123",
        provider="telegram",
        text="Hello",
        correlation_id="c-2",
    )
    result = action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=envelope), "ctx": Bus()})
    assert result.intention_key == "MESSAGE_READY"
    assert "temporarily unable to process messages" in str(result.payload.get("message") or "")
    assert called == {}
