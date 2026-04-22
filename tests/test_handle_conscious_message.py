from __future__ import annotations

from alphonse.agent.actions.conscious_message_handler import IncomingMessageEnvelope
from alphonse.agent.actions.conscious_message_handler import build_incoming_message_envelope
from alphonse.agent.actions.handle_conscious_message import HandleConsciousMessageAction
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.agent.services.pdca_ingress import BufferedTaskInput


def test_from_payload_resolves_actor_user_id_from_external_user_id(monkeypatch) -> None:
    monkeypatch.setattr(
        "alphonse.agent.actions.conscious_message_handler.identity.resolve_service_id",
        lambda service_key: 1 if str(service_key or "").strip() == "telegram" else None,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.conscious_message_handler.identity.resolve_user_id",
        lambda **kwargs: "owner-1" if kwargs == {"service_id": 1, "service_user_id": "u-ext"} else None,
    )

    raw_payload = build_incoming_message_envelope(
        message_id="m-env-1",
        channel_type="telegram",
        channel_target="123",
        provider="telegram",
        text="Hello",
        correlation_id="c-env-1",
        actor_external_user_id="u-ext",
    )
    envelope = IncomingMessageEnvelope.from_payload(raw_payload)

    assert envelope.actor == {
        "external_user_id": "u-ext",
        "display_name": None,
        "user_id": "owner-1",
    }
    runtime_payload = envelope.runtime_payload()
    assert runtime_payload["user_id"] == "owner-1"
    assert runtime_payload["external_user_id"] == "u-ext"


def test_from_payload_preserves_actor_user_id() -> None:
    raw_payload = build_incoming_message_envelope(
        message_id="m-env-2",
        channel_type="telegram",
        channel_target="123",
        provider="telegram",
        text="Hello",
        correlation_id="c-env-2",
        actor_external_user_id="u-ext",
        actor_user_id="owner-1",
    )

    envelope = IncomingMessageEnvelope.from_payload(raw_payload)

    assert envelope.actor == {
        "external_user_id": "u-ext",
        "display_name": None,
        "user_id": "owner-1",
    }


def test_from_payload_returns_stable_actor_keys_when_missing() -> None:
    raw_payload = build_incoming_message_envelope(
        message_id="m-env-3",
        channel_type="telegram",
        channel_target="123",
        provider="telegram",
        text="Hello",
        correlation_id="c-env-3",
    )

    envelope = IncomingMessageEnvelope.from_payload(raw_payload)

    assert envelope.actor == {
        "external_user_id": None,
        "display_name": None,
        "user_id": None,
    }


def test_handle_conscious_message_enqueues_pdca_slice(monkeypatch) -> None:
    action = HandleConsciousMessageAction()
    called: dict[str, object] = {}
    presence: dict[str, object] = {}

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
        actor_user_id="u-1",
    )
    result = action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=envelope), "ctx": Bus()})
    assert result.intention_key == "NOOP"
    assert result.payload.get("task_id") == "task-123"
    assert set(called) == {"task_record", "buffered_input", "bus", "force_new_task"}
    task_record = called.get("task_record")
    assert isinstance(task_record, TaskRecord)
    buffered_input = called.get("buffered_input")
    assert isinstance(buffered_input, BufferedTaskInput)
    assert task_record.user_id == "u-1"
    assert task_record.correlation_id == "c-1"
    assert task_record.goal == "Hello"
    assert buffered_input.text == "Hello"
    assert buffered_input.message_id == "m-1"
    assert buffered_input.channel_type == "telegram"
    assert buffered_input.channel_target == "123"
    assert presence.get("phase") == "acknowledged"
    assert presence.get("channel_type") == "telegram"
    assert presence.get("channel_target") == "123"
    assert presence.get("message_id") == "m-1"


def test_handle_conscious_message_does_not_parse_control_intent(monkeypatch) -> None:
    action = HandleConsciousMessageAction()
    captured: dict[str, object] = {}

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
    task_record = captured.get("task_record")
    assert isinstance(task_record, TaskRecord)
    assert task_record.goal == "Please cancel task"
    assert isinstance(captured.get("buffered_input"), BufferedTaskInput)


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


def test_handle_conscious_message_uses_signal_correlation_fallback(monkeypatch) -> None:
    action = HandleConsciousMessageAction()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.enqueue_pdca_slice",
        lambda **kwargs: captured.update(kwargs) or "task-correlation",
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
        message_id="m-correlation",
        channel_type="telegram",
        channel_target="123",
        provider="telegram",
        text="Hello",
        correlation_id=None,
        actor_external_user_id="u-ext",
    )
    result = action.execute(
        {
            "signal": Signal(
                type="sense.telegram.message.user.received",
                payload=envelope,
                correlation_id="signal-correlation",
            ),
            "ctx": Bus(),
        }
    )
    assert result.payload.get("task_id") == "task-correlation"
    task_record = captured.get("task_record")
    assert isinstance(task_record, TaskRecord)
    assert task_record.correlation_id == "signal-correlation"
    buffered_input = captured.get("buffered_input")
    assert isinstance(buffered_input, BufferedTaskInput)
    assert buffered_input.correlation_id == "signal-correlation"


def test_handle_conscious_message_preserves_attachment_only_payload(monkeypatch) -> None:
    action = HandleConsciousMessageAction()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.enqueue_pdca_slice",
        lambda **kwargs: captured.update(kwargs) or "task-attachment-only",
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
        message_id="m-attach-only",
        channel_type="telegram",
        channel_target="123",
        provider="telegram",
        text="",
        attachments=[{"kind": "voice", "provider": "telegram", "file_id": "voice-1"}],
        correlation_id="cid-attach-only",
    )
    result = action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=envelope), "ctx": Bus()})
    assert result.payload.get("task_id") == "task-attachment-only"
    task_record = captured.get("task_record")
    assert isinstance(task_record, TaskRecord)
    assert task_record.goal.startswith("[attachments:")
    buffered_input = captured.get("buffered_input")
    assert isinstance(buffered_input, BufferedTaskInput)
    assert buffered_input.text.startswith("[attachments:")
