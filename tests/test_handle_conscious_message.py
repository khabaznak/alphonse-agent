from __future__ import annotations

import pytest

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

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_user_id_by_service_user_id",
        lambda **kwargs: "u-1" if kwargs == {"service_id": 2, "service_user_id": "u-ext"} else None,
    )

    payload = {
        "contract_type": "canonical_inbound_event",
        "contract_version": "1.0",
        "service_key": "telegram",
        "provider_user_id_from": "u-ext",
        "provider_message_id": "m-1",
        "channel_target": "123",
        "occurred_at": "2026-05-04T12:00:00+00:00",
        "event_kind": "message",
        "provider_raw_message": {"update_id": 1},
        "text": "Hello",
        "attachments": [],
        "dedupe_key": "c-1",
        "display_name": "Alex",
        "metadata": {},
    }
    result = action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=payload), "ctx": Bus()})
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

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_user_id_by_service_user_id",
        lambda **kwargs: "u-1" if kwargs == {"service_id": 2, "service_user_id": "u-ext"} else None,
    )

    payload = {
        "contract_type": "canonical_inbound_event",
        "contract_version": "1.0",
        "service_key": "telegram",
        "provider_user_id_from": "u-ext",
        "provider_message_id": "m-1",
        "channel_target": "123",
        "occurred_at": "2026-05-04T12:00:00+00:00",
        "event_kind": "message",
        "provider_raw_message": {"update_id": 1},
        "text": "Please cancel task",
        "attachments": [],
    }
    result = action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=payload), "ctx": Bus()})
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

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_user_id_by_service_user_id",
        lambda **kwargs: "u-1" if kwargs == {"service_id": 2, "service_user_id": "u-ext"} else None,
    )

    payload = {
        "contract_type": "canonical_inbound_event",
        "contract_version": "1.0",
        "service_key": "telegram",
        "provider_user_id_from": "u-ext",
        "provider_message_id": "m-2",
        "channel_target": "123",
        "occurred_at": "2026-05-04T12:00:00+00:00",
        "event_kind": "message",
        "provider_raw_message": {"update_id": 2},
        "text": "Hello",
        "attachments": [],
        "dedupe_key": "c-2",
    }
    result = action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=payload), "ctx": Bus()})
    assert result.delivers_message is True
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

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_user_id_by_service_user_id",
        lambda **kwargs: "u-1" if kwargs == {"service_id": 2, "service_user_id": "u-ext"} else None,
    )

    payload = {
        "contract_type": "canonical_inbound_event",
        "contract_version": "1.0",
        "service_key": "telegram",
        "provider_user_id_from": "u-ext",
        "provider_message_id": "m-correlation",
        "channel_target": "123",
        "occurred_at": "2026-05-04T12:00:00+00:00",
        "event_kind": "message",
        "provider_raw_message": {"update_id": 3},
        "text": "Hello",
        "attachments": [],
    }
    result = action.execute(
        {
            "signal": Signal(
                type="sense.telegram.message.user.received",
                payload=payload,
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


def test_handle_conscious_message_accepts_canonical_timed_conscious_payload(monkeypatch) -> None:
    action = HandleConsciousMessageAction()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.enqueue_pdca_slice",
        lambda **kwargs: captured.update(kwargs) or "task-timed-1",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.is_pdca_slicing_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.emit_presence_phase_changed",
        lambda **_: None,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_user_id_by_service_user_id",
        lambda **kwargs: "u-1" if kwargs == {"service_id": 1, "service_user_id": "me"} else None,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_service_id_by_channel_type",
        lambda service_key: 1 if str(service_key or "").strip() == "api" else None,
    )

    payload = {
        "contract_type": "canonical_inbound_event",
        "contract_version": "1.0",
        "service_key": "api",
        "provider_user_id_from": "me",
        "provider_message_id": "tsig_1",
        "channel_target": "me",
        "occurred_at": "2026-05-07T12:00:00+00:00",
        "event_kind": "message",
        "provider_raw_message": {"timed_signal": {"timed_signal_id": "tsig_1"}},
        "text": "Take a shower now.",
        "attachments": [],
        "dedupe_key": "corr-timed-1",
        "metadata": {"mind_layer": "conscious"},
    }
    result = action.execute({"signal": Signal(type="timed_signal.conscious_payload", payload=payload), "ctx": Bus()})
    assert result.intention_key == "NOOP"
    assert result.payload.get("task_id") == "task-timed-1"
    task_record = captured.get("task_record")
    assert isinstance(task_record, TaskRecord)
    assert task_record.user_id == "u-1"
    assert task_record.goal == "Take a shower now."
    buffered_input = captured.get("buffered_input")
    assert isinstance(buffered_input, BufferedTaskInput)
    assert buffered_input.channel_type == "api"
    assert buffered_input.channel_target == "me"
    assert buffered_input.text == "Take a shower now."


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

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_user_id_by_service_user_id",
        lambda **kwargs: "u-1" if kwargs == {"service_id": 2, "service_user_id": "u-ext"} else None,
    )

    payload = {
        "contract_type": "canonical_inbound_event",
        "contract_version": "1.0",
        "service_key": "telegram",
        "provider_user_id_from": "u-ext",
        "provider_message_id": "m-attach-only",
        "channel_target": "123",
        "occurred_at": "2026-05-04T12:00:00+00:00",
        "event_kind": "message",
        "provider_raw_message": {"update_id": 4},
        "text": "",
        "attachments": [{"kind": "voice", "provider": "telegram", "file_id": "voice-1"}],
        "dedupe_key": "cid-attach-only",
    }
    result = action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=payload), "ctx": Bus()})
    assert result.payload.get("task_id") == "task-attachment-only"
    task_record = captured.get("task_record")
    assert isinstance(task_record, TaskRecord)
    assert task_record.goal.startswith("[attachments:")
    buffered_input = captured.get("buffered_input")
    assert isinstance(buffered_input, BufferedTaskInput)
    assert buffered_input.text.startswith("[attachments:")


def test_handle_conscious_message_accepts_canonical_telegram_payload(monkeypatch) -> None:
    action = HandleConsciousMessageAction()

    payload = {
        "contract_type": "canonical_inbound_message",
        "contract_version": "1.0",
        "message_id": "m-can-1",
        "correlation_id": "c-can-1",
        "occurred_at": "2026-05-04T12:00:00+00:00",
        "service_id": 2,
        "service_key": "telegram",
        "channel_type": "telegram",
        "channel_target": "123",
        "external_user_id": "u-ext",
        "display_name": "Alex",
        "resolved_user_id": "u-1",
        "text": "Hello canonical",
        "attachments": [],
        "metadata": {"provider_event": {"update_id": 7}},
    }
    with pytest.raises(ValueError, match="unsupported contract_type"):
        action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=payload), "ctx": Bus()})


def test_handle_conscious_message_accepts_mechanical_telegram_payload(monkeypatch) -> None:
    action = HandleConsciousMessageAction()
    captured: dict[str, object] = {}
    presence: dict[str, object] = {}

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.enqueue_pdca_slice",
        lambda **kwargs: captured.update(kwargs) or "task-mechanical",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.is_pdca_slicing_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.emit_presence_phase_changed",
        lambda **kwargs: presence.update(kwargs),
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_user_id_by_service_user_id",
        lambda **kwargs: "u-1" if kwargs == {"service_id": 2, "service_user_id": "u-ext"} else None,
    )

    payload = {
        "contract_type": "canonical_inbound_event",
        "contract_version": "1.0",
        "service_key": "telegram",
        "provider_user_id_from": "u-ext",
        "provider_message_id": "m-evt-1",
        "channel_target": "123",
        "occurred_at": "2026-05-04T12:00:00+00:00",
        "event_kind": "message",
        "provider_raw_message": {"update_id": 7},
        "text": "Hello mechanical",
        "attachments": [],
        "dedupe_key": "d-1",
        "display_name": "Alex",
        "metadata": {},
    }
    result = action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=payload), "ctx": Bus()})

    assert result.payload.get("task_id") == "task-mechanical"
    task_record = captured.get("task_record")
    assert isinstance(task_record, TaskRecord)
    assert task_record.user_id == "u-1"
    assert task_record.goal == "Hello mechanical"
    buffered_input = captured.get("buffered_input")
    assert isinstance(buffered_input, BufferedTaskInput)
    assert buffered_input.channel_type == "telegram"
    assert buffered_input.channel_target == "123"
    assert buffered_input.message_id == "m-evt-1"
    assert presence.get("channel_type") == "telegram"
    assert presence.get("message_id") == "m-evt-1"


def test_handle_conscious_message_canonical_payload_with_unresolved_user_raises(monkeypatch) -> None:
    action = HandleConsciousMessageAction()

    payload = {
        "contract_type": "canonical_inbound_message",
        "contract_version": "1.0",
        "message_id": "m-can-2",
        "correlation_id": "c-can-2",
        "occurred_at": "2026-05-04T12:01:00+00:00",
        "service_id": None,
        "service_key": "unknown",
        "channel_type": "telegram",
        "channel_target": "123",
        "external_user_id": "u-ext",
        "display_name": "Alex",
        "resolved_user_id": None,
        "text": "Hello unresolved",
        "attachments": [],
        "metadata": {},
    }
    with pytest.raises(ValueError, match="unsupported contract_type"):
        action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=payload), "ctx": Bus()})


def test_handle_conscious_message_mechanical_payload_with_unresolved_service_key_raises(monkeypatch) -> None:
    action = HandleConsciousMessageAction()

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.is_pdca_slicing_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.emit_presence_phase_changed",
        lambda **_: None,
    )

    payload = {
        "contract_type": "canonical_inbound_event",
        "contract_version": "1.0",
        "service_key": "unknown",
        "provider_user_id_from": "u-ext",
        "provider_message_id": "m-evt-2",
        "channel_target": "123",
        "occurred_at": "2026-05-04T12:01:00+00:00",
        "event_kind": "message",
        "provider_raw_message": {},
        "text": "Hello unresolved",
        "attachments": [],
    }
    with pytest.raises(ValueError, match="unknown service_key"):
        action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=payload), "ctx": Bus()})


def test_handle_conscious_message_mechanical_payload_missing_provider_message_id_raises(monkeypatch) -> None:
    action = HandleConsciousMessageAction()

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.is_pdca_slicing_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.emit_presence_phase_changed",
        lambda **_: None,
    )

    payload = {
        "contract_type": "canonical_inbound_event",
        "contract_version": "1.0",
        "service_key": "telegram",
        "provider_user_id_from": "u-ext",
        "channel_target": "123",
        "occurred_at": "2026-05-04T12:01:00+00:00",
        "event_kind": "message",
        "provider_raw_message": {},
        "text": "Hello unresolved",
        "attachments": [],
    }
    with pytest.raises(ValueError, match="missing provider_message_id"):
        action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=payload), "ctx": Bus()})


def test_handle_conscious_message_mechanical_payload_invalid_provider_raw_message_raises(monkeypatch) -> None:
    action = HandleConsciousMessageAction()

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.is_pdca_slicing_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.emit_presence_phase_changed",
        lambda **_: None,
    )

    payload = {
        "contract_type": "canonical_inbound_event",
        "contract_version": "1.0",
        "service_key": "telegram",
        "provider_user_id_from": "u-ext",
        "provider_message_id": "m-evt-raw",
        "channel_target": "123",
        "occurred_at": "2026-05-04T12:01:00+00:00",
        "event_kind": "message",
        "provider_raw_message": "not-an-object",
        "text": "Hello unresolved",
        "attachments": [],
    }
    with pytest.raises(ValueError, match="provider_raw_message must be an object"):
        action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=payload), "ctx": Bus()})


def test_handle_conscious_message_canonical_payload_with_unmapped_user_raises(monkeypatch) -> None:
    action = HandleConsciousMessageAction()

    payload = {
        "contract_type": "canonical_inbound_message",
        "contract_version": "1.0",
        "message_id": "m-can-3",
        "correlation_id": "c-can-3",
        "occurred_at": "2026-05-04T12:02:00+00:00",
        "service_id": 2,
        "service_key": "telegram",
        "channel_type": "telegram",
        "channel_target": "123",
        "external_user_id": "u-ext",
        "display_name": "Alex",
        "resolved_user_id": None,
        "text": "Hello unresolved",
        "attachments": [],
        "metadata": {},
    }
    with pytest.raises(ValueError, match="unsupported contract_type"):
        action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=payload), "ctx": Bus()})


def test_handle_conscious_message_legacy_payload_without_identity_raises(monkeypatch) -> None:
    action = HandleConsciousMessageAction()

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.is_pdca_slicing_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.emit_presence_phase_changed",
        lambda **_: None,
    )

    envelope = build_incoming_message_envelope(
        message_id="m-legacy-missing-user",
        channel_type="telegram",
        channel_target="123",
        provider="telegram",
        text="Hello",
        correlation_id="c-legacy-missing-user",
    )

    with pytest.raises(ValueError, match="unsupported contract_type"):
        action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=envelope), "ctx": Bus()})
