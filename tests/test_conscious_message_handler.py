from __future__ import annotations

from alphonse.agent.actions.conscious_message_handler import IncomingMessageEnvelope
from alphonse.agent.actions.conscious_message_handler import build_incoming_message_envelope


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
