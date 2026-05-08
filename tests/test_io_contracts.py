from __future__ import annotations

from alphonse.agent.nervous_system.senses.api import ApiSense, build_api_signal


class FakeBus:
    def __init__(self) -> None:
        self.emitted: list[object] = []

    def emit(self, signal: object) -> None:
        self.emitted.append(signal)


def test_api_sense_emits_canonical_event_from_text_payload() -> None:
    sense = ApiSense()
    bus = FakeBus()

    signal = build_api_signal(
        "sense.api.message.user.received",
        {
            "text": "hi",
            "channel": "webui",
            "target": "webui",
            "user_id": "u1",
            "user_name": "User",
            "timestamp": 1.0,
        },
        None,
    )
    sense.emit(bus, signal)

    assert len(bus.emitted) == 1
    emitted = bus.emitted[0]
    payload = getattr(emitted, "payload", {})
    assert payload["contract_type"] == "canonical_inbound_event"
    assert payload["service_key"] == "webui"
    assert payload["provider_user_id_from"] == "u1"
    assert payload["channel_target"] == "webui"
    assert payload["text"] == "hi"
    assert payload["display_name"] == "User"


def test_api_sense_emits_canonical_event_from_asset_payload() -> None:
    sense = ApiSense()
    bus = FakeBus()

    signal = build_api_signal(
        "sense.api.message.user.received",
        {
            "channel": "webui",
            "target": "webui",
            "content": {
                "type": "asset",
                "assets": [{"asset_id": "asset-1", "kind": "audio"}],
            },
        },
        "cid-asset",
    )
    sense.emit(bus, signal)

    payload = getattr(bus.emitted[0], "payload", {})
    assert payload["contract_type"] == "canonical_inbound_event"
    assert payload["service_key"] == "webui"
    assert payload["channel_target"] == "webui"
    assert payload["text"] == "[audio asset message]"
    assert payload["attachments"] == [{"asset_id": "asset-1", "kind": "audio"}]
    assert payload["dedupe_key"] == "cid-asset"


def test_api_sense_passes_through_canonical_payload() -> None:
    sense = ApiSense()
    bus = FakeBus()
    canonical_payload = {
        "contract_type": "canonical_inbound_event",
        "contract_version": "1.0",
        "service_key": "api",
        "provider_user_id_from": "u1",
        "provider_message_id": "m-1",
        "channel_target": "u1",
        "occurred_at": "2026-05-07T00:00:00+00:00",
        "event_kind": "message",
        "provider_raw_message": {"x": 1},
        "text": "hello",
    }

    signal = build_api_signal("sense.api.message.user.received", canonical_payload, "cid-pass")
    sense.emit(bus, signal)

    payload = getattr(bus.emitted[0], "payload", {})
    assert payload == canonical_payload
