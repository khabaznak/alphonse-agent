from __future__ import annotations

import pytest

from alphonse.agent.io.contracts import NormalizedOutboundMessage
from alphonse.agent.io import telegram_channel


def test_telegram_extremity_adapter_deliver_emits_send_message(monkeypatch) -> None:
    captured: list[dict] = []

    class FakeTelegramAdapter:
        def __init__(self, config: dict) -> None:
            self.config = config

        def handle_action(self, action: dict) -> None:
            captured.append(action)

    monkeypatch.setattr(
        telegram_channel,
        "build_telegram_adapter_config",
        lambda: {"bot_token": "fake-token", "poll_interval_sec": 1.0},
    )
    monkeypatch.setattr(telegram_channel, "TelegramAdapter", FakeTelegramAdapter)
    monkeypatch.setattr(
        telegram_channel,
        "resolve_telegram_chat_id_for_user",
        lambda _: None,
    )
    monkeypatch.setattr(telegram_channel, "can_deliver_to_chat", lambda _chat_id: True)

    adapter = telegram_channel.TelegramExtremityAdapter()
    adapter.deliver(
        NormalizedOutboundMessage(
            message="hola",
            channel_type="telegram",
            channel_target="12345",
            audience={"kind": "system", "id": "system"},
            correlation_id="cid-1",
            metadata={},
        )
    )

    assert len(captured) == 1
    assert captured[0]["type"] == "send_message"
    assert captured[0]["payload"]["chat_id"] == "12345"
    assert captured[0]["payload"]["text"] == "hola"
    assert captured[0]["payload"]["correlation_id"] == "cid-1"
    assert captured[0]["target_integration_id"] == "telegram"


def test_telegram_extremity_adapter_deliver_emits_send_audio(monkeypatch) -> None:
    captured: list[dict] = []

    class FakeTelegramAdapter:
        def __init__(self, config: dict) -> None:
            self.config = config

        def handle_action(self, action: dict) -> None:
            captured.append(action)

    monkeypatch.setattr(
        telegram_channel,
        "build_telegram_adapter_config",
        lambda: {"bot_token": "fake-token", "poll_interval_sec": 1.0},
    )
    monkeypatch.setattr(telegram_channel, "TelegramAdapter", FakeTelegramAdapter)
    monkeypatch.setattr(
        telegram_channel,
        "resolve_telegram_chat_id_for_user",
        lambda _: None,
    )
    monkeypatch.setattr(telegram_channel, "can_deliver_to_chat", lambda _chat_id: True)

    adapter = telegram_channel.TelegramExtremityAdapter()
    adapter.deliver(
        NormalizedOutboundMessage(
            message="hola",
            channel_type="telegram",
            channel_target="12345",
            audience={"kind": "system", "id": "system"},
            correlation_id="cid-audio-1",
            metadata={
                "delivery_mode": "audio",
                "audio_file_path": "/tmp/alphonse-audio/response-1.m4a",
                "as_voice": False,
                "caption": "Hola por audio",
            },
        )
    )

    assert len(captured) == 1
    assert captured[0]["type"] == "send_audio"
    assert captured[0]["payload"]["chat_id"] == "12345"
    assert captured[0]["payload"]["file_path"] == "/tmp/alphonse-audio/response-1.m4a"
    assert captured[0]["payload"]["as_voice"] is False
    assert captured[0]["payload"]["caption"] == "Hola por audio"


def test_telegram_extremity_adapter_audio_without_file_path_raises(monkeypatch) -> None:
    class FakeTelegramAdapter:
        def __init__(self, config: dict) -> None:
            self.config = config

        def handle_action(self, action: dict) -> None:
            _ = action

    monkeypatch.setattr(
        telegram_channel,
        "build_telegram_adapter_config",
        lambda: {"bot_token": "fake-token", "poll_interval_sec": 1.0},
    )
    monkeypatch.setattr(telegram_channel, "TelegramAdapter", FakeTelegramAdapter)
    monkeypatch.setattr(telegram_channel, "can_deliver_to_chat", lambda _chat_id: True)

    adapter = telegram_channel.TelegramExtremityAdapter()
    with pytest.raises(ValueError, match="missing_audio_file_path"):
        adapter.deliver(
            NormalizedOutboundMessage(
                message="hola",
                channel_type="telegram",
                channel_target="12345",
                audience={"kind": "system", "id": "system"},
                correlation_id="cid-audio-missing",
                metadata={"delivery_mode": "audio"},
            )
        )


def test_telegram_extremity_adapter_disabled_without_config(monkeypatch) -> None:
    monkeypatch.setattr(telegram_channel, "build_telegram_adapter_config", lambda: None)

    adapter = telegram_channel.TelegramExtremityAdapter()

    assert adapter._adapter is None
    adapter.deliver(
        NormalizedOutboundMessage(
            message="ignored",
            channel_type="telegram",
            channel_target="12345",
            audience={"kind": "system", "id": "system"},
            correlation_id="cid-2",
            metadata={},
        )
    )


def test_telegram_extremity_adapter_resolves_internal_user_to_chat_id(monkeypatch) -> None:
    captured: list[dict] = []

    class FakeTelegramAdapter:
        def __init__(self, config: dict) -> None:
            self.config = config

        def handle_action(self, action: dict) -> None:
            captured.append(action)

    monkeypatch.setattr(
        telegram_channel,
        "build_telegram_adapter_config",
        lambda: {"bot_token": "fake-token", "poll_interval_sec": 1.0},
    )
    monkeypatch.setattr(telegram_channel, "TelegramAdapter", FakeTelegramAdapter)
    monkeypatch.setattr(
        telegram_channel,
        "resolve_telegram_chat_id_for_user",
        lambda user_ref: "8553589429" if user_ref in {"e64-user", "Alex"} else None,
    )
    monkeypatch.setattr(telegram_channel, "can_deliver_to_chat", lambda _chat_id: True)

    adapter = telegram_channel.TelegramExtremityAdapter()
    adapter.deliver(
        NormalizedOutboundMessage(
            message="hola",
            channel_type="telegram",
            channel_target="Alex",
            audience={"kind": "person", "id": "e64-user"},
            correlation_id="cid-map",
            metadata={},
        )
    )

    assert len(captured) == 1
    assert captured[0]["type"] == "send_message"
    assert captured[0]["payload"]["chat_id"] == "8553589429"


def test_telegram_waiting_user_reaction_is_safe_and_deduped(monkeypatch) -> None:
    captured: list[dict] = []

    class FakeTelegramAdapter:
        def __init__(self, config: dict) -> None:
            self.config = config

        def handle_action(self, action: dict) -> None:
            captured.append(action)

    monkeypatch.setattr(
        telegram_channel,
        "build_telegram_adapter_config",
        lambda: {"bot_token": "fake-token", "poll_interval_sec": 1.0},
    )
    monkeypatch.setattr(telegram_channel, "TelegramAdapter", FakeTelegramAdapter)
    monkeypatch.setattr(telegram_channel, "can_deliver_to_chat", lambda _chat_id: True)

    adapter = telegram_channel.TelegramExtremityAdapter()
    adapter.emit_transition(
        channel_target="12345",
        phase="waiting_user",
        correlation_id="cid-1",
        message_id="999",
    )
    adapter.emit_transition(
        channel_target="12345",
        phase="waiting_user",
        correlation_id="cid-1",
        message_id="999",
    )

    reactions = [
        action for action in captured if action.get("type") == "set_message_reaction"
    ]
    assert len(reactions) == 1
    assert reactions[0]["payload"]["emoji"] == "🤔"


def test_telegram_extremity_adapter_blocks_unauthorized_chat(monkeypatch) -> None:
    captured: list[dict] = []

    class FakeTelegramAdapter:
        def __init__(self, config: dict) -> None:
            self.config = config

        def handle_action(self, action: dict) -> None:
            captured.append(action)

    monkeypatch.setattr(
        telegram_channel,
        "build_telegram_adapter_config",
        lambda: {"bot_token": "fake-token", "poll_interval_sec": 1.0},
    )
    monkeypatch.setattr(telegram_channel, "TelegramAdapter", FakeTelegramAdapter)
    monkeypatch.setattr(telegram_channel, "can_deliver_to_chat", lambda _chat_id: False)

    adapter = telegram_channel.TelegramExtremityAdapter()
    adapter.deliver(
        NormalizedOutboundMessage(
            message="blocked",
            channel_type="telegram",
            channel_target="12345",
            audience={"kind": "system", "id": "system"},
            correlation_id="cid-block",
            metadata={},
        )
    )

    assert captured == []


def test_telegram_transition_event_wip_update_sends_message(monkeypatch) -> None:
    captured: list[dict] = []

    class FakeTelegramAdapter:
        def __init__(self, config: dict) -> None:
            self.config = config

        def handle_action(self, action: dict) -> None:
            captured.append(action)

    monkeypatch.setattr(
        telegram_channel,
        "build_telegram_adapter_config",
        lambda: {"bot_token": "fake-token", "poll_interval_sec": 1.0},
    )
    monkeypatch.setattr(telegram_channel, "TelegramAdapter", FakeTelegramAdapter)
    monkeypatch.setattr(telegram_channel, "can_deliver_to_chat", lambda _chat_id: True)

    adapter = telegram_channel.TelegramExtremityAdapter()
    adapter.emit_transition_event(
        channel_target="12345",
        event={
            "phase": "wip_update",
            "detail": {"text": "Working on your request. Cycle 5."},
        },
        correlation_id="cid-wip",
        message_id="11",
    )

    assert len(captured) == 1
    assert captured[0]["type"] == "send_message"
    assert captured[0]["payload"]["chat_id"] == "12345"
    assert captured[0]["payload"]["text"] == "Working on your request. Cycle 5."
