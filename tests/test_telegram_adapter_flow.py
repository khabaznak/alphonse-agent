from __future__ import annotations

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
