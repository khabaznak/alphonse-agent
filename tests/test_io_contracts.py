from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alphonse.agent.io.api_channel import ApiSenseAdapter
from alphonse.agent.io.cli_channel import CliSenseAdapter
from alphonse.agent.io.telegram_channel import TelegramSenseAdapter
from alphonse.agent.io.web_channel import WebSenseAdapter
from alphonse.agent.io.contracts import NormalizedInboundMessage
from alphonse.agent.nervous_system.senses.api import ApiSense, build_api_signal


class FakeBus:
    def __init__(self) -> None:
        self.emitted = []

    def emit(self, signal: object) -> None:
        self.emitted.append(signal)


@dataclass(frozen=True)
class StubSenseAdapter:
    channel_type: str = "webui"
    normalized: NormalizedInboundMessage | None = None

    def normalize(self, payload: dict[str, Any]) -> NormalizedInboundMessage:
        if self.normalized is None:
            raise RuntimeError("missing normalized payload")
        return self.normalized


def test_cli_sense_adapter_normalizes() -> None:
    adapter = CliSenseAdapter()
    result = adapter.normalize({"text": "hello", "user_name": "Alex", "timestamp": 123.0})
    assert result.text == "hello"
    assert result.channel_type == "cli"
    assert result.channel_target == "cli"
    assert result.user_name == "Alex"


def test_telegram_sense_adapter_normalizes() -> None:
    adapter = TelegramSenseAdapter()
    result = adapter.normalize(
        {
            "text": "hi",
            "chat_id": 42,
            "from_user": "u1",
            "from_user_name": "User",
            "timestamp": 10.0,
            "message_id": 7,
            "update_id": 9,
        }
    )
    assert result.text == "hi"
    assert result.channel_type == "telegram"
    assert result.channel_target == "42"
    assert result.user_id == "u1"
    assert result.user_name == "User"


def test_web_sense_adapter_normalizes() -> None:
    adapter = WebSenseAdapter()
    result = adapter.normalize({"text": "ping", "user_id": "u", "timestamp": 1.0})
    assert result.text == "ping"
    assert result.channel_type == "webui"
    assert result.channel_target == "webui"


def test_api_sense_emits_normalized_payload(monkeypatch) -> None:
    normalized = NormalizedInboundMessage(
        text="ok",
        channel_type="webui",
        channel_target="webui",
        user_id="u1",
        user_name="User",
        timestamp=1.0,
        correlation_id="cid",
        metadata={"raw": {"x": 1}},
    )

    stub_adapter = StubSenseAdapter(normalized=normalized)

    class StubRegistry:
        def get_sense(self, channel_type: str):
            return stub_adapter

    monkeypatch.setattr(
        "alphonse.agent.nervous_system.senses.api.get_io_registry",
        lambda: StubRegistry(),
    )

    sense = ApiSense()
    bus = FakeBus()
    signal = build_api_signal("api.message_received", {"text": "hi", "channel": "webui"}, None)
    sense.emit(bus, signal)

    assert len(bus.emitted) == 1
    emitted = bus.emitted[0]
    payload = getattr(emitted, "payload", {})
    assert payload["text"] == "ok"
    assert payload["channel"] == "webui"
    assert payload["target"] == "webui"
    assert payload["user_id"] == "u1"
    assert payload["user_name"] == "User"
    assert payload["correlation_id"] == "cid"
