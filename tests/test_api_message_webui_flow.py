from __future__ import annotations

import threading
from typing import Any

from fastapi.testclient import TestClient

from alphonse.agent.nervous_system.senses.bus import Bus
from alphonse.infrastructure.api import app
from alphonse.infrastructure.api_exchange import ApiExchange
from alphonse.infrastructure.api_gateway import gateway


def test_agent_message_webui_flow_returns_exchange_response() -> None:
    original = (gateway.bus, gateway.exchange, gateway.sense)
    bus = Bus()
    exchange = ApiExchange()
    gateway.configure(bus, exchange)
    captured: dict[str, Any] = {}

    def worker() -> None:
        signal = bus.get(timeout=2)
        assert signal is not None
        captured["signal_type"] = signal.type
        captured["payload"] = signal.payload
        exchange.publish(
            str(signal.correlation_id),
            {"message": "hello from worker", "data": {"ok": True}},
        )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        client = TestClient(app)
        response = client.post(
            "/agent/message",
            json={
                "text": "hello",
                "channel": "webui",
                "content": {"type": "asset", "assets": [{"asset_id": "asset-1", "kind": "audio"}]},
                "controls": {"audio_mode": "local_audio"},
                "metadata": {"user_name": "UI Tester"},
            },
        )
    finally:
        thread.join(timeout=3)
        gateway.bus, gateway.exchange, gateway.sense = original

    assert response.status_code == 200
    assert response.json() == {"message": "hello from worker", "data": {"ok": True}}

    assert captured["signal_type"] == "api.message_received"
    payload = captured["payload"]
    assert payload["text"] == "hello"
    assert payload["channel"] == "webui"
    assert payload["target"] == "webui"
    assert payload["content"]["type"] == "asset"
    assert payload["content"]["assets"][0]["asset_id"] == "asset-1"
    assert payload["controls"]["audio_mode"] == "local_audio"
    assert payload["user_name"] is None
    assert payload["metadata"]["raw"]["metadata"]["user_name"] == "UI Tester"
    assert payload["origin"] == "api"
    assert isinstance(payload["correlation_id"], str)
