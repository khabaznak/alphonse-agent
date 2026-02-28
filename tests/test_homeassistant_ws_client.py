from __future__ import annotations

import queue

from alphonse.integrations.homeassistant.config import HomeAssistantConfig
from alphonse.integrations.homeassistant.ws_client import (
    HomeAssistantWsClient,
    _PendingRequest,
    _Subscription,
)


def _config() -> HomeAssistantConfig:
    return HomeAssistantConfig(base_url="http://ha.local:8123", token="token")


def test_handle_message_resolves_pending_request(monkeypatch) -> None:
    monkeypatch.setattr("alphonse.integrations.homeassistant.ws_client.websocket", object())
    client = HomeAssistantWsClient(_config())

    q: queue.Queue[dict | None] = queue.Queue(maxsize=1)
    client._pending_requests[7] = _PendingRequest(response_queue=q)

    payload = {"id": 7, "type": "result", "success": True}
    client._handle_message(payload)

    resolved = q.get(timeout=0.1)
    assert resolved == payload
    assert 7 not in client._pending_requests


def test_handle_message_dispatches_subscription_event(monkeypatch) -> None:
    monkeypatch.setattr("alphonse.integrations.homeassistant.ws_client.websocket", object())
    client = HomeAssistantWsClient(_config())

    received = []
    sub = _Subscription(local_id="sub-1", event_type="state_changed", callback=lambda event: received.append(event), ha_subscription_id=42)
    client._subscriptions["sub-1"] = sub
    client._subscriptions_by_ha_id[42] = "sub-1"

    event = {
        "id": 42,
        "type": "event",
        "event": {"event_type": "state_changed", "data": {"entity_id": "light.kitchen"}},
    }
    client._handle_message(event)

    assert len(received) == 1
    assert received[0]["event_type"] == "state_changed"
