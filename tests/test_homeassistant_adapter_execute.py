from __future__ import annotations

from typing import Any

from alphonse.integrations.domotics.contracts import ActionRequest
from alphonse.integrations.homeassistant.adapter import HomeAssistantAdapter
from alphonse.integrations.homeassistant.config import (
    DebounceConfig,
    HomeAssistantConfig,
    RetryConfig,
    WsReconnectConfig,
)


class FakeRestClient:
    def __init__(self) -> None:
        self.called = False
        self._state: dict[str, Any] | None = None

    def call_service(self, domain: str, service: str, data=None, target=None):
        self.called = True
        _ = (domain, service, data, target)
        return []

    def get_state(self, entity_id: str):
        _ = entity_id
        return self._state

    def get_states(self):
        return []


class FakeWsClient:
    def __init__(self) -> None:
        self.subscriptions: list[tuple[str, Any]] = []

    def subscribe_events(self, event_type: str, callback):
        self.subscriptions.append((event_type, callback))
        return "sub-1"

    def unsubscribe(self, local_subscription_id: str) -> None:
        _ = local_subscription_id

    def stop(self) -> None:
        return None


def _config() -> HomeAssistantConfig:
    return HomeAssistantConfig(
        base_url="http://ha.local:8123",
        token="token",
        request_timeout_sec=5.0,
        retry=RetryConfig(),
        ws=WsReconnectConfig(),
        allowed_domains=frozenset({"light"}),
        allowed_entity_ids=frozenset({"light.kitchen"}),
        debounce=DebounceConfig(enabled=False),
    )


def test_execute_transport_ok_but_effect_false(monkeypatch) -> None:
    fake_rest = FakeRestClient()
    fake_rest._state = {
        "entity_id": "light.kitchen",
        "state": "off",
        "attributes": {"brightness": 20},
    }
    monkeypatch.setattr("alphonse.integrations.homeassistant.adapter.HomeAssistantRestClient", lambda _cfg: fake_rest)
    monkeypatch.setattr("alphonse.integrations.homeassistant.adapter.HomeAssistantWsClient", lambda _cfg: FakeWsClient())

    adapter = HomeAssistantAdapter(_config())
    result = adapter.execute(
        ActionRequest(
            action_type="call_service",
            domain="light",
            service="turn_on",
            data={"brightness": 120},
            target={"entity_id": "light.kitchen"},
            readback=True,
            expected_state="on",
        )
    )

    assert result.transport_ok is True
    assert result.readback_performed is True
    assert result.effect_applied_ok is False


def test_execute_readback_false(monkeypatch) -> None:
    fake_rest = FakeRestClient()
    monkeypatch.setattr("alphonse.integrations.homeassistant.adapter.HomeAssistantRestClient", lambda _cfg: fake_rest)
    monkeypatch.setattr("alphonse.integrations.homeassistant.adapter.HomeAssistantWsClient", lambda _cfg: FakeWsClient())

    adapter = HomeAssistantAdapter(_config())
    result = adapter.execute(
        ActionRequest(
            action_type="call_service",
            domain="light",
            service="turn_off",
            target={"entity_id": "light.kitchen"},
            readback=False,
        )
    )

    assert result.transport_ok is True
    assert result.readback_performed is False
    assert result.effect_applied_ok is None
