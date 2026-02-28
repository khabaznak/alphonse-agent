from __future__ import annotations

import logging
from dataclasses import dataclass

from alphonse.agent.io.contracts import NormalizedOutboundMessage
from alphonse.agent.io.homeassistant_channel import HomeAssistantExtremityAdapter
from alphonse.agent.nervous_system.senses.bus import Bus
from alphonse.agent.nervous_system.senses.homeassistant import HomeAssistantSense
from alphonse.integrations.domotics.contracts import (
    ActionResult,
    NormalizedEvent,
    SubscriptionHandle,
)


@dataclass
class StubFacade:
    executed: list = None
    callback: object | None = None

    def __post_init__(self) -> None:
        if self.executed is None:
            self.executed = []

    def subscribe(self, spec, on_event):
        _ = spec
        self.callback = on_event
        return SubscriptionHandle(subscription_id="sub-1", unsubscribe=lambda: None)

    def execute(self, action_request):
        self.executed.append(action_request)
        return ActionResult(
            transport_ok=True,
            effect_applied_ok=True,
            readback_performed=True,
            readback_state={"entity_id": "light.kitchen", "state": "on"},
        )


def test_homeassistant_sense_disabled_logs_health(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        "alphonse.agent.nervous_system.senses.homeassistant.get_domotics_facade",
        lambda: None,
    )

    caplog.set_level(logging.INFO)
    sense = HomeAssistantSense()
    sense.start(Bus())

    assert any(
        "HomeAssistant integration disabled (missing config)" in rec.getMessage()
        for rec in caplog.records
    )


def test_homeassistant_sense_emits_state_changed(monkeypatch) -> None:
    facade = StubFacade()
    monkeypatch.setattr(
        "alphonse.agent.nervous_system.senses.homeassistant.get_domotics_facade",
        lambda: facade,
    )

    bus = Bus()
    sense = HomeAssistantSense()
    sense.start(bus)

    assert callable(facade.callback)
    facade.callback(
        NormalizedEvent(
            event_type="state_changed",
            entity_id="light.kitchen",
            domain="light",
            area_id=None,
            old_state="off",
            new_state="on",
            attributes={"brightness": 120},
            changed_at="2026-02-28T00:00:00Z",
            raw_event={"foo": "bar"},
        )
    )

    signal = bus.get(timeout=0.1)
    assert signal is not None
    assert signal.type == "homeassistant.state_changed"
    assert signal.payload["entity_id"] == "light.kitchen"


def test_homeassistant_extremity_executes_action(monkeypatch) -> None:
    facade = StubFacade()
    monkeypatch.setattr(
        "alphonse.agent.io.homeassistant_channel.get_domotics_facade",
        lambda: facade,
    )

    adapter = HomeAssistantExtremityAdapter()
    adapter.deliver(
        NormalizedOutboundMessage(
            message="turn on kitchen",
            channel_type="homeassistant",
            channel_target="light.kitchen",
            audience={"kind": "system", "id": "system"},
            correlation_id="cid-1",
            metadata={
                "domain": "light",
                "service": "turn_on",
                "target": {"entity_id": "light.kitchen"},
                "readback": True,
                "expected_state": "on",
            },
        )
    )

    assert len(facade.executed) == 1
    action = facade.executed[0]
    assert action.domain == "light"
    assert action.service == "turn_on"
    assert action.readback is True
