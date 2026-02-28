from __future__ import annotations

from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.nervous_system.senses.base import Sense, SignalSpec
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.integrations.domotics import SubscribeSpec, SubscriptionHandle, get_domotics_facade

logger = get_component_logger("senses.homeassistant")


class HomeAssistantSense(Sense):
    key = "homeassistant"
    name = "Home Assistant Sense"
    description = "Emits signals when Home Assistant state changes"
    source_type = "service"
    signals = [
        SignalSpec(
            key="homeassistant.state_changed",
            name="Home Assistant State Changed",
            description="Home Assistant state_changed normalized event",
        )
    ]

    def __init__(self) -> None:
        self._bus: Bus | None = None
        self._running = False
        self._subscription: SubscriptionHandle | None = None

    def start(self, bus: Bus) -> None:
        if self._running:
            return
        facade = get_domotics_facade()
        if facade is None:
            logger.info("HomeAssistant integration disabled (missing config)")
            return

        self._bus = bus
        self._subscription = facade.subscribe(SubscribeSpec(event_type="state_changed"), self._on_event)
        self._running = True
        logger.info("HomeAssistantSense started")

    def stop(self) -> None:
        if not self._running:
            return
        if self._subscription:
            try:
                self._subscription.unsubscribe()
            except Exception:
                pass
        self._subscription = None
        self._running = False
        logger.info("HomeAssistantSense stopped")

    def _on_event(self, event) -> None:
        if not self._bus:
            return
        self._bus.emit(
            Signal(
                type="homeassistant.state_changed",
                payload={
                    "event_type": event.event_type,
                    "entity_id": event.entity_id,
                    "domain": event.domain,
                    "area_id": event.area_id,
                    "old_state": event.old_state,
                    "new_state": event.new_state,
                    "attributes": event.attributes,
                    "changed_at": event.changed_at,
                    "raw_event": event.raw_event,
                },
                source="homeassistant",
                correlation_id=event.entity_id,
            )
        )
