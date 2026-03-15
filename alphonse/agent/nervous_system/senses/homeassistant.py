from __future__ import annotations

from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.nervous_system.senses.base import Sense, SignalSpec
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.integrations.domotics import SubscribeSpec, SubscriptionHandle, get_domotics_facade
from alphonse.integrations.homeassistant.config import load_homeassistant_config

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
        ),
        SignalSpec(
            key="homeassistant.connection_status",
            name="Home Assistant Connection Status",
            description="Home Assistant websocket status and recovery events",
        ),
    ]

    def __init__(self) -> None:
        self._bus: Bus | None = None
        self._running = False
        self._subscription: SubscriptionHandle | None = None
        self._facade = None

    def start(self, bus: Bus) -> None:
        if self._running:
            return
        try:
            config = load_homeassistant_config()
        except Exception as exc:
            logger.warning("HomeAssistant integration disabled (invalid config): %s", exc)
            return
        if config is None:
            logger.info(
                "HomeAssistant integration disabled (missing HA_BASE_URL/HA_TOKEN)"
            )
            return
        try:
            facade = get_domotics_facade()
        except Exception as exc:
            logger.warning("HomeAssistant integration disabled (init error): %s", exc)
            return
        if facade is None:
            logger.warning("HomeAssistant integration disabled (facade unavailable)")
            return

        self._bus = bus
        self._facade = facade
        self._bind_ws_health_callback(facade)
        self._subscription = facade.subscribe(SubscribeSpec(event_type="state_changed"), self._on_event)
        self._running = True
        logger.info(
            "HomeAssistantSense started base_url=%s allowed_domains=%s allowed_entities=%s",
            config.base_url,
            len(config.allowed_domains),
            len(config.allowed_entity_ids),
        )

    def stop(self) -> None:
        if not self._running:
            return
        if self._subscription:
            try:
                self._subscription.unsubscribe()
            except Exception:
                pass
        self._unbind_ws_health_callback()
        self._facade = None
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

    def _bind_ws_health_callback(self, facade) -> None:
        adapter = getattr(facade, "adapter", None)
        set_health_cb = getattr(adapter, "set_ws_health_callback", None)
        if callable(set_health_cb):
            set_health_cb(self._on_health_event)

    def _unbind_ws_health_callback(self) -> None:
        facade = self._facade
        adapter = getattr(facade, "adapter", None) if facade is not None else None
        set_health_cb = getattr(adapter, "set_ws_health_callback", None)
        if callable(set_health_cb):
            set_health_cb(None)

    def _on_health_event(self, payload: dict) -> None:
        if not self._bus:
            return
        status = str(payload.get("status") or "").strip()
        self._bus.emit(
            Signal(
                type="homeassistant.connection_status",
                payload={
                    "status": status,
                    "detail": payload.get("detail"),
                    "local_id": payload.get("local_id"),
                    "ws_url": payload.get("ws_url"),
                },
                source="homeassistant",
                correlation_id=f"homeassistant.ws.{status or 'unknown'}",
            )
        )
