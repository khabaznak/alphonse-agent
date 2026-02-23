from __future__ import annotations

from typing import Any

from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.nervous_system.senses.base import Sense, SignalSpec
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.agent.nervous_system.location_profiles import upsert_location_profile
from alphonse.agent.tools.registry import build_default_tool_registry

logger = get_component_logger("senses.location")


class LocationSense(Sense):
    key = "location"
    name = "Location Sense"
    description = "Normalizes user-provided addresses into location profiles."
    source_type = "internal"
    owner = "core"
    signals = [
        SignalSpec(
            key="location.profile_updated",
            name="Location Profile Updated",
            description="A user location profile was created or updated",
        )
    ]

    def __init__(self) -> None:
        self._bus: Bus | None = None

    def start(self, bus: Bus) -> None:
        self._bus = bus
        logger.info("LocationSense started")

    def stop(self) -> None:
        self._bus = None
        logger.info("LocationSense stopped")

    def ingest_address(
        self,
        *,
        principal_id: str,
        label: str,
        address_text: str,
        source: str = "user",
        confidence: float | None = None,
        language: str | None = None,
        region: str | None = None,
    ) -> str:
        tool_registry = build_default_tool_registry()
        geocoder = tool_registry.get("geocoder")
        lat = None
        lng = None
        if geocoder is not None:
            try:
                result = geocoder.geocode(address_text, language=language, region=region)
                if result and result.get("location"):
                    lat = result["location"].get("lat")
                    lng = result["location"].get("lng")
            except Exception as exc:
                logger.warning("LocationSense geocode failed error=%s", exc)
        location_id = upsert_location_profile(
            {
                "principal_id": principal_id,
                "label": label,
                "address_text": address_text,
                "latitude": lat,
                "longitude": lng,
                "source": source,
                "confidence": confidence,
            }
        )
        if self._bus:
            self._bus.emit(
                Signal(
                    type="location.profile_updated",
                    payload={
                        "principal_id": principal_id,
                        "label": label,
                        "location_id": location_id,
                        "address_text": address_text,
                        "latitude": lat,
                        "longitude": lng,
                        "source": source,
                    },
                    source="location",
                )
            )
        return location_id
