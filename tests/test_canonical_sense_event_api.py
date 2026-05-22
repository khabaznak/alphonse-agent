from __future__ import annotations

from alphonse.agent.extremities.interfaces.integrations._contracts import CanonicalSenseEvent
from alphonse.agent.nervous_system.senses.canonical_sense_event_api import (
    CanonicalSenseEventApiSense,
    build_canonical_sense_event_api_signal,
)


class FakeBus:
    def __init__(self) -> None:
        self.emitted: list[object] = []

    def emit(self, signal: object) -> None:
        self.emitted.append(signal)


def test_canonical_sense_event_api_passes_through_payload() -> None:
    sense = CanonicalSenseEventApiSense()
    bus = FakeBus()
    payload = CanonicalSenseEvent(
        source_key="office_thermostat_monitor",
        provider="local_program",
        event_id="evt-temp-1",
        event_kind="temperature_threshold_crossed",
        occurred_at="2026-05-22T12:01:00+00:00",
        subject={"kind": "thermostat", "id": "sensor.office_temperature"},
        summary="Office temperature reached 30 Celsius.",
        data={"temperature_c": 30, "threshold_c": 30},
        raw_event={"poll_interval_seconds": 60},
        dedupe_key="office-temp-above-30",
    ).to_payload()

    signal = build_canonical_sense_event_api_signal("sense.api.event.received", payload, "cid-temp")
    sense.emit(bus, signal)

    assert len(bus.emitted) == 1
    emitted = bus.emitted[0]
    assert getattr(emitted, "type") == "sense.api.event.received"
    assert getattr(emitted, "source") == "canonical_sense_event_api"
    assert getattr(emitted, "correlation_id") == "office-temp-above-30"
    assert getattr(emitted, "payload") == payload
