from __future__ import annotations

from alphonse.agent.extremities.interfaces.integrations._contracts import CanonicalSenseEvent


def test_canonical_sense_event_payload_shape() -> None:
    payload = CanonicalSenseEvent(
        source_key="front_door_camera",
        provider="homeassistant",
        event_id="evt-1",
        event_kind="person_detected",
        occurred_at="2026-05-22T12:00:00+00:00",
        subject={"kind": "camera", "id": "camera.front_door", "label": "Front door"},
        summary="A person was detected at the front door.",
        data={"confidence": 0.91},
        raw_event={"entity_id": "camera.front_door"},
        dedupe_key="front-door-person-1",
    ).to_payload()

    assert payload == {
        "contract_type": "canonical_sense_event",
        "contract_version": "1.0",
        "source_key": "front_door_camera",
        "provider": "homeassistant",
        "event_id": "evt-1",
        "event_kind": "person_detected",
        "occurred_at": "2026-05-22T12:00:00+00:00",
        "subject": {"kind": "camera", "id": "camera.front_door", "label": "Front door"},
        "summary": "A person was detected at the front door.",
        "data": {"confidence": 0.91},
        "raw_event": {"entity_id": "camera.front_door"},
        "dedupe_key": "front-door-person-1",
    }
