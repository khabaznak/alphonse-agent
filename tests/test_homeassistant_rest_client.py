from __future__ import annotations

from alphonse.integrations.homeassistant.rest_client import _build_service_payload


def test_build_service_payload_flattens_target_ids() -> None:
    payload = _build_service_payload(
        data={"brightness": 120},
        target={"entity_id": "light.kitchen"},
    )

    assert payload == {"brightness": 120, "entity_id": "light.kitchen"}


def test_build_service_payload_prefers_explicit_data_over_target() -> None:
    payload = _build_service_payload(
        data={"entity_id": "light.office"},
        target={"entity_id": "light.kitchen", "area_id": "living_room"},
    )

    assert payload == {"entity_id": "light.office", "area_id": "living_room"}


def test_build_service_payload_ignores_non_targeting_keys() -> None:
    payload = _build_service_payload(
        data={},
        target={"entity_id": "switch.fan", "unexpected": "value"},
    )

    assert payload == {"entity_id": "switch.fan"}
