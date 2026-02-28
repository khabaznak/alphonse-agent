from __future__ import annotations

from alphonse.integrations.homeassistant.anti_spam import EventDebouncer, EventFilter
from alphonse.integrations.homeassistant.config import DebounceConfig


def _event(entity_id: str, state: str, *, brightness: int | None = None) -> dict:
    attrs = {}
    if brightness is not None:
        attrs["brightness"] = brightness
    return {
        "event_type": "state_changed",
        "data": {
            "new_state": {
                "entity_id": entity_id,
                "state": state,
                "attributes": attrs,
            }
        },
    }


def test_event_filter_by_domain_and_entity() -> None:
    filt = EventFilter(
        allowed_domains=frozenset({"light"}),
        allowed_entity_ids=frozenset({"light.kitchen"}),
    )

    assert filt.allows(domain="light", entity_id="light.kitchen") is True
    assert filt.allows(domain="sensor", entity_id="sensor.temp") is False
    assert filt.allows(domain="light", entity_id="light.bedroom") is False


def test_debouncer_suppresses_repeated_same_state() -> None:
    debouncer = EventDebouncer(
        DebounceConfig(enabled=True, window_ms=2_000, key_strategy="entity_state")
    )

    first = debouncer.is_suppressed(_event("binary_sensor.motion", "on"))
    second = debouncer.is_suppressed(_event("binary_sensor.motion", "on"))

    assert first is False
    assert second is True


def test_debouncer_keeps_meaningful_transition() -> None:
    debouncer = EventDebouncer(
        DebounceConfig(enabled=True, window_ms=2_000, key_strategy="entity_state")
    )

    _ = debouncer.is_suppressed(_event("binary_sensor.motion", "on"))
    changed = debouncer.is_suppressed(_event("binary_sensor.motion", "off"))

    assert changed is False


def test_debouncer_attribute_strategy_uses_subset() -> None:
    debouncer = EventDebouncer(
        DebounceConfig(
            enabled=True,
            window_ms=2_000,
            key_strategy="entity_state_attributes",
            attributes=("brightness",),
        )
    )

    first = debouncer.is_suppressed(_event("light.kitchen", "on", brightness=20))
    second = debouncer.is_suppressed(_event("light.kitchen", "on", brightness=20))
    third = debouncer.is_suppressed(_event("light.kitchen", "on", brightness=40))

    assert first is False
    assert second is True
    assert third is False
