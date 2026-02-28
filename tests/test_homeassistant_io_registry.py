from __future__ import annotations

from alphonse.agent.io.registry import build_default_io_registry


def test_default_io_registry_contains_homeassistant_adapters() -> None:
    registry = build_default_io_registry()

    assert registry.get_sense("homeassistant") is not None
    assert registry.get_extremity("homeassistant") is not None
