"""Local runner for the fake integration adapter."""

from __future__ import annotations

import logging

from alphonse.extremities.interfaces.integrations.loader import IntegrationLoader
from alphonse.senses.bus import Bus


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    bus = Bus()
    config = {
        "integrations": [
            {
                "name": "fake",
                "enabled": True,
                "module": "alphonse.extremities.interfaces.integrations.fake.fake_adapter",
                "class": "FakeAdapter",
                "config": {},
            }
        ]
    }

    loader = IntegrationLoader(config, bus)
    registry = loader.load_all()
    loader.start_all(registry)

    signal = bus.get(timeout=1)
    if signal is None:
        logging.error("No signal received from fake adapter")
    else:
        logging.info(
            "Received signal: type=%s source=%s payload=%s id=%s",
            signal.type,
            signal.source,
            signal.payload,
            signal.id,
        )

    adapter = registry.get("fake")
    if adapter is None:
        logging.error("Fake adapter not found in registry")
        return

    adapter.handle_action(
        {
            "type": "test.action",
            "payload": {"msg": "hello fake adapter"},
        }
    )
    logging.info("Sent action to fake adapter")


if __name__ == "__main__":
    main()
