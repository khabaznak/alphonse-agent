"""Local runner to demonstrate a pure heart loop with integrations."""

from __future__ import annotations

import logging

from alphonse.agent.extremities.interfaces.integrations.loader import IntegrationLoader
from alphonse.agent.nervous_system.senses.bus import Bus, Signal


def heart_step(signal: Signal) -> dict[str, object] | None:
    signal_type = signal.type or ""
    if signal_type.endswith(".message") or "message" in signal_type:
        text = ""
        if isinstance(signal.payload, dict):
            text = str(signal.payload.get("text", ""))
        return {
            "type": "echo",
            "payload": {"text": f"Echo: {text}"},
            "target_integration_id": signal.source or "fake",
        }
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    bus = Bus()
    config = {
        "integrations": [
            {
                "name": "fake",
                "enabled": True,
                "module": "alphonse.agent.extremities.interfaces.integrations.fake.fake_adapter",
                "class": "FakeAdapter",
                "config": {},
            }
        ]
    }

    loader = IntegrationLoader(config, bus)
    registry = loader.load_all()
    loader.start_all(registry)

    signal = bus.get(timeout=2)
    if signal is None:
        logging.error("No signal received from bus")
        return

    logging.info(
        "Received signal: type=%s source=%s payload=%s id=%s",
        signal.type,
        signal.source,
        signal.payload,
        signal.id,
    )

    action = heart_step(signal)
    if action is None:
        logging.info("No action produced by heart_step")
        return

    target_id = str(action.get("target_integration_id"))
    adapter = registry.get(target_id)
    if adapter is None:
        logging.error("No adapter found for target: %s", target_id)
        return

    adapter.handle_action(action)
    logging.info("Delivered action to adapter: %s", target_id)


if __name__ == "__main__":
    main()
