"""Fake integration adapter for testing and observability."""

from __future__ import annotations

from alphonse.agent.observability.log_manager import get_component_logger
from typing import Any

from alphonse.agent.extremities.interfaces.integrations._contracts import IntegrationAdapter
from alphonse.agent.nervous_system.senses.bus import Signal as BusSignal

logger = get_component_logger("extremities.interfaces.integrations.fake.fake_adapter")


class FakeAdapter(IntegrationAdapter):
    """A minimal adapter that emits a test signal on start()."""

    @property
    def id(self) -> str:
        return "fake"

    @property
    def io_type(self) -> str:
        return "io"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.last_action: dict[str, Any] | None = None

    def start(self) -> None:
        logger.info("FakeAdapter start()")
        signal = BusSignal(
            type="external.fake.message",
            payload={"text": "hello from fake adapter"},
            source="fake",
        )
        logger.info("FakeAdapter emitting signal: %s", signal.type)
        self.emit_signal(signal)

    def stop(self) -> None:
        logger.info("FakeAdapter stop()")

    def handle_action(self, action: dict[str, Any]) -> None:
        logger.info("FakeAdapter handle_action: %s", action)
        self.last_action = action
