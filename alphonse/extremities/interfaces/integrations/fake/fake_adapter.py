"""Fake integration adapter for testing and observability."""

from __future__ import annotations

import logging
from typing import Any

from alphonse.extremities.interfaces.integrations._contracts import IntegrationAdapter
from alphonse.senses.bus import Signal as BusSignal

logger = logging.getLogger(__name__)


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
