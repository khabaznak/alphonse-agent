from __future__ import annotations

import logging
import threading
import time

from alphonse.agent.nervous_system.senses.base import Sense, SignalSpec
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.agent.io import CliSenseAdapter

logger = logging.getLogger(__name__)


class CliSense(Sense):
    key = "cli"
    name = "CLI Sense"
    description = "Reads CLI input and emits cli.message_received"
    source_type = "system"
    signals = [
        SignalSpec(key="cli.message_received", name="CLI Message Received"),
    ]

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._bus: Bus | None = None
        self._sense_adapter = CliSenseAdapter()

    def start(self, bus: Bus) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._bus = bus
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("CliSense started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        logger.info("CliSense stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                text = input("alphonse> ").strip()
            except EOFError:
                self._stop_event.set()
                return
            if not text:
                continue
            if not self._bus:
                continue
            normalized = self._sense_adapter.normalize(
                {
                    "text": text,
                    "origin": "cli",
                    "user_name": "Alex",
                    "timestamp": time.time(),
                }
            )
            self._bus.emit(
                Signal(
                    type="cli.message_received",
                    payload={
                        "text": normalized.text,
                        "channel": normalized.channel_type,
                        "target": normalized.channel_target,
                        "user_id": normalized.user_id,
                        "user_name": normalized.user_name,
                        "timestamp": normalized.timestamp,
                        "correlation_id": normalized.correlation_id,
                        "metadata": normalized.metadata,
                    },
                    source="cli",
                    correlation_id=normalized.correlation_id,
                )
            )
