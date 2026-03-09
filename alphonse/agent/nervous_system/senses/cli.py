from __future__ import annotations

import threading
import time

from alphonse.agent.actions.conscious_message_handler import build_incoming_message_envelope
from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.nervous_system.senses.base import Sense, SignalSpec
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.agent.io import CliSenseAdapter

logger = get_component_logger("senses.cli")


class CliSense(Sense):
    key = "cli"
    name = "CLI Sense"
    description = "Reads CLI input and emits sense.cli.message.user.received"
    source_type = "system"
    signals = [
        SignalSpec(key="sense.cli.message.user.received", name="CLI User Message Received"),
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
                    type="sense.cli.message.user.received",
                    payload=build_incoming_message_envelope(
                        message_id=str(normalized.correlation_id or normalized.timestamp),
                        channel_type=normalized.channel_type,
                        channel_target=str(normalized.channel_target or "cli"),
                        provider="cli",
                        text=normalized.text,
                        occurred_at=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(float(normalized.timestamp))),
                        correlation_id=normalized.correlation_id,
                        actor_external_user_id=normalized.user_id,
                        actor_display_name=normalized.user_name,
                        metadata={"normalized_metadata": normalized.metadata},
                    ),
                    source="cli",
                    correlation_id=normalized.correlation_id,
                )
            )
