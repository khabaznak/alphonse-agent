from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone

from alphonse.agent.nervous_system.senses.base import Sense, SignalSpec
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.config import settings

logger = logging.getLogger(__name__)


class ClockSense(Sense):
    key = "clock"
    name = "Clock Sense"
    description = "Emits periodic time.tick signals with current time context."
    source_type = "system"
    signals = [
        SignalSpec(key="time.tick", name="Time Tick", description="Periodic clock tick"),
    ]

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._bus: Bus | None = None
        self._interval_seconds = _parse_float(os.getenv("CLOCK_SENSE_INTERVAL_SECONDS"), 60.0)
        self._enabled = os.getenv("CLOCK_SENSE_ENABLED", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def start(self, bus: Bus) -> None:
        if not self._enabled:
            logger.info("ClockSense disabled")
            return
        if self._thread and self._thread.is_alive():
            return
        self._bus = bus
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("ClockSense started interval=%.2fs", self._interval_seconds)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("ClockSense stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            now = datetime.now(tz=timezone.utc)
            if self._bus:
                self._bus.emit(
                    Signal(
                        type="time.tick",
                        payload={
                            "now_utc": now.isoformat(),
                            "timezone": settings.get_timezone(),
                        },
                        source="clock",
                        correlation_id=None,
                    )
                )
            self._stop_event.wait(timeout=self._interval_seconds)


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return max(1.0, float(raw))
    except ValueError:
        return default
