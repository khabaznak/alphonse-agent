"""Durable scheduled-task polling service for the v2 runtime host."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskRunner
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore


@dataclass(frozen=True)
class ScheduledWorkerStats:
    ticks: int = 0
    queued: int = 0
    errors: int = 0


class ScheduledTaskWorker:
    """Polls durable schedules and wakes the host after dispatching due work."""

    def __init__(
        self,
        *,
        store: ScheduledTaskStore,
        messages: Any,
        on_message_queued: Callable[[], None] | None = None,
        poll_interval_sec: float = 1.0,
    ) -> None:
        self.store = store
        self.runner = ScheduledTaskRunner(store=store, messages=messages)
        self.on_message_queued = on_message_queued
        self.poll_interval_sec = max(0.1, float(poll_interval_sec))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats = ScheduledWorkerStats()

    @property
    def stats(self) -> ScheduledWorkerStats:
        return self._stats

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="alphonse-v2-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)

    def run_once(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
        outcomes = self.runner.run_due_once(now=now or datetime.now(timezone.utc))
        queued = sum(1 for outcome in outcomes if outcome.get("status") == "queued")
        errors = sum(1 for outcome in outcomes if outcome.get("status") == "error")
        self._stats = ScheduledWorkerStats(
            ticks=self._stats.ticks + 1,
            queued=self._stats.queued + queued,
            errors=self._stats.errors + errors,
        )
        if queued and self.on_message_queued is not None:
            try:
                self.on_message_queued()
            except Exception:
                pass
        return outcomes

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:
                self._stats = ScheduledWorkerStats(
                    ticks=self._stats.ticks + 1,
                    queued=self._stats.queued,
                    errors=self._stats.errors + 1,
                )
            self._stop.wait(self.poll_interval_sec)
