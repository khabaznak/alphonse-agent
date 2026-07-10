"""Foreground v2 daemon host."""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass
from typing import Any

from alphonse.agent_v2.core.core import LoopStepStatus
from alphonse.agent_v2.core.io import project_snapshot_to_outbox
from alphonse.agent_v2.core.messages import SQLiteMessageQueue
from alphonse.agent_v2.ipc import V2DaemonServer
from alphonse.agent_v2.runtime import V2RuntimeHost
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.runtime import start_runtime_integrations
from alphonse.agent_v2.runtime import stop_runtime_integrations
from alphonse.agent_v2.services.scheduled_worker import ScheduledTaskWorker


@dataclass
class V2Daemon:
    runtime: V2RuntimeHost
    poll_interval_sec: float = 0.05

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._processor_thread: threading.Thread | None = None
        self.scheduler = ScheduledTaskWorker(
            store=self.runtime.schedule_store,
            messages=self.runtime.queue,
            on_message_queued=lambda: None,
        )
        self.ipc = V2DaemonServer(self)

    def start(self) -> None:
        if self._processor_thread is not None and self._processor_thread.is_alive():
            return
        self._stop.clear()
        start_runtime_integrations(self.runtime)
        self.scheduler.start()
        self.ipc.start()
        self._processor_thread = threading.Thread(target=self._process_loop, name="alphonse-v2-core", daemon=True)
        self._processor_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.ipc.stop()
        self.scheduler.stop()
        stop_runtime_integrations(self.runtime)
        if self._processor_thread is not None and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=5)

    def restart_integrations(self) -> None:
        start_runtime_integrations(self.runtime)

    def pop_activity_events(self) -> list[dict[str, Any]]:
        events = list(self.runtime.activity_events)
        self.runtime.activity_events.clear()
        return [
            {
                "phase": event.phase.value,
                "label": event.label,
                "message": event.message,
                "speaker": event.speaker,
            }
            for event in events
        ]

    def run_once(self) -> Any:
        queued = self.runtime.queue.peek()
        with self.runtime.presence_projector.processing(queued):
            step = self.runtime.core.step()
            if step.queued_message_id:
                acknowledge = getattr(self.runtime.queue, "ack", None)
                if callable(acknowledge):
                    acknowledge(step.queued_message_id)
            if step.status in {
                LoopStepStatus.PROCESSED,
                LoopStepStatus.PARKED,
                LoopStepStatus.WAITING,
                LoopStepStatus.FAILED,
            }:
                self.runtime.presence_projector.finish(
                    failed=step.status == LoopStepStatus.FAILED,
                    waiting=step.status in {LoopStepStatus.PARKED, LoopStepStatus.WAITING},
                )
        snapshot = self.runtime.visible_state.snapshot()
        if step.status in {LoopStepStatus.PROCESSED, LoopStepStatus.PARKED, LoopStepStatus.WAITING}:
            project_snapshot_to_outbox(snapshot=snapshot, outbox=self.runtime.outbox)
        return step

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop.wait(0.5):
                pass
        finally:
            self.stop()

    def _process_loop(self) -> None:
        while not self._stop.is_set():
            step = self.run_once()
            if step.status in {LoopStepStatus.EMPTY, LoopStepStatus.BUSY}:
                self._stop.wait(max(0.01, self.poll_interval_sec))


def main() -> None:
    daemon = V2Daemon(build_runtime_host(user="local", messages=SQLiteMessageQueue.default()))
    previous = {}

    def _request_stop(signum: int, frame: Any) -> None:
        _ = signum, frame
        daemon.stop()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, _request_stop)
    try:
        daemon.run_forever()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
