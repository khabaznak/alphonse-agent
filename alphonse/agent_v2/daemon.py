"""Foreground v2 daemon host."""

from __future__ import annotations

import signal
import fcntl
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from alphonse.agent_v2.core.core import LoopStepStatus
from alphonse.agent_v2.core.io import project_snapshot_to_outbox
from alphonse.agent_v2.core.messages import SQLiteMessageQueue
from alphonse.agent_v2.ipc import V2DaemonClient
from alphonse.agent_v2.ipc import V2DaemonServer
from alphonse.agent_v2.ipc import default_socket_path
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
        self.daemon_id = f"daemon-{uuid4().hex[:12]}"
        if hasattr(self.runtime.queue, "lease_owner"):
            self.runtime.queue.lease_owner = self.daemon_id
        self._stop = threading.Event()
        self._processor_thread: threading.Thread | None = None
        self._lock_file: Any | None = None
        self.scheduler = ScheduledTaskWorker(
            store=self.runtime.schedule_store,
            messages=self.runtime.queue,
            worker_id=self.daemon_id,
            on_message_queued=lambda: None,
        )
        self.ipc = V2DaemonServer(self)

    def start(self) -> None:
        if self._processor_thread is not None and self._processor_thread.is_alive():
            return
        self._acquire_single_instance_lock()
        self._stop.clear()
        reclaim_expired = getattr(self.runtime.queue, "reclaim_expired", None)
        if callable(reclaim_expired):
            reclaim_expired()
        start_runtime_integrations(
            self.runtime,
            on_outbox_delivered=self._on_outbox_delivered,
            on_outbox_failed=self._on_outbox_failed,
        )
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
        self._release_single_instance_lock()

    def restart_integrations(self) -> None:
        start_runtime_integrations(
            self.runtime,
            on_outbox_delivered=self._on_outbox_delivered,
            on_outbox_failed=self._on_outbox_failed,
        )

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
        claim_next = getattr(self.runtime.queue, "claim_next", None)
        queued = claim_next(lease_owner=self.daemon_id) if callable(claim_next) else self.runtime.queue.peek()
        with self.runtime.presence_projector.processing(queued):
            step = self.runtime.core.step()
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
            projected = project_snapshot_to_outbox(snapshot=snapshot, outbox=self.runtime.outbox)
            if projected is not None:
                occurrence_key = str(projected.metadata.get("occurrence_key") or "").strip()
                if occurrence_key:
                    self.runtime.schedule_store.mark_occurrence_response_pending(
                        occurrence_key,
                        response_outbox_id=projected.outbox_message_id,
                    )
            if step.queued_message_id:
                acknowledge = getattr(self.runtime.queue, "ack", None)
                if callable(acknowledge):
                    acknowledge(step.queued_message_id, lease_owner=self.daemon_id)
        elif step.status == LoopStepStatus.FAILED and step.queued_message_id:
            retry = getattr(self.runtime.queue, "retry", None)
            if callable(retry):
                retry(
                    step.queued_message_id,
                    error=str(getattr(step, "error", "") or "capd_processing_failed"),
                    next_attempt_at=datetime.now(timezone.utc) + timedelta(seconds=5),
                    lease_owner=self.daemon_id,
                )
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

    def _acquire_single_instance_lock(self) -> None:
        lock_path = default_socket_path().with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = lock_path.open("a+")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                V2DaemonClient(default_socket_path(), timeout_sec=0.5).ping()
            except Exception as exc:
                raise RuntimeError("alphonse_v2_daemon_lock_held") from exc
            raise RuntimeError("alphonse_v2_daemon_already_running")
        self._lock_file.seek(0)
        self._lock_file.truncate()
        self._lock_file.write(self.daemon_id)
        self._lock_file.flush()

    def _release_single_instance_lock(self) -> None:
        if self._lock_file is None:
            return
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
        finally:
            self._lock_file = None

    def _on_outbox_delivered(self, outbound: Any) -> None:
        metadata = getattr(outbound, "metadata", {}) if outbound is not None else {}
        occurrence_key = str(metadata.get("occurrence_key") or "").strip() if isinstance(metadata, dict) else ""
        if occurrence_key:
            self.runtime.schedule_store.mark_occurrence_delivered(
                occurrence_key,
                response_outbox_id=str(getattr(outbound, "outbox_message_id", "") or ""),
            )

    def _on_outbox_failed(self, outbound: Any, error: str) -> None:
        metadata = getattr(outbound, "metadata", {}) if outbound is not None else {}
        occurrence_key = str(metadata.get("occurrence_key") or "").strip() if isinstance(metadata, dict) else ""
        if occurrence_key:
            self.runtime.schedule_store.mark_occurrence_failed(occurrence_key, error=error)


def main() -> None:
    daemon_id = f"daemon-{uuid4().hex[:12]}"
    daemon = V2Daemon(build_runtime_host(user="local", messages=SQLiteMessageQueue.default(lease_owner=daemon_id)))
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
