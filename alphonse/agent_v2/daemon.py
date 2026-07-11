"""Foreground v2 daemon host."""

from __future__ import annotations

import signal
import fcntl
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from alphonse.agent_v2.core.core import LoopStepStatus
from alphonse.agent_v2.core.io import project_snapshot_to_outbox
from alphonse.agent_v2.core.io import SQLiteOutboundStore
from alphonse.agent_v2.core.messages import SQLiteMessageQueue
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.integrations import SQLiteIntegrationStore
from alphonse.agent_v2.inference_settings import SQLiteInferenceSettingsStore
from alphonse.agent_v2.ipc import V2DaemonClient
from alphonse.agent_v2.ipc import V2DaemonServer
from alphonse.agent_v2.ipc import default_socket_path
from alphonse.agent_v2.runtime import V2RuntimeHost
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.runtime import start_runtime_integrations
from alphonse.agent_v2.runtime import stop_runtime_integrations
from alphonse.agent_v2.runtime import refresh_runtime_inference
from alphonse.agent_v2.inference_settings import validate_and_save_inference_settings
from alphonse.agent_v2.services.scheduled_worker import ScheduledTaskWorker


@dataclass
class V2Daemon:
    runtime: V2RuntimeHost
    poll_interval_sec: float = 0.05
    inbound_max_attempts: int = 5

    def __post_init__(self) -> None:
        self.daemon_id = f"daemon-{uuid4().hex[:12]}"
        if hasattr(self.runtime.queue, "lease_owner"):
            self.runtime.queue.lease_owner = self.daemon_id
        self._stop = threading.Event()
        self._processor_thread: threading.Thread | None = None
        self._lock_file: Any | None = None
        self._lifecycle_lock = threading.RLock()
        self._stopped = False
        self._last_processor_error = ""
        self._active_work_lock = threading.RLock()
        self._active_work: dict[str, str] = {}
        self._activity_status: dict[str, str] = {"state": "idle", "updated_at": datetime.now(timezone.utc).isoformat()}
        self.scheduler = ScheduledTaskWorker(
            store=self.runtime.schedule_store,
            messages=self.runtime.queue,
            worker_id=self.daemon_id,
            on_message_queued=lambda: None,
        )
        self.ipc = V2DaemonServer(self)

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._stopped:
                raise RuntimeError("alphonse_v2_daemon_stopped")
            if self._processor_thread is not None and self._processor_thread.is_alive():
                return
            self._acquire_single_instance_lock()
        try:
            self._stop.clear()
            reclaim_expired = getattr(self.runtime.queue, "reclaim_expired", None)
            if callable(reclaim_expired):
                reclaim_expired()
            # Publish the health socket before optional providers initialize so
            # clients can distinguish a live daemon from a failed startup.
            self.ipc.start()
            start_runtime_integrations(
                self.runtime,
                on_outbox_delivered=self._on_outbox_delivered,
                on_outbox_failed=self._on_outbox_failed,
            )
            self.scheduler.start()
            self._processor_thread = threading.Thread(target=self._process_loop, name="alphonse-v2-core", daemon=True)
            self._processor_thread.start()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        with self._lifecycle_lock:
            if self._stopped:
                return
            self._stopped = True
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

    def update_inference_settings(self, *, provider_key: str, model_id: str) -> dict[str, str]:
        settings = validate_and_save_inference_settings(
            self.runtime.inference_settings_store,
            provider_key=provider_key,
            model_id=model_id,
        )
        refresh_runtime_inference(self.runtime, settings)
        return settings.to_dict()

    def pop_activity_events(self) -> list[dict[str, Any]]:
        events = list(self.runtime.activity_events)
        self.runtime.activity_events.clear()
        return [
            {
                "phase": event.phase.value,
                "label": event.label,
                "message": event.message,
                "speaker": event.speaker,
                "task_id": event.task_id,
                "message_id": event.message_id,
                "user": event.user,
                "integration_id": event.integration_id,
                "channel_target": event.channel_target,
            }
            for event in events
        ]

    def active_work(self) -> dict[str, str]:
        with self._active_work_lock:
            return dict(self._active_work)

    def activity_status(self) -> dict[str, str]:
        with self._active_work_lock:
            return dict(self._activity_status)

    def run_once(self) -> Any:
        queued = self.runtime.queue.peek()
        self._set_active_work(queued)
        if queued is not None:
            self._set_activity_status("working")
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
                    max_attempts=self.inbound_max_attempts,
                )
            self.runtime.core.clear_failure()
        if step.status != LoopStepStatus.BUSY:
            self._set_active_work(None)
            terminal_state = {
                LoopStepStatus.WAITING: "waiting",
                LoopStepStatus.PARKED: "waiting",
                LoopStepStatus.FAILED: "error",
            }.get(step.status, "idle")
            self._set_activity_status(terminal_state)
        return step

    def _set_activity_status(self, state: str) -> None:
        with self._active_work_lock:
            self._activity_status = {"state": state, "updated_at": datetime.now(timezone.utc).isoformat()}

    def _set_active_work(self, queued: Any | None) -> None:
        with self._active_work_lock:
            if queued is None:
                self._active_work = {}
                return
            message = getattr(queued, "message", None)
            self._active_work = {
                "message_id": str(getattr(queued, "message_id", "") or ""),
                "user": str(getattr(message, "user", "") or ""),
                "prompt": str(getattr(message, "prompt", "") or ""),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop.wait(0.5):
                pass
        finally:
            self.stop()

    def _process_loop(self) -> None:
        while not self._stop.is_set():
            try:
                step = self.run_once()
                self._last_processor_error = ""
            except Exception as exc:
                self._last_processor_error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                self._stop.wait(max(0.1, self.poll_interval_sec))
                continue
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
        with self._lifecycle_lock:
            lock_file = self._lock_file
            self._lock_file = None
        if lock_file is None:
            return
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
        finally:
            pass

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
    daemon = V2Daemon(
        build_runtime_host(
            user="local",
            messages=SQLiteMessageQueue.default(lease_owner=daemon_id),
            question_store=SQLiteQuestionStore.default(),
            project_store=ProjectStore.default(),
            schedule_store=ScheduledTaskStore.default(),
            outbox=SQLiteOutboundStore.default(),
            integration_store=SQLiteIntegrationStore.default(),
            inference_settings_store=SQLiteInferenceSettingsStore.default(),
        )
    )
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
