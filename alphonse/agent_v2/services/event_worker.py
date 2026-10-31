"""Bounded durable event handoff/recovery, independent of event publication."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable

from alphonse.agent_v2.automations import EventAutomationStore

logger = logging.getLogger(__name__)


class EventDispatchWorker:
    def __init__(self, *, store: EventAutomationStore, messages: Any,
                 dispatch: Callable[[dict[str, Any]], str], worker_id: str,
                 poll_interval_sec: float = 1, batch_size: int = 100,
                 lease_seconds: float = 30, max_attempts: int = 5) -> None:
        self.store = store
        self.messages = messages
        self.dispatch = dispatch
        self.worker_id = worker_id
        self.poll_interval_sec = max(.1, poll_interval_sec)
        self.batch_size = max(1, min(batch_size, 1000))
        self.lease_seconds = max(1, lease_seconds)
        self.max_attempts = max(1, max_attempts)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="alphonse-v2-events", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def run_once(self, *, now: datetime | None = None) -> list[dict[str, str]]:
        # Publication and the periodic tick may both request a bounded drain.
        with self._lock:
            return self._tick(now or datetime.now(timezone.utc))

    def _tick(self, now: datetime) -> list[dict[str, str]]:
        status_for = getattr(self.messages, "status_for", lambda _: "")
        for execution in self.store.active_executions(limit=self.batch_size):
            self.reconcile(execution["execution_id"], status_for(execution["queued_message_id"]))
        outcomes = []
        for execution in self.store.claim_event_executions(
                limit=self.batch_size, worker_id=self.worker_id,
                lease_seconds=self.lease_seconds, now=now):
            identity = execution["execution_id"]
            token = execution["lease_token"]
            try:
                # Accepted executions are replayable for 90 days of trusted store time.
                accepted = datetime.fromisoformat(execution["created_at"])
                state = status_for(f"event:{identity}")
                if not state and (now - accepted).total_seconds() > 90 * 86400:
                    self.store.cancel_execution(identity)
                    cancel = getattr(self.messages, "cancel_event", None)
                    if callable(cancel):
                        cancel(f"event:{identity}")
                    outcomes.append({"execution_id": identity, "status": "cancelled"})
                    continue
                # Queue receipts survive ack, so even a completed handoff can recover.
                if not state and execution["attempt_count"] > self.max_attempts:
                    self.store.retry_execution(identity, lease_token=token, now=now, max_attempts=self.max_attempts)
                    outcomes.append({"execution_id": identity, "status": "failed"})
                    continue
                message_id = f"event:{identity}" if state else self.dispatch(execution)
                marked = self.store.mark_execution_enqueued(identity, message_id, lease_token=token)
                if marked:
                    self.reconcile(identity, status_for(message_id))
                else:
                    current = self.store.get_execution(identity)
                    if current and current["status"] == "cancelled":
                        cancel = getattr(self.messages, "cancel_event", None)
                        if callable(cancel):
                            cancel(message_id)
                outcomes.append({"execution_id": identity, "status": "queued" if marked else "superseded"})
            except Exception:
                self.store.retry_execution(identity, lease_token=token, now=now, max_attempts=self.max_attempts)
                current = self.store.get_execution(identity)
                outcomes.append({"execution_id": identity, "status": current["status"] if current else "failed"})
        self.store.prune_history()
        return outcomes

    def reconcile(self, execution_id: str, queue_status: str) -> None:
        status = {"pending": "queued", "retry_wait": "queued", "processing": "processing",
                  "completed": "completed", "cancelled": "cancelled", "failed": "failed"}.get(queue_status)
        if status:
            self.store.execution_lifecycle(execution_id, status)

    def cancel(self, execution_id: str) -> bool:
        # Queue tombstone first prevents late insertion by a concurrent dispatcher.
        cancel = getattr(self.messages, "cancel_event", None)
        if not callable(cancel) or not cancel(f"event:{execution_id}"):
            return False
        return self.store.cancel_execution(execution_id)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:
                # Never log exception/payload text from private event dispatch.
                logger.warning("Event recovery tick failed")
            self._stop.wait(self.poll_interval_sec)
