"""Durable scheduled-task polling service for the v2 runtime host."""

from __future__ import annotations

import threading
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskRunner
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledOccurrence


@dataclass(frozen=True)
class ScheduledWorkerStats:
    ticks: int = 0
    queued: int = 0
    errors: int = 0
    claimed: int = 0
    retried: int = 0


class ScheduledTaskWorker:
    """Polls durable schedules and wakes the host after dispatching due work."""

    def __init__(
        self,
        *,
        store: ScheduledTaskStore,
        messages: Any,
        on_message_queued: Callable[[], None] | None = None,
        poll_interval_sec: float = 1.0,
        lease_seconds: float = 30.0,
        max_attempts: int = 5,
        worker_id: str | None = None,
        on_failure: Callable[[ScheduledOccurrence, str], None] | None = None,
        on_direct_delivery: Callable[[ScheduledOccurrence], str] | None = None,
    ) -> None:
        self.store = store
        self.messages = messages
        self.runner = ScheduledTaskRunner(store=store, messages=messages)
        self.on_message_queued = on_message_queued
        self.poll_interval_sec = max(0.1, float(poll_interval_sec))
        self.lease_seconds = max(1.0, float(lease_seconds))
        self.max_attempts = max(1, int(max_attempts))
        self.worker_id = str(worker_id or f"scheduler-{uuid.uuid4().hex[:12]}")
        self.on_failure = on_failure
        self.on_direct_delivery = on_direct_delivery
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
        current = now or datetime.now(timezone.utc)
        occurrences = self.store.claim_due_occurrences(
            worker_id=self.worker_id,
            now=current,
            lease_seconds=self.lease_seconds,
        )
        occurrences.extend(
            self.store.claim_expired_occurrences(
                worker_id=self.worker_id,
                now=current,
                lease_seconds=self.lease_seconds,
            )
        )
        occurrences.extend(
            self.store.claim_retry_occurrences(
                worker_id=self.worker_id,
                now=current,
                lease_seconds=self.lease_seconds,
            )
        )
        outcomes = [self._dispatch(occurrence, current) for occurrence in occurrences]
        queued = sum(1 for outcome in outcomes if outcome.get("status") == "queued")
        errors = sum(1 for outcome in outcomes if outcome.get("status") == "error")
        retried = sum(1 for outcome in outcomes if outcome.get("status") == "retry_wait")
        self._stats = ScheduledWorkerStats(
            ticks=self._stats.ticks + 1,
            queued=self._stats.queued + queued,
            errors=self._stats.errors + errors,
            claimed=self._stats.claimed + len(occurrences),
            retried=self._stats.retried + retried,
        )
        if queued and self.on_message_queued is not None:
            try:
                self.on_message_queued()
            except Exception:
                pass
        return outcomes

    def _dispatch(self, occurrence: ScheduledOccurrence, now: datetime) -> dict[str, Any]:
        try:
            if occurrence.task.delivery_mode == "direct":
                if self.on_direct_delivery is None:
                    raise RuntimeError("scheduled_direct_delivery_unavailable")
                outbox_message_id = str(self.on_direct_delivery(occurrence) or "").strip()
                if not outbox_message_id:
                    raise RuntimeError("scheduled_direct_delivery_unavailable")
                self.store.mark_occurrence_enqueued(
                    occurrence.occurrence_key,
                    worker_id=self.worker_id,
                    message_id=f"direct:{occurrence.occurrence_key}",
                )
                self.store.mark_occurrence_response_pending(
                    occurrence.occurrence_key,
                    response_outbox_id=outbox_message_id,
                )
                self.store.update_after_run(occurrence.task, now=now)
                return {
                    "scheduled_task_id": occurrence.task.scheduled_task_id,
                    "project_id": occurrence.task.project_id,
                    "occurrence_key": occurrence.occurrence_key,
                    "status": "delivered",
                    "response_outbox_id": outbox_message_id,
                }
            metadata: dict[str, Any] = {
                "source": "scheduled_task",
                "scheduled_task_id": occurrence.task.scheduled_task_id,
                "project_id": occurrence.task.project_id,
                "scheduled_run_id": occurrence.run_id,
                "occurrence_key": occurrence.occurrence_key,
            }
            if occurrence.task.origin_channel:
                metadata["channel"] = dict(occurrence.task.origin_channel)
            message_id = f"scheduled:{occurrence.occurrence_key}"
            queued = self.runner.channel.queue_message(
                prompt=occurrence.task.prompt,
                user=occurrence.task.owner_user_id,
                project_id=occurrence.task.project_id,
                metadata=metadata,
                message_id=message_id,
            )
            self.store.mark_occurrence_enqueued(
                occurrence.occurrence_key,
                worker_id=self.worker_id,
                message_id=queued.message_id,
            )
            self.store.update_after_run(occurrence.task, now=now)
            return {
                "scheduled_task_id": occurrence.task.scheduled_task_id,
                "project_id": occurrence.task.project_id,
                "occurrence_key": occurrence.occurrence_key,
                "status": "queued",
                "queued_message_id": queued.message_id,
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if occurrence.attempt_count >= self.max_attempts:
                self.store.mark_occurrence_failed(occurrence.occurrence_key, error=error)
                if self.on_failure is not None:
                    self.on_failure(occurrence, error)
                return {"occurrence_key": occurrence.occurrence_key, "status": "error", "error": error}
            delay = min(3600.0, 5.0 * (2 ** max(0, occurrence.attempt_count - 1)))
            delay += random.uniform(0.0, min(5.0, delay * 0.1))
            next_attempt = (now + timedelta(seconds=delay)).isoformat()
            self.store.mark_occurrence_retry(
                occurrence.occurrence_key,
                worker_id=self.worker_id,
                error=error,
                next_attempt_at=next_attempt,
            )
            return {
                "occurrence_key": occurrence.occurrence_key,
                "status": "retry_wait",
                "error": error,
                "next_attempt_at": next_attempt,
            }

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
