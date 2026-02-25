from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from alphonse.agent.nervous_system.pdca_queue_store import (
    acquire_pdca_task_lease,
    append_pdca_event,
    list_runnable_pdca_tasks,
    release_pdca_task_lease,
    upsert_pdca_task,
)
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.agent.observability.log_manager import get_component_logger

logger = get_component_logger("services.pdca_queue_runner")


class PdcaQueueRunner:
    def __init__(
        self,
        *,
        bus: Bus,
        poll_seconds: float | None = None,
        lease_seconds: int | None = None,
        dispatch_cooldown_seconds: int | None = None,
        worker_id: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._bus = bus
        self._poll_seconds = _as_float(poll_seconds, env_name="ALPHONSE_PDCA_QUEUE_POLL_SECONDS", default=5.0, minimum=0.2)
        self._lease_seconds = _as_int(lease_seconds, env_name="ALPHONSE_PDCA_QUEUE_LEASE_SECONDS", default=30, minimum=1)
        self._dispatch_cooldown_seconds = _as_int(
            dispatch_cooldown_seconds,
            env_name="ALPHONSE_PDCA_QUEUE_DISPATCH_COOLDOWN_SECONDS",
            default=30,
            minimum=1,
        )
        self._worker_id = str(worker_id or os.getenv("ALPHONSE_PDCA_QUEUE_WORKER_ID") or "pdca-queue-runner").strip()
        self._enabled = _is_enabled(enabled)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        if not self._enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(
            "PDCA queue runner started worker_id=%s poll_seconds=%.2f lease_seconds=%s cooldown_seconds=%s",
            self._worker_id,
            self._poll_seconds,
            self._lease_seconds,
            self._dispatch_cooldown_seconds,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("PDCA queue runner stopped worker_id=%s", self._worker_id)

    def run_once(self, *, now: str | None = None) -> int:
        if not self._enabled:
            return 0
        now_dt = _parse_or_now(now)
        now_text = now_dt.isoformat()
        candidates = list_runnable_pdca_tasks(now=now_text, limit=20)
        for task in candidates:
            task_id = str(task.get("task_id") or "").strip()
            if not task_id:
                continue
            acquired = acquire_pdca_task_lease(
                task_id=task_id,
                worker_id=self._worker_id,
                lease_seconds=self._lease_seconds,
                now=now_text,
            )
            if not acquired:
                continue
            try:
                emitted = self._emit_slice_requested(task=task, now_dt=now_dt)
                if emitted:
                    return 1
            finally:
                release_pdca_task_lease(task_id=task_id, worker_id=self._worker_id)
        return 0

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                _ = self.run_once()
            except Exception as exc:
                logger.warning("PDCA queue runner iteration failed: %s", exc)
            self._stop_event.wait(self._poll_seconds)

    def _emit_slice_requested(self, *, task: dict[str, Any], now_dt: datetime) -> bool:
        task_id = str(task.get("task_id") or "").strip()
        owner_id = str(task.get("owner_id") or "").strip()
        conversation_key = str(task.get("conversation_key") or "").strip()
        if not task_id or not owner_id or not conversation_key:
            return False
        correlation_id = f"pdca.slice.requested:{task_id}:{int(now_dt.timestamp())}"
        self._bus.emit(
            Signal(
                type="pdca.slice.requested",
                payload={
                    "task_id": task_id,
                    "owner_id": owner_id,
                    "conversation_key": conversation_key,
                    "session_id": task.get("session_id"),
                    "correlation_id": correlation_id,
                },
                source="pdca_queue_runner",
                correlation_id=correlation_id,
            )
        )
        append_pdca_event(
            task_id=task_id,
            event_type="slice.requested",
            payload={"worker_id": self._worker_id, "scheduled_for": now_dt.isoformat()},
            correlation_id=correlation_id,
        )
        next_run = now_dt + timedelta(seconds=self._dispatch_cooldown_seconds)
        upsert_pdca_task(
            {
                "task_id": task_id,
                "owner_id": owner_id,
                "conversation_key": conversation_key,
                "session_id": task.get("session_id"),
                "status": "running",
                "priority": task.get("priority", 100),
                "next_run_at": next_run.isoformat(),
                "slice_cycles": task.get("slice_cycles", 5),
                "max_cycles": task.get("max_cycles"),
                "max_runtime_seconds": task.get("max_runtime_seconds"),
                "token_budget_remaining": task.get("token_budget_remaining"),
                "failure_streak": task.get("failure_streak", 0),
                "last_error": task.get("last_error"),
                "metadata": task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
                "created_at": task.get("created_at"),
            }
        )
        logger.info(
            "PDCA queue runner emitted slice request task_id=%s owner_id=%s conversation_key=%s",
            task_id,
            owner_id,
            conversation_key,
        )
        return True


def _is_enabled(value: bool | None) -> bool:
    if isinstance(value, bool):
        return value
    raw = str(os.getenv("ALPHONSE_PDCA_SLICING_ENABLED", "false")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _as_float(value: float | None, *, env_name: str, default: float, minimum: float) -> float:
    raw: Any = value
    if raw is None:
        raw = os.getenv(env_name)
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        parsed = float(default)
    return max(float(minimum), parsed)


def _as_int(value: int | None, *, env_name: str, default: int, minimum: int) -> int:
    raw: Any = value
    if raw is None:
        raw = os.getenv(env_name)
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(int(minimum), parsed)


def _parse_or_now(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
