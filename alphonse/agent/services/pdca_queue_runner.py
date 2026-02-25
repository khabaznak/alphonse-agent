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
        interactive_boost: int | None = None,
        starvation_warn_seconds: int | None = None,
        starvation_warn_cooldown_seconds: int | None = None,
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
        self._interactive_boost = _as_int(
            interactive_boost,
            env_name="ALPHONSE_PDCA_QUEUE_INTERACTIVE_BOOST",
            default=40,
            minimum=0,
        )
        self._starvation_warn_seconds = _as_int(
            starvation_warn_seconds,
            env_name="ALPHONSE_PDCA_QUEUE_STARVATION_WARN_SECONDS",
            default=300,
            minimum=1,
        )
        self._starvation_warn_cooldown_seconds = _as_int(
            starvation_warn_cooldown_seconds,
            env_name="ALPHONSE_PDCA_QUEUE_STARVATION_WARN_COOLDOWN_SECONDS",
            default=60,
            minimum=1,
        )
        self._worker_id = str(worker_id or os.getenv("ALPHONSE_PDCA_QUEUE_WORKER_ID") or "pdca-queue-runner").strip()
        self._enabled = _is_enabled(enabled)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_owner_id: str | None = None
        self._last_starvation_warning_at: datetime | None = None

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
        candidates = self._fair_order_candidates(list_runnable_pdca_tasks(now=now_text, limit=20))
        self._emit_queue_health(candidates=candidates, now_dt=now_dt)
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
                    self._last_owner_id = str(task.get("owner_id") or "").strip() or None
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

    def _fair_order_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        scored = sorted(
            candidates,
            key=lambda item: (
                -self._effective_priority(item),
                _iso_for_order(item.get("next_run_at")),
                _iso_for_order(item.get("updated_at")),
            ),
        )
        top_priority = max(self._effective_priority(item) for item in scored)
        top_bucket = [item for item in scored if self._effective_priority(item) == top_priority]
        tail = [item for item in scored if self._effective_priority(item) != top_priority]
        if not self._last_owner_id:
            return top_bucket + tail
        for index, item in enumerate(top_bucket):
            owner_id = str(item.get("owner_id") or "").strip()
            if owner_id and owner_id != self._last_owner_id:
                preferred = top_bucket[index]
                remaining = top_bucket[:index] + top_bucket[index + 1 :]
                return [preferred, *remaining, *tail]
        return top_bucket + tail

    def _effective_priority(self, task: dict[str, Any]) -> int:
        base = _task_priority(task)
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        task_class = str(metadata.get("task_class") or metadata.get("kind") or "").strip().lower()
        interactive = bool(metadata.get("interactive"))
        if task_class == "interactive" or interactive:
            return base + self._interactive_boost
        return base

    def _emit_queue_health(self, *, candidates: list[dict[str, Any]], now_dt: datetime) -> None:
        if not candidates:
            return
        oldest = max(candidates, key=lambda item: _wait_seconds(item=item, now_dt=now_dt))
        oldest_wait = _wait_seconds(item=oldest, now_dt=now_dt)
        if oldest_wait < self._starvation_warn_seconds:
            return
        if self._last_starvation_warning_at is not None:
            cooldown = timedelta(seconds=self._starvation_warn_cooldown_seconds)
            if (now_dt - self._last_starvation_warning_at) < cooldown:
                return
        task_id = str(oldest.get("task_id") or "").strip()
        if not task_id:
            return
        self._last_starvation_warning_at = now_dt
        append_pdca_event(
            task_id=task_id,
            event_type="queue.starvation_warning",
            payload={
                "worker_id": self._worker_id,
                "queue_depth": len(candidates),
                "oldest_wait_seconds": oldest_wait,
                "warn_threshold_seconds": self._starvation_warn_seconds,
                "owner_id": str(oldest.get("owner_id") or "").strip() or None,
            },
            correlation_id=f"pdca.queue.starvation:{task_id}:{int(now_dt.timestamp())}",
        )
        logger.warning(
            "PDCA queue starvation warning worker_id=%s task_id=%s wait_seconds=%s depth=%s",
            self._worker_id,
            task_id,
            oldest_wait,
            len(candidates),
        )


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


def _task_priority(task: dict[str, Any]) -> int:
    try:
        return int(task.get("priority") or 0)
    except (TypeError, ValueError):
        return 0


def _parse_iso_utc(value: Any) -> datetime | None:
    rendered = str(value or "").strip()
    if not rendered:
        return None
    try:
        parsed = datetime.fromisoformat(rendered)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _wait_seconds(*, item: dict[str, Any], now_dt: datetime) -> int:
    baseline = _parse_iso_utc(item.get("next_run_at")) or _parse_iso_utc(item.get("created_at"))
    if baseline is None:
        return 0
    return max(int((now_dt - baseline).total_seconds()), 0)


def _iso_for_order(value: Any) -> str:
    parsed = _parse_iso_utc(value)
    if parsed is None:
        return "9999-12-31T23:59:59+00:00"
    return parsed.isoformat()
