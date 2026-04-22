from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path

_ACTIVE_TASK_STATUSES = {"queued", "running", "waiting_user", "paused"}
_ALL_TASK_STATUSES = _ACTIVE_TASK_STATUSES | {"done", "failed", "cancelled"}
_CANONICAL_METADATA_KEYS = {
    "initial_message_id",
    "initial_correlation_id",
    "input_dirty",
    "next_unconsumed_index",
    "inputs",
    "state",
    "pending_user_text",
    "last_user_message",
    "last_user_channel",
    "last_user_target",
    "last_user_message_id",
    "last_enqueue_correlation_id",
    "last_enqueued_at",
    "last_input_dequeued_at",
    "invoke_inflight",
    "invoke_inflight_started_at",
    "invoke_inflight_correlation_id",
    "task_class",
    "interactive",
    "started_at",
}
_RUNTIMES: dict[str, "PdcaRuntime"] = {}
_RUNTIMES_LOCK = threading.RLock()


@dataclass
class PdcaRuntimeEvent:
    event_id: str
    task_id: str
    event_type: str
    payload: dict[str, Any]
    correlation_id: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "task_id": self.task_id,
            "event_type": self.event_type,
            "payload": dict(self.payload),
            "correlation_id": self.correlation_id,
            "created_at": self.created_at,
        }


@dataclass
class PdcaCheckpoint:
    state: dict[str, Any]
    version: int = 1
    created_at: str = field(default_factory=lambda: _now_iso())
    updated_at: str = field(default_factory=lambda: _now_iso())

    def to_dict(self, *, task_id: str) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "state": dict(self.state),
            "version": int(self.version),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class ActivePdcaTask:
    task_id: str
    owner_id: str
    conversation_key: str
    session_id: str | None
    status: str
    priority: int
    next_run_at: str | None
    lease_until: str | None
    worker_id: str | None
    slice_cycles: int
    max_cycles: int | None
    max_runtime_seconds: int | None
    token_budget_remaining: int | None
    failure_streak: int
    last_error: str | None
    created_at: str
    updated_at: str
    state: dict[str, Any] = field(default_factory=dict)
    buffered_inputs: list[dict[str, Any]] = field(default_factory=list)
    next_unconsumed_index: int = 0
    input_dirty: bool = False
    last_user_message: str | None = None
    pending_user_text: str | None = None
    last_user_channel: str | None = None
    last_user_target: str | None = None
    last_user_message_id: str | None = None
    last_enqueue_correlation_id: str | None = None
    last_enqueued_at: str | None = None
    last_input_dequeued_at: str | None = None
    initial_message_id: str | None = None
    initial_correlation_id: str | None = None
    invoke_inflight: bool = False
    invoke_inflight_started_at: str | None = None
    invoke_inflight_correlation_id: str | None = None
    task_class: str | None = None
    interactive: bool = False
    started_at: str | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)
    checkpoint: PdcaCheckpoint | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "owner_id": self.owner_id,
            "conversation_key": self.conversation_key,
            "session_id": self.session_id,
            "status": self.status,
            "priority": self.priority,
            "next_run_at": self.next_run_at,
            "lease_until": self.lease_until,
            "worker_id": self.worker_id,
            "slice_cycles": self.slice_cycles,
            "max_cycles": self.max_cycles,
            "max_runtime_seconds": self.max_runtime_seconds,
            "token_budget_remaining": self.token_budget_remaining,
            "failure_streak": self.failure_streak,
            "last_error": self.last_error,
            "metadata": self._metadata_snapshot(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def _metadata_snapshot(self) -> dict[str, Any]:
        metadata = dict(self.extra_metadata)
        metadata["inputs"] = [dict(item) for item in self.buffered_inputs]
        metadata["next_unconsumed_index"] = int(self.next_unconsumed_index)
        metadata["input_dirty"] = bool(self.input_dirty)
        metadata["state"] = dict(self.state)
        for key, value in (
            ("pending_user_text", self.pending_user_text),
            ("last_user_message", self.last_user_message),
            ("last_user_channel", self.last_user_channel),
            ("last_user_target", self.last_user_target),
            ("last_user_message_id", self.last_user_message_id),
            ("last_enqueue_correlation_id", self.last_enqueue_correlation_id),
            ("last_enqueued_at", self.last_enqueued_at),
            ("last_input_dequeued_at", self.last_input_dequeued_at),
            ("initial_message_id", self.initial_message_id),
            ("initial_correlation_id", self.initial_correlation_id),
            ("invoke_inflight_started_at", self.invoke_inflight_started_at),
            ("invoke_inflight_correlation_id", self.invoke_inflight_correlation_id),
            ("task_class", self.task_class),
            ("started_at", self.started_at),
        ):
            if value is not None:
                metadata[key] = value
        if self.invoke_inflight:
            metadata["invoke_inflight"] = True
        else:
            metadata.pop("invoke_inflight", None)
        if self.interactive:
            metadata["interactive"] = True
        return metadata


class PdcaRuntime:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tasks: dict[str, ActivePdcaTask] = {}
        self._events_by_task: dict[str, list[PdcaRuntimeEvent]] = {}

    def upsert_task(self, record: dict[str, Any]) -> str:
        task_id = str(record.get("task_id") or uuid.uuid4())
        now = _now_iso()
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                task = ActivePdcaTask(
                    task_id=task_id,
                    owner_id=str(record.get("owner_id") or "").strip(),
                    conversation_key=str(record.get("conversation_key") or "").strip(),
                    session_id=_none_if_blank(record.get("session_id")),
                    status=_normalize_status(str(record.get("status") or "queued")),
                    priority=_as_int(record.get("priority"), default=100, minimum=0),
                    next_run_at=_none_if_blank(record.get("next_run_at")),
                    lease_until=_none_if_blank(record.get("lease_until")),
                    worker_id=_none_if_blank(record.get("worker_id")),
                    slice_cycles=_as_int(record.get("slice_cycles"), default=5, minimum=1),
                    max_cycles=_as_optional_int(record.get("max_cycles")),
                    max_runtime_seconds=_as_optional_int(record.get("max_runtime_seconds")),
                    token_budget_remaining=_as_optional_int(record.get("token_budget_remaining")),
                    failure_streak=_as_int(record.get("failure_streak"), default=0, minimum=0),
                    last_error=_none_if_blank(record.get("last_error")),
                    created_at=str(record.get("created_at") or now),
                    updated_at=now,
                )
                self._tasks[task_id] = task
            else:
                task.owner_id = str(record.get("owner_id") or task.owner_id or "").strip()
                task.conversation_key = str(record.get("conversation_key") or task.conversation_key or "").strip()
                task.session_id = _none_if_blank(record.get("session_id")) if "session_id" in record else task.session_id
                task.status = _normalize_status(str(record.get("status") or task.status or "queued"))
                task.priority = _as_int(record.get("priority"), default=task.priority, minimum=0)
                task.next_run_at = _none_if_blank(record.get("next_run_at")) if "next_run_at" in record else task.next_run_at
                task.lease_until = _none_if_blank(record.get("lease_until")) if "lease_until" in record else task.lease_until
                task.worker_id = _none_if_blank(record.get("worker_id")) if "worker_id" in record else task.worker_id
                task.slice_cycles = _as_int(record.get("slice_cycles"), default=task.slice_cycles, minimum=1)
                task.max_cycles = _as_optional_int(record.get("max_cycles")) if "max_cycles" in record else task.max_cycles
                task.max_runtime_seconds = _as_optional_int(record.get("max_runtime_seconds")) if "max_runtime_seconds" in record else task.max_runtime_seconds
                task.token_budget_remaining = _as_optional_int(record.get("token_budget_remaining")) if "token_budget_remaining" in record else task.token_budget_remaining
                task.failure_streak = _as_int(record.get("failure_streak"), default=task.failure_streak, minimum=0)
                task.last_error = _none_if_blank(record.get("last_error")) if "last_error" in record else task.last_error
                task.updated_at = now
            if isinstance(record.get("metadata"), dict):
                self._apply_metadata(task, record["metadata"])
            return task_id

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(str(task_id or "").strip())
            return task.snapshot() if task is not None else None

    def get_latest_task_for_conversation(self, *, conversation_key: str, statuses: list[str] | None = None) -> dict[str, Any] | None:
        key = str(conversation_key or "").strip()
        if not key:
            return None
        allowed = {str(item or "").strip().lower() for item in (statuses or list(_ACTIVE_TASK_STATUSES))}
        with self._lock:
            candidates = [
                task for task in self._tasks.values()
                if task.conversation_key == key and task.status in allowed
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda item: item.updated_at, reverse=True)
            return candidates[0].snapshot()

    def get_latest_task_for_owner(self, *, owner_id: str, statuses: list[str] | None = None) -> dict[str, Any] | None:
        owner = str(owner_id or "").strip()
        if not owner:
            return None
        allowed = {str(item or "").strip().lower() for item in (statuses or list(_ACTIVE_TASK_STATUSES))}
        with self._lock:
            candidates = [task for task in self._tasks.values() if task.owner_id == owner and task.status in allowed]
            if not candidates:
                return None
            candidates.sort(key=lambda item: item.updated_at, reverse=True)
            return candidates[0].snapshot()

    def list_runnable(self, *, now: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        now_dt = _parse_or_now(now)
        safe_limit = _as_int(limit, default=20, minimum=1)
        with self._lock:
            tasks = []
            for task in self._tasks.values():
                if task.status not in {"queued", "running"}:
                    continue
                next_run = _parse_utc(task.next_run_at)
                if next_run is not None and next_run > now_dt:
                    continue
                lease_until = _parse_utc(task.lease_until)
                if lease_until is not None and lease_until > now_dt:
                    continue
                tasks.append(task)
            tasks.sort(
                key=lambda item: (
                    -item.priority,
                    _iso_for_order(item.next_run_at or item.created_at),
                    _iso_for_order(item.updated_at),
                )
            )
            return [task.snapshot() for task in tasks[:safe_limit]]

    def update_task_status(self, *, task_id: str, status: str, last_error: str | None = None) -> bool:
        with self._lock:
            task = self._tasks.get(str(task_id or "").strip())
            if task is None:
                return False
            task.status = _normalize_status(status)
            if last_error is not None:
                task.last_error = _none_if_blank(last_error)
            task.updated_at = _now_iso()
            return True

    def acquire_lease(self, *, task_id: str, worker_id: str, lease_seconds: int = 30, now: str | None = None) -> bool:
        task_key = str(task_id or "").strip()
        worker = str(worker_id or "").strip()
        if not task_key or not worker:
            return False
        now_dt = _parse_or_now(now)
        with self._lock:
            task = self._tasks.get(task_key)
            if task is None or task.status not in {"queued", "running"}:
                return False
            lease_until = _parse_utc(task.lease_until)
            if lease_until is not None and lease_until > now_dt:
                return False
            task.lease_until = (now_dt + timedelta(seconds=max(int(lease_seconds or 30), 1))).isoformat()
            task.worker_id = worker
            if task.status == "queued":
                task.status = "running"
            task.updated_at = now_dt.isoformat()
            return True

    def release_lease(self, *, task_id: str, worker_id: str) -> bool:
        task_key = str(task_id or "").strip()
        worker = str(worker_id or "").strip()
        if not task_key or not worker:
            return False
        with self._lock:
            task = self._tasks.get(task_key)
            if task is None or str(task.worker_id or "").strip() != worker:
                return False
            task.lease_until = None
            task.worker_id = None
            task.updated_at = _now_iso()
            return True

    def update_task_metadata(self, *, task_id: str, metadata: dict[str, Any]) -> bool:
        with self._lock:
            task = self._tasks.get(str(task_id or "").strip())
            if task is None:
                return False
            self._apply_metadata(task, metadata if isinstance(metadata, dict) else {})
            task.updated_at = _now_iso()
            return True

    def save_checkpoint(self, *, task_id: str, state: dict[str, Any], expected_version: int | None = None) -> int | None:
        normalized_state = _normalize_checkpoint_state(state or {})
        with self._lock:
            task = self._tasks.get(str(task_id or "").strip())
            if task is None:
                return None
            now = _now_iso()
            current = task.checkpoint
            if expected_version is None:
                version = int(current.version if current is not None else 0) + 1
                created_at = current.created_at if current is not None else now
                task.checkpoint = PdcaCheckpoint(state=normalized_state, version=version, created_at=created_at, updated_at=now)
                return version
            if int(expected_version) == 0:
                if current is not None:
                    return None
                task.checkpoint = PdcaCheckpoint(state=normalized_state, version=1, created_at=now, updated_at=now)
                return 1
            if current is None or int(current.version) != int(expected_version):
                return None
            task.checkpoint = PdcaCheckpoint(
                state=normalized_state,
                version=int(current.version) + 1,
                created_at=current.created_at,
                updated_at=now,
            )
            return task.checkpoint.version

    def load_checkpoint(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(str(task_id or "").strip())
            if task is None or task.checkpoint is None:
                return None
            return task.checkpoint.to_dict(task_id=task.task_id)

    def append_event(self, *, task_id: str, event_type: str, payload: dict[str, Any] | None = None, correlation_id: str | None = None) -> str:
        event = PdcaRuntimeEvent(
            event_id=str(uuid.uuid4()),
            task_id=str(task_id or "").strip(),
            event_type=str(event_type or "").strip(),
            payload=_json_safe_dict(payload or {}),
            correlation_id=_none_if_blank(correlation_id),
            created_at=_now_iso(),
        )
        with self._lock:
            self._events_by_task.setdefault(event.task_id, []).append(event)
        return event.event_id

    def list_events(self, *, task_id: str, limit: int = 100) -> list[dict[str, Any]]:
        safe_limit = _as_int(limit, default=100, minimum=1)
        with self._lock:
            events = list(self._events_by_task.get(str(task_id or "").strip(), []))
            return [item.to_dict() for item in events[:safe_limit]]

    def has_event(self, *, task_id: str, event_type: str, correlation_id: str | None = None) -> bool:
        task_key = str(task_id or "").strip()
        event_key = str(event_type or "").strip()
        corr = _none_if_blank(correlation_id)
        with self._lock:
            for item in self._events_by_task.get(task_key, []):
                if item.event_type != event_key:
                    continue
                if corr is not None and item.correlation_id != corr:
                    continue
                return True
        return False

    def clear(self) -> dict[str, int]:
        with self._lock:
            counts = self.describe_counts(include_signal_queue=True)
            self._tasks.clear()
            self._events_by_task.clear()
            return counts

    def describe_counts(self, *, include_signal_queue: bool = True) -> dict[str, int]:
        with self._lock:
            counts = {
                "pdca_tasks": len(self._tasks),
                "pdca_checkpoints": sum(1 for task in self._tasks.values() if task.checkpoint is not None),
                "pdca_events": sum(len(items) for items in self._events_by_task.values()),
            }
            if include_signal_queue:
                counts["signal_queue"] = 0
            return counts

    def get_metrics(self, *, now: str | None = None, lookback_minutes: int = 15, top_owners: int = 5) -> dict[str, Any]:
        now_dt = _parse_or_now(now)
        window_minutes = max(int(lookback_minutes or 15), 1)
        top_limit = max(int(top_owners or 5), 1)
        window_start = now_dt - timedelta(minutes=window_minutes)
        with self._lock:
            tasks = list(self._tasks.values())
            events = [item for items in self._events_by_task.values() for item in items]
        by_status: dict[str, int] = {}
        for task in tasks:
            by_status[task.status] = by_status.get(task.status, 0) + 1
        queue_by_owner: dict[str, int] = {}
        for task in tasks:
            if task.status not in {"queued", "running"}:
                continue
            queue_by_owner[task.owner_id] = queue_by_owner.get(task.owner_id, 0) + 1
        top_queue_by_owner = dict(sorted(queue_by_owner.items(), key=lambda item: (-item[1], item[0]))[:top_limit])
        wait_by_owner: dict[str, int] = {}
        oldest_wait = 0
        for task in tasks:
            if task.status not in {"queued", "running"}:
                continue
            wait_seconds = _wait_seconds(task=task, now_dt=now_dt)
            oldest_wait = max(oldest_wait, wait_seconds)
            owner = task.owner_id
            wait_by_owner[owner] = max(wait_by_owner.get(owner, 0), wait_seconds)
        dispatch_counts_by_owner: dict[str, int] = {}
        dispatch_count = 0
        budget_by_reason: dict[str, int] = {}
        starvation_warnings = 0
        terminal_outcomes = {"done": 0, "waiting_user": 0, "failed": 0}
        for event in events:
            created = _parse_utc(event.created_at)
            if created is None or created < window_start:
                continue
            if event.event_type == "slice.requested":
                dispatch_count += 1
                owner = self._owner_for_task(event.task_id)
                if owner:
                    dispatch_counts_by_owner[owner] = dispatch_counts_by_owner.get(owner, 0) + 1
            elif event.event_type == "slice.blocked.budget_exhausted":
                reason = str(event.payload.get("reason") or "unknown")
                budget_by_reason[reason] = budget_by_reason.get(reason, 0) + 1
                terminal_outcomes["failed"] += 1
            elif event.event_type == "queue.starvation_warning":
                starvation_warnings += 1
            elif event.event_type == "slice.completed.done":
                terminal_outcomes["done"] += 1
            elif event.event_type == "slice.blocked.missing_text":
                terminal_outcomes["waiting_user"] += 1
            elif event.event_type == "slice.failed":
                terminal_outcomes["failed"] += 1
        fairness = {
            owner: (
                float(dispatch_counts_by_owner.get(owner, 0)) / float(queue_by_owner.get(owner, 1))
                if queue_by_owner.get(owner, 0) > 0
                else 0.0
            )
            for owner in sorted(queue_by_owner)
        }
        return {
            "generated_at": now_dt.isoformat(),
            "lookback_minutes": window_minutes,
            "queue_depth_total": sum(by_status.values()),
            "queue_depth_by_status": by_status,
            "queue_depth_by_owner": top_queue_by_owner,
            "oldest_wait_seconds": {"global": oldest_wait, "by_owner": wait_by_owner},
            "dispatch_rate_per_minute": {
                "global": dispatch_count / float(window_minutes),
                "by_owner": {owner: count / float(window_minutes) for owner, count in dispatch_counts_by_owner.items()},
            },
            "budget_exhaustions_total": {"all": sum(budget_by_reason.values()), "by_reason": budget_by_reason},
            "starvation_warnings_total": starvation_warnings,
            "owner_fairness_ratio": {
                "by_owner": fairness,
                "max_skew": max(fairness.values(), default=0.0),
            },
            "terminal_outcomes_total": terminal_outcomes,
        }

    def mark_inflight(self, *, task_id: str, correlation_id: str | None = None) -> bool:
        with self._lock:
            task = self._tasks.get(str(task_id or "").strip())
            if task is None:
                return False
            task.invoke_inflight = True
            task.invoke_inflight_started_at = _now_iso()
            task.invoke_inflight_correlation_id = _none_if_blank(correlation_id)
            task.updated_at = _now_iso()
            return True

    def clear_inflight(self, *, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(str(task_id or "").strip())
            if task is None:
                return False
            task.invoke_inflight = False
            task.invoke_inflight_started_at = None
            task.invoke_inflight_correlation_id = None
            task.updated_at = _now_iso()
            return True

    def consume_inputs_for_check(self, *, task_id: str) -> list[dict[str, Any]]:
        with self._lock:
            task = self._tasks.get(str(task_id or "").strip())
            if task is None:
                return []
            inputs = [dict(item) for item in task.buffered_inputs]
            if not inputs:
                return []
            next_unconsumed = max(0, min(int(task.next_unconsumed_index or 0), len(inputs)))
            consumed: list[dict[str, Any]] = []
            scanned_any = False
            now = _now_iso()
            for idx in range(next_unconsumed, len(inputs)):
                scanned_any = True
                record = inputs[idx]
                message_id = str(record.get("message_id") or "").strip()
                record_correlation_id = str(record.get("correlation_id") or "").strip()
                if _is_bootstrap_record(
                    message_id=message_id,
                    correlation_id=record_correlation_id,
                    initial_message_id=str(task.initial_message_id or "").strip(),
                    initial_correlation_id=str(task.initial_correlation_id or "").strip(),
                ):
                    continue
                text = str(record.get("text") or "").strip()
                attachments = [dict(item) for item in record.get("attachments", []) if isinstance(item, dict)]
                if not text and not attachments:
                    continue
                consumed.append(
                    {
                        "text": text,
                        "actor_id": str(record.get("actor_id") or "").strip() or None,
                        "message_id": message_id or None,
                        "correlation_id": record_correlation_id or None,
                        "channel": str(record.get("channel") or "").strip() or None,
                        "attachments": attachments,
                        "received_at": str(record.get("received_at") or "").strip() or None,
                        "consumed_at": now,
                    }
                )
                record["consumed_at"] = now
                inputs[idx] = record
            if not consumed and not scanned_any:
                return []
            task.buffered_inputs = inputs
            task.next_unconsumed_index = len(inputs)
            task.input_dirty = False
            if consumed:
                latest_text = str(consumed[-1].get("text") or "").strip()
                task.last_user_message = latest_text
                task.pending_user_text = latest_text
                task.last_input_dequeued_at = now
            task.updated_at = now
            return consumed

    def create_or_append(
        self,
        *,
        task_record: TaskRecord,
        buffered_input: Any,
        force_new_task: bool = False,
    ) -> str:
        from alphonse.agent.services.pdca_ingress import normalize_buffered_attachments

        now = str(buffered_input.normalized_received_at()).strip()
        session_key = f"{str(buffered_input.channel_type or '').strip()}:{str(buffered_input.channel_target or '').strip()}"
        attachments = normalize_buffered_attachments(getattr(buffered_input, "attachments", []))
        user_text = str(buffered_input.normalized_text() or "").strip()
        with self._lock:
            task: ActivePdcaTask | None = None
            if not force_new_task:
                for candidate in sorted(self._tasks.values(), key=lambda item: item.updated_at, reverse=True):
                    if candidate.conversation_key != session_key:
                        continue
                    if candidate.owner_id != str(task_record.user_id or "").strip():
                        continue
                    if candidate.status not in {"running", "waiting_user"}:
                        continue
                    task = candidate
                    break
            if task is None:
                task_id = self.upsert_task(
                    {
                        "owner_id": task_record.user_id,
                        "conversation_key": session_key,
                        "session_id": None,
                        "status": "queued",
                        "priority": 100,
                        "next_run_at": now,
                        "slice_cycles": 5,
                    }
                )
                task = self._tasks[task_id]
                task_record.task_id = task_id
                task.state = {
                    "conversation_key": session_key,
                    "chat_id": session_key,
                    "correlation_id": str(task_record.correlation_id or "").strip(),
                    "task_record": task_record.to_dict(),
                    "channel_type": str(buffered_input.channel_type or "").strip(),
                    "channel_target": str(buffered_input.channel_target or "").strip(),
                    "message_id": str(buffered_input.message_id or "").strip() or None,
                    "actor_person_id": str(buffered_input.actor_id or "").strip() or None,
                    "timezone": str(buffered_input.timezone or "").strip() or None,
                    "locale": str(buffered_input.locale or "").strip() or None,
                    "check_provenance": "entry",
                }
                task.initial_message_id = str(buffered_input.message_id or "").strip() or None
                task.initial_correlation_id = str(task_record.correlation_id or "").strip() or None
            else:
                state = dict(task.state)
                state["task_record"] = task_record.to_dict()
                state["check_provenance"] = "slice_resume"
                task.state = state
                if task.status in {"waiting_user", "paused"}:
                    task.status = "queued"
                elif task.status == "running":
                    task.input_dirty = True
            task.buffered_inputs = _append_input_record(
                inputs=task.buffered_inputs,
                message_id=str(buffered_input.message_id or "").strip() or None,
                channel_type=str(buffered_input.channel_type or "").strip(),
                correlation_id=str(task_record.correlation_id or "").strip(),
                user_text=user_text,
                attachments=attachments,
                actor_id=str(buffered_input.actor_id or "").strip() or None,
                now=now,
            )
            task.last_user_message = user_text
            task.pending_user_text = user_text
            task.last_user_channel = str(buffered_input.channel_type or "").strip()
            task.last_user_target = str(buffered_input.channel_target or "").strip()
            task.last_user_message_id = str(buffered_input.message_id or "").strip() or None
            task.last_enqueue_correlation_id = str(task_record.correlation_id or "").strip() or None
            task.last_enqueued_at = now
            task.updated_at = _now_iso()
            return task.task_id

    def _owner_for_task(self, task_id: str) -> str | None:
        task = self._tasks.get(str(task_id or "").strip())
        if task is None:
            return None
        return task.owner_id or None

    def _apply_metadata(self, task: ActivePdcaTask, metadata: dict[str, Any]) -> None:
        metadata = dict(metadata or {})
        task.buffered_inputs = [dict(item) for item in metadata.get("inputs", []) if isinstance(item, dict)]
        task.next_unconsumed_index = max(0, int(metadata.get("next_unconsumed_index") or 0))
        task.input_dirty = bool(metadata.get("input_dirty"))
        task.state = dict(metadata.get("state") or {}) if isinstance(metadata.get("state"), dict) else {}
        task.pending_user_text = _none_if_blank(metadata.get("pending_user_text"))
        task.last_user_message = _none_if_blank(metadata.get("last_user_message"))
        task.last_user_channel = _none_if_blank(metadata.get("last_user_channel"))
        task.last_user_target = _none_if_blank(metadata.get("last_user_target"))
        task.last_user_message_id = _none_if_blank(metadata.get("last_user_message_id"))
        task.last_enqueue_correlation_id = _none_if_blank(metadata.get("last_enqueue_correlation_id"))
        task.last_enqueued_at = _none_if_blank(metadata.get("last_enqueued_at"))
        task.last_input_dequeued_at = _none_if_blank(metadata.get("last_input_dequeued_at"))
        task.initial_message_id = _none_if_blank(metadata.get("initial_message_id"))
        task.initial_correlation_id = _none_if_blank(metadata.get("initial_correlation_id"))
        task.invoke_inflight = bool(metadata.get("invoke_inflight"))
        task.invoke_inflight_started_at = _none_if_blank(metadata.get("invoke_inflight_started_at"))
        task.invoke_inflight_correlation_id = _none_if_blank(metadata.get("invoke_inflight_correlation_id"))
        task.task_class = _none_if_blank(metadata.get("task_class"))
        task.interactive = bool(metadata.get("interactive"))
        task.started_at = _none_if_blank(metadata.get("started_at"))
        task.extra_metadata = {key: _json_safe_copy(value) for key, value in metadata.items() if key not in _CANONICAL_METADATA_KEYS}


def get_pdca_runtime() -> PdcaRuntime:
    runtime_key = str(resolve_nervous_system_db_path())
    with _RUNTIMES_LOCK:
        runtime = _RUNTIMES.get(runtime_key)
        if runtime is None:
            runtime = PdcaRuntime()
            _RUNTIMES[runtime_key] = runtime
        return runtime


def _append_input_record(
    *,
    inputs: list[dict[str, Any]],
    message_id: str | None,
    channel_type: str,
    correlation_id: str,
    user_text: str,
    attachments: list[dict[str, Any]],
    actor_id: str | None,
    now: str,
) -> list[dict[str, Any]]:
    normalized_inputs = [dict(item) for item in inputs if isinstance(item, dict)]
    message_key = str(message_id or "").strip()
    cid = str(correlation_id or "").strip()
    text = str(user_text or "").strip()
    attachment_fp = _attachment_fingerprint(attachments)
    dedupe_key = f"{message_key}|{cid}|{text}|{attachment_fp}"
    seen = {
        f"{str(item.get('message_id') or '').strip()}|{str(item.get('correlation_id') or '').strip()}|"
        f"{str(item.get('text') or '').strip()}|{_attachment_fingerprint([dict(att) for att in item.get('attachments', []) if isinstance(att, dict)])}"
        for item in normalized_inputs
    }
    if dedupe_key in seen:
        return normalized_inputs
    sequence = max((int(item.get("sequence") or 0) for item in normalized_inputs), default=0) + 1
    normalized_inputs.append(
        {
            "message_id": message_key,
            "correlation_id": cid,
            "text": text,
            "channel": str(channel_type or "").strip(),
            "actor_id": str(actor_id or "").strip() or None,
            "attachments": [dict(item) for item in attachments],
            "received_at": now,
            "consumed_at": None,
            "sequence": sequence,
        }
    )
    return normalized_inputs


def _attachment_fingerprint(attachments: list[dict[str, Any]]) -> str:
    return ",".join(
        sorted(
            f"{str(item.get('kind') or '').strip()}:{str(item.get('file_id') or item.get('url') or '').strip()}"
            for item in attachments
        )
    )


def _wait_seconds(*, task: ActivePdcaTask, now_dt: datetime) -> int:
    baseline = _parse_utc(task.next_run_at) or _parse_utc(task.created_at) or now_dt
    return max(int((now_dt - baseline).total_seconds()), 0)


def _is_bootstrap_record(
    *,
    message_id: str,
    correlation_id: str,
    initial_message_id: str,
    initial_correlation_id: str,
) -> bool:
    return bool(
        (initial_message_id and message_id and message_id == initial_message_id)
        or (initial_correlation_id and correlation_id and correlation_id == initial_correlation_id)
    )


def _normalize_checkpoint_state(state: dict[str, Any]) -> dict[str, Any]:
    task_record_value = state.get("task_record")
    if isinstance(task_record_value, dict):
        task_record = _json_safe_dict(task_record_value)
    elif isinstance(task_record_value, TaskRecord):
        task_record = task_record_value.to_dict()
    else:
        raise ValueError("pdca_checkpoint.missing_task_record")
    payload: dict[str, Any] = {"task_record": task_record}
    if "last_user_message" in state:
        payload["last_user_message"] = _json_safe_copy(state.get("last_user_message"))
    if "check_provenance" in state:
        payload["check_provenance"] = _json_safe_copy(state.get("check_provenance"))
    if "cycle_index" in state:
        payload["cycle_index"] = _json_safe_copy(state.get("cycle_index"))
    return payload


def _json_safe_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe_copy(value) for key, value in payload.items()}


def _json_safe_copy(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, TaskRecord):
        return value.to_dict()
    if isinstance(value, list):
        return [_json_safe_copy(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_copy(item) for key, item in value.items()}
    raise TypeError(f"pdca_runtime.unsupported_value_type:{type(value).__name__}")


def _none_if_blank(value: Any) -> str | None:
    rendered = str(value or "").strip()
    return rendered or None


def _normalize_status(value: str) -> str:
    rendered = str(value or "").strip().lower()
    return rendered if rendered in _ALL_TASK_STATUSES else "queued"


def _as_int(value: Any, *, default: int, minimum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(parsed, minimum)


def _as_optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_utc(value: Any) -> datetime | None:
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


def _parse_or_now(value: str | None) -> datetime:
    parsed = _parse_utc(value)
    return parsed if parsed is not None else datetime.now(timezone.utc)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_for_order(value: str | None) -> str:
    return str(value or "9999-12-31T23:59:59+00:00")
