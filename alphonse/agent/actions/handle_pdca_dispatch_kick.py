from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.nervous_system.pdca_queue_store import (
    acquire_pdca_task_lease,
    append_pdca_event,
    get_pdca_task,
    has_pdca_event,
    release_pdca_task_lease,
    upsert_pdca_task,
)
from alphonse.agent.nervous_system.senses.bus import Signal as BusSignal
from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.observability.log_manager import get_log_manager

logger = get_component_logger("actions.handle_pdca_dispatch_kick")
_LOG = get_log_manager()
_TERMINAL_STATUSES = {"done", "failed", "cancelled"}
_RUNNABLE_STATUSES = {"queued", "running"}


class HandlePdcaDispatchKickAction(Action):
    key = "handle_pdca_dispatch_kick"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        payload = getattr(signal, "payload", {}) if signal else {}
        if not isinstance(payload, dict):
            payload = {}
        task_id = str(payload.get("task_id") or "").strip()
        correlation_id = str(payload.get("correlation_id") or getattr(signal, "correlation_id", "") or "").strip() or None
        if not task_id:
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)

        if correlation_id and has_pdca_event(
            task_id=task_id,
            event_type="pdca.dispatch.kick.received",
            correlation_id=correlation_id,
        ):
            _emit_skip(
                task_id=task_id,
                correlation_id=correlation_id,
                reason="duplicate_kick_signal",
            )
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)

        append_pdca_event(
            task_id=task_id,
            event_type="pdca.dispatch.kick.received",
            payload={"source": getattr(signal, "source", None)},
            correlation_id=correlation_id,
        )
        _LOG.emit(
            event="pdca.dispatch.kick.received",
            component="actions.handle_pdca_dispatch_kick",
            correlation_id=correlation_id,
            payload={"task_id": task_id},
        )

        task = get_pdca_task(task_id)
        if not isinstance(task, dict):
            _emit_skip(task_id=task_id, correlation_id=correlation_id, reason="task_not_found")
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)

        status = str(task.get("status") or "").strip().lower()
        if status in _TERMINAL_STATUSES:
            _emit_skip(task_id=task_id, correlation_id=correlation_id, reason="terminal_task")
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)
        if status not in _RUNNABLE_STATUSES:
            _emit_skip(task_id=task_id, correlation_id=correlation_id, reason="status_not_runnable")
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)

        now = datetime.now(timezone.utc)
        next_run_at = _parse_utc(task.get("next_run_at"))
        if next_run_at is not None and next_run_at > now:
            _emit_skip(task_id=task_id, correlation_id=correlation_id, reason="cooldown_not_elapsed")
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)

        worker_id = "pdca-dispatcher"
        acquired = acquire_pdca_task_lease(
            task_id=task_id,
            worker_id=worker_id,
            lease_seconds=_dispatch_lease_seconds(),
            now=now.isoformat(),
        )
        if not acquired:
            _emit_skip(task_id=task_id, correlation_id=correlation_id, reason="lease_held")
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)

        try:
            latest = get_pdca_task(task_id)
            if not isinstance(latest, dict):
                _emit_skip(task_id=task_id, correlation_id=correlation_id, reason="task_not_found_after_lease")
                return ActionResult(intention_key="NOOP", payload={}, urgency=None)

            latest_status = str(latest.get("status") or "").strip().lower()
            if latest_status in _TERMINAL_STATUSES:
                _emit_skip(task_id=task_id, correlation_id=correlation_id, reason="terminal_task_after_lease")
                return ActionResult(intention_key="NOOP", payload={}, urgency=None)
            if latest_status not in _RUNNABLE_STATUSES:
                _emit_skip(task_id=task_id, correlation_id=correlation_id, reason="status_not_runnable_after_lease")
                return ActionResult(intention_key="NOOP", payload={}, urgency=None)

            dispatch_corr = f"pdca.slice.requested:{task_id}:{int(now.timestamp() * 1000)}"
            append_pdca_event(
                task_id=task_id,
                event_type="slice.requested",
                payload={"worker_id": worker_id, "scheduled_for": now.isoformat()},
                correlation_id=dispatch_corr,
            )
            append_pdca_event(
                task_id=task_id,
                event_type="pdca.dispatch.slice_emitted",
                payload={"from_kick_correlation_id": correlation_id},
                correlation_id=dispatch_corr,
            )

            next_run = now + timedelta(seconds=_dispatch_cooldown_seconds())
            upsert_pdca_task(
                {
                    "task_id": latest.get("task_id"),
                    "owner_id": latest.get("owner_id"),
                    "conversation_key": latest.get("conversation_key"),
                    "session_id": latest.get("session_id"),
                    "status": "running",
                    "priority": latest.get("priority", 100),
                    "next_run_at": next_run.isoformat(),
                    "slice_cycles": latest.get("slice_cycles", 5),
                    "max_cycles": latest.get("max_cycles"),
                    "max_runtime_seconds": latest.get("max_runtime_seconds"),
                    "token_budget_remaining": latest.get("token_budget_remaining"),
                    "failure_streak": latest.get("failure_streak", 0),
                    "last_error": latest.get("last_error"),
                    "metadata": latest.get("metadata") if isinstance(latest.get("metadata"), dict) else {},
                    "created_at": latest.get("created_at"),
                }
            )

            bus = context.get("ctx")
            if hasattr(bus, "emit"):
                bus.emit(
                    BusSignal(
                        type="pdca.slice.requested",
                        payload={
                            "task_id": task_id,
                            "owner_id": str(latest.get("owner_id") or "").strip() or None,
                            "conversation_key": str(latest.get("conversation_key") or "").strip() or None,
                            "correlation_id": dispatch_corr,
                        },
                        source="pdca_dispatcher",
                        correlation_id=dispatch_corr,
                    )
                )
            _LOG.emit(
                event="pdca.dispatch.slice_emitted",
                component="actions.handle_pdca_dispatch_kick",
                correlation_id=dispatch_corr,
                payload={"task_id": task_id},
            )
            return ActionResult(intention_key="NOOP", payload={"task_id": task_id}, urgency=None)
        finally:
            release_pdca_task_lease(task_id=task_id, worker_id=worker_id)


def _dispatch_cooldown_seconds() -> int:
    import os

    raw = str(os.getenv("ALPHONSE_PDCA_QUEUE_DISPATCH_COOLDOWN_SECONDS") or "30").strip()
    try:
        parsed = int(raw)
    except ValueError:
        parsed = 30
    return max(parsed, 1)


def _dispatch_lease_seconds() -> int:
    import os

    raw = str(os.getenv("ALPHONSE_PDCA_QUEUE_LEASE_SECONDS") or "30").strip()
    try:
        parsed = int(raw)
    except ValueError:
        parsed = 30
    return max(parsed, 1)


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


def _emit_skip(*, task_id: str, correlation_id: str | None, reason: str) -> None:
    append_pdca_event(
        task_id=task_id,
        event_type="pdca.dispatch.kick.skipped",
        payload={"reason": reason},
        correlation_id=correlation_id,
    )
    _LOG.emit(
        event="pdca.dispatch.kick.skipped",
        component="actions.handle_pdca_dispatch_kick",
        correlation_id=correlation_id,
        payload={"task_id": task_id, "reason": reason},
    )

