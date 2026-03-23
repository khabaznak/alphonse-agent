from __future__ import annotations

import os
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from alphonse.agent.actions.conscious_message_handler import build_incoming_message_envelope
from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.cognition.memory import MemoryService
from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
from alphonse.agent.nervous_system.senses.bus import Signal as BusSignal
from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.services.job_runner import JobRunner
from alphonse.agent.services.scheduled_jobs_reconciler import ScheduledJobsReconciler
from alphonse.agent.services.job_store import JobStore


logger = get_component_logger("actions.handle_timed_signals")
_MEMORY_MAINTENANCE_SIGNAL_ID = "runtime.memory.weekly.maintenance"
_MEMORY_MAINTENANCE_KIND = "memory_maintenance"


class HandleTimedSignalsAction(Action):
    key = "handle_timed_dispatch"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        if getattr(signal, "type", "") == "timer.fired":
            return _handle_timer_reconcile_tick(context=context)
        payload = getattr(signal, "payload", {}) if signal else {}
        inner = _payload_from_signal(payload)
        if _is_memory_maintenance_payload(payload=payload, inner=inner):
            return _run_memory_maintenance(payload=payload)
        if _is_job_trigger_payload(signal_payload=payload, inner=inner):
            return _execute_job_trigger(context=context, payload=payload, inner=inner, signal=signal)
        if _is_legacy_daily_report(payload=payload, inner=inner):
            logger.info("HandleTimedSignalsAction skipped legacy_daily_report timed_signal_id=%s", payload.get("timed_signal_id"))
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)
        mind_layer = str(payload.get("mind_layer") or inner.get("mind_layer") or "subconscious").strip().lower()
        prompt = _extract_prompt_text(inner=inner)
        target = str(inner.get("chat_id") or payload.get("target") or inner.get("delivery_target") or "").strip()
        user_id = str(inner.get("person_id") or target or "").strip()
        channel_type = str(
            inner.get("origin_channel") or payload.get("origin_channel") or payload.get("origin") or inner.get("origin") or "api"
        ).strip()
        logger.info(
            "HandleTimedSignalsAction invoked signal_id=%s timed_signal_id=%s correlation_id=%s mind_layer=%s",
            getattr(signal, "id", None),
            payload.get("timed_signal_id"),
            getattr(signal, "correlation_id", None),
            mind_layer,
        )
        bus = context.get("ctx")
        if not hasattr(bus, "emit"):
            logger.warning("HandleTimedSignalsAction route skipped reason=no_bus")
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)

        if mind_layer != "conscious":
            logger.info(
                "HandleTimedSignalsAction deterministic_non_conscious_drop timed_signal_id=%s",
                payload.get("timed_signal_id"),
            )
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)
        routed_signal_type = "timed_signal.conscious_payload"
        routed_payload: dict[str, Any] = build_incoming_message_envelope(
            message_id=str(payload.get("timed_signal_id") or _correlation_id(payload, signal) or "timed"),
            channel_type=channel_type,
            channel_target=target or user_id or channel_type,
            provider=channel_type,
            text=prompt,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            correlation_id=_correlation_id(payload, signal),
            actor_external_user_id=user_id or None,
            metadata={
                "timed_signal": payload,
                "mind_layer": mind_layer,
                "channel_hint": channel_type,
            },
        )
        bus.emit(
            BusSignal(
                type=routed_signal_type,
                payload=routed_payload,
                source="timer",
                correlation_id=_correlation_id(payload, signal),
            )
        )
        logger.info("HandleTimedSignalsAction routed signal_type=%s", routed_signal_type)
        return ActionResult(intention_key="NOOP", payload={}, urgency=None)


def _handle_timer_reconcile_tick(*, context: dict[str, Any]) -> ActionResult:
    bus = context.get("ctx")
    signal = context.get("signal")

    def _brain_sink(brain_payload: dict[str, Any]) -> None:
        _emit_brain_payload_to_bus(
            bus=bus,
            signal_payload={},
            inner={},
            user_id=str((brain_payload or {}).get("user_id") or "").strip(),
            signal=signal,
            brain_payload=brain_payload,
        )

    summary = ScheduledJobsReconciler().reconcile(
        brain_event_sink=_brain_sink if hasattr(bus, "emit") else None
    )
    logger.info(
        "HandleTimedSignalsAction jobs_reconciled scanned=%s updated=%s stale_removed=%s executed=%s advanced_without_run=%s failed=%s overdue_active=%s due_pending_signals=%s",
        int(summary.get("scanned") or 0),
        int(summary.get("updated") or 0),
        int(summary.get("stale_removed") or 0),
        int(summary.get("executed") or 0),
        int(summary.get("advanced_without_run") or 0),
        int(summary.get("failed") or 0),
        int(summary.get("overdue_active_jobs") or 0),
        int(summary.get("due_pending_timed_signals") or 0),
    )
    _ensure_weekly_memory_maintenance_signal()
    return ActionResult(intention_key="NOOP", payload={}, urgency=None)


def _payload_from_signal(payload: dict) -> dict:
    signal_payload = payload.get("payload") if isinstance(payload, dict) else None
    return signal_payload if isinstance(signal_payload, dict) else {}


def _correlation_id(payload: dict, signal: object | None) -> str | None:
    if isinstance(payload, dict):
        cid = payload.get("correlation_id")
        if cid:
            return str(cid)
    return getattr(signal, "correlation_id", None) if signal else None


def _extract_prompt_text(*, inner: dict) -> str:
    text = str(
        inner.get("prompt")
        or inner.get("agent_internal_prompt")
        or inner.get("prompt_text")
        or inner.get("message_text")
        or inner.get("message")
        or inner.get("reminder_text_raw")
        or "You just remembered something important."
    ).strip()
    return text or "You just remembered something important."


def _is_job_trigger_payload(*, signal_payload: dict[str, Any], inner: dict[str, Any]) -> bool:
    timed_signal_id = str((signal_payload or {}).get("timed_signal_id") or "").strip()
    if timed_signal_id.startswith("job_trigger:"):
        return True
    if str((inner or {}).get("kind") or "").strip().lower() == "job_trigger":
        return True
    return bool(str((inner or {}).get("job_id") or "").strip() and str((inner or {}).get("user_id") or "").strip())


def _is_legacy_daily_report(*, payload: dict[str, Any], inner: dict[str, Any]) -> bool:
    timed_signal_id = str((payload or {}).get("timed_signal_id") or "").strip()
    kind = str((inner or {}).get("kind") or "").strip().lower()
    return timed_signal_id == "daily_report" or kind == "daily_report"


def _is_memory_maintenance_payload(*, payload: dict[str, Any], inner: dict[str, Any]) -> bool:
    timed_signal_id = str((payload or {}).get("timed_signal_id") or "").strip()
    kind = str((inner or {}).get("kind") or "").strip().lower()
    return timed_signal_id == _MEMORY_MAINTENANCE_SIGNAL_ID or kind == _MEMORY_MAINTENANCE_KIND


def _run_memory_maintenance(*, payload: dict[str, Any]) -> ActionResult:
    service = MemoryService()
    try:
        summary = service.run_maintenance(generate_weekly=True)
        logger.info(
            "HandleTimedSignalsAction memory_maintenance users_scanned=%s summaries_written=%s deleted_daily=%s deleted_weekly=%s",
            int(summary.get("users_scanned") or 0),
            int(summary.get("summaries_written") or 0),
            int(summary.get("deleted_daily") or 0),
            int(summary.get("deleted_weekly") or 0),
        )
    except Exception as exc:
        logger.warning(
            "HandleTimedSignalsAction memory_maintenance_failed error=%s",
            type(exc).__name__,
        )
    _reschedule_weekly_memory_maintenance(signal_id=str(payload.get("timed_signal_id") or _MEMORY_MAINTENANCE_SIGNAL_ID))
    return ActionResult(intention_key="NOOP", payload={}, urgency=None)


def _ensure_weekly_memory_maintenance_signal() -> None:
    db_path = resolve_nervous_system_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT status, trigger_at FROM timed_signals WHERE id = ?",
            (_MEMORY_MAINTENANCE_SIGNAL_ID,),
        ).fetchone()
        if row is None:
            trigger_at = _next_weekly_trigger_at(now).isoformat()
            payload = json.dumps({"kind": _MEMORY_MAINTENANCE_KIND}, ensure_ascii=False)
            conn.execute(
                """
                INSERT INTO timed_signals (
                  id, trigger_at, timezone, status, fired_at, signal_type, payload, target, origin, correlation_id, created_at, updated_at
                ) VALUES (?, ?, ?, 'pending', NULL, 'timed_signal', ?, ?, ?, ?, datetime('now'), datetime('now'))
                """,
                (
                    _MEMORY_MAINTENANCE_SIGNAL_ID,
                    trigger_at,
                    "UTC",
                    payload,
                    "system",
                    "runtime",
                    "runtime.memory.weekly",
                ),
            )
            conn.commit()
            return
        status = str(row[0] or "").strip().lower()
        trigger_at = str(row[1] or "").strip()
        is_future_pending = status == "pending" and bool(trigger_at) and trigger_at > now.isoformat()
        if is_future_pending:
            return
    _reschedule_weekly_memory_maintenance(signal_id=_MEMORY_MAINTENANCE_SIGNAL_ID)


def _reschedule_weekly_memory_maintenance(*, signal_id: str) -> None:
    db_path = resolve_nervous_system_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    next_trigger = _next_weekly_trigger_at(datetime.now(timezone.utc)).isoformat()
    payload = json.dumps({"kind": _MEMORY_MAINTENANCE_KIND}, ensure_ascii=False)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE timed_signals
            SET trigger_at = ?, timezone = ?, status = 'pending', fired_at = NULL,
                signal_type = 'timed_signal', payload = ?, target = ?, origin = ?, correlation_id = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (
                next_trigger,
                "UTC",
                payload,
                "system",
                "runtime",
                "runtime.memory.weekly",
                signal_id or _MEMORY_MAINTENANCE_SIGNAL_ID,
            ),
        )
        conn.commit()


def _next_weekly_trigger_at(now: datetime) -> datetime:
    base = now.astimezone(timezone.utc).replace(minute=5, second=0, microsecond=0, hour=0)
    days_until_monday = (7 - base.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    return base + timedelta(days=days_until_monday)


def _execute_job_trigger(*, context: dict[str, Any], payload: dict[str, Any], inner: dict[str, Any], signal: object | None) -> ActionResult:
    bus = context.get("ctx")
    if not hasattr(bus, "emit"):
        logger.warning("HandleTimedSignalsAction job_trigger skipped reason=no_bus")
        return ActionResult(intention_key="NOOP", payload={}, urgency=None)
    job_id = str(inner.get("job_id") or "").strip()
    user_id = str(inner.get("user_id") or payload.get("target") or "").strip()
    if not job_id or not user_id:
        logger.warning(
            "HandleTimedSignalsAction job_trigger skipped reason=missing_identifiers job_id=%s user_id=%s",
            job_id,
            user_id,
        )
        return ActionResult(intention_key="NOOP", payload={}, urgency=None)
    store_root = str(os.getenv("ALPHONSE_JOBS_ROOT") or "").strip() or None
    runner = JobRunner(
        job_store=JobStore(root=store_root),
        tick_seconds=45,
    )

    def _brain_sink(brain_payload: dict[str, Any]) -> None:
        _emit_brain_payload_to_bus(
            bus=bus,
            signal_payload=payload,
            inner=inner,
            user_id=user_id,
            signal=signal,
            brain_payload=brain_payload,
        )

    try:
        runner.run_job_now(
            user_id=user_id,
            job_id=job_id,
            now=datetime.now(timezone.utc),
            brain_event_sink=_brain_sink,
        )
    except Exception as exc:
        logger.warning(
            "HandleTimedSignalsAction job_trigger failed job_id=%s user_id=%s error=%s",
            job_id,
            user_id,
            type(exc).__name__,
        )
    else:
        logger.info("HandleTimedSignalsAction executed job_trigger job_id=%s user_id=%s", job_id, user_id)
    return ActionResult(intention_key="NOOP", payload={}, urgency=None)


def _emit_brain_payload_to_bus(
    *,
    bus: Any,
    signal_payload: dict[str, Any],
    inner: dict[str, Any],
    user_id: str,
    signal: object | None,
    brain_payload: dict[str, Any],
) -> None:
    if not hasattr(bus, "emit"):
        return
    channel = str(
        signal_payload.get("origin")
        or inner.get("origin_channel")
        or signal_payload.get("channel")
        or "api"
    ).strip() or "api"
    target = str(signal_payload.get("target") or inner.get("delivery_target") or user_id).strip() or user_id
    correlation_id = _correlation_id(signal_payload, signal)
    nested = brain_payload.get("payload") if isinstance(brain_payload, dict) else {}
    nested_payload = nested if isinstance(nested, dict) else {}
    payload_type = str((brain_payload or {}).get("payload_type") or "").strip().lower()
    text = _extract_job_execution_prompt(nested_payload)
    if not text and payload_type != "prompt_to_brain":
        text = str(
            nested_payload.get("agent_internal_prompt")
            or nested_payload.get("source_instruction")
            or "You just remembered something important."
        ).strip()
    text = text or "You just remembered something important."
    metadata = {
        "timed_signal": signal_payload,
        "source": "job_trigger" if signal_payload else "jobs_reconcile",
    }
    job_id = str((brain_payload or {}).get("job_id") or "").strip()
    if job_id:
        metadata["job_id"] = job_id
    bus.emit(
        BusSignal(
            type="timed_signal.conscious_payload",
            payload=build_incoming_message_envelope(
                message_id=str((signal_payload or {}).get("message_id") or correlation_id),
                channel_type=channel,
                channel_target=target,
                provider=str(channel or "timer"),
                text=text,
                correlation_id=correlation_id,
                actor_external_user_id=user_id,
                metadata=metadata,
            ),
            source="timer",
            correlation_id=correlation_id,
        )
    )


def _extract_job_execution_prompt(payload: dict[str, Any] | None) -> str:
    rows = payload if isinstance(payload, dict) else {}
    for key in ("prompt_text", "text", "message", "prompt"):
        value = str(rows.get(key) or "").strip()
        if value:
            return value
    return ""
