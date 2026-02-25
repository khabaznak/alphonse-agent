from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.cognition.providers.factory import build_llm_client
from alphonse.agent.cortex.graph import CortexGraph
from alphonse.agent.cortex.state_store import load_state, save_state
from alphonse.agent.nervous_system.pdca_queue_store import (
    append_pdca_event,
    get_pdca_task,
    load_pdca_checkpoint,
    save_pdca_checkpoint,
    upsert_pdca_task,
    update_pdca_task_status,
)
from alphonse.agent.nervous_system.senses.bus import Signal as BusSignal
from alphonse.agent.observability.log_manager import get_component_logger

logger = get_component_logger("actions.handle_pdca_slice_request")
_CORTEX_GRAPH = CortexGraph()


class HandlePdcaSliceRequestAction(Action):
    key = "handle_pdca_slice_request"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        payload = getattr(signal, "payload", {}) if signal else {}
        if not isinstance(payload, dict):
            payload = {}
        task_id = str(payload.get("task_id") or "").strip()
        correlation_id = str(payload.get("correlation_id") or getattr(signal, "correlation_id", "") or "").strip() or None
        if not task_id:
            logger.warning("HandlePdcaSliceRequestAction skipped reason=missing_task_id")
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)

        task = get_pdca_task(task_id)
        if not isinstance(task, dict):
            logger.warning("HandlePdcaSliceRequestAction skipped reason=task_not_found task_id=%s", task_id)
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)

        updated = update_pdca_task_status(task_id=task_id, status="running")
        append_pdca_event(
            task_id=task_id,
            event_type="slice.request.signal_received",
            payload={
                "updated": updated,
                "source": getattr(signal, "source", None),
            },
            correlation_id=correlation_id,
        )

        checkpoint = load_pdca_checkpoint(task_id)
        base_state = _base_state(task=task, checkpoint=checkpoint)
        text = _resolve_slice_text(task=task, checkpoint=checkpoint, payload=payload)
        if not text:
            update_pdca_task_status(task_id=task_id, status="waiting_user")
            append_pdca_event(
                task_id=task_id,
                event_type="slice.blocked.missing_text",
                payload={"reason": "missing_input_text"},
                correlation_id=correlation_id,
            )
            _emit_slice_signal(
                context=context,
                event_type="pdca.waiting_user",
                task_id=task_id,
                correlation_id=correlation_id,
            )
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)

        try:
            llm_client = build_llm_client()
        except Exception:
            llm_client = None

        try:
            result = _CORTEX_GRAPH.invoke(base_state, text, llm_client=llm_client)
        except Exception as exc:
            _finalize_failure(task=task, correlation_id=correlation_id, error=str(exc))
            _emit_slice_signal(
                context=context,
                event_type="pdca.failed",
                task_id=task_id,
                correlation_id=correlation_id,
            )
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)

        cognition_state = result.cognition_state if isinstance(result.cognition_state, dict) else {}
        merged_state = _merge_state(base=base_state, cognition_state=cognition_state, reply_text=str(result.reply_text or ""))
        _ = save_pdca_checkpoint(
            task_id=task_id,
            state=merged_state,
            task_state=cognition_state.get("task_state") if isinstance(cognition_state.get("task_state"), dict) else {},
            expected_version=int(checkpoint.get("version") or 0) if isinstance(checkpoint, dict) else 0,
        )
        conversation_key = str(task.get("conversation_key") or "").strip()
        if conversation_key:
            save_state(conversation_key, merged_state)

        next_status = _next_status(cognition_state=cognition_state, reply_text=str(result.reply_text or ""))
        _upsert_task_after_slice(task=task, status=next_status)
        append_pdca_event(
            task_id=task_id,
            event_type=f"slice.completed.{next_status}",
            payload={
                "reply_text": str(result.reply_text or "").strip(),
                "has_task_state": isinstance(cognition_state.get("task_state"), dict),
            },
            correlation_id=correlation_id,
        )
        _emit_slice_signal(
            context=context,
            event_type=_status_signal(next_status),
            task_id=task_id,
            correlation_id=correlation_id,
        )
        logger.info(
            "HandlePdcaSliceRequestAction completed task_id=%s status=%s correlation_id=%s",
            task_id,
            next_status,
            correlation_id,
        )
        return ActionResult(intention_key="NOOP", payload={}, urgency=None)


def _base_state(*, task: dict[str, Any], checkpoint: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state"), dict):
        return dict(checkpoint.get("state") or {})
    conversation_key = str(task.get("conversation_key") or "").strip()
    loaded = load_state(conversation_key) if conversation_key else None
    if isinstance(loaded, dict):
        return dict(loaded)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    seeded = metadata.get("state") if isinstance(metadata.get("state"), dict) else {}
    out = dict(seeded)
    out.setdefault("conversation_key", conversation_key)
    out.setdefault("chat_id", conversation_key)
    out.setdefault("correlation_id", f"pdca:{task.get('task_id')}")
    return out


def _resolve_slice_text(*, task: dict[str, Any], checkpoint: dict[str, Any] | None, payload: dict[str, Any]) -> str:
    direct = str(payload.get("text") or "").strip()
    if direct:
        return direct
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    for key in ("pending_user_text", "user_text", "last_user_message"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    if isinstance(checkpoint, dict):
        state = checkpoint.get("state") if isinstance(checkpoint.get("state"), dict) else {}
        value = str(state.get("last_user_message") or "").strip()
        if value:
            return value
    return ""


def _merge_state(*, base: dict[str, Any], cognition_state: dict[str, Any], reply_text: str) -> dict[str, Any]:
    merged = dict(base)
    slots_collected = cognition_state.get("slots_collected")
    if isinstance(slots_collected, dict):
        merged["slots"] = dict(slots_collected)
    for source_key, target_key in (
        ("last_intent", "intent"),
        ("locale", "locale"),
        ("autonomy_level", "autonomy_level"),
        ("planning_mode", "planning_mode"),
        ("intent_category", "intent_category"),
        ("route_decision", "route_decision"),
        ("pending_interaction", "pending_interaction"),
        ("ability_state", "ability_state"),
        ("task_state", "task_state"),
        ("planning_context", "planning_context"),
    ):
        if source_key in cognition_state:
            merged[target_key] = cognition_state.get(source_key)
    merged["response_text"] = str(reply_text or "").strip() or None
    merged["last_updated_at"] = datetime.now(timezone.utc).isoformat()
    return merged


def _next_status(*, cognition_state: dict[str, Any], reply_text: str) -> str:
    task_state = cognition_state.get("task_state") if isinstance(cognition_state.get("task_state"), dict) else {}
    task_status = str(task_state.get("status") or "").strip().lower()
    if task_status in {"done", "failed", "waiting_user"}:
        return task_status
    if str(reply_text or "").strip():
        return "done"
    return "queued"


def _upsert_task_after_slice(*, task: dict[str, Any], status: str) -> None:
    now = datetime.now(timezone.utc)
    next_run_at = None
    if status == "queued":
        cooldown = _dispatch_cooldown_seconds()
        next_run_at = (now + timedelta(seconds=cooldown)).isoformat()
    upsert_pdca_task(
        {
            "task_id": task.get("task_id"),
            "owner_id": task.get("owner_id"),
            "conversation_key": task.get("conversation_key"),
            "session_id": task.get("session_id"),
            "status": status,
            "priority": task.get("priority", 100),
            "next_run_at": next_run_at,
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


def _finalize_failure(*, task: dict[str, Any], correlation_id: str | None, error: str) -> None:
    task_id = str(task.get("task_id") or "").strip()
    update_pdca_task_status(task_id=task_id, status="failed", last_error=error)
    append_pdca_event(
        task_id=task_id,
        event_type="slice.failed",
        payload={"error": error},
        correlation_id=correlation_id,
    )


def _status_signal(status: str) -> str:
    if status == "waiting_user":
        return "pdca.waiting_user"
    if status == "failed":
        return "pdca.failed"
    if status == "done":
        return "pdca.slice.completed"
    return "pdca.slice.persisted"


def _emit_slice_signal(*, context: dict[str, Any], event_type: str, task_id: str, correlation_id: str | None) -> None:
    bus = context.get("ctx")
    if not hasattr(bus, "emit"):
        return
    payload: dict[str, Any] = {"task_id": task_id}
    if correlation_id:
        payload["correlation_id"] = correlation_id
    bus.emit(
        BusSignal(
            type=event_type,
            payload=payload,
            source="handle_pdca_slice_request",
            correlation_id=correlation_id,
        )
    )


def _dispatch_cooldown_seconds() -> int:
    raw = str(os.getenv("ALPHONSE_PDCA_QUEUE_DISPATCH_COOLDOWN_SECONDS") or "30").strip()
    try:
        parsed = int(raw)
    except ValueError:
        parsed = 30
    return max(parsed, 1)
