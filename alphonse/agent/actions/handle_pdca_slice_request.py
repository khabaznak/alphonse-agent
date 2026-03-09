from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from math import ceil
from typing import Any

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.actions.presence_projection import emit_channel_transition_event
from alphonse.agent.actions.session_context import IncomingContext
from alphonse.agent.cognition.providers.factory import build_llm_client
from alphonse.agent.cortex.graph import CortexGraph
from alphonse.agent.cortex.state_store import load_state, save_state
from alphonse.agent.nervous_system.pdca_queue_store import (
    append_pdca_event,
    get_pdca_task,
    has_pdca_event,
    load_pdca_checkpoint,
    save_pdca_checkpoint,
    upsert_pdca_task,
    update_pdca_task_status,
)
from alphonse.agent.nervous_system.senses.bus import Signal as BusSignal
from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.observability.log_manager import get_log_manager

logger = get_component_logger("actions.handle_pdca_slice_request")
_LOG = get_log_manager()
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
        if correlation_id and has_pdca_event(
            task_id=task_id,
            event_type="slice.request.signal_received",
            correlation_id=correlation_id,
        ):
            append_pdca_event(
                task_id=task_id,
                event_type="slice.request.duplicate_ignored",
                payload={"source": getattr(signal, "source", None)},
                correlation_id=correlation_id,
            )
            logger.info(
                "HandlePdcaSliceRequestAction duplicate ignored task_id=%s correlation_id=%s",
                task_id,
                correlation_id,
            )
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
        _stamp_check_provenance_for_slice(base_state=base_state, signal=signal, checkpoint=checkpoint)
        incoming = _resolve_incoming_context(task=task, correlation_id=correlation_id)
        text = _resolve_slice_text(task=task, checkpoint=checkpoint, payload=payload)
        if not text:
            update_pdca_task_status(task_id=task_id, status="waiting_user")
            append_pdca_event(
                task_id=task_id,
                event_type="slice.blocked.missing_text",
                payload={"reason": "missing_input_text"},
                correlation_id=correlation_id,
            )
            _emit_presence_lifecycle_event(
                incoming=incoming,
                event_family="presence.waiting_input",
                phase="waiting_user",
                correlation_id=correlation_id,
            )
            _emit_slice_signal(
                context=context,
                event_type="pdca.waiting_user",
                task_id=task_id,
                correlation_id=correlation_id,
            )
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)
        budget_block = _budget_block(task=task, checkpoint=checkpoint, request_text=text)
        if budget_block is not None:
            _finalize_budget_exhausted(
                task=task,
                correlation_id=correlation_id,
                reason=str(budget_block.get("reason") or "budget_exhausted"),
                details=budget_block,
            )
            _emit_presence_lifecycle_event(
                incoming=incoming,
                event_family="presence.failed",
                phase="failed",
                correlation_id=correlation_id,
            )
            _emit_slice_signal(
                context=context,
                event_type="pdca.failed",
                task_id=task_id,
                correlation_id=correlation_id,
            )
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)

        try:
            llm_client = build_llm_client()
        except Exception:
            llm_client = None

        _emit_presence_lifecycle_event(
            incoming=incoming,
            event_family="presence.phase_changed",
            phase="executing",
            correlation_id=correlation_id,
        )
        invoke_state = dict(base_state)
        if incoming is not None:
            invoke_state["_transition_sink"] = lambda event: emit_channel_transition_event(incoming, event)
        try:
            result = _CORTEX_GRAPH.invoke(invoke_state, text, llm_client=llm_client)
        except Exception as exc:
            failure_code = _classify_failure_code(exc)
            user_notice_required = failure_code == "engine_unavailable"
            _LOG.emit(
                event="pdca.slice.failed.classified",
                component="actions.handle_pdca_slice_request",
                correlation_id=correlation_id,
                payload={
                    "task_id": task_id,
                    "failure_code": failure_code,
                    "user_notice_required": user_notice_required,
                },
            )
            _finalize_failure(task=task, correlation_id=correlation_id, error=str(exc))
            _emit_presence_lifecycle_event(
                incoming=incoming,
                event_family="presence.failed",
                phase="failed",
                correlation_id=correlation_id,
            )
            _emit_slice_signal(
                context=context,
                event_type="pdca.failed",
                task_id=task_id,
                correlation_id=correlation_id,
                extra_payload={
                    "failure_code": failure_code,
                    "user_notice_required": user_notice_required,
                },
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
        if next_status == "waiting_user":
            _emit_presence_lifecycle_event(
                incoming=incoming,
                event_family="presence.waiting_input",
                phase="waiting_user",
                correlation_id=correlation_id,
            )
        elif next_status == "done":
            _emit_presence_lifecycle_event(
                incoming=incoming,
                event_family="presence.completed",
                phase="done",
                correlation_id=correlation_id,
            )
        elif next_status == "failed":
            _emit_presence_lifecycle_event(
                incoming=incoming,
                event_family="presence.failed",
                phase="failed",
                correlation_id=correlation_id,
            )
        _upsert_task_after_slice(task=task, status=next_status, request_text=text, reply_text=str(result.reply_text or ""))
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


def _upsert_task_after_slice(*, task: dict[str, Any], status: str, request_text: str, reply_text: str) -> None:
    now = datetime.now(timezone.utc)
    next_run_at = None
    if status == "queued":
        cooldown = _dispatch_cooldown_seconds()
        next_run_at = (now + timedelta(seconds=cooldown)).isoformat()
    token_budget = task.get("token_budget_remaining")
    tokens_used = _estimate_token_burn(request_text, reply_text)
    next_token_budget = int(token_budget) - tokens_used if token_budget is not None else None
    failure_streak = int(task.get("failure_streak") or 0)
    if status == "failed":
        failure_streak += 1
    else:
        failure_streak = 0
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
            "token_budget_remaining": next_token_budget,
            "failure_streak": failure_streak,
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


def _finalize_budget_exhausted(*, task: dict[str, Any], correlation_id: str | None, reason: str, details: dict[str, Any]) -> None:
    task_id = str(task.get("task_id") or "").strip()
    update_pdca_task_status(task_id=task_id, status="failed", last_error=reason)
    append_pdca_event(
        task_id=task_id,
        event_type="slice.blocked.budget_exhausted",
        payload=details,
        correlation_id=correlation_id,
    )


def _budget_block(*, task: dict[str, Any], checkpoint: dict[str, Any] | None, request_text: str) -> dict[str, Any] | None:
    max_cycles = _as_optional_int(task.get("max_cycles"))
    cycle_index = _current_cycle_index(checkpoint)
    if max_cycles is not None and cycle_index >= max_cycles:
        return {
            "reason": "max_cycles_reached",
            "cycle_index": cycle_index,
            "max_cycles": max_cycles,
        }
    runtime_limit = _as_optional_int(task.get("max_runtime_seconds"))
    if runtime_limit is not None:
        elapsed = _elapsed_runtime_seconds(task)
        if elapsed is not None and elapsed >= runtime_limit:
            return {
                "reason": "max_runtime_reached",
                "elapsed_seconds": elapsed,
                "max_runtime_seconds": runtime_limit,
            }
    token_budget = _as_optional_int(task.get("token_budget_remaining"))
    if token_budget is not None:
        projected_burn = _estimate_token_burn(request_text, "")
        if token_budget <= 0 or token_budget < projected_burn:
            return {
                "reason": "token_budget_exhausted",
                "token_budget_remaining": token_budget,
                "projected_slice_burn": projected_burn,
            }
    return None


def _current_cycle_index(checkpoint: dict[str, Any] | None) -> int:
    if not isinstance(checkpoint, dict):
        return 0
    task_state = checkpoint.get("task_state") if isinstance(checkpoint.get("task_state"), dict) else {}
    return _as_optional_int(task_state.get("cycle_index")) or 0


def _elapsed_runtime_seconds(task: dict[str, Any]) -> int | None:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    for key in ("started_at", "last_enqueued_at"):
        started = _parse_utc(metadata.get(key))
        if started is not None:
            now = datetime.now(timezone.utc)
            return max(int((now - started).total_seconds()), 0)
    created = _parse_utc(task.get("created_at"))
    if created is None:
        return None
    now = datetime.now(timezone.utc)
    return max(int((now - created).total_seconds()), 0)


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


def _as_optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _stamp_check_provenance_for_slice(
    *,
    base_state: dict[str, Any],
    signal: object | None,
    checkpoint: dict[str, Any] | None,
) -> None:
    signal_type = str(getattr(signal, "type", "") or "").strip().lower()
    task_state = base_state.get("task_state")
    if not isinstance(task_state, dict):
        task_state = {}
        base_state["task_state"] = task_state
    if signal_type == "pdca.resume_requested":
        task_state["check_provenance"] = "slice_resume"
        return
    if isinstance(checkpoint, dict):
        task_state["check_provenance"] = "do"


def _estimate_token_burn(request_text: str, reply_text: str) -> int:
    chars = len(str(request_text or "").strip()) + len(str(reply_text or "").strip())
    return max(1, int(ceil(chars / 4)))


def _status_signal(status: str) -> str:
    if status == "waiting_user":
        return "pdca.waiting_user"
    if status == "failed":
        return "pdca.failed"
    if status == "done":
        return "pdca.slice.completed"
    return "pdca.slice.persisted"


def _emit_slice_signal(
    *,
    context: dict[str, Any],
    event_type: str,
    task_id: str,
    correlation_id: str | None,
    extra_payload: dict[str, Any] | None = None,
) -> None:
    bus = context.get("ctx")
    if not hasattr(bus, "emit"):
        return
    payload: dict[str, Any] = {"task_id": task_id}
    if correlation_id:
        payload["correlation_id"] = correlation_id
    if isinstance(extra_payload, dict):
        payload.update(extra_payload)
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


def _classify_failure_code(exc: Exception) -> str:
    rendered = str(exc or "").strip().lower()
    matchers = (
        "connection refused",
        "failed to establish a new connection",
        "max retries exceeded",
        "timed out",
        "connect timeout",
        "read timeout",
        "unable to connect",
        "endpoint unavailable",
        "/session",
    )
    if any(token in rendered for token in matchers):
        return "engine_unavailable"
    return "execution_failed"


def _resolve_incoming_context(*, task: dict[str, Any], correlation_id: str | None) -> IncomingContext | None:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    state = metadata.get("state") if isinstance(metadata.get("state"), dict) else {}
    channel_type = (
        str(metadata.get("last_user_channel") or "").strip()
        or str(state.get("channel_type") or "").strip()
    )
    channel_target = (
        str(metadata.get("last_user_target") or "").strip()
        or str(state.get("channel_target") or "").strip()
    )
    if (not channel_type or not channel_target) and ":" in str(task.get("conversation_key") or ""):
        inferred_channel, inferred_target = str(task.get("conversation_key") or "").split(":", 1)
        if not channel_type:
            channel_type = str(inferred_channel or "").strip()
        if not channel_target:
            channel_target = str(inferred_target or "").strip()
    if not channel_type or not channel_target:
        return None
    resolved_correlation = str(correlation_id or "").strip() or f"pdca:{str(task.get('task_id') or '').strip()}"
    return IncomingContext(
        channel_type=channel_type,
        address=channel_target,
        person_id=str(state.get("actor_person_id") or "").strip() or None,
        correlation_id=resolved_correlation,
    )


def _emit_presence_lifecycle_event(
    *,
    incoming: IncomingContext | None,
    event_family: str,
    phase: str,
    correlation_id: str | None,
) -> None:
    if incoming is None:
        return
    corr = str(correlation_id or incoming.correlation_id or "").strip() or None
    emit_channel_transition_event(
        incoming,
        {
            "type": "agent.transition",
            "phase": str(phase or "").strip(),
            "correlation_id": corr,
            "detail": {"presence_event_family": str(event_family or "").strip()},
        },
    )
