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
    update_pdca_task_metadata,
    update_pdca_task_status,
)
from alphonse.agent.nervous_system.senses.bus import Signal as BusSignal
from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.observability.log_manager import get_log_manager
from alphonse.agent.services.pdca_queue_runner import emit_pdca_dispatch_kick
from alphonse.agent.services.session_identity_resolution import resolve_assistant_session_message
from alphonse.agent.session.day_state import build_next_session_state
from alphonse.agent.session.day_state import commit_session_state
from alphonse.agent.session.day_state import render_recent_conversation_block
from alphonse.agent.session.day_state import resolve_day_session
from alphonse.config import settings

logger = get_component_logger("actions.handle_pdca_slice_request")
_LOG = get_log_manager()
_CORTEX_GRAPH = CortexGraph()
_TERMINAL_TASK_STATUSES = {"done", "failed", "cancelled"}


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
        status = str(task.get("status") or "").strip().lower()
        if status in _TERMINAL_TASK_STATUSES:
            append_pdca_event(
                task_id=task_id,
                event_type="slice.request.terminal_ignored",
                payload={"status": status},
                correlation_id=correlation_id,
            )
            _LOG.emit(
                event="pdca.slice.ignored_terminal_task",
                component="actions.handle_pdca_slice_request",
                correlation_id=correlation_id,
                payload={"task_id": task_id, "status": status},
            )
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
        base_state = _base_state(task=task, checkpoint=checkpoint, correlation_id=correlation_id)
        _inject_session_memory_context(base_state=base_state, task=task)
        _stamp_check_provenance_for_slice(base_state=base_state, signal=signal, checkpoint=checkpoint)
        incoming = _resolve_incoming_context(task=task, correlation_id=correlation_id)
        consumed_inputs = _consume_buffered_inputs(task_id=task_id, correlation_id=correlation_id)
        if consumed_inputs:
            _mark_slice_resume(base_state=base_state)
            _inject_steering_facts(base_state=base_state, consumed_inputs=consumed_inputs)
            _LOG.emit(
                event="pdca.slice.resume_with_input",
                component="actions.handle_pdca_slice_request",
                correlation_id=correlation_id,
                payload={
                    "task_id": task_id,
                    "input_count": len(consumed_inputs),
                },
            )
        text = _resolve_slice_text(
            task=task,
            checkpoint=checkpoint,
            payload=payload,
            buffered_inputs=consumed_inputs,
        )
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
        invoke_state.setdefault("task_id", task_id)
        invoke_state.setdefault("pdca_task_id", task_id)
        invoke_state["_consume_task_inputs"] = lambda: _consume_buffered_inputs(
            task_id=task_id,
            correlation_id=correlation_id,
        )
        task_state_for_id = invoke_state.get("task_state")
        if isinstance(task_state_for_id, dict):
            task_state_for_id.setdefault("task_id", task_id)
        _emit_context_continuity_gap_if_any(
            task=task,
            invoke_state=invoke_state,
            correlation_id=correlation_id,
        )
        invoke_state["_bus"] = context.get("ctx")
        if incoming is not None:
            invoke_state["_transition_sink"] = lambda event: emit_channel_transition_event(incoming, event)
        slice_started_at = datetime.now(timezone.utc)
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
        should_preempt = _should_preempt_after_step(
            task_id=task_id,
            slice_started_at=slice_started_at,
        )
        if should_preempt:
            append_pdca_event(
                task_id=task_id,
                event_type="pdca.slice.preempt_after_step",
                payload={"reason": "fresh_user_input_buffered"},
                correlation_id=correlation_id,
            )
            _LOG.emit(
                event="pdca.slice.preempt_after_step",
                component="actions.handle_pdca_slice_request",
                correlation_id=correlation_id,
                payload={"task_id": task_id},
            )
        effective_reply_text = "" if should_preempt else str(result.reply_text or "")
        merged_state = _merge_state(base=base_state, cognition_state=cognition_state, reply_text=effective_reply_text)
        _ = save_pdca_checkpoint(
            task_id=task_id,
            state=merged_state,
            task_state=cognition_state.get("task_state") if isinstance(cognition_state.get("task_state"), dict) else {},
            expected_version=int(checkpoint.get("version") or 0) if isinstance(checkpoint, dict) else 0,
        )
        conversation_key = str(task.get("conversation_key") or "").strip()
        if conversation_key:
            save_state(conversation_key, merged_state)
        _update_day_session_memory(
            task=task,
            user_message=text,
            reply_text=effective_reply_text,
            cognition_state=cognition_state,
            merged_state=merged_state,
        )

        next_status = "queued" if should_preempt else _next_status(cognition_state=cognition_state, reply_text=effective_reply_text)
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
        _upsert_task_after_slice(task=task, status=next_status, request_text=text, reply_text=effective_reply_text)
        if next_status == "queued":
            _emit_dispatch_kick_signal(context=context, task_id=task_id, correlation_id=correlation_id, reason="slice_completed_queued")
        append_pdca_event(
            task_id=task_id,
            event_type=f"slice.completed.{next_status}",
            payload={
                "reply_text": effective_reply_text.strip(),
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


def _base_state(
    *,
    task: dict[str, Any],
    checkpoint: dict[str, Any] | None,
    correlation_id: str | None,
) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state"), dict):
        out = dict(checkpoint.get("state") or {})
        _ensure_state_correlation_id(
            base=out,
            task=task,
            seeded=(task.get("metadata") or {}).get("state")
            if isinstance(task.get("metadata"), dict)
            else {},
            incoming_correlation_id=correlation_id,
        )
        return out
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    seeded = metadata.get("state") if isinstance(metadata.get("state"), dict) else {}
    conversation_key = str(task.get("conversation_key") or "").strip()
    loaded = load_state(conversation_key) if conversation_key else None
    if isinstance(loaded, dict):
        sanitized, removed_keys = _sanitize_loaded_state_for_new_task(
            loaded,
            preserve_pending_interaction=_should_preserve_pending_interaction(task=task),
        )
        _overlay_ingress_context(base=sanitized, seeded=seeded)
        _ensure_state_correlation_id(
            base=sanitized,
            task=task,
            seeded=seeded,
            incoming_correlation_id=correlation_id,
        )
        if removed_keys:
            _LOG.emit(
                event="pdca.slice.base_state.sanitized",
                component="actions.handle_pdca_slice_request",
                correlation_id=correlation_id,
                payload={
                    "task_id": str(task.get("task_id") or "").strip() or None,
                    "conversation_key": conversation_key or None,
                    "removed_keys": removed_keys,
                },
            )
        return sanitized
    out = dict(seeded)
    out.setdefault("conversation_key", conversation_key)
    out.setdefault("chat_id", conversation_key)
    _ensure_state_correlation_id(
        base=out,
        task=task,
        seeded=seeded,
        incoming_correlation_id=correlation_id,
    )
    return out


def _inject_session_memory_context(*, base_state: dict[str, Any], task: dict[str, Any]) -> None:
    owner_id = str(task.get("owner_id") or "").strip()
    if not owner_id:
        return
    channel_type = str(base_state.get("channel_type") or "").strip() or "api"
    timezone_name = str(base_state.get("timezone") or "").strip() or settings.get_timezone()
    try:
        day_session = resolve_day_session(
            user_id=owner_id,
            channel=channel_type,
            timezone_name=timezone_name,
        )
    except Exception:
        return
    base_state["session_id"] = str(day_session.get("session_id") or "").strip() or base_state.get("session_id")
    base_state["session_state"] = day_session
    base_state["recent_conversation_block"] = render_recent_conversation_block(day_session)


def _update_day_session_memory(
    *,
    task: dict[str, Any],
    user_message: str,
    reply_text: str,
    cognition_state: dict[str, Any],
    merged_state: dict[str, Any],
) -> None:
    owner_id = str(task.get("owner_id") or "").strip()
    if not owner_id:
        return
    channel = str(merged_state.get("channel_type") or "").strip() or "api"
    timezone_name = str(merged_state.get("timezone") or "").strip() or settings.get_timezone()
    try:
        day_session = resolve_day_session(
            user_id=owner_id,
            channel=channel,
            timezone_name=timezone_name,
        )
        assistant_message = resolve_assistant_session_message(reply_text=reply_text, plans=[])
        updated = build_next_session_state(
            previous=day_session,
            channel=channel,
            user_message=str(user_message or ""),
            assistant_message=assistant_message,
            ability_state=cognition_state.get("ability_state")
            if isinstance(cognition_state.get("ability_state"), dict)
            else None,
            task_state=cognition_state.get("task_state")
            if isinstance(cognition_state.get("task_state"), dict)
            else None,
            planning_context=cognition_state.get("planning_context")
            if isinstance(cognition_state.get("planning_context"), dict)
            else None,
            pending_interaction=merged_state.get("pending_interaction")
            if isinstance(merged_state.get("pending_interaction"), dict)
            else None,
        )
        commit_session_state(updated)
        merged_state["session_id"] = str(updated.get("session_id") or "").strip() or merged_state.get("session_id")
        merged_state["session_state"] = updated
        merged_state["recent_conversation_block"] = render_recent_conversation_block(updated)
    except Exception:
        return


def _resolve_slice_text(
    *,
    task: dict[str, Any],
    checkpoint: dict[str, Any] | None,
    payload: dict[str, Any],
    buffered_inputs: list[dict[str, Any]] | None = None,
) -> str:
    direct = str(payload.get("text") or "").strip()
    if direct:
        return direct
    if isinstance(buffered_inputs, list):
        merged = [
            str(item.get("steering_text") or item.get("text") or "").strip()
            for item in buffered_inputs
            if isinstance(item, dict)
        ]
        merged = [item for item in merged if item]
        if merged:
            return "\n".join(merged)
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


def _sanitize_loaded_state_for_new_task(
    loaded: dict[str, Any],
    *,
    preserve_pending_interaction: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    sanitized = dict(loaded)
    removed_keys: list[str] = []
    for key in (
        "task_state",
        "pending_interaction",
        "response_text",
        "events",
        "utterance",
        "render_error",
    ):
        if key == "pending_interaction" and preserve_pending_interaction:
            continue
        if key in sanitized:
            removed_keys.append(key)
            sanitized.pop(key, None)
    return sanitized, removed_keys


def _should_preserve_pending_interaction(*, task: dict[str, Any]) -> bool:
    status = str(task.get("status") or "").strip().lower()
    if status in {"waiting_user", "running", "paused"}:
        return True
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    return bool(str(metadata.get("last_user_message") or "").strip())


def _mark_slice_resume(*, base_state: dict[str, Any]) -> None:
    task_state = base_state.get("task_state")
    if not isinstance(task_state, dict):
        task_state = {}
        base_state["task_state"] = task_state
    task_state["check_provenance"] = "slice_resume"


def _consume_buffered_inputs(*, task_id: str, correlation_id: str | None) -> list[dict[str, Any]]:
    latest = get_pdca_task(task_id)
    if not isinstance(latest, dict):
        return []
    metadata = dict(latest.get("metadata") or {}) if isinstance(latest.get("metadata"), dict) else {}
    raw_inputs = metadata.get("inputs")
    inputs = [dict(item) for item in raw_inputs if isinstance(item, dict)] if isinstance(raw_inputs, list) else []
    if not inputs:
        return []
    next_unconsumed = _as_optional_int(metadata.get("next_unconsumed_index")) or 0
    next_unconsumed = max(0, min(next_unconsumed, len(inputs)))
    consumed: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    for idx in range(next_unconsumed, len(inputs)):
        record = inputs[idx]
        text = str(record.get("text") or "").strip()
        steering_text = str(record.get("steering_text") or text).strip()
        attachments = record.get("attachments")
        normalized_attachments = [dict(item) for item in attachments if isinstance(item, dict)] if isinstance(attachments, list) else []
        if not steering_text and not normalized_attachments:
            continue
        consumed.append(
            {
                "text": text,
                "steering_text": steering_text,
                "actor_id": str(record.get("actor_id") or "").strip() or None,
                "message_id": str(record.get("message_id") or "").strip() or None,
                "correlation_id": str(record.get("correlation_id") or "").strip() or None,
                "channel": str(record.get("channel") or "").strip() or None,
                "attachments": normalized_attachments,
                "received_at": str(record.get("received_at") or "").strip() or None,
                "consumed_at": now,
            }
        )
        record["consumed_at"] = now
    if not consumed:
        return []
    metadata["inputs"] = inputs
    metadata["next_unconsumed_index"] = len(inputs)
    metadata["input_dirty"] = False
    metadata["last_user_message"] = str(consumed[-1].get("steering_text") or consumed[-1].get("text") or "").strip()
    metadata["pending_user_text"] = str(consumed[-1].get("steering_text") or consumed[-1].get("text") or "").strip()
    metadata["last_input_dequeued_at"] = now
    update_pdca_task_metadata(task_id=task_id, metadata=metadata)
    _LOG.emit(
        event="pdca.input.dequeued",
        component="actions.handle_pdca_slice_request",
        correlation_id=correlation_id,
        payload={"task_id": task_id, "count": len(consumed)},
    )
    return consumed


def _inject_steering_facts(*, base_state: dict[str, Any], consumed_inputs: list[dict[str, Any]]) -> None:
    if not consumed_inputs:
        return
    task_state = base_state.get("task_state")
    if not isinstance(task_state, dict):
        task_state = {}
        base_state["task_state"] = task_state
    facts = task_state.get("facts")
    fact_bucket = dict(facts) if isinstance(facts, dict) else {}
    max_suffix = 0
    for key in fact_bucket.keys():
        rendered = str(key or "").strip()
        if not rendered.startswith("user_reply_"):
            continue
        suffix = rendered.split("user_reply_", 1)[1]
        try:
            max_suffix = max(max_suffix, int(suffix))
        except ValueError:
            continue
    for index, item in enumerate(consumed_inputs, start=1):
        text = str(item.get("steering_text") or item.get("text") or "").strip()
        attachments = item.get("attachments")
        normalized_attachments = [dict(att) for att in attachments if isinstance(att, dict)] if isinstance(attachments, list) else []
        if not text and not normalized_attachments:
            continue
        fact_bucket[f"user_reply_{max_suffix + index}"] = {
            "source": "user_reply",
            "text": text,
            "attachments": normalized_attachments,
            "actor_id": str(item.get("actor_id") or "").strip() or None,
            "message_id": str(item.get("message_id") or "").strip() or None,
            "received_at": str(item.get("received_at") or "").strip() or None,
            "consumed_at": str(item.get("consumed_at") or "").strip() or None,
        }
    task_state["facts"] = fact_bucket


def _should_preempt_after_step(*, task_id: str, slice_started_at: datetime) -> bool:
    latest = get_pdca_task(task_id)
    if not isinstance(latest, dict):
        return False
    metadata = latest.get("metadata") if isinstance(latest.get("metadata"), dict) else {}
    if not bool(metadata.get("input_dirty")):
        return False
    enqueued_at = _parse_utc(metadata.get("last_enqueued_at"))
    if enqueued_at is None:
        return True
    return enqueued_at >= slice_started_at


def _overlay_ingress_context(*, base: dict[str, Any], seeded: dict[str, Any]) -> None:
    for key in (
        "conversation_key",
        "chat_id",
        "channel_type",
        "channel_target",
        "actor_person_id",
        "incoming_user_id",
        "incoming_user_name",
        "locale",
        "tone",
        "address_style",
        "timezone",
        "correlation_id",
    ):
        if str(base.get(key) or "").strip():
            continue
        value = seeded.get(key)
        if value is None:
            continue
        rendered = str(value).strip()
        if not rendered:
            continue
        base[key] = value


def _ensure_state_correlation_id(
    *,
    base: dict[str, Any],
    task: dict[str, Any],
    seeded: dict[str, Any] | None,
    incoming_correlation_id: str | None,
) -> None:
    current = str(base.get("correlation_id") or "").strip()
    if current:
        return
    incoming = str(incoming_correlation_id or "").strip()
    if incoming:
        base["correlation_id"] = incoming
        return
    seeded_map = seeded if isinstance(seeded, dict) else {}
    seeded_corr = str(seeded_map.get("correlation_id") or "").strip()
    if seeded_corr:
        base["correlation_id"] = seeded_corr
        return
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    enqueued_corr = str(metadata.get("last_enqueue_correlation_id") or "").strip()
    if enqueued_corr:
        base["correlation_id"] = enqueued_corr
        return
    task_id = str(task.get("task_id") or "").strip()
    base["correlation_id"] = f"pdca:{task_id}" if task_id else "pdca:unknown"


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
    task_id = str(task.get("task_id") or "").strip()
    latest = get_pdca_task(task_id) if task_id else None
    base_task = latest if isinstance(latest, dict) else task
    now = datetime.now(timezone.utc)
    next_run_at = None
    if status == "queued":
        cooldown = _dispatch_cooldown_seconds()
        next_run_at = (now + timedelta(seconds=cooldown)).isoformat()
    token_budget = base_task.get("token_budget_remaining")
    tokens_used = _estimate_token_burn(request_text, reply_text)
    next_token_budget = int(token_budget) - tokens_used if token_budget is not None else None
    failure_streak = int(base_task.get("failure_streak") or 0)
    if status == "failed":
        failure_streak += 1
    else:
        failure_streak = 0
    upsert_pdca_task(
        {
            "task_id": base_task.get("task_id"),
            "owner_id": base_task.get("owner_id"),
            "conversation_key": base_task.get("conversation_key"),
            "session_id": base_task.get("session_id"),
            "status": status,
            "priority": base_task.get("priority", 100),
            "next_run_at": next_run_at,
            "slice_cycles": base_task.get("slice_cycles", 5),
            "max_cycles": base_task.get("max_cycles"),
            "max_runtime_seconds": base_task.get("max_runtime_seconds"),
            "token_budget_remaining": next_token_budget,
            "failure_streak": failure_streak,
            "last_error": base_task.get("last_error"),
            "metadata": base_task.get("metadata") if isinstance(base_task.get("metadata"), dict) else {},
            "created_at": base_task.get("created_at"),
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


def _emit_dispatch_kick_signal(
    *,
    context: dict[str, Any],
    task_id: str,
    correlation_id: str | None,
    reason: str,
) -> None:
    bus = context.get("ctx")
    emitted = emit_pdca_dispatch_kick(
        bus=bus if hasattr(bus, "emit") else None,
        task_id=task_id,
        reason=reason,
        correlation_id=correlation_id,
        source="handle_pdca_slice_request",
    )
    if not emitted:
        return
    _LOG.emit(
        event="pdca.dispatch.kick_emitted",
        component="actions.handle_pdca_slice_request",
        correlation_id=correlation_id,
        payload={"task_id": task_id, "reason": reason},
    )


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
    message_id = (
        str(metadata.get("last_user_message_id") or "").strip()
        or str(state.get("message_id") or "").strip()
        or None
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
        message_id=message_id,
    )


def _emit_context_continuity_gap_if_any(
    *,
    task: dict[str, Any],
    invoke_state: dict[str, Any],
    correlation_id: str | None,
) -> None:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    seeded = metadata.get("state") if isinstance(metadata.get("state"), dict) else {}
    present_at_ingress = {
        key: str(seeded.get(key) or "").strip()
        for key in ("channel_type", "channel_target", "actor_person_id", "incoming_user_id", "incoming_user_name")
    }
    missing_in_invoke: list[str] = []
    for key, seeded_value in present_at_ingress.items():
        if not seeded_value:
            continue
        current = str(invoke_state.get(key) or "").strip()
        if not current:
            missing_in_invoke.append(key)
    if not missing_in_invoke:
        return
    _LOG.emit(
        level="warning",
        event="pdca.context.continuity_gap",
        component="actions.handle_pdca_slice_request",
        correlation_id=correlation_id,
        payload={
            "task_id": str(task.get("task_id") or "").strip() or None,
            "missing_in_invoke_state": missing_in_invoke,
            "ingress_channel_type": present_at_ingress.get("channel_type") or None,
            "ingress_channel_target": present_at_ingress.get("channel_target") or None,
        },
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
