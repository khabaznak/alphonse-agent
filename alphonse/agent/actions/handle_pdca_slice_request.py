from __future__ import annotations

import os
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from math import ceil
from typing import Any

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.actions.presence_projection import emit_channel_transition_event
from alphonse.agent.actions.session_context import IncomingContext
from alphonse.agent.cognition.memory import append_conversation_transcript
from alphonse.agent.cognition.providers.factory import build_llm_client
from alphonse.agent.cortex.graph import CortexGraph
from alphonse.agent.nervous_system.pdca_queue_store import (
    append_pdca_event,
    get_pdca_task,
    has_pdca_event,
    load_pdca_checkpoint,
    save_pdca_checkpoint,
    update_pdca_task_metadata,
    upsert_pdca_task,
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
_SLICE_INVOKE_EXECUTOR = ThreadPoolExecutor(max_workers=max(int(os.getenv("ALPHONSE_PDCA_SLICE_INVOKE_WORKERS", "2") or "2"), 1))
_INFLIGHT_STALE_SECONDS = max(int(os.getenv("ALPHONSE_PDCA_INVOKE_INFLIGHT_STALE_SECONDS", "300") or "300"), 30)
_INFLIGHT_METADATA_KEYS = (
    "invoke_inflight",
    "invoke_inflight_started_at",
    "invoke_inflight_correlation_id",
)


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

        inflight_state = _read_inflight_state(task=task)
        if inflight_state["active"] and not inflight_state["stale"]:
            append_pdca_event(
                task_id=task_id,
                event_type="slice.request.inflight_ignored",
                payload={"started_at": inflight_state["started_at"]},
                correlation_id=correlation_id,
            )
            logger.info(
                "HandlePdcaSliceRequestAction inflight ignored task_id=%s correlation_id=%s",
                task_id,
                correlation_id,
            )
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)
        if inflight_state["active"] and inflight_state["stale"]:
            _clear_inflight_metadata(task_id=task_id)
            append_pdca_event(
                task_id=task_id,
                event_type="slice.request.inflight_stale_cleared",
                payload={"previous_started_at": inflight_state["started_at"]},
                correlation_id=correlation_id,
            )

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
        text = _resolve_slice_text(
            task=task,
            checkpoint=checkpoint,
            payload=payload,
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

        _emit_presence_lifecycle_event(
            incoming=incoming,
            event_family="presence.phase_changed",
            phase="executing",
            correlation_id=correlation_id,
        )
        invoke_state = dict(base_state)
        invoke_state.setdefault("task_id", task_id)
        invoke_state.setdefault("pdca_task_id", task_id)
        task_record_for_id = invoke_state.get("task_record")
        if isinstance(task_record_for_id, dict) and not str(task_record_for_id.get("task_id") or "").strip():
            task_record_for_id["task_id"] = task_id
        _emit_context_continuity_gap_if_any(
            task=task,
            invoke_state=invoke_state,
            correlation_id=correlation_id,
        )
        _mark_inflight_metadata(task_id=task_id, correlation_id=correlation_id)
        scheduled = _schedule_slice_invoke(
            task_id=task_id,
            correlation_id=correlation_id,
            task=task,
            checkpoint=checkpoint,
            base_state=base_state,
            invoke_state=invoke_state,
            incoming=incoming,
            text=text,
            bus=context.get("ctx"),
        )
        if not scheduled:
            _clear_inflight_metadata(task_id=task_id)
            _finalize_failure(
                task=task,
                correlation_id=correlation_id,
                error="slice_invoke_schedule_failed",
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
        append_pdca_event(
            task_id=task_id,
            event_type="slice.request.invoke_scheduled",
            payload={"mode": "background"},
            correlation_id=correlation_id,
        )
        return ActionResult(intention_key="NOOP", payload={}, urgency=None)


def _schedule_slice_invoke(
    *,
    task_id: str,
    correlation_id: str | None,
    task: dict[str, Any],
    checkpoint: dict[str, Any] | None,
    base_state: dict[str, Any],
    invoke_state: dict[str, Any],
    incoming: IncomingContext | None,
    text: str,
    bus: Any,
) -> bool:
    try:
        future = _SLICE_INVOKE_EXECUTOR.submit(
            _run_slice_invoke_job,
            task_id=task_id,
            correlation_id=correlation_id,
            task=task,
            checkpoint=checkpoint,
            base_state=base_state,
            invoke_state=invoke_state,
            incoming=incoming,
            text=text,
            bus=bus,
        )
    except Exception:
        logger.exception("failed to schedule pdca slice invoke task_id=%s", task_id)
        return False
    future.add_done_callback(
        lambda done: _log_slice_worker_failure(task_id=task_id, correlation_id=correlation_id, future=done)
    )
    return True


def _log_slice_worker_failure(*, task_id: str, correlation_id: str | None, future: Future[None]) -> None:
    try:
        _ = future.result()
    except Exception:
        logger.exception(
            "pdca slice worker crashed task_id=%s correlation_id=%s",
            task_id,
            correlation_id,
        )


def _run_slice_invoke_job(
    *,
    task_id: str,
    correlation_id: str | None,
    task: dict[str, Any],
    checkpoint: dict[str, Any] | None,
    base_state: dict[str, Any],
    invoke_state: dict[str, Any],
    incoming: IncomingContext | None,
    text: str,
    bus: Any,
) -> None:
    context = {"ctx": bus}
    try:       
        invoke_state["_bus"] = bus
        if incoming is not None: # TODO: what if inconing IS none? exception? 
            invoke_state["_transition_sink"] = lambda event: emit_channel_transition_event(
                event=event,
                channel_type=incoming.channel_type,
                channel_target=incoming.address,
                user_id=incoming.person_id,
                message_id=incoming.message_id,
                correlation_id=incoming.correlation_id,
            )
        slice_started_at = datetime.now(timezone.utc)
        try:
            result = _CORTEX_GRAPH.invoke(invoke_state, text)
        except Exception as exc:
            failure_code = _classify_failure_code(exc)
            user_notice_required = failure_code == "engine_unavailable" # TODO this was already part of the _classify_failure_code why whould we put this additional?
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
            return
        
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
        effective_reply_text = "" if should_preempt else str(_result_field(result, "reply_text") or "")
        cognition_state = (
            dict(_result_field(result, "cognition_state"))
            if isinstance(_result_field(result, "cognition_state"), dict)
            else dict(invoke_state)
        )
        merged_state = _merge_state(base=cognition_state, reply_text=effective_reply_text)
        try:
            checkpoint_state = _build_checkpoint_payload(
                cognition_state=cognition_state,
                request_text=text,
                previous_checkpoint=checkpoint,
            )
            _ = save_pdca_checkpoint(
                task_id=task_id,
                state=checkpoint_state,
                expected_version=int(checkpoint.get("version") or 0) if isinstance(checkpoint, dict) else 0,
            )
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            _LOG.emit(
                event="pdca.slice.checkpoint_save_failed",
                component="actions.handle_pdca_slice_request",
                correlation_id=correlation_id,
                payload={"task_id": task_id, "error": error},
            )
            _finalize_failure(task=task, correlation_id=correlation_id, error=error)
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
                extra_payload={"failure_code": "checkpoint_save_failed"},
            )
            return
        _update_day_session_memory(
            task=task,
            user_message=text,
            reply_text=effective_reply_text,
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
                "has_task_record": isinstance(cognition_state.get("task_record"), dict),
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
    finally:
        _clear_inflight_metadata(task_id=task_id)


def _read_inflight_state(*, task: dict[str, Any]) -> dict[str, Any]:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    inflight = bool(metadata.get("invoke_inflight"))
    started_text = str(metadata.get("invoke_inflight_started_at") or "").strip()
    started_dt = _parse_utc(started_text)
    stale = False
    if inflight and started_dt is not None:
        stale = (datetime.now(timezone.utc) - started_dt) > timedelta(seconds=_INFLIGHT_STALE_SECONDS)
    return {
        "active": inflight,
        "stale": stale,
        "started_at": started_text or None,
    }


def _result_field(result: Any, key: str) -> Any:
    if isinstance(result, dict):
        return result.get(key)
    return getattr(result, key, None)


def _mark_inflight_metadata(*, task_id: str, correlation_id: str | None) -> None:
    task = get_pdca_task(task_id)
    if not isinstance(task, dict):
        return
    metadata = dict(task.get("metadata") or {})
    metadata["invoke_inflight"] = True
    metadata["invoke_inflight_started_at"] = datetime.now(timezone.utc).isoformat()
    metadata["invoke_inflight_correlation_id"] = correlation_id
    update_pdca_task_metadata(task_id=task_id, metadata=metadata)


def _clear_inflight_metadata(*, task_id: str) -> None:
    task = get_pdca_task(task_id)
    if not isinstance(task, dict):
        return
    metadata = dict(task.get("metadata") or {})
    changed = False
    for key in _INFLIGHT_METADATA_KEYS:
        if key in metadata:
            metadata.pop(key, None)
            changed = True
    if changed:
        update_pdca_task_metadata(task_id=task_id, metadata=metadata)


def _base_state(
    *,
    task: dict[str, Any],
    checkpoint: dict[str, Any] | None,
    correlation_id: str | None,
) -> dict[str, Any]:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    seeded = metadata.get("state") if isinstance(metadata.get("state"), dict) else {}
    conversation_key = str(task.get("conversation_key") or "").strip()
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state"), dict):
        out = _base_state_from_checkpoint(
            checkpoint_state=checkpoint.get("state") or {},
            seeded=seeded,
            conversation_key=conversation_key,
        )
        _ensure_state_correlation_id(
            base=out,
            task=task,
            seeded=seeded,
            incoming_correlation_id=correlation_id,
        )
        return out
    if isinstance(seeded.get("task_record"), dict):
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
    out = dict(seeded)
    out.setdefault("conversation_key", conversation_key)
    out.setdefault("chat_id", conversation_key)
    if not isinstance(out.get("task_record"), dict):
        synthesized = _synthesize_task_record(task=task, seeded=seeded, correlation_id=correlation_id)
        if synthesized is not None:
            out["task_record"] = synthesized
    _ensure_state_correlation_id(
        base=out,
        task=task,
        seeded=seeded,
        incoming_correlation_id=correlation_id,
    )
    return out


def _base_state_from_checkpoint(
    *,
    checkpoint_state: dict[str, Any],
    seeded: dict[str, Any],
    conversation_key: str,
) -> dict[str, Any]:
    out = dict(seeded)
    out.setdefault("conversation_key", conversation_key)
    out.setdefault("chat_id", conversation_key)
    for key in ("task_record", "last_user_message", "check_provenance", "cycle_index"):
        value = checkpoint_state.get(key)
        if value is not None:
            out[key] = value
    return out


def _synthesize_task_record(
    *,
    task: dict[str, Any],
    seeded: dict[str, Any] | None,
    correlation_id: str | None,
) -> dict[str, Any] | None:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    goal = ""
    for key in ("pending_user_text", "last_user_message", "user_text"):
        goal = str(metadata.get(key) or "").strip()
        if goal:
            break
    rendered_correlation = str(correlation_id or "").strip() or str(metadata.get("last_enqueue_correlation_id") or "").strip()
    record = {
        "task_id": str(task.get("task_id") or "").strip() or None,
        "user_id": str(task.get("owner_id") or "").strip() or None,
        "correlation_id": rendered_correlation,
        "goal": goal,
        "facts_md": "- (none)",
        "recent_conversation_md": "- (none)",
        "plan_md": "- (none)",
        "acceptance_criteria_md": "- (none)",
        "memory_facts_md": "- (none)",
        "tool_call_history_md": "- (none)",
        "status": "running",
        "outcome": None,
    }
    for key, value in (
        ("channel_type", str((seeded or {}).get("channel_type") or metadata.get("last_user_channel") or "").strip()),
        ("channel_target", str((seeded or {}).get("channel_target") or metadata.get("last_user_target") or "").strip()),
        ("message_id", str(metadata.get("last_user_message_id") or "").strip()),
    ):
        if value:
            record["facts_md"] = f"{record['facts_md']}\n- {key}: {value}"
    return record if record["correlation_id"] or record["task_id"] else None


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
    merged_state: dict[str, Any],
) -> None:
    owner_id = str(task.get("owner_id") or "").strip() # TODO is this even correct? is it "owner_id"?
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
        task_record = merged_state.get("task_record") if isinstance(merged_state.get("task_record"), dict) else {}
        updated = build_next_session_state(
            previous=day_session,
            channel=channel,
            user_message="",
            assistant_message=assistant_message,
            task_record=task_record if isinstance(task_record, dict) else None,
            pending_interaction=merged_state.get("pending_interaction")
            if isinstance(merged_state.get("pending_interaction"), dict)
            else None,
            assistant_event_meta={
                "correlation_id": str(merged_state.get("correlation_id") or "").strip() or None,
                "channel": channel,
            },
        )
        commit_session_state(updated)
        if assistant_message:
            append_conversation_transcript(
                user_id=owner_id,
                session_id=str(updated.get("session_id") or "").strip() or owner_id,
                role="assistant",
                text=assistant_message,
                channel=channel,
                correlation_id=str(merged_state.get("correlation_id") or "").strip() or None,
            )
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
) -> str:
    direct = str(payload.get("text") or "").strip()
    if direct:
        return direct
    content = payload.get("content") if isinstance(payload.get("content"), dict) else {}
    payload_attachments = content.get("attachments") if isinstance(content.get("attachments"), list) else []
    payload_attachment_text = _render_attachment_summary(payload_attachments)
    if payload_attachment_text:
        return payload_attachment_text
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


def _render_attachment_summary(attachments: list[Any]) -> str:
    normalized = [dict(item) for item in attachments if isinstance(item, dict)]
    if not normalized:
        return ""
    parts: list[str] = []
    for item in normalized:
        kind = str(item.get("kind") or "").strip().lower() or "file"
        if kind == "contact":
            contact = item.get("contact") if isinstance(item.get("contact"), dict) else {}
            first = str(contact.get("first_name") or "").strip()
            last = str(contact.get("last_name") or "").strip()
            full_name = " ".join(value for value in (first, last) if value).strip()
            if full_name:
                parts.append(f"contact ({full_name})")
                continue
        file_id = str(item.get("file_id") or "").strip()
        if file_id:
            parts.append(f"{kind}({file_id})")
            continue
        parts.append(kind)
    return f"[attachments: {', '.join(parts)}]"


def _sanitize_loaded_state_for_new_task(
    loaded: dict[str, Any],
    *,
    preserve_pending_interaction: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    sanitized = dict(loaded)
    removed_keys: list[str] = []
    for key in (
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


def _merge_state(*, base: dict[str, Any], reply_text: str) -> dict[str, Any]:
    merged = dict(base)        
    merged["response_text"] = str(reply_text or "").strip() or None
    merged["last_updated_at"] = datetime.now(timezone.utc).isoformat()
    return merged


def _build_checkpoint_payload(
    *,
    cognition_state: dict[str, Any],
    request_text: str,
    previous_checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    task_record = _task_record_dict(cognition_state.get("task_record"))
    if task_record is None:
        raise ValueError("pdca_checkpoint.missing_task_record")
    previous_state = previous_checkpoint.get("state") if isinstance(previous_checkpoint, dict) and isinstance(previous_checkpoint.get("state"), dict) else {}
    cycle_index = _as_optional_int(cognition_state.get("cycle_index"))
    if cycle_index is None:
        cycle_index = _as_optional_int(previous_state.get("cycle_index"))
    payload: dict[str, Any] = {
        "task_record": task_record,
        "last_user_message": str(cognition_state.get("last_user_message") or request_text or "").strip() or None,
    }
    check_provenance = str(cognition_state.get("check_provenance") or "").strip()
    if check_provenance:
        payload["check_provenance"] = check_provenance
    if cycle_index is not None:
        payload["cycle_index"] = cycle_index
    return payload


def _next_status(*, cognition_state: dict[str, Any], reply_text: str) -> str:
    task_record = _task_record_dict(cognition_state.get("task_record")) or {}
    task_status = str(task_record.get("status") or "").strip().lower()
    if task_status in {"done", "failed", "waiting_user"}:
        return task_status
    if str(reply_text or "").strip():
        return "done"
    return "queued"


def _task_record_dict(task_record_value: Any) -> dict[str, Any] | None:
    if isinstance(task_record_value, dict):
        return dict(task_record_value)
    if hasattr(task_record_value, "to_dict"):
        serialized = task_record_value.to_dict()
        if isinstance(serialized, dict):
            return dict(serialized)
    return None


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
    state = checkpoint.get("state") if isinstance(checkpoint.get("state"), dict) else {}
    value = _as_optional_int(state.get("cycle_index"))
    if value is not None:
        return value
    return 0


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
    if signal_type == "pdca.resume_requested":
        base_state["check_provenance"] = "slice_resume"
        return
    if isinstance(checkpoint, dict):
        base_state["check_provenance"] = "do"


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
        event={
            "type": "agent.transition",
            "phase": str(phase or "").strip(),
            "correlation_id": corr,
            "detail": {"presence_event_family": str(event_family or "").strip()},
        },
        channel_type=incoming.channel_type,
        channel_target=incoming.address,
        user_id=incoming.person_id,
        message_id=incoming.message_id,
        correlation_id=corr,
    )
