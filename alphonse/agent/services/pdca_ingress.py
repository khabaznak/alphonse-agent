from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from alphonse.agent.actions.session_context import IncomingContext
from alphonse.agent.nervous_system.pdca_queue_store import (
    append_pdca_event,
    get_latest_pdca_task_for_conversation,
    update_pdca_task_metadata,
    update_pdca_task_status,
    upsert_pdca_task,
)
from alphonse.agent.observability.log_manager import get_log_manager
from alphonse.agent.services.pdca_queue_runner import emit_pdca_dispatch_kick
from alphonse.agent.session.day_state import render_recent_conversation_block

_LOG = get_log_manager()
_STEERING_SCOPES = {"conversation", "owner_only", "targeted"}


def _normalize_inputs(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    raw = metadata.get("inputs")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        record = {
            "message_id": str(item.get("message_id") or "").strip(),
            "correlation_id": str(item.get("correlation_id") or "").strip(),
            "text": str(item.get("text") or "").strip(),
            "channel": str(item.get("channel") or "").strip(),
            "actor_id": str(item.get("actor_id") or "").strip() or None,
            "received_at": str(item.get("received_at") or "").strip(),
            "consumed_at": str(item.get("consumed_at") or "").strip() or None,
            "sequence": int(item.get("sequence") or 0),
        }
        if not record["text"]:
            continue
        out.append(record)
    return out


def _append_input_record(
    *,
    metadata: dict[str, Any],
    incoming: IncomingContext,
    correlation_id: str,
    user_text: str,
    actor_id: str | None,
    now: str,
) -> tuple[dict[str, Any], int]:
    inputs = _normalize_inputs(metadata)
    message_id = str(incoming.message_id or "").strip()
    cid = str(correlation_id or "").strip()
    dedupe_id = f"{message_id}|{cid}|{user_text}"
    seen = {
        f"{str(item.get('message_id') or '').strip()}|{str(item.get('correlation_id') or '').strip()}|{str(item.get('text') or '').strip()}"
        for item in inputs
    }
    if dedupe_id not in seen:
        max_seq = max((int(item.get("sequence") or 0) for item in inputs), default=0)
        inputs.append(
            {
                "message_id": message_id,
                "correlation_id": cid,
                "text": user_text,
                "channel": str(incoming.channel_type or "").strip(),
                "actor_id": str(actor_id or "").strip() or None,
                "received_at": now,
                "consumed_at": None,
                "sequence": max_seq + 1,
            }
        )
    metadata["inputs"] = inputs
    next_unconsumed = int(metadata.get("next_unconsumed_index") or 0)
    if next_unconsumed < 0:
        next_unconsumed = 0
    metadata["next_unconsumed_index"] = min(next_unconsumed, len(inputs))
    return metadata, len(inputs)


def _normalize_steering_policy(*, metadata: dict[str, Any], owner_id: str) -> dict[str, Any]:
    scope = str(metadata.get("steering_scope") or "").strip().lower()
    if scope not in _STEERING_SCOPES:
        scope = "conversation"
    metadata["steering_scope"] = scope
    allowed = metadata.get("allowed_actor_ids")
    if isinstance(allowed, list):
        metadata["allowed_actor_ids"] = [str(item).strip() for item in allowed if str(item).strip()]
    else:
        metadata["allowed_actor_ids"] = []
    metadata["target_actor_id"] = str(metadata.get("target_actor_id") or "").strip() or None
    metadata["target_wait_timeout_at"] = str(metadata.get("target_wait_timeout_at") or "").strip() or None
    if not isinstance(metadata.get("steering_decision_log"), list):
        metadata["steering_decision_log"] = []
    metadata.setdefault("steering_owner_id", str(owner_id or "").strip() or None)
    return metadata


def _resolve_actor_id(
    *,
    incoming: IncomingContext,
    payload: dict[str, Any],
    session_user_id: str,
) -> str:
    for value in (
        payload.get("person_id"),
        payload.get("user_id"),
        incoming.person_id,
        session_user_id,
        incoming.address,
    ):
        rendered = str(value or "").strip()
        if rendered:
            return rendered
    return "unknown"


def _append_steering_decision(
    *,
    metadata: dict[str, Any],
    now: str,
    incoming: IncomingContext,
    correlation_id: str,
    actor_id: str,
    decision: str,
    reason: str,
) -> None:
    scope = str(metadata.get("steering_scope") or "").strip().lower() or "conversation"
    target_actor = str(metadata.get("target_actor_id") or "").strip() or None
    timeout_at = str(metadata.get("target_wait_timeout_at") or "").strip() or None
    log = metadata.get("steering_decision_log") if isinstance(metadata.get("steering_decision_log"), list) else []
    next_sequence = max((int(item.get("sequence") or 0) for item in log if isinstance(item, dict)), default=0) + 1
    log.append(
        {
            "sequence": next_sequence,
            "occurred_at": now,
            "message_id": str(incoming.message_id or "").strip() or None,
            "correlation_id": str(correlation_id or "").strip() or None,
            "actor_id": actor_id,
            "decision": decision,
            "reason": reason,
            "scope": scope,
            "target_actor_id": target_actor,
            "target_wait_timeout_at": timeout_at,
        }
    )
    metadata["steering_decision_log"] = log[-100:]


def _parse_iso_utc(value: object | None) -> datetime | None:
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


def _is_steering_eligible(
    *,
    metadata: dict[str, Any],
    owner_id: str,
    actor_id: str,
    now: str,
) -> tuple[bool, str]:
    scope = str(metadata.get("steering_scope") or "").strip().lower() or "conversation"
    allowed = metadata.get("allowed_actor_ids") if isinstance(metadata.get("allowed_actor_ids"), list) else []
    allowed_set = {str(item).strip() for item in allowed if str(item).strip()}
    if allowed_set and actor_id not in allowed_set:
        return False, "actor_not_allowed"
    if scope == "owner_only":
        owner = str(owner_id or "").strip()
        if owner and actor_id != owner:
            return False, "owner_only_scope"
        return True, "owner_scope_ok"
    if scope == "targeted":
        target_actor = str(metadata.get("target_actor_id") or "").strip()
        if not target_actor:
            return True, "target_missing_fallback_to_conversation"
        if actor_id == target_actor:
            return True, "target_actor_match"
        timeout_at = _parse_iso_utc(metadata.get("target_wait_timeout_at"))
        now_dt = _parse_iso_utc(now) or datetime.now(timezone.utc)
        if timeout_at is not None and now_dt >= timeout_at:
            return True, "target_timeout_fallback"
        return False, "target_waiting_for_actor"
    return True, "conversation_scope"


def enqueue_pdca_slice(
    *,
    context: dict[str, Any],
    incoming: IncomingContext,
    state: dict[str, Any],
    session_key: str,
    session_user_id: str,
    day_session: dict[str, Any],
    payload: dict[str, Any],
    correlation_id: str,
) -> str:
    now = now_iso()
    user_text = str(payload.get("text") or "").strip()
    existing = get_latest_pdca_task_for_conversation(
        conversation_key=session_key,
        statuses=["waiting_user", "queued", "running", "paused"],
    )
    if isinstance(existing, dict):
        task_id = str(existing.get("task_id") or "").strip()
        owner_id = str(existing.get("owner_id") or "").strip()
        metadata = dict(existing.get("metadata") or {}) if isinstance(existing.get("metadata"), dict) else {}
        metadata = _normalize_steering_policy(metadata=metadata, owner_id=owner_id)
        actor_id = _resolve_actor_id(incoming=incoming, payload=payload, session_user_id=session_user_id)
        accepted, reason = _is_steering_eligible(
            metadata=metadata,
            owner_id=owner_id,
            actor_id=actor_id,
            now=now,
        )
        _append_steering_decision(
            metadata=metadata,
            now=now,
            incoming=incoming,
            correlation_id=correlation_id,
            actor_id=actor_id,
            decision="accepted" if accepted else "rejected",
            reason=reason,
        )
        if not accepted:
            update_pdca_task_metadata(task_id=task_id, metadata=metadata)
            append_pdca_event(
                task_id=task_id,
                event_type="incoming.user_message.rejected",
                payload={
                    "text": user_text,
                    "channel": incoming.channel_type,
                    "actor_id": actor_id,
                    "reason": reason,
                },
                correlation_id=correlation_id,
            )
            _LOG.emit(
                event="pdca.input.rejected",
                component="services.pdca_ingress",
                correlation_id=correlation_id or None,
                channel=str(incoming.channel_type or "").strip() or None,
                payload={"task_id": task_id, "actor_id": actor_id, "reason": reason},
            )
            return task_id
        metadata, buffered_count = _append_input_record(
            metadata=metadata,
            incoming=incoming,
            correlation_id=correlation_id,
            user_text=user_text,
            actor_id=actor_id,
            now=now,
        )
        metadata["pending_user_text"] = user_text
        metadata["last_user_message"] = user_text
        metadata["last_user_channel"] = incoming.channel_type
        metadata["last_user_target"] = incoming.address
        metadata["last_user_message_id"] = incoming.message_id
        metadata["last_enqueue_correlation_id"] = correlation_id
        metadata["last_enqueued_at"] = now
        status = str(existing.get("status") or "").strip().lower()
        if status == "running":
            metadata["input_dirty"] = True
        update_pdca_task_metadata(task_id=task_id, metadata=metadata)
        if status in {"waiting_user", "paused"}:
            update_pdca_task_status(task_id=task_id, status="queued")
        _emit_dispatch_kick(
            context=context,
            task_id=task_id,
            correlation_id=correlation_id,
            reason="ingress_existing_task",
        )
        append_pdca_event(
            task_id=task_id,
            event_type="incoming.user_message",
            payload={
                "text": user_text,
                "channel": incoming.channel_type,
                "buffered_count": buffered_count,
                "input_dirty": bool(metadata.get("input_dirty")),
            },
            correlation_id=correlation_id,
        )
        _LOG.emit(
            event="pdca.input.buffered",
            component="services.pdca_ingress",
            correlation_id=correlation_id or None,
            channel=str(incoming.channel_type or "").strip() or None,
            payload={
                "task_id": task_id,
                "buffered_count": buffered_count,
                "input_dirty": bool(metadata.get("input_dirty")),
            },
        )
        return task_id

    metadata = {
        "steering_scope": "conversation",
        "allowed_actor_ids": [],
        "target_actor_id": None,
        "target_wait_timeout_at": None,
        "steering_decision_log": [],
        "pending_user_text": user_text,
        "last_user_message": user_text,
        "last_user_channel": incoming.channel_type,
        "last_user_target": incoming.address,
        "last_user_message_id": incoming.message_id,
        "last_enqueue_correlation_id": correlation_id,
        "input_dirty": False,
        "next_unconsumed_index": 0,
        "inputs": [
            {
                "message_id": str(incoming.message_id or "").strip(),
                "correlation_id": str(correlation_id or "").strip(),
                "text": user_text,
                "channel": str(incoming.channel_type or "").strip(),
                "actor_id": _resolve_actor_id(incoming=incoming, payload=payload, session_user_id=session_user_id),
                "received_at": now,
                "consumed_at": None,
                "sequence": 1,
            }
        ],
        "state": {
            "conversation_key": session_key,
            "chat_id": session_key,
            "channel_type": incoming.channel_type,
            "channel_target": incoming.address,
            "actor_person_id": incoming.person_id,
            "message_id": incoming.message_id,
            "incoming_user_id": str(payload.get("user_id") or "").strip() or None,
            "incoming_user_name": str(payload.get("user_name") or "").strip() or None,
            "locale": state.get("locale"),
            "tone": state.get("tone"),
            "address_style": state.get("address_style"),
            "timezone": state.get("timezone"),
            "session_state": day_session,
            "recent_conversation_block": render_recent_conversation_block(day_session),
        },
    }
    task_id = upsert_pdca_task(
        {
            "owner_id": session_user_id,
            "conversation_key": session_key,
            "session_id": str(day_session.get("session_id") or "").strip() or None,
            "status": "queued",
            "priority": 100,
            "next_run_at": now,
            "slice_cycles": 5,
            "metadata": metadata,
        }
    )
    append_pdca_event(
        task_id=task_id,
        event_type="incoming.task_created",
        payload={"channel": incoming.channel_type, "target": incoming.address},
        correlation_id=correlation_id,
    )
    _emit_dispatch_kick(
        context=context,
        task_id=task_id,
        correlation_id=correlation_id,
        reason="ingress_task_created",
    )
    return task_id


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit_dispatch_kick(
    *,
    context: dict[str, Any],
    task_id: str,
    correlation_id: str,
    reason: str,
) -> None:
    bus = context.get("ctx")
    emitted = emit_pdca_dispatch_kick(
        bus=bus if hasattr(bus, "emit") else None,
        task_id=task_id,
        reason=reason,
        correlation_id=correlation_id,
        source="pdca_ingress",
    )
    if not emitted:
        return
    _LOG.emit(
        event="pdca.dispatch.kick_emitted",
        component="services.pdca_ingress",
        correlation_id=correlation_id or None,
        payload={"task_id": task_id, "reason": reason},
    )
