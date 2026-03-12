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
from alphonse.agent.session.day_state import render_recent_conversation_block

_LOG = get_log_manager()


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
        metadata = dict(existing.get("metadata") or {}) if isinstance(existing.get("metadata"), dict) else {}
        metadata, buffered_count = _append_input_record(
            metadata=metadata,
            incoming=incoming,
            correlation_id=correlation_id,
            user_text=user_text,
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
    return task_id


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
