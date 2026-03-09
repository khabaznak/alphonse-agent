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
        metadata["pending_user_text"] = user_text
        metadata["last_user_message"] = user_text
        metadata["last_user_channel"] = incoming.channel_type
        metadata["last_user_target"] = incoming.address
        metadata["last_enqueue_correlation_id"] = correlation_id
        metadata["last_enqueued_at"] = now
        update_pdca_task_metadata(task_id=task_id, metadata=metadata)
        if str(existing.get("status") or "").strip().lower() in {"waiting_user", "paused"}:
            update_pdca_task_status(task_id=task_id, status="queued")
        append_pdca_event(
            task_id=task_id,
            event_type="incoming.user_message",
            payload={"text": user_text, "channel": incoming.channel_type},
            correlation_id=correlation_id,
        )
        return task_id

    metadata = {
        "pending_user_text": user_text,
        "last_user_message": user_text,
        "last_user_channel": incoming.channel_type,
        "last_user_target": incoming.address,
        "last_enqueue_correlation_id": correlation_id,
        "state": {
            "conversation_key": session_key,
            "chat_id": session_key,
            "channel_type": incoming.channel_type,
            "channel_target": incoming.address,
            "actor_person_id": incoming.person_id,
            "incoming_user_id": str(payload.get("user_id") or "").strip() or None,
            "incoming_user_name": str(payload.get("user_name") or "").strip() or None,
            "locale": state.get("locale"),
            "tone": state.get("tone"),
            "address_style": state.get("address_style"),
            "timezone": state.get("timezone"),
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
