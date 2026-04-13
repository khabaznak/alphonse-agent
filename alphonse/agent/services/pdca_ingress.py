from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from alphonse.agent.actions.conscious_message_handler import IncomingMessageEnvelope
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.nervous_system.pdca_queue_store import (
    append_pdca_event,
    update_pdca_task_metadata,
    update_pdca_task_status,
    upsert_pdca_task,
)
from alphonse.agent.observability.log_manager import get_log_manager
from alphonse.agent.services.pdca_queue_runner import emit_pdca_dispatch_kick
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
            "actor_id": str(item.get("actor_id") or "").strip() or None,
            "attachments": _normalize_attachments(item.get("attachments")),
            "received_at": str(item.get("received_at") or "").strip(),
            "consumed_at": str(item.get("consumed_at") or "").strip() or None,
            "sequence": int(item.get("sequence") or 0),
        }
        if not record["text"] and not record["attachments"]:
            continue
        out.append(record)
    return out


def _normalize_attachments(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    canonical_keys = {
        "kind",
        "provider",
        "file_id",
        "url",
        "mime_type",
        "size_bytes",
        "duration_seconds",
        "width",
        "height",
        "caption",
        "provider_event_ref",
        "contact",
        "contact_user_id",
    }
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip().lower()
        provider = str(item.get("provider") or "").strip().lower()
        file_id = str(item.get("file_id") or "").strip() or None
        if not kind:
            continue
        normalized: dict[str, Any] = {
            "kind": kind,
            "provider": provider or None,
            "file_id": file_id,
            "url": str(item.get("url") or "").strip() or None,
            "mime_type": str(item.get("mime_type") or "").strip() or None,
            "size_bytes": int(item.get("size_bytes") or 0) or None,
            "duration_seconds": int(item.get("duration_seconds") or 0) or None,
            "width": int(item.get("width") or 0) or None,
            "height": int(item.get("height") or 0) or None,
            "caption": str(item.get("caption") or "").strip() or None,
            "provider_event_ref": item.get("provider_event_ref") if isinstance(item.get("provider_event_ref"), dict) else None,
            "contact": item.get("contact") if isinstance(item.get("contact"), dict) else None,
            "contact_user_id": str(item.get("contact_user_id")).strip()
            if item.get("contact_user_id") is not None
            else None,
        }
        for key, value in item.items():
            if key in canonical_keys:
                continue
            safe = _json_safe_copy(value)
            if safe is not _JSON_UNSAFE:
                normalized[key] = safe
        out.append(normalized)
    return out


_JSON_UNSAFE = object()


def _json_safe_copy(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        copied: list[Any] = []
        for item in value:
            safe = _json_safe_copy(item)
            if safe is _JSON_UNSAFE:
                return _JSON_UNSAFE
            copied.append(safe)
        return copied
    if isinstance(value, dict):
        copied_dict: dict[str, Any] = {}
        for key, item in value.items():
            safe = _json_safe_copy(item)
            if safe is _JSON_UNSAFE:
                return _JSON_UNSAFE
            copied_dict[str(key)] = safe
        return copied_dict
    return _JSON_UNSAFE


def _extract_envelope_attachments(envelope: IncomingMessageEnvelope) -> list[dict[str, Any]]:
    content = envelope.content if isinstance(envelope.content, dict) else {}
    attachments = content.get("attachments") if isinstance(content.get("attachments"), list) else []
    return _normalize_attachments(attachments)


def _render_attachment_summary(attachments: list[dict[str, Any]]) -> str:
    if not attachments:
        return ""
    parts: list[str] = []
    for item in attachments:
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


def _attachment_fingerprint(attachments: list[dict[str, Any]]) -> str:
    return ",".join(
        sorted(
            [
                f"{str(item.get('kind') or '').strip()}:{str(item.get('file_id') or item.get('url') or '').strip()}"
                for item in attachments
            ]
        )
    )


def _append_input_record(
    *,
    metadata: dict[str, Any],
    message_id: str | None,
    channel_type: str,
    correlation_id: str,
    user_text: str,
    attachments: list[dict[str, Any]],
    actor_id: str | None,
    now: str,
) -> tuple[dict[str, Any], int]:
    inputs = _normalize_inputs(metadata)
    message_id = str(message_id or "").strip()
    cid = str(correlation_id or "").strip()
    text = str(user_text or "").strip()
    attachment_fp = _attachment_fingerprint(attachments)
    dedupe_id = f"{message_id}|{cid}|{text}|{attachment_fp}"
    seen = {
        f"{str(item.get('message_id') or '').strip()}|{str(item.get('correlation_id') or '').strip()}|"
        f"{str(item.get('text') or '').strip()}|{_attachment_fingerprint(_normalize_attachments(item.get('attachments')))}"
        for item in inputs
    }
    if dedupe_id not in seen:
        max_seq = max((int(item.get("sequence") or 0) for item in inputs), default=0)
        inputs.append(
            {
                "message_id": message_id,
                "correlation_id": cid,
                "text": text,
                "channel": str(channel_type or "").strip(),
                "actor_id": str(actor_id or "").strip() or None,
                "attachments": attachments,
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


def _resolve_actor_id(
    *,
    envelope: IncomingMessageEnvelope,
    session_user_id: str,
) -> str:
    actor = envelope.actor if isinstance(envelope.actor, dict) else {}
    for value in (actor.get("user_id"), session_user_id):
        rendered = str(value or "").strip()
        if rendered:
            return rendered
    return "unknown"


def _ensure_initial_identity(
    *,
    metadata: dict[str, Any],
    initial_message_id: str,
    initial_correlation_id: str,
) -> None:
    fallback_message_id = str(initial_message_id or "").strip()
    fallback_correlation_id = str(initial_correlation_id or "").strip()
    if not fallback_message_id or not fallback_correlation_id:
        inputs = metadata.get("inputs") if isinstance(metadata.get("inputs"), list) else []
        if inputs and isinstance(inputs[0], dict):
            first = inputs[0]
            if not fallback_message_id:
                fallback_message_id = str(first.get("message_id") or "").strip()
            if not fallback_correlation_id:
                fallback_correlation_id = str(first.get("correlation_id") or "").strip()
    message_id = fallback_message_id
    correlation_id = fallback_correlation_id
    if message_id and not str(metadata.get("initial_message_id") or "").strip():
        metadata["initial_message_id"] = message_id
    if correlation_id and not str(metadata.get("initial_correlation_id") or "").strip():
        metadata["initial_correlation_id"] = correlation_id


def enqueue_pdca_slice(
    *,
    context: dict[str, Any],
    session_user_id: str,
    day_session: dict[str, Any],
    envelope: IncomingMessageEnvelope,
    correlation_id: str,
    task_record: TaskRecord,
    existing_task: dict[str, Any] | None = None,
) -> str:
    now = now_iso()
    channel = envelope.channel if isinstance(envelope.channel, dict) else {}
    content = envelope.content if isinstance(envelope.content, dict) else {}
    channel_type = str(channel.get("type") or "").strip()
    channel_target = str(channel.get("target") or "").strip()
    message_id = str(envelope.message_id or "").strip()
    session_key = f"{channel_type}:{channel_target}"
    attachments = _extract_envelope_attachments(envelope)
    user_text = str(content.get("text") or "").strip()
    if not user_text:
        user_text = _render_attachment_summary(attachments)
    force_new_task = _force_new_task(envelope=envelope)
    existing = (
        existing_task
        if isinstance(existing_task, dict)
        and not force_new_task
        and _existing_task_matches_ingress(
            existing_task=existing_task,
            session_user_id=session_user_id,
            session_key=session_key,
        )
        else None
    )
    if isinstance(existing, dict):
        task_id = str(existing.get("task_id") or "").strip()
        metadata = dict(existing.get("metadata") or {}) if isinstance(existing.get("metadata"), dict) else {}
        actor_id = _resolve_actor_id(envelope=envelope, session_user_id=session_user_id)
        _ensure_initial_identity(
            metadata=metadata,
            initial_message_id="",
            initial_correlation_id="",
        )
        metadata, buffered_count = _append_input_record(
            metadata=metadata,
            message_id=message_id,
            channel_type=channel_type,
            correlation_id=correlation_id,
            user_text=user_text,
            attachments=attachments,
            actor_id=actor_id,
            now=now,
        )
        metadata["pending_user_text"] = user_text
        metadata["last_user_message"] = user_text
        metadata["last_user_channel"] = channel_type
        metadata["last_user_target"] = channel_target
        metadata["last_user_message_id"] = message_id
        metadata["last_enqueue_correlation_id"] = correlation_id
        metadata["last_enqueued_at"] = now
        state_snapshot = metadata.get("state") if isinstance(metadata.get("state"), dict) else {}
        state_snapshot = dict(state_snapshot)
        state_snapshot["conversation_key"] = state_snapshot.get("conversation_key") or session_key
        state_snapshot["chat_id"] = state_snapshot.get("chat_id") or session_key
        state_snapshot["correlation_id"] = str(correlation_id or "").strip() or state_snapshot.get("correlation_id")
        state_snapshot["task_record"] = task_record.to_dict()
        state_snapshot["message_id"] = message_id or state_snapshot.get("message_id")
        state_snapshot["session_state"] = state_snapshot.get("session_state") or day_session
        state_snapshot["recent_conversation_block"] = (
            state_snapshot.get("recent_conversation_block") or render_recent_conversation_block(day_session)
        )
        state_snapshot["check_provenance"] = "slice_resume"
        metadata["state"] = state_snapshot
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
                "channel": channel_type,
                "buffered_count": buffered_count,
                "input_dirty": bool(metadata.get("input_dirty")),
                "attachment_count": len(attachments),
            },
            correlation_id=correlation_id,
        )
        _LOG.emit(
            event="pdca.input.buffered",
            component="services.pdca_ingress",
            correlation_id=correlation_id or None,
            channel=channel_type or None,
            payload={
                "task_id": task_id,
                "buffered_count": buffered_count,
                "input_dirty": bool(metadata.get("input_dirty")),
            },
        )
        return task_id

    task_record.task_id = None
    task_record.user_id = task_record.user_id or str(session_user_id or "").strip() or None
    task_record.correlation_id = task_record.correlation_id or str(correlation_id or "").strip()
    task_record.goal = task_record.goal or user_text
    task_record.status = task_record.status or "running"
    metadata = {
        "pending_user_text": user_text,
        "last_user_message": user_text,
        "last_user_channel": channel_type,
        "last_user_target": channel_target,
        "last_user_message_id": message_id,
        "last_enqueue_correlation_id": correlation_id,
        "initial_message_id": message_id or None,
        "initial_correlation_id": str(correlation_id or "").strip() or None,
        "input_dirty": False,
        "next_unconsumed_index": 0,
        "inputs": [
            {
                "message_id": message_id,
                "correlation_id": str(correlation_id or "").strip(),
                "text": user_text,
                "channel": channel_type,
                "actor_id": _resolve_actor_id(envelope=envelope, session_user_id=session_user_id),
                "attachments": attachments,
                "received_at": now,
                "consumed_at": None,
                "sequence": 1,
            }
        ],
        "state": {
            "conversation_key": session_key,
            "chat_id": session_key,
            "correlation_id": str(correlation_id or "").strip() or None,
            "task_record": task_record.to_dict(),
            "message_id": message_id,
            "session_state": day_session,
            "recent_conversation_block": render_recent_conversation_block(day_session),
            "check_provenance": "entry",
        },
    }
    source = _payload_source(envelope=envelope)
    if source:
        metadata["trigger_source"] = source
    job_id = _payload_job_id(envelope=envelope)
    if job_id:
        metadata["trigger_job_id"] = job_id
    metadata["force_new_task"] = bool(force_new_task)
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
    task_record.task_id = task_id
    metadata_state = metadata.get("state") if isinstance(metadata.get("state"), dict) else {}
    metadata_state["task_record"] = task_record.to_dict()
    metadata["state"] = metadata_state
    update_pdca_task_metadata(task_id=task_id, metadata=metadata)
    append_pdca_event(
        task_id=task_id,
        event_type="incoming.task_created",
        payload={"channel": channel_type, "target": channel_target},
        correlation_id=correlation_id,
    )
    _emit_dispatch_kick(
        context=context,
        task_id=task_id,
        correlation_id=correlation_id,
        reason="ingress_task_created",
    )
    return task_id


def _force_new_task(*, envelope: IncomingMessageEnvelope) -> bool:
    controls = envelope.controls if isinstance(envelope.controls, dict) else {}
    if _as_bool(controls.get("force_new_task")):
        return True
    source = _payload_source(envelope=envelope)
    return source in {"job_runner", "job_trigger", "jobs_reconcile"}


def _existing_task_matches_ingress(
    *,
    existing_task: dict[str, Any],
    session_user_id: str,
    session_key: str,
) -> bool:
    if str(existing_task.get("owner_id") or "").strip() != str(session_user_id or "").strip():
        return False
    if str(existing_task.get("conversation_key") or "").strip() != str(session_key or "").strip():
        return False
    return True


def _payload_source(*, envelope: IncomingMessageEnvelope) -> str:
    metadata = envelope.metadata if isinstance(envelope.metadata, dict) else {}
    rendered = str(metadata.get("source") or "").strip().lower()
    return rendered


def _payload_job_id(*, envelope: IncomingMessageEnvelope) -> str:
    metadata = envelope.metadata if isinstance(envelope.metadata, dict) else {}
    job = metadata.get("job") if isinstance(metadata.get("job"), dict) else {}
    candidate = str(metadata.get("job_id") or job.get("job_id") or "").strip()
    return candidate


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


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
