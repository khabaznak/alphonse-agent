from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.nervous_system.pdca_queue_store import (
    append_pdca_event,
    get_latest_pdca_task_for_conversation,
    update_pdca_task_metadata,
    update_pdca_task_status,
    upsert_pdca_task,
)
from alphonse.agent.observability.log_manager import get_log_manager
from alphonse.agent.nervous_system.senses.bus import Bus
from alphonse.agent.services.pdca_queue_runner import emit_pdca_dispatch_kick

_LOG = get_log_manager()


@dataclass(frozen=True)
class BufferedTaskInput:
    message_id: str | None
    correlation_id: str
    channel_type: str
    channel_target: str
    actor_id: str | None
    text: str
    attachments: list[dict[str, Any]]
    received_at: str | None = None
    timezone: str | None = None
    locale: str | None = None

    def normalized_text(self) -> str:
        text = str(self.text or "").strip()
        if text:
            return text
        return _render_attachment_summary(self.attachments)

    def normalized_received_at(self) -> str:
        rendered = str(self.received_at or "").strip()
        if rendered:
            return rendered
        return now_iso()


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


def normalize_buffered_attachments(raw: Any) -> list[dict[str, Any]]:
    return _normalize_attachments(raw)


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
    task_record: TaskRecord,
    buffered_input: BufferedTaskInput,
    bus: Bus | None = None,
    force_new_task: bool = False,
) -> str:
    now = buffered_input.normalized_received_at()
    channel_type = str(buffered_input.channel_type or "").strip()
    channel_target = str(buffered_input.channel_target or "").strip()
    if not channel_type:
        raise ValueError("invalid_conscious_payload: missing channel type")
    if not channel_target:
        raise ValueError("invalid_conscious_payload: missing channel target")
    message_id = str(buffered_input.message_id or "").strip()
    session_key = f"{channel_type}:{channel_target}"
    attachments = normalize_buffered_attachments(buffered_input.attachments)
    user_text = buffered_input.normalized_text()

    existing_task: dict[str, Any] | None = None
    if not force_new_task:
        try:
            candidate = get_latest_pdca_task_for_conversation(
                conversation_key=session_key,
                statuses=["running", "waiting_user"],
            )
        except Exception:
            candidate = None
        existing_task = candidate if isinstance(candidate, dict) else None
    existing = (
        existing_task
        if isinstance(existing_task, dict)
        and _existing_task_matches_ingress(
            existing_task=existing_task,
            session_user_id=task_record.user_id,
            session_key=session_key,
        )
        else None
    )
    if isinstance(existing, dict):
        task_id = str(existing.get("task_id") or "").strip()
        metadata, buffered_count = _build_ingress_metadata(
            existing=existing,
            task_record=task_record,
            buffered_input=buffered_input,
            session_key=session_key,
            message_id=message_id,
            user_text=user_text,
            attachments=attachments,
            now=now,
        )
        status = str(existing.get("status") or "").strip().lower()
        update_pdca_task_metadata(task_id=task_id, metadata=metadata)
        if status in {"waiting_user", "paused"}:
            update_pdca_task_status(task_id=task_id, status="queued")
        _emit_dispatch_kick(
            bus=bus,
            task_id=task_id,
            correlation_id=task_record.correlation_id or "",
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
            correlation_id=task_record.correlation_id,
        )
        _LOG.emit(
            event="pdca.input.buffered",
            component="services.pdca_ingress",
            correlation_id=task_record.correlation_id or None,
            channel=channel_type or None,
            payload={
                "task_id": task_id,
                "buffered_count": buffered_count,
                "input_dirty": bool(metadata.get("input_dirty")),
            },
        )
        return task_id

    metadata, _ = _build_ingress_metadata(
        existing=None,
        task_record=task_record,
        buffered_input=buffered_input,
        session_key=session_key,
        message_id=message_id,
        user_text=user_text,
        attachments=attachments,
        now=now,
    )
    task_id = upsert_pdca_task(
        {
            "owner_id": task_record.user_id,
            "conversation_key": session_key,
            "session_id": None,
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
        correlation_id=task_record.correlation_id,
    )
    _emit_dispatch_kick(
        bus=bus,
        task_id=task_id,
        correlation_id=task_record.correlation_id,
        reason="ingress_task_created",
    )
    return task_id


def _build_ingress_metadata(
    *,
    existing: dict[str, Any] | None,
    task_record: TaskRecord,
    buffered_input: BufferedTaskInput,
    session_key: str,
    message_id: str,
    user_text: str,
    attachments: list[dict[str, Any]],
    now: str,
) -> tuple[dict[str, Any], int]:
    correlation_id = str(task_record.correlation_id or "").strip()
    actor_id = task_record.user_id
    channel_type = str(buffered_input.channel_type or "").strip()
    channel_target = str(buffered_input.channel_target or "").strip()
    if isinstance(existing, dict):
        metadata = dict(existing.get("metadata") or {}) if isinstance(existing.get("metadata"), dict) else {}
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
        metadata["last_enqueued_at"] = now
        existing_state = metadata.get("state") if isinstance(metadata.get("state"), dict) else {}
        metadata["state"] = _build_metadata_state(
            existing_state=existing_state,
            task_record=task_record,
            buffered_input=buffered_input,
            session_key=session_key,
            message_id=message_id,
            check_provenance="slice_resume",
        )
        status = str(existing.get("status") or "").strip().lower()
        if status == "running":
            metadata["input_dirty"] = True
    else:
        metadata = {
            "initial_message_id": message_id or None,
            "initial_correlation_id": correlation_id or None,
            "input_dirty": False,
            "next_unconsumed_index": 0,
            "inputs": [
                {
                    "message_id": message_id,
                    "correlation_id": correlation_id,
                    "text": user_text,
                    "channel": channel_type,
                    "actor_id": actor_id,
                    "attachments": attachments,
                    "received_at": now,
                    "consumed_at": None,
                    "sequence": 1,
                }
            ],
            "state": _build_metadata_state(
                existing_state=None,
                task_record=task_record,
                buffered_input=buffered_input,
                session_key=session_key,
                message_id=message_id,
                check_provenance="entry",
            ),
        }
        buffered_count = len(metadata["inputs"])

    metadata["pending_user_text"] = user_text
    metadata["last_user_message"] = user_text
    metadata["last_user_channel"] = channel_type
    metadata["last_user_target"] = channel_target
    metadata["last_user_message_id"] = message_id
    metadata["last_enqueue_correlation_id"] = correlation_id
    return metadata, buffered_count


def _build_metadata_state(
    *,
    existing_state: dict[str, Any] | None,
    task_record: TaskRecord,
    buffered_input: BufferedTaskInput,
    session_key: str,
    message_id: str,
    check_provenance: str,
) -> dict[str, Any]:
    state = dict(existing_state or {}) if isinstance(existing_state, dict) else {}
    state["conversation_key"] = state.get("conversation_key") or session_key
    state["chat_id"] = state.get("chat_id") or session_key
    state["correlation_id"] = str(task_record.correlation_id or "").strip() or state.get("correlation_id")
    state["task_record"] = task_record.to_dict()
    state["channel_type"] = str(buffered_input.channel_type or "").strip() or state.get("channel_type")
    state["channel_target"] = str(buffered_input.channel_target or "").strip() or state.get("channel_target")
    if message_id:
        state["message_id"] = message_id
    elif isinstance(existing_state, dict):
        state["message_id"] = state.get("message_id")
    else:
        state["message_id"] = message_id
    if str(buffered_input.actor_id or "").strip():
        state["actor_person_id"] = str(buffered_input.actor_id or "").strip()
    if str(buffered_input.timezone or "").strip():
        state["timezone"] = str(buffered_input.timezone or "").strip()
    if str(buffered_input.locale or "").strip():
        state["locale"] = str(buffered_input.locale or "").strip()
    state["check_provenance"] = check_provenance
    return state


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

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _emit_dispatch_kick(
    *,
    bus: Bus | None,
    task_id: str,
    correlation_id: str,
    reason: str,
) -> None:
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
