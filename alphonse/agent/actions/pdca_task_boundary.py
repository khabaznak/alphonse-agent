from __future__ import annotations

from typing import Any

from alphonse.agent.actions.conscious_message_handler import IncomingMessageEnvelope
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.nervous_system.pdca_queue_store import get_latest_pdca_task_for_conversation
from alphonse.agent.session.day_state import render_recent_conversation_block


def select_pending_pdca_task_for_message(
    *,
    envelope: IncomingMessageEnvelope,
    session_user_id: str,
) -> dict[str, Any] | None:
    if _force_new_task(envelope=envelope):
        return None
    conversation_key = _conversation_key(envelope)
    if not conversation_key:
        return None
    try:
        existing = get_latest_pdca_task_for_conversation(
            conversation_key=conversation_key,
            statuses=["waiting_user"],
        )
    except Exception:
        return None
    if not isinstance(existing, dict):
        return None
    if str(existing.get("owner_id") or "").strip() != str(session_user_id or "").strip():
        return None
    return existing


def build_task_record_for_message(
    *,
    envelope: IncomingMessageEnvelope,
    session_user_id: str,
    day_session: dict[str, Any],
    correlation_id: str,
    existing_task: dict[str, Any] | None = None,
) -> TaskRecord:
    if isinstance(existing_task, dict):
        existing_record = _task_record_from_existing_task(existing_task)
        if existing_record is not None:
            existing_record.task_id = str(existing_task.get("task_id") or "").strip() or existing_record.task_id
            existing_record.user_id = existing_record.user_id or str(session_user_id or "").strip() or None
            existing_record.correlation_id = str(correlation_id or "").strip() or existing_record.correlation_id
            return existing_record
        record = _build_new_task_record(
            envelope=envelope,
            session_user_id=session_user_id,
            day_session=day_session,
            correlation_id=correlation_id,
        )
        record.task_id = str(existing_task.get("task_id") or "").strip() or None
        return record
    return _build_new_task_record(
        envelope=envelope,
        session_user_id=session_user_id,
        day_session=day_session,
        correlation_id=correlation_id,
    )


def _task_record_from_existing_task(existing_task: dict[str, Any]) -> TaskRecord | None:
    metadata = existing_task.get("metadata") if isinstance(existing_task.get("metadata"), dict) else {}
    state = metadata.get("state") if isinstance(metadata.get("state"), dict) else {}
    raw_task_record = state.get("task_record") if isinstance(state.get("task_record"), dict) else None
    if not isinstance(raw_task_record, dict):
        return None
    return TaskRecord.from_dict(raw_task_record)


def _build_new_task_record(
    *,
    envelope: IncomingMessageEnvelope,
    session_user_id: str,
    day_session: dict[str, Any],
    correlation_id: str,
) -> TaskRecord:
    channel = envelope.channel if isinstance(envelope.channel, dict) else {}
    actor = envelope.actor if isinstance(envelope.actor, dict) else {}
    envelope_context = envelope.context if isinstance(envelope.context, dict) else {}
    record = TaskRecord(
        user_id=str(session_user_id or "").strip() or None,
        correlation_id=str(correlation_id or "").strip(),
        goal=_render_envelope_user_text(envelope),
        status="running",
    )
    record.set_recent_conversation_md(render_recent_conversation_block(day_session))
    for key, value in (
        ("channel_type", channel.get("type")),
        ("channel_target", channel.get("target")),
        ("message_id", envelope.message_id),
        ("external_user_id", actor.get("external_user_id")),
        ("display_name", actor.get("display_name")),
        ("locale", envelope_context.get("locale")),
        ("timezone", envelope_context.get("timezone")),
    ):
        rendered = str(value or "").strip()
        if rendered:
            record.append_fact(f"{key}: {rendered}")
    return record


def _render_envelope_user_text(envelope: IncomingMessageEnvelope) -> str:
    content = envelope.content if isinstance(envelope.content, dict) else {}
    text = str(content.get("text") or "").strip()
    if text:
        return text
    attachments = content.get("attachments") if isinstance(content.get("attachments"), list) else []
    parts: list[str] = []
    for item in attachments:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip().lower() or "file"
        if kind == "contact":
            contact = item.get("contact") if isinstance(item.get("contact"), dict) else {}
            name = " ".join(
                value
                for value in (
                    str(contact.get("first_name") or "").strip(),
                    str(contact.get("last_name") or "").strip(),
                )
                if value
            ).strip()
            if name:
                parts.append(f"contact ({name})")
                continue
        file_id = str(item.get("file_id") or "").strip()
        parts.append(f"{kind}({file_id})" if file_id else kind)
    return f"[attachments: {', '.join(parts)}]" if parts else ""


def _conversation_key(envelope: IncomingMessageEnvelope) -> str:
    channel = envelope.channel if isinstance(envelope.channel, dict) else {}
    channel_type = str(channel.get("type") or "").strip()
    channel_target = str(channel.get("target") or "").strip()
    if not channel_type or not channel_target:
        return ""
    return f"{channel_type}:{channel_target}"


def _force_new_task(*, envelope: IncomingMessageEnvelope) -> bool:
    controls = envelope.controls if isinstance(envelope.controls, dict) else {}
    if _as_bool(controls.get("force_new_task")):
        return True
    metadata = envelope.metadata if isinstance(envelope.metadata, dict) else {}
    source = str(metadata.get("source") or "").strip().lower()
    return source in {"job_runner", "job_trigger", "jobs_reconcile"}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
