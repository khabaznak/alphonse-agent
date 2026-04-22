from __future__ import annotations

from typing import Any

from alphonse.agent.actions.conscious_message_handler import IncomingMessageEnvelope
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.nervous_system.pdca_queue_store import get_latest_pdca_task_for_conversation


def build_task_record_for_message(
    *,
    envelope: IncomingMessageEnvelope,
    session_user_id: str,
    day_session: dict[str, Any] | None,
    correlation_id: str,
    existing_task: dict[str, Any] | None = None,
) -> TaskRecord:
    _ = existing_task
    text = str(envelope.content.get("text") or "").strip()
    attachments = envelope.content.get("attachments") if isinstance(envelope.content.get("attachments"), list) else []
    if not text and attachments:
        text = _render_attachment_summary(attachments)
    record = TaskRecord(
        task_id=str((existing_task or {}).get("task_id") or "").strip() or None,
        user_id=str(session_user_id or "").strip() or None,
        correlation_id=str(correlation_id or "").strip(),
        goal=text,
        status="running",
        recent_conversation_md=str((day_session or {}).get("recent_conversation_md") or "- (none)"),
    )
    for key, value in (
        ("channel_type", envelope.channel.get("type")),
        ("channel_target", envelope.channel.get("target")),
        ("message_id", envelope.message_id),
        ("external_user_id", envelope.actor.get("external_user_id")),
        ("display_name", envelope.actor.get("display_name")),
        ("locale", envelope.context.get("locale")),
        ("timezone", envelope.context.get("timezone")),
    ):
        rendered = str(value or "").strip()
        if rendered:
            record.append_fact(f"{key}: {rendered}")
    return record


def select_pending_pdca_task_for_message(
    *,
    envelope: IncomingMessageEnvelope,
    session_user_id: str,
) -> dict[str, Any] | None:
    conversation_key = f"{str(envelope.channel.get('type') or '').strip()}:{str(envelope.channel.get('target') or '').strip()}"
    candidate = get_latest_pdca_task_for_conversation(
        conversation_key=conversation_key,
        statuses=["waiting_user"],
    )
    if not isinstance(candidate, dict):
        return None
    if str(candidate.get("owner_id") or "").strip() != str(session_user_id or "").strip():
        return None
    if str(candidate.get("conversation_key") or "").strip() != conversation_key:
        return None
    return candidate


def _render_attachment_summary(attachments: list[dict[str, Any]]) -> str:
    if not attachments:
        return ""
    parts: list[str] = []
    for item in attachments:
        kind = str(item.get("kind") or "").strip().lower() or "file"
        if kind == "contact":
            contact = item.get("contact") if isinstance(item.get("contact"), dict) else {}
            full_name = " ".join(
                value
                for value in (
                    str(contact.get("first_name") or "").strip(),
                    str(contact.get("last_name") or "").strip(),
                )
                if value
            ).strip()
            if full_name:
                parts.append(f"contact ({full_name})")
                continue
        file_id = str(item.get("file_id") or "").strip()
        parts.append(f"{kind}({file_id})" if file_id else kind)
    return f"[attachments: {', '.join(parts)}]"
