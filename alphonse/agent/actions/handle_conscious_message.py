from __future__ import annotations

import uuid
from typing import Any

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.conscious_message_context_adapter import (
    build_incoming_context_from_envelope,
)
from alphonse.agent.actions.conscious_message_handler import IncomingMessageEnvelope
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.actions.presence_projection import emit_presence_phase_changed
from alphonse.agent.actions.session_context import build_session_key
from alphonse.agent.cognition.memory import append_conversation_transcript
from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.observability.log_manager import get_log_manager
from alphonse.agent.services.pdca_ingress import enqueue_pdca_slice
from alphonse.agent.services.pdca_queue_runner import is_pdca_slicing_enabled
from alphonse.agent.services.session_identity_resolution import resolve_session_timezone
from alphonse.agent.services.session_identity_resolution import resolve_session_user_id
from alphonse.agent.session.day_state import build_next_session_state
from alphonse.agent.session.day_state import commit_session_state
from alphonse.agent.session.day_state import resolve_day_session
from alphonse.agent.tools.telegram_files import TranscribeTelegramAudioTool

logger = get_component_logger("actions.handle_conscious_message")
_LOG = get_log_manager()


class HandleConsciousMessageAction(Action):
    key = "handle_conscious_message"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        raw_payload = getattr(signal, "payload", {}) if signal else {}
        if not isinstance(raw_payload, dict):
            raise ValueError("invalid_envelope: payload must be an object")

        envelope = IncomingMessageEnvelope.from_payload(raw_payload)
        correlation_id = envelope.correlation_id or getattr(signal, "correlation_id", None) or str(uuid.uuid4())
        payload = envelope.runtime_payload()
        payload = _enrich_multimodal_payload(
            payload=payload,
            envelope=envelope,
            correlation_id=str(correlation_id),
            incoming_channel=str((envelope.channel or {}).get("type") or "").strip() or "telegram",
        )
        missing_actor_fields = [
            key
            for key in ("external_user_id", "display_name", "person_id")
            if not str((envelope.actor or {}).get(key) or "").strip()
        ]
        incoming = build_incoming_context_from_envelope(
            envelope=envelope,
            correlation_id=str(correlation_id),
        )
        _LOG.emit(
            event="incoming_message.accepted",
            component="actions.handle_conscious_message",
            correlation_id=str(correlation_id),
            channel=incoming.channel_type,
            user_id=incoming.person_id,
            payload={
                "address": incoming.address,
                "message_id": incoming.message_id,
            },
        )
        if missing_actor_fields:
            _LOG.emit(
                event="incoming_message.context_missing_fields",
                component="actions.handle_conscious_message",
                correlation_id=str(correlation_id),
                channel=incoming.channel_type,
                user_id=incoming.person_id,
                payload={
                    "missing_fields": missing_actor_fields,
                    "has_display_name": "display_name" not in missing_actor_fields,
                },
            )
        emit_presence_phase_changed(
            incoming=incoming,
            phase="acknowledged",
            correlation_id=str(correlation_id),
        )

        if not is_pdca_slicing_enabled():
            _LOG.emit(
                level="warning",
                event="incoming_message.rejected",
                component="actions.handle_conscious_message",
                correlation_id=str(correlation_id),
                channel=incoming.channel_type,
                user_id=incoming.person_id,
                error_code="pdca_slicing_disabled",
                payload={
                    "reason": "pdca_slicing_disabled",
                    "address": incoming.address,
                },
            )
            text = "I am temporarily unable to process messages right now. Please try again in a moment."
            return ActionResult(
                intention_key="MESSAGE_READY",
                payload={
                    "message": text,
                    "channel_hint": incoming.channel_type,
                    "target": incoming.address,
                    "correlation_id": str(correlation_id),
                    "direct_reply": {
                        "channel_type": incoming.channel_type,
                        "target": incoming.address,
                        "text": text,
                        "correlation_id": str(correlation_id),
                    },
                },
                urgency="normal",
            )

        session_key = build_session_key(incoming)
        session_user_id = resolve_session_user_id(incoming=incoming, payload=payload)
        day_session = resolve_day_session(
            user_id=session_user_id,
            channel=incoming.channel_type,
            timezone_name=resolve_session_timezone(incoming),
        )
        _write_through_user_message(
            day_session=day_session,
            channel=incoming.channel_type,
            user_text=str(payload.get("text") or "").strip(),
            payload=payload,
            correlation_id=str(correlation_id),
        )

        task_id = enqueue_pdca_slice(
            context=context,
            incoming=incoming,
            state={
                "conversation_key": session_key,
                "chat_id": session_key,
                "channel_type": incoming.channel_type,
                "channel_target": incoming.address,
                "actor_person_id": incoming.person_id,
                "message_id": incoming.message_id,
                "incoming_user_id": str(payload.get("user_id") or "").strip() or None,
                "incoming_user_name": str(payload.get("user_name") or "").strip() or None,
                "locale": envelope.context.get("locale"),
                "timezone": envelope.context.get("timezone"),
            },
            session_key=session_key,
            session_user_id=session_user_id,
            day_session=day_session,
            payload=payload,
            correlation_id=str(correlation_id),
        )
        logger.info(
            "HandleConsciousMessageAction enqueued task_id=%s channel=%s target=%s",
            task_id,
            incoming.channel_type,
            incoming.address,
        )
        _LOG.emit(
            event="incoming_message.enqueued",
            component="actions.handle_conscious_message",
            correlation_id=str(correlation_id),
            channel=incoming.channel_type,
            user_id=incoming.person_id,
            payload={"task_id": task_id},
        )
        return ActionResult(
            intention_key="NOOP",
            payload={"task_id": task_id},
            urgency=None,
        )


def _write_through_user_message(
    *,
    day_session: dict[str, object],
    channel: str,
    user_text: str,
    payload: dict[str, Any] | None = None,
    correlation_id: str,
) -> None:
    text = str(user_text or "").strip()
    if not text and isinstance(payload, dict):
        text = _render_attachment_summary(payload)
    if not text or not isinstance(day_session, dict):
        return
    try:
        append_conversation_transcript(
            user_id=str(day_session.get("user_id") or "").strip() or "anonymous",
            session_id=str(day_session.get("session_id") or "").strip() or "unknown",
            role="user",
            text=text,
            channel=str(channel or "").strip() or "api",
            correlation_id=correlation_id,
        )
        updated = build_next_session_state(
            previous=day_session,
            channel=str(channel or "").strip() or "api",
            user_message=text,
            assistant_message="",
            task_record=None,
            pending_interaction=None,
            user_event_meta={
                "correlation_id": correlation_id,
                "message_id": str((payload or {}).get("message_id") or "").strip() or None,
                "channel": str(channel or "").strip() or "api",
                "attachments": (
                    (payload or {}).get("content", {}).get("attachments")
                    if isinstance((payload or {}).get("content"), dict)
                    else []
                ),
            },
        )
        commit_session_state(updated)
    except Exception:
        _LOG.emit(
            level="warning",
            event="pdca.input.history_append_failed",
            component="actions.handle_conscious_message",
            correlation_id=correlation_id or None,
            payload={"channel": str(channel or "").strip() or None},
        )


def _enrich_multimodal_payload(
    *,
    payload: dict[str, Any],
    envelope: IncomingMessageEnvelope,
    correlation_id: str,
    incoming_channel: str,
) -> dict[str, Any]:
    content = payload.get("content") if isinstance(payload.get("content"), dict) else {}
    attachments_raw = content.get("attachments") if isinstance(content.get("attachments"), list) else []
    attachments = [dict(item) for item in attachments_raw if isinstance(item, dict)]
    if not attachments:
        return payload
    text = str(payload.get("text") or "").strip()
    if text:
        return payload
    transcript = _transcribe_audio_attachments(
        attachments=attachments,
        locale_hint=str((envelope.context or {}).get("locale") or "").strip() or None,
        correlation_id=correlation_id,
        channel=incoming_channel,
    )
    if transcript:
        payload["text"] = transcript
        content["text"] = transcript
    payload["content"] = content
    return payload


def _transcribe_audio_attachments(
    *,
    attachments: list[dict[str, Any]],
    locale_hint: str | None,
    correlation_id: str,
    channel: str,
) -> str:
    lines: list[str] = []
    tool = TranscribeTelegramAudioTool()
    for item in attachments:
        kind = str(item.get("kind") or "").strip().lower()
        provider = str(item.get("provider") or "").strip().lower()
        file_id = str(item.get("file_id") or "").strip()
        if kind not in {"voice", "audio"} or provider != "telegram" or not file_id:
            continue
        result = tool.execute(file_id=file_id, language=locale_hint)
        exception = result.get("exception") if isinstance(result, dict) else None
        if isinstance(exception, dict):
            _LOG.emit(
                level="warning",
                event="incoming_message.audio_transcription_failed",
                component="actions.handle_conscious_message",
                correlation_id=correlation_id or None,
                channel=str(channel or "").strip() or None,
                error_code=str(exception.get("code") or "transcribe_failed"),
                payload={
                    "file_id": file_id,
                    "kind": kind,
                },
            )
            continue
        output = result.get("output") if isinstance(result, dict) else None
        transcript_text = str((output or {}).get("text") or "").strip() if isinstance(output, dict) else ""
        if transcript_text:
            lines.append(transcript_text)
    return "\n".join(lines).strip()


def _render_attachment_summary(payload: dict[str, Any]) -> str:
    content = payload.get("content") if isinstance(payload.get("content"), dict) else {}
    attachments = content.get("attachments") if isinstance(content.get("attachments"), list) else []
    if not isinstance(attachments, list) or not attachments:
        return ""
    counts: dict[str, int] = {}
    for item in attachments:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "attachment").strip().lower() or "attachment"
        counts[kind] = int(counts.get(kind) or 0) + 1
    if not counts:
        return ""
    parts = [f"{value} {key}" for key, value in sorted(counts.items())]
    return f"[attachments: {', '.join(parts)}]"
