from __future__ import annotations

import uuid
from typing import Any

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.actions.presence_projection import emit_presence_phase_changed
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.identity.service_resolvers import resolve_user_id_by_service_user_id, resolve_service_id_by_channel_type
from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.observability.log_manager import get_log_manager
from alphonse.agent.services.pdca_ingress import BufferedTaskInput
from alphonse.agent.services.pdca_ingress import enqueue_pdca_slice
from alphonse.agent.services.pdca_ingress import normalize_buffered_attachments
from alphonse.agent.services.pdca_queue_runner import is_pdca_slicing_enabled

logger = get_component_logger("actions.handle_conscious_message")
_LOG = get_log_manager()
_SIGNAL_CORRELATION_KEY = "correlation_id"
_PAYLOAD_KEY = "payload"
_SIGNAL_KEY = "signal"


class HandleConsciousMessageAction(Action):
    key = "handle_conscious_message"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get(_SIGNAL_KEY)
        raw_payload = getattr(signal, _PAYLOAD_KEY, {}) if signal else {}
        if not isinstance(raw_payload, dict):
            raise ValueError("invalid_envelope: payload must be an object")

        payload = dict(raw_payload)
        _validate_canonical_inbound_event_payload(payload)
        correlation_id = (
            str(payload.get(_SIGNAL_CORRELATION_KEY) or "").strip()
            or str(payload.get("dedupe_key") or "").strip()
            or str(getattr(signal, _SIGNAL_CORRELATION_KEY, "") or "").strip()
            or str(uuid.uuid4())
        )
        channel_type = channel_type_for_payload(payload)
        channel_target = channel_target_for_payload(payload)
        message_id = message_id_for_payload(payload) or None
        user_id = resolved_user_id_for_payload(payload) or None
        if not channel_type:
            raise ValueError("invalid_conscious_payload: missing channel type")
        if not channel_target:
            raise ValueError("invalid_conscious_payload: missing channel target")

        _LOG.emit(
            event="incoming_message.accepted",
            component="actions.handle_conscious_message",
            correlation_id=str(correlation_id),
            channel=channel_type,
            user_id=user_id,
            payload={
                "address": channel_target,
                "message_id": message_id,
            },
        )
        emit_presence_phase_changed(
            channel_type=channel_type,
            channel_target=channel_target,
            user_id=user_id,
            message_id=message_id,
            phase="acknowledged",
            correlation_id=str(correlation_id),
        )

        if not is_pdca_slicing_enabled():
            _LOG.emit(
                level="warning",
                event="incoming_message.rejected",
                component="actions.handle_conscious_message",
                correlation_id=str(correlation_id),
                channel=channel_type,
                user_id=user_id,
                error_code="pdca_slicing_disabled",
                payload={
                    "reason": "pdca_slicing_disabled",
                    "address": channel_target,
                },
            )
            text = "I am temporarily unable to process messages right now. Please try again in a moment."
            return ActionResult(
                intention_key="MESSAGE_READY",
                payload={
                    "message": text,
                    "channel_hint": channel_type,
                    "target": channel_target,
                    "correlation_id": str(correlation_id),
                    "direct_reply": {
                        "channel_type": channel_type,
                        "target": channel_target,
                        "text": text,
                        "correlation_id": str(correlation_id),
                    },
                },
                urgency="normal",
                delivers_message=True,
            )
        
        task_record = _build_task_record_from_payload(
            payload=payload,
            session_user_id=user_id,
            correlation_id=str(correlation_id),
        )
        buffered_input = _build_buffered_input_from_payload(
            payload=payload,
            session_user_id=user_id,
            correlation_id=str(correlation_id),
        )

        task_id = enqueue_pdca_slice(
            task_record=task_record,
            buffered_input=buffered_input,
            bus=context.get("ctx") if isinstance(context, dict) else None,
            force_new_task=_force_new_task_for_payload(payload),
        )
        logger.info(
            "HandleConsciousMessageAction enqueued task_id=%s channel=%s target=%s",
            task_id,
            channel_type,
            channel_target,
        )
        _LOG.emit(
            event="incoming_message.enqueued",
            component="actions.handle_conscious_message",
            correlation_id=str(correlation_id),
            channel=channel_type,
            user_id=user_id,
            payload={"task_id": task_id},
        )
        return ActionResult(
            intention_key="NOOP",
            payload={"task_id": task_id},
            urgency=None,
        )


def _build_task_record_from_payload(
    *,
    payload: dict[str, Any],
    session_user_id: str,
    correlation_id: str,
) -> TaskRecord:
    record = TaskRecord(
        user_id=str(session_user_id or "").strip() or None,
        correlation_id=str(correlation_id or "").strip(),
        goal=_message_text_for_payload(payload),
        status="running",
    )
    for key, value in (
        ("channel_type", channel_type_for_payload(payload)),
        ("channel_target", channel_target_for_payload(payload)),
        ("message_id", message_id_for_payload(payload)),
        ("provider_user_id_from", provider_user_id_from_for_payload(payload)),
        ("display_name", display_name_for_payload(payload)),
        ("locale", locale_for_payload(payload)),
        ("timezone", timezone_for_payload(payload)),
    ):
        rendered = str(value or "").strip()
        if rendered:
            record.append_fact(f"{key}: {rendered}")
    return record


def _build_buffered_input_from_payload(
    *,
    payload: dict[str, Any],
    session_user_id: str | None,
    correlation_id: str,
) -> BufferedTaskInput:
    attachments = normalize_buffered_attachments(attachments_for_payload(payload))
    text = _message_text_for_payload(payload)
    return BufferedTaskInput(
        message_id=message_id_for_payload(payload) or None,
        correlation_id=str(correlation_id or "").strip(),
        channel_type=channel_type_for_payload(payload),
        channel_target=channel_target_for_payload(payload),
        actor_id=str(session_user_id or "").strip() or None,
        text=text,
        attachments=attachments,
    )


def _message_text_for_payload(payload: dict[str, Any]) -> str:
    text = text_for_payload(payload)
    if text:
        return text
    parts: list[str] = []
    for item in attachments_for_payload(payload):
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


def channel_type_for_payload(payload: dict[str, Any]) -> str:
    return service_key_for_payload(payload)


def channel_target_for_payload(payload: dict[str, Any]) -> str:
    return _required_str_field(payload, "channel_target")


def message_id_for_payload(payload: dict[str, Any]) -> str:
    return _required_str_field(payload, "provider_message_id")


def attachments_for_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("attachments")
    return [dict(item) for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def provider_user_id_from_for_payload(payload: dict[str, Any]) -> str:
    return _required_str_field(payload, "provider_user_id_from")


def display_name_for_payload(payload: dict[str, Any]) -> str:
    return str(payload.get("display_name") or "").strip()


def locale_for_payload(payload: dict[str, Any]) -> str:
    return str(payload.get("locale") or "").strip()


def timezone_for_payload(payload: dict[str, Any]) -> str:
    return str(payload.get("timezone") or "").strip()


def _force_new_task_for_payload(payload: dict[str, Any]) -> bool:
    if _as_bool(payload.get("force_new_task")):
        return True
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    source = str(payload.get("source") or metadata.get("source") or "").strip().lower()
    return source in {"job_runner", "job_trigger", "jobs_reconcile"}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def resolved_user_id_for_payload(payload: dict[str, Any]) -> str:
    direct = str(payload.get("alphonse_user_id") or "").strip()
    if direct:
        return direct
    service_id = service_id_for_payload(payload)
    if service_id is None:
        raise ValueError("invalid_conscious_payload: unknown service_key")
    provider_user_id_from = provider_user_id_from_for_payload(payload)
    user_id = resolve_user_id_by_service_user_id(
        service_id=service_id,
        service_user_id=provider_user_id_from,
    )
    resolved = str(user_id or "").strip()
    if not resolved:
        raise ValueError("invalid_conscious_payload: user_id unresolved")
    return resolved


def service_id_for_payload(payload: dict[str, Any]) -> int | None:
    resolved = resolve_service_id_by_channel_type(service_key_for_payload(payload))
    return int(resolved) if resolved is not None else None


def _is_canonical_inbound_event(payload: dict[str, Any]) -> bool:
    return str(payload.get("contract_type") or "").strip() == "canonical_inbound_event"


def _validate_canonical_inbound_event_payload(payload: dict[str, Any]) -> None:
    if not _is_canonical_inbound_event(payload):
        raise ValueError("invalid_conscious_payload: unsupported contract_type")
    for legacy_field_name in ("external_user_id", "resolved_user_id"):
        if legacy_field_name in payload:
            raise ValueError(f"invalid_conscious_payload: legacy field {legacy_field_name} is not allowed")
    for field_name in (
        "service_key",
        "provider_user_id_from",
        "provider_message_id",
        "channel_target",
        "occurred_at",
        "event_kind",
    ):
        _required_str_field(payload, field_name)
    if not isinstance(payload.get("provider_raw_message"), dict):
        raise ValueError("invalid_conscious_payload: provider_raw_message must be an object")


def _required_str_field(payload: dict[str, Any], field_name: str) -> str:
    rendered = str(payload.get(field_name) or "").strip()
    if not rendered:
        raise ValueError(f"invalid_conscious_payload: missing {field_name}")
    return rendered


def service_key_for_payload(payload: dict[str, Any]) -> str:
    return _required_str_field(payload, "service_key")


def text_for_payload(payload: dict[str, Any]) -> str:
    return str(payload.get("text") or "").strip()
