from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from alphonse.agent import identity
from alphonse.agent.extremities.interfaces.integrations._contracts import CanonicalInboundEvent
from alphonse.agent.nervous_system.senses.base import Sense, SignalSpec
from alphonse.agent.nervous_system.senses.bus import Bus, Signal


@dataclass(frozen=True)
class ApiSignal:
    type: str
    payload: dict[str, object]
    correlation_id: str


class ApiSense(Sense):
    key = "api"
    name = "API Sense"
    description = "Emits api.* signals from internal runtime requests"
    source_type = "service"
    signals = [
        SignalSpec(key="sense.api.message.user.received", name="API User Message Received"),
    ]

    def start(self, bus: Bus) -> None:
        self._bus = bus

    def stop(self) -> None:
        self._bus = None

    def emit(self, bus: Bus, api_signal: ApiSignal) -> None:
        signal_type = _canonical_api_signal_type(api_signal.type)
        if str(api_signal.payload.get("contract_type") or "").strip() == "canonical_inbound_event":
            correlation_id = str(api_signal.payload.get("correlation_id") or api_signal.correlation_id or "").strip() or api_signal.correlation_id
            bus.emit(
                Signal(
                    type=signal_type,
                    payload=dict(api_signal.payload),
                    source="api",
                    correlation_id=correlation_id,
                )
            )
            return
        correlation_id = (
            str(api_signal.payload.get("correlation_id") or "").strip()
            or str(api_signal.correlation_id or "").strip()
            or str(uuid.uuid4())
        )
        payload = _canonical_inbound_event_payload_from_raw(
            raw_payload=dict(api_signal.payload),
            correlation_id=correlation_id,
        )
        bus.emit(
            Signal(
                type=signal_type,
                payload=payload,
                source="api",
                correlation_id=correlation_id,
            )
        )


def build_api_signal(signal_type: str, payload: dict[str, object] | None, correlation_id: str | None) -> ApiSignal:
    cid = correlation_id or str(uuid.uuid4())
    return ApiSignal(type=signal_type, payload=payload or {}, correlation_id=cid)


def _canonical_api_signal_type(value: str) -> str:
    rendered = str(value or "").strip()
    if rendered in {"api.message_received", "sense.api.message.user.received"}:
        return "sense.api.message.user.received"
    return rendered


def _canonical_inbound_event_payload_from_raw(
    raw_payload: dict[str, object],
    *,
    correlation_id: str,
) -> dict[str, Any]:
    raw = dict(raw_payload)
    content = raw.get("content") if isinstance(raw.get("content"), dict) else None
    controls = raw.get("controls") if isinstance(raw.get("controls"), dict) else None
    provider_event = raw.get("provider_event") if isinstance(raw.get("provider_event"), dict) else None
    service_key = str(raw.get("channel") or raw.get("origin") or raw.get("provider") or "webui").strip() or "webui"
    channel_target = str(raw.get("target") or service_key).strip() or service_key
    provider_user_id_from = str(raw.get("user_id") or "").strip()
    display_name = str(raw.get("user_name") or "").strip() or None
    text = _extract_api_text(raw)
    attachments = _extract_api_attachments(raw)
    alphonse_user_id = _resolved_alphonse_user_id(raw, provider_user_id_from)
    if not provider_user_id_from:
        provider_user_id_from = alphonse_user_id or channel_target or service_key
    message_id = str(raw.get("message_id") or raw.get("update_id") or correlation_id or uuid.uuid4()).strip()
    occurred_at = _occurred_at_for_raw(raw)
    payload = CanonicalInboundEvent(
        service_key=service_key,
        provider_user_id_from=provider_user_id_from,
        provider_message_id=message_id,
        channel_target=channel_target,
        occurred_at=occurred_at,
        event_kind="message",
        provider_raw_message=raw,
        text=text or None,
        attachments=attachments,
        reply_to_provider_message_id=str(raw.get("reply_to_message_id") or "").strip() or None,
        dedupe_key=correlation_id,
        display_name=display_name,
    ).to_payload()
    if alphonse_user_id:
        payload["alphonse_user_id"] = alphonse_user_id
    if isinstance(content, dict):
        payload["content"] = dict(content)
    if isinstance(raw.get("locale"), str) and str(raw.get("locale") or "").strip():
        payload["locale"] = str(raw.get("locale") or "").strip()
    if isinstance(raw.get("timezone"), str) and str(raw.get("timezone") or "").strip():
        payload["timezone"] = str(raw.get("timezone") or "").strip()
    if isinstance(controls, dict) and str(controls.get("force_new_task") or "").strip():
        payload["force_new_task"] = _as_bool(controls.get("force_new_task"))
    payload["metadata"] = {
        "provider_event": provider_event if isinstance(provider_event, dict) else None,
        "controls": dict(controls) if isinstance(controls, dict) else None,
        "session_hint": str(raw.get("session_hint") or "").strip() or None,
        "source": str(raw.get("source") or "api").strip() or "api",
        "raw": raw,
    }
    return payload


def _resolved_alphonse_user_id(raw: dict[str, Any], provider_user_id_from: str) -> str | None:
    person_id = str(raw.get("person_id") or "").strip()
    if identity.get_user(person_id):
        return person_id
    user_id = str(provider_user_id_from or "").strip()
    if identity.get_user(user_id):
        return user_id
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _occurred_at_for_raw(raw: dict[str, Any]) -> str:
    timestamp = raw.get("timestamp")
    try:
        if timestamp is None:
            raise ValueError
        return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return datetime.now(timezone.utc).isoformat()


def _extract_api_text(raw: dict[str, Any]) -> str:
    text = str(raw.get("text") or "").strip()
    if text:
        return text
    content = raw.get("content")
    if not isinstance(content, dict):
        return ""
    content_type = str(content.get("type") or "").strip().lower()
    if content_type == "text":
        return str(content.get("text") or "").strip()
    if content_type == "asset":
        assets = content.get("assets")
        if isinstance(assets, list):
            for item in assets:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("kind") or "").strip().lower()
                if kind == "audio":
                    return "[audio asset message]"
        return "[asset message]"
    return ""


def _extract_api_attachments(raw: dict[str, Any]) -> list[dict[str, Any]]:
    attachments = raw.get("attachments")
    if isinstance(attachments, list):
        return [dict(item) for item in attachments if isinstance(item, dict)]
    content = raw.get("content")
    if not isinstance(content, dict):
        return []
    assets = content.get("assets")
    if not isinstance(assets, list):
        return []
    return [dict(item) for item in assets if isinstance(item, dict)]
