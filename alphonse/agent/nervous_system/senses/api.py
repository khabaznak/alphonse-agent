from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass

from alphonse.agent.actions.conscious_message_handler import build_incoming_message_envelope
from alphonse.agent.io import get_io_registry
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
    description = "Emits api.* signals from HTTP requests"
    source_type = "service"
    signals = [
        SignalSpec(key="sense.api.message.user.received", name="API User Message Received"),
    ]

    def start(self, bus: Bus) -> None:
        self._bus = bus

    def stop(self) -> None:
        self._bus = None

    def emit(self, bus: Bus, api_signal: ApiSignal) -> None:
        _assert_api_token(api_signal.payload)
        signal_type = _canonical_api_signal_type(api_signal.type)
        if str(api_signal.payload.get("schema_version") or "").strip() == "1.0":
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
        registry = get_io_registry()
        channel = api_signal.payload.get("channel") or api_signal.payload.get("origin") or "webui"
        adapter = registry.get_sense(str(channel))
        if not adapter:
            raise ValueError(f"No sense adapter for channel={channel}")
        normalized = adapter.normalize({**api_signal.payload, "channel": channel})
        correlation_id = normalized.correlation_id or api_signal.correlation_id
        raw_payload = normalized.metadata.get("raw") if isinstance(normalized.metadata, dict) else None
        content = None
        controls = None
        provider = None
        provider_event = None
        if isinstance(raw_payload, dict):
            if isinstance(raw_payload.get("content"), dict):
                content = raw_payload.get("content")
            if isinstance(raw_payload.get("controls"), dict):
                controls = raw_payload.get("controls")
            if raw_payload.get("provider") is not None:
                provider = raw_payload.get("provider")
            if isinstance(raw_payload.get("provider_event"), dict):
                provider_event = raw_payload.get("provider_event")
        message_id = str(
            (raw_payload or {}).get("message_id")
            or (raw_payload or {}).get("update_id")
            or correlation_id
            or uuid.uuid4()
        )
        envelope = build_incoming_message_envelope(
            message_id=message_id,
            channel_type=str(normalized.channel_type or "webui"),
            channel_target=str(normalized.channel_target or normalized.channel_type or "webui"),
            provider=str(provider or normalized.channel_type or "api"),
            text=str(normalized.text or ""),
            occurred_at=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(float(normalized.timestamp))),
            correlation_id=correlation_id,
            actor_external_user_id=normalized.user_id,
            actor_display_name=normalized.user_name,
            attachments=[dict(item) for item in normalized.attachments if isinstance(item, dict)],
            controls=controls,
            metadata={
                "normalized_metadata": normalized.metadata,
                "provider_event": provider_event if isinstance(provider_event, dict) else None,
                "content": content if isinstance(content, dict) else None,
            },
            locale=str((raw_payload or {}).get("locale") or "").strip() or None,
            timezone_name=str((raw_payload or {}).get("timezone") or "").strip() or None,
            reply_to_message_id=str((raw_payload or {}).get("reply_to_message_id") or "").strip() or None,
            session_hint=str((raw_payload or {}).get("session_hint") or "").strip() or None,
        )
        bus.emit(
            Signal(
                type=signal_type,
                payload=envelope,
                source="api",
                correlation_id=correlation_id,
            )
        )


def build_api_signal(signal_type: str, payload: dict[str, object] | None, correlation_id: str | None) -> ApiSignal:
    cid = correlation_id or str(uuid.uuid4())
    return ApiSignal(type=signal_type, payload=payload or {}, correlation_id=cid)


def _assert_api_token(payload: dict[str, object]) -> None:
    expected = os.getenv("ALPHONSE_API_TOKEN")
    if not expected:
        return
    provided = payload.get("api_token")
    if provided != expected:
        raise PermissionError("Invalid API token")


def _canonical_api_signal_type(value: str) -> str:
    rendered = str(value or "").strip()
    if rendered in {"api.message_received", "sense.api.message.user.received"}:
        return "sense.api.message.user.received"
    return rendered
