from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass

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
        SignalSpec(key="api.message_received", name="API Message Received"),
        SignalSpec(key="api.status_requested", name="API Status Requested"),
        SignalSpec(key="api.timed_signals_requested", name="API Timed Signals Requested"),
    ]

    def start(self, bus: Bus) -> None:
        self._bus = bus

    def stop(self) -> None:
        self._bus = None

    def emit(self, bus: Bus, api_signal: ApiSignal) -> None:
        _assert_api_token(api_signal.payload)
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
        bus.emit(
            Signal(
                type=api_signal.type,
                payload={
                    "text": normalized.text,
                    "channel": normalized.channel_type,
                    "target": normalized.channel_target,
                    "user_id": normalized.user_id,
                    "user_name": normalized.user_name,
                    "timestamp": normalized.timestamp,
                    "correlation_id": correlation_id,
                    "metadata": normalized.metadata,
                    "content": content,
                    "controls": controls,
                    "provider": provider,
                    "provider_event": provider_event,
                    "origin": "api",
                },
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
