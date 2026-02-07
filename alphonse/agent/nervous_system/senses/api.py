from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass

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
        bus.emit(
            Signal(
                type=api_signal.type,
                payload={
                    **api_signal.payload,
                    "origin": "api",
                    "timestamp": time.time(),
                },
                source="api",
                correlation_id=api_signal.correlation_id,
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
