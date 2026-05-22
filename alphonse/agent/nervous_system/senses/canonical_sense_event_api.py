from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from alphonse.agent.nervous_system.senses.base import Sense, SignalSpec
from alphonse.agent.nervous_system.senses.bus import Bus, Signal


@dataclass(frozen=True)
class CanonicalSenseEventApiSignal:
    type: str
    payload: dict[str, object]
    correlation_id: str


class CanonicalSenseEventApiSense(Sense):
    key = "canonical_sense_event_api"
    name = "Canonical Sense Event API"
    description = "Emits canonical sense events submitted by external deterministic programs"
    source_type = "service"
    signals = [
        SignalSpec(
            key="sense.api.event.received",
            name="API Sense Event Received",
            description="Canonical non-message sense event submitted through API ingress",
        ),
    ]

    def start(self, bus: Bus) -> None:
        self._bus = bus

    def stop(self) -> None:
        self._bus = None

    def emit(self, bus: Bus, api_signal: CanonicalSenseEventApiSignal) -> None:
        payload = dict(api_signal.payload)
        _validate_canonical_sense_event_payload(payload)
        bus.emit(
            Signal(
                type=_canonical_signal_type(api_signal.type),
                payload=payload,
                source=self.key,
                correlation_id=_correlation_id_for_payload(payload, api_signal.correlation_id),
            )
        )


def build_canonical_sense_event_api_signal(
    signal_type: str,
    payload: dict[str, object] | None,
    correlation_id: str | None,
) -> CanonicalSenseEventApiSignal:
    cid = correlation_id or str(uuid.uuid4())
    return CanonicalSenseEventApiSignal(type=signal_type, payload=payload or {}, correlation_id=cid)


def _canonical_signal_type(value: str) -> str:
    rendered = str(value or "").strip()
    if rendered in {"api.sense_event_received", "sense.api.event.received"}:
        return "sense.api.event.received"
    return rendered


def _correlation_id_for_payload(payload: dict[str, Any], fallback: str) -> str:
    return (
        str(payload.get("dedupe_key") or "").strip()
        or str(payload.get("event_id") or "").strip()
        or str(fallback or "").strip()
        or str(uuid.uuid4())
    )


def _validate_canonical_sense_event_payload(payload: dict[str, Any]) -> None:
    if str(payload.get("contract_type") or "").strip() != "canonical_sense_event":
        raise ValueError("invalid_sense_event_payload: unsupported contract_type")
    for field_name in (
        "source_key",
        "provider",
        "event_id",
        "event_kind",
        "occurred_at",
        "summary",
    ):
        _required_str_field(payload, field_name)
    if not isinstance(payload.get("subject"), dict):
        raise ValueError("invalid_sense_event_payload: subject must be an object")
    if not isinstance(payload.get("data"), dict):
        raise ValueError("invalid_sense_event_payload: data must be an object")
    if not isinstance(payload.get("raw_event"), dict):
        raise ValueError("invalid_sense_event_payload: raw_event must be an object")


def _required_str_field(payload: dict[str, Any], field_name: str) -> str:
    rendered = str(payload.get(field_name) or "").strip()
    if not rendered:
        raise ValueError(f"invalid_sense_event_payload: missing {field_name}")
    return rendered
