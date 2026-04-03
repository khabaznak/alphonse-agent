from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alphonse.agent.nervous_system.senses.base import Sense, SignalSpec
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.agent.services import communication_directory


@dataclass(frozen=True)
class CanonicalUserCommunication:
    """Provider-agnostic inbound user-message payload contract."""

    message_id: str
    correlation_id: str | None
    occurred_at: str
    service_id: int | None
    service_key: str
    channel_type: str
    channel_target: str
    external_user_id: str | None
    display_name: str | None
    text: str
    attachments: list[dict[str, Any]]
    metadata: dict[str, Any]
    resolved_user_id: str | None = None

    @property
    def identity_resolved(self) -> bool:
        return bool(str(self.resolved_user_id or "").strip())

    def to_payload(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "occurred_at": self.occurred_at,
            "identity": {
                "resolved": self.identity_resolved,
                "alphonse_user_id": self.resolved_user_id,
                "user_id": self.resolved_user_id,
                "external_user_id": self.external_user_id,
                "display_name": self.display_name,
            },
            "transport": {
                "service_id": self.service_id,
                "service_key": self.service_key,
                "channel_type": self.channel_type,
                "channel_target": self.channel_target,
            },
            "service_id": self.service_id,
            "service_key": self.service_key,
            "identity_resolved": self.identity_resolved,
            "alphonse_user_id": self.resolved_user_id,
            "content": {
                "text": self.text,
                "attachments": [dict(item) for item in self.attachments if isinstance(item, dict)],
            },
            "metadata": dict(self.metadata),
        }


def build_canonical_user_message(
    *,
    message_id: str,
    occurred_at: str,
    service_key: str,
    channel_type: str | None,
    channel_target: str,
    text: str,
    correlation_id: str | None = None,
    external_user_id: str | None = None,
    display_name: str | None = None,
    attachments: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    service_id: int | None = None,
    resolved_user_id: str | None = None,
) -> dict[str, Any]:
    normalized_service_key = str(service_key or "").strip().lower()
    resolved_service_id = service_id
    if resolved_service_id is None:
        resolved_service_id = communication_directory.resolve_service_id(normalized_service_key)
    resolved_service_key = (
        communication_directory.resolve_service_key(resolved_service_id)
        or normalized_service_key
    )
    canonical_external_user_id = str(external_user_id or "").strip() or None
    canonical_user_id = str(resolved_user_id or "").strip() or None
    if canonical_user_id is None and resolved_service_id is not None and canonical_external_user_id:
        canonical_user_id = communication_directory.resolve_user_id(
            service_id=resolved_service_id,
            service_user_id=canonical_external_user_id,
        )
    payload = CanonicalUserCommunication(
        message_id=str(message_id or "").strip(),
        correlation_id=str(correlation_id or "").strip() or None,
        occurred_at=str(occurred_at or "").strip(),
        service_id=resolved_service_id,
        service_key=resolved_service_key,
        channel_type=str(channel_type or resolved_service_key or "").strip(),
        channel_target=str(channel_target or "").strip(),
        external_user_id=canonical_external_user_id,
        display_name=str(display_name or "").strip() or None,
        text=str(text or "").strip(),
        attachments=[dict(item) for item in (attachments or []) if isinstance(item, dict)],
        metadata=dict(metadata or {}),
        resolved_user_id=canonical_user_id,
    )
    return payload.to_payload()


class UserCommunicationSense(Sense):
    """Canonical inbound user communication routing and identity boundary."""

    key = "user_communication"
    name = "User Communication Sense"
    description = "Canonical provider-agnostic inbound user-message sense"
    source_type = "system"
    signals = [
        SignalSpec(
            key="sense.user_communication.message.user.received",
            name="User Communication Message Received",
        ),
    ]

    def __init__(self) -> None:
        self._bus: Bus | None = None

    def start(self, bus: Bus) -> None:
        self._bus = bus

    def stop(self) -> None:
        self._bus = None

    def canonicalize_message(
        self,
        *,
        message_id: str,
        occurred_at: str,
        service_key: str,
        service_user_id: str | None,
        channel_target: str,
        text: str,
        correlation_id: str | None = None,
        display_name: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return build_canonical_user_message(
            message_id=message_id,
            correlation_id=correlation_id,
            occurred_at=occurred_at,
            service_key=service_key,
            channel_type=service_key,
            channel_target=channel_target,
            external_user_id=service_user_id,
            display_name=display_name,
            text=text,
            attachments=attachments,
            metadata=metadata,
        )

    def emit(self, payload: dict[str, Any], *, correlation_id: str | None = None) -> Signal:
        signal = Signal(
            type="sense.user_communication.message.user.received",
            payload=dict(payload),
            source=self.key,
            correlation_id=str(correlation_id or payload.get("correlation_id") or "").strip() or None,
        )
        if self._bus is not None:
            self._bus.emit(signal)
        return signal
