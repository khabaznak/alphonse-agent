"""Integration adapter contract for external I/O systems.

Adapters are extremities: they translate external events into internal signals
and internal actions into external effects. They do not make decisions and do
not know about Core, Heart, or the SignalBus.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

Signal = Dict[str, Any]
Action = Dict[str, Any]


@dataclass(frozen=True)
class CanonicalInboundEvent:
    """Provider-agnostic inbound event contract emitted by integrations."""

    service_key: str
    provider_user_id_from: str
    provider_message_id: str
    channel_target: str
    occurred_at: str
    event_kind: str
    provider_raw_message: dict[str, Any]
    text: str | None = None
    attachments: list[dict[str, Any]] = field(default_factory=list)
    reply_to_provider_message_id: str | None = None
    dedupe_key: str | None = None
    display_name: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "contract_type": "canonical_inbound_event",
            "contract_version": "1.0",
            "service_key": str(self.service_key or "").strip(),
            "provider_user_id_from": str(self.provider_user_id_from or "").strip(),
            "provider_message_id": str(self.provider_message_id or "").strip(),
            "channel_target": str(self.channel_target or "").strip(),
            "occurred_at": str(self.occurred_at or "").strip(),
            "event_kind": str(self.event_kind or "").strip(),
            "provider_raw_message": dict(self.provider_raw_message),
            "text": str(self.text or "").strip() or None,
            "attachments": [dict(item) for item in self.attachments if isinstance(item, dict)],
            "reply_to_provider_message_id": str(self.reply_to_provider_message_id or "").strip() or None,
            "dedupe_key": str(self.dedupe_key or "").strip() or None,
            "display_name": str(self.display_name or "").strip() or None,
        }


class IntegrationAdapter(ABC):
    """Base interface for I/O integrations (Telegram, WhatsApp, etc)."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._signal_callback: Callable[[Signal], None] | None = None

    @property
    @abstractmethod
    def id(self) -> str:
        """Stable adapter identifier (e.g., 'telegram')."""

    @property
    @abstractmethod
    def io_type(self) -> str:
        """I/O mode: 'input', 'output', or 'io'."""

    def on_signal(self, callback: Callable[[Signal], None]) -> None:
        """Register a callback to emit internal signals."""
        self._signal_callback = callback

    def emit_signal(self, signal: Signal) -> None:
        """Emit a signal via the registered callback.

        The callback is provided by the core runtime at boot time.
        """
        if self._signal_callback is None:
            raise RuntimeError("Signal callback not registered. Call on_signal() first.")
        self._signal_callback(signal)

    @abstractmethod
    def start(self) -> None:
        """Start the adapter and begin listening to external events."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the adapter and release any external resources."""

    @abstractmethod
    def handle_action(self, action: Action) -> None:
        """Handle an internal action by producing an external effect."""


__all__ = ["IntegrationAdapter", "Signal", "Action", "CanonicalInboundEvent"]
