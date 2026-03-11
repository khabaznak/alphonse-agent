from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from alphonse.agent.io.contracts import (
    NormalizedInboundMessage,
    NormalizedOutboundMessage,
)


class SenseAdapter(Protocol):
    channel_type: str

    def normalize(self, payload: dict[str, Any]) -> NormalizedInboundMessage:
        ...


class ExtremityAdapter(Protocol):
    channel_type: str

    def deliver(self, message: NormalizedOutboundMessage) -> None:
        ...

    # Optional channel primitives used by orchestration for transition UX.
    def send_chat_action(
        self,
        *,
        channel_target: str | None,
        action: str,
        correlation_id: str | None = None,
    ) -> None:
        ...

    def set_reaction(
        self,
        *,
        channel_target: str | None,
        message_id: str | None,
        emoji: str,
        correlation_id: str | None = None,
    ) -> None:
        ...


@dataclass
class AdapterRegistry:
    senses: dict[str, SenseAdapter] = field(default_factory=dict)
    extremities: dict[str, ExtremityAdapter] = field(default_factory=dict)

    def register_sense(self, adapter: SenseAdapter) -> None:
        self.senses[str(adapter.channel_type)] = adapter

    def register_extremity(self, adapter: ExtremityAdapter) -> None:
        self.extremities[str(adapter.channel_type)] = adapter

    def get_sense(self, channel_type: str) -> SenseAdapter | None:
        return self.senses.get(str(channel_type))

    def get_extremity(self, channel_type: str) -> ExtremityAdapter | None:
        return self.extremities.get(str(channel_type))
