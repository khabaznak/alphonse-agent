from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from alphonse.agent.io.contracts import NormalizedOutboundMessage


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

    def send_intent_update(
        self,
        *,
        channel_target: str | None,
        text: str,
        correlation_id: str | None = None,
    ) -> None:
        ...


@dataclass
class AdapterRegistry:
    extremities: dict[str, ExtremityAdapter] = field(default_factory=dict)

    def register_extremity(self, adapter: ExtremityAdapter) -> None:
        self.extremities[str(adapter.channel_type)] = adapter

    def get_extremity(self, channel_type: str) -> ExtremityAdapter | None:
        return self.extremities.get(str(channel_type))
