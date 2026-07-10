"""Canonical v2 channel metadata.

This module intentionally models provider addresses without importing any
provider SDKs. Integrations can translate their payloads into this shape before
queueing work, and deterministic outbox consumers can use the same shape for
delivery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChannelAddress:
    """Concrete address for an integration instance and conversation target."""

    integration_id: str
    provider_key: str
    channel_target: str
    alphonse_user_id: str = ""
    provider_user_id: str = ""
    provider_message_id: str = ""
    reply_to_provider_message_id: str = ""
    thread_id: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "integration_id": self.integration_id,
            "provider_key": self.provider_key,
            "channel_target": self.channel_target,
            "alphonse_user_id": self.alphonse_user_id,
            "provider_user_id": self.provider_user_id,
            "provider_message_id": self.provider_message_id,
            "reply_to_provider_message_id": self.reply_to_provider_message_id,
            "thread_id": self.thread_id,
        }


def channel_metadata(
    *,
    integration_id: str = "tui",
    provider_key: str = "tui",
    channel_target: str = "",
    alphonse_user_id: str = "",
    provider_user_id: str = "",
    provider_message_id: str = "",
    reply_to_provider_message_id: str = "",
    thread_id: str = "",
) -> dict[str, str]:
    """Build normalized metadata suitable for `CoreMessage.metadata["channel"]`."""
    integration = str(integration_id or "").strip() or "tui"
    provider = str(provider_key or "").strip().lower() or integration.lower()
    target = str(channel_target or "").strip()
    user = str(alphonse_user_id or "").strip()
    provider_user = str(provider_user_id or "").strip() or user
    return ChannelAddress(
        integration_id=integration,
        provider_key=provider,
        channel_target=target or provider_user or user,
        alphonse_user_id=user,
        provider_user_id=provider_user,
        provider_message_id=str(provider_message_id or "").strip(),
        reply_to_provider_message_id=str(reply_to_provider_message_id or "").strip(),
        thread_id=str(thread_id or "").strip(),
    ).to_dict()


def channel_address_from_metadata(metadata: dict[str, Any] | None) -> ChannelAddress | None:
    """Extract a channel address from task/message metadata."""
    raw = (metadata or {}).get("channel")
    if not isinstance(raw, dict):
        return None
    integration_id = str(raw.get("integration_id") or "").strip()
    provider_key = str(raw.get("provider_key") or "").strip().lower()
    channel_target = str(raw.get("channel_target") or "").strip()
    if not integration_id or not provider_key or not channel_target:
        return None
    return ChannelAddress(
        integration_id=integration_id,
        provider_key=provider_key,
        channel_target=channel_target,
        alphonse_user_id=str(raw.get("alphonse_user_id") or "").strip(),
        provider_user_id=str(raw.get("provider_user_id") or "").strip(),
        provider_message_id=str(raw.get("provider_message_id") or "").strip(),
        reply_to_provider_message_id=str(raw.get("reply_to_provider_message_id") or "").strip(),
        thread_id=str(raw.get("thread_id") or "").strip(),
    )
