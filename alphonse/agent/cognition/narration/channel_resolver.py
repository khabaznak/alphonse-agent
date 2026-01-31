from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alphonse.agent.cognition.narration.models import AudienceRef, NarrationIntent
from alphonse.agent.identity import store as identity_store


@dataclass(frozen=True)
class ChannelResolution:
    channel_type: str
    address: str | None
    person_id: str | None


def resolve_channel(intent: NarrationIntent, fallback_address: str | None = None) -> ChannelResolution:
    if intent.channel_type == "silent":
        return ChannelResolution(channel_type="silent", address=None, person_id=None)

    if intent.audience.kind == "person":
        channels = identity_store.list_channels_for_person(intent.audience.id, intent.channel_type)
        if channels:
            channel = channels[0]
            return ChannelResolution(
                channel_type=channel["channel_type"],
                address=str(channel["address"]),
                person_id=channel.get("person_id"),
            )
    if intent.audience.kind == "group":
        channels = identity_store.list_channels_for_group(intent.audience.id, intent.channel_type)
        if channels:
            channel = channels[0]
            return ChannelResolution(
                channel_type=channel["channel_type"],
                address=str(channel["address"]),
                person_id=channel.get("person_id"),
            )

    if intent.channel_type in {"cli", "api"}:
        return ChannelResolution(channel_type=intent.channel_type, address=None, person_id=None)
    if fallback_address:
        return ChannelResolution(channel_type=intent.channel_type, address=fallback_address, person_id=None)
    return ChannelResolution(channel_type="silent", address=None, person_id=None)
