from __future__ import annotations

from dataclasses import dataclass

from alphonse.agent import identity
from alphonse.agent.cognition.narration.models import AudienceRef, NarrationIntent


@dataclass(frozen=True)
class ChannelResolution:
    channel_type: str
    address: str | None
    person_id: str | None


def resolve_channel(intent: NarrationIntent) -> ChannelResolution:
    if intent.channel_type == "silent":
        return ChannelResolution(channel_type="silent", address=None, person_id=None)

    if intent.audience.kind == "person":
        service_id = identity.resolve_service_id(intent.channel_type)
        target = identity.resolve_delivery_target(user_id=intent.audience.id, service_id=service_id)
        if target:
            return ChannelResolution(
                channel_type=intent.channel_type,
                address=target,
                person_id=intent.audience.id,
            )
        return ChannelResolution(channel_type="silent", address=None, person_id=intent.audience.id)
    return ChannelResolution(channel_type="silent", address=None, person_id=None)
