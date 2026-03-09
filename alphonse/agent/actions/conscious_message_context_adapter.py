from __future__ import annotations

from alphonse.agent.actions.conscious_message_handler import IncomingMessageEnvelope
from alphonse.agent.actions.session_context import IncomingContext


def build_incoming_context_from_envelope(
    *,
    envelope: IncomingMessageEnvelope,
    correlation_id: str,
) -> IncomingContext:
    channel = envelope.channel if isinstance(envelope.channel, dict) else {}
    actor = envelope.actor if isinstance(envelope.actor, dict) else {}
    metadata = envelope.metadata if isinstance(envelope.metadata, dict) else {}
    channel_type = str(channel.get("type") or "").strip()
    if not channel_type:
        raise ValueError("invalid_envelope: missing channel.type")
    address = str(channel.get("target") or "").strip()
    if not address:
        raise ValueError("invalid_envelope: missing channel.target")
    person_id = str(actor.get("person_id") or "").strip() or None
    message_id = str(envelope.message_id or "").strip() or None
    update_id = str(metadata.get("provider_event_id") or "").strip() or None
    return IncomingContext(
        channel_type=channel_type,
        address=address,
        person_id=person_id,
        correlation_id=correlation_id,
        update_id=update_id,
        message_id=message_id,
    )
