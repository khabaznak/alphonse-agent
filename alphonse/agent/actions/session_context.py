from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alphonse.agent.identity import store as identity_store


@dataclass(frozen=True)
class IncomingContext:
    channel_type: str
    address: str | None
    person_id: str | None
    correlation_id: str
    update_id: str | None = None
    message_id: str | None = None


def build_incoming_context_from_normalized(
    normalized: object,
    correlation_id: str,
) -> IncomingContext:
    channel_type = str(getattr(normalized, "channel_type", "") or "system")
    address = as_optional_str(getattr(normalized, "channel_target", None))
    metadata = getattr(normalized, "metadata", {}) or {}
    person_id = _resolve_person_id_from_normalized(channel_type, address, metadata)
    update_id = metadata.get("update_id") if isinstance(metadata, dict) else None
    return IncomingContext(
        channel_type=channel_type,
        address=address,
        person_id=person_id,
        correlation_id=correlation_id,
        update_id=str(update_id) if update_id is not None else None,
        message_id=as_optional_str(metadata.get("message_id")) if isinstance(metadata, dict) else None,
    )


def build_session_key(incoming: IncomingContext) -> str:
    if incoming.channel_type and incoming.address:
        return f"{incoming.channel_type}:{incoming.address}"
    if incoming.channel_type:
        return f"{incoming.channel_type}:{incoming.channel_type}"
    raise ValueError("incoming.channel_type is required to build session key")


def as_optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _resolve_person_id_from_normalized(
    channel_type: str,
    address: str | None,
    metadata: dict[str, Any],
) -> str | None:
    person_id = metadata.get("person_id") if isinstance(metadata, dict) else None
    if person_id:
        return str(person_id)
    if channel_type and address:
        person = identity_store.resolve_person_by_channel(channel_type, address)
        if person:
            return str(person.get("person_id"))
    return None
