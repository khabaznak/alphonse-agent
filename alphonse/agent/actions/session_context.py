from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alphonse.agent import identity


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
    person_id = _resolve_person_id_from_normalized(
        channel_type=channel_type,
        address=address,
        metadata=metadata,
        raw_user_id=as_optional_str(getattr(normalized, "user_id", None)),
    )
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
    *,
    channel_type: str,
    address: str | None,
    metadata: dict[str, Any],
    raw_user_id: str | None,
) -> str | None:
    person_id = metadata.get("person_id") if isinstance(metadata, dict) else None
    direct = identity.get_user(str(person_id or "").strip() or None)
    if direct:
        return str(direct.get("user_id") or "")
    service_id = identity.resolve_service_id(channel_type)
    for candidate in (
        raw_user_id,
        metadata.get("user_id") if isinstance(metadata, dict) else None,
        metadata.get("from_user") if isinstance(metadata, dict) else None,
        metadata.get("service_user_id") if isinstance(metadata, dict) else None,
    ):
        mapped = identity.resolve_user_id(
            service_id=service_id,
            service_user_id=str(candidate or "").strip() or None,
        )
        if mapped:
            return mapped
    return None
