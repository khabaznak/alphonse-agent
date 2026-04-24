from __future__ import annotations

from typing import Any

from alphonse.agent import identity
from alphonse.agent.actions.session_context import IncomingContext
from alphonse.agent.cognition.preferences.store import get_user_preference
from alphonse.config import settings


def resolve_session_timezone(incoming: IncomingContext) -> str:
    user_id = _resolve_incoming_user_id(incoming)
    if user_id:
        timezone_name = get_user_preference(user_id, "timezone")
        if isinstance(timezone_name, str) and timezone_name.strip():
            return timezone_name.strip()
    return settings.get_timezone()


def resolve_session_user_id(*, incoming: IncomingContext, payload: dict[str, Any]) -> str:
    resolved = _resolve_incoming_user_id(incoming)
    if resolved:
        return resolved

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    for candidate in (
        incoming.person_id,
        metadata.get("person_id"),
        payload.get("person_id"),
        payload.get("resolved_user_id"),
        _nested_get(payload, "actor", "person_id"),
    ):
        validated = _validated_user_id(candidate)
        if validated:
            return validated

    service_id = _resolve_service_id_from_incoming(incoming=incoming, payload=payload, metadata=metadata)
    for candidate in (
        payload.get("user_id"),
        payload.get("from_user"),
        metadata.get("user_id"),
        metadata.get("from_user"),
        metadata.get("service_user_id"),
        _nested_get(payload, "identity", "external_user_id"),
        _nested_get(payload, "actor", "external_user_id"),
        _nested_get(payload, "metadata", "raw", "user_id"),
        _nested_get(payload, "metadata", "raw", "from_user"),
        _nested_get(payload, "metadata", "raw", "metadata", "user_id"),
        _nested_get(payload, "metadata", "raw", "metadata", "from_user"),
    ):
        validated = _validated_user_id(candidate)
        if validated:
            return validated
        if service_id is None:
            continue
        mapped = identity.resolve_user_id(service_id=service_id, service_user_id=str(candidate or "").strip() or None)
        if mapped:
            return mapped
    raise ValueError("unresolved_session_user_id")


def _resolve_incoming_user_id(incoming: IncomingContext) -> str | None:
    service_id = identity.resolve_service_id(str(incoming.channel_type or "").strip() or None)
    address = str(incoming.address or incoming.channel_type or "").strip()
    if service_id is not None and address:
        resolved = identity.resolve_user_id(service_id=service_id, service_user_id=address)
        if resolved:
            return resolved
    validated = _validated_user_id(incoming.person_id)
    if validated:
        return validated
    return None


def _validated_user_id(value: object | None) -> str | None:
    rendered = str(value or "").strip()
    if not rendered:
        return None
    user = identity.get_user(rendered)
    if not isinstance(user, dict) or not user:
        return None
    user_id = str(user.get("user_id") or "").strip()
    return user_id or None


def _resolve_service_id_from_incoming(
    *,
    incoming: IncomingContext,
    payload: dict[str, Any],
    metadata: dict[str, Any],
) -> int | None:
    for candidate in (
        payload.get("service_id"),
        metadata.get("service_id"),
        payload.get("service_key"),
        metadata.get("service_key"),
        payload.get("provider"),
        payload.get("channel"),
        incoming.channel_type,
    ):
        resolved = identity.resolve_service_id(str(candidate or "").strip() or None)
        if resolved is not None:
            return resolved
    return None


def _nested_get(payload: dict[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
