from __future__ import annotations

from alphonse.agent import identity
from alphonse.agent.actions.session_context import IncomingContext
from alphonse.agent.cognition.preferences.store import get_or_create_principal_for_channel
from alphonse.agent.cognition.preferences.store import resolve_preference_with_precedence
from alphonse.config import settings


def resolve_session_timezone(incoming: IncomingContext) -> str:
    principal_id = _principal_id_for_incoming(incoming)
    if principal_id:
        timezone_name = resolve_preference_with_precedence(
            key="timezone",
            default=settings.get_timezone(),
            channel_principal_id=principal_id,
        )
        if isinstance(timezone_name, str) and timezone_name.strip():
            return timezone_name.strip()
    return settings.get_timezone()


def resolve_session_user_id(*, incoming: IncomingContext, payload: dict[str, Any]) -> str:
    principal_id = _principal_id_for_incoming(incoming)
    if principal_id:
        principal_user = identity.get_user_by_principal_id(principal_id)
        if principal_user:
            return str(principal_user["user_id"])

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    for candidate in (
        incoming.person_id,
        metadata.get("person_id"),
        payload.get("person_id"),
        payload.get("resolved_user_id"),
        _nested_get(payload, "actor", "person_id"),
    ):
        resolved = _validated_user_id(candidate)
        if resolved:
            return resolved

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
        resolved = _validated_user_id(candidate)
        if resolved:
            return resolved
        if service_id is None:
            continue
        mapped = identity.resolve_user_id(service_id=service_id, service_user_id=str(candidate or "").strip() or None)
        if mapped:
            return mapped
    raise ValueError("unresolved_session_user_id")


def _principal_id_for_incoming(incoming: IncomingContext) -> str | None:
    if incoming.channel_type and (incoming.address or incoming.channel_type):
        channel_id = str(incoming.address or incoming.channel_type)
        return get_or_create_principal_for_channel(str(incoming.channel_type), channel_id)
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
