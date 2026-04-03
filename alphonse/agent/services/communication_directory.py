from __future__ import annotations

from typing import Any

from alphonse.agent.cognition.preferences.store import get_preference
from alphonse.agent.nervous_system import user_service_resolvers as resolver_store
from alphonse.agent.nervous_system import users as users_store
from alphonse.agent.nervous_system.services import get_service, get_service_by_key


def resolve_service_id(service_key: str | None) -> int | None:
    key = str(service_key or "").strip().lower()
    if not key:
        return None
    if key.isdigit():
        return int(key)
    service = get_service_by_key(key)
    if not isinstance(service, dict):
        return None
    value = service.get("service_id")
    return int(value) if value is not None else None


def resolve_service(service_id: int | None) -> dict[str, Any] | None:
    if service_id is None:
        return None
    try:
        return get_service(int(service_id))
    except (TypeError, ValueError):
        return None


def resolve_service_key(service_id: int | None) -> str | None:
    service = resolve_service(service_id)
    if not isinstance(service, dict):
        return None
    value = str(service.get("service_key") or "").strip().lower()
    return value or None


def resolve_user_id(*, service_id: int | None, service_user_id: str | None) -> str | None:
    if service_id is None:
        return None
    value = str(service_user_id or "").strip()
    if not value:
        return None
    return resolver_store.resolve_user_id_by_service_user_id(
        service_id=int(service_id),
        service_user_id=value,
    )


def resolve_service_user_id(*, user_id: str | None, service_id: int | None) -> str | None:
    rendered_user_id = str(user_id or "").strip()
    if not rendered_user_id or service_id is None:
        return None
    return resolver_store.resolve_service_user_id(
        user_id=rendered_user_id,
        service_id=int(service_id),
    )


def get_preferred_service_id(user_id: str | None) -> int | None:
    rendered_user_id = str(user_id or "").strip()
    if not rendered_user_id:
        return None
    user = users_store.get_user(rendered_user_id)
    if not isinstance(user, dict):
        return None
    principal_id = str(user.get("principal_id") or "").strip()
    if not principal_id:
        return None
    preferred = str(get_preference(principal_id, "preferred_communication_channel") or "").strip().lower()
    if not preferred:
        return None
    return resolve_service_id(preferred)


def resolve_delivery_target(*, user_id: str | None, service_id: int | None) -> str | None:
    return resolve_service_user_id(user_id=user_id, service_id=service_id)

