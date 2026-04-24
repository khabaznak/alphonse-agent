from __future__ import annotations

from typing import Any

from alphonse.agent.cognition.preferences.store import get_user_preference
from alphonse.agent.identity import service_resolvers as resolver_store
from alphonse.agent.identity import users as users_store
from alphonse.agent.nervous_system.services import get_service, get_service_by_key


def get_user(user_id: str | None) -> dict[str, Any] | None:
    rendered = str(user_id or "").strip()
    if not rendered:
        return None
    user = users_store.get_user(rendered)
    return user if isinstance(user, dict) and user else None


def list_users(*, active_only: bool = False, limit: int = 200) -> list[dict[str, Any]]:
    return users_store.list_users(active_only=active_only, limit=limit)


def get_user_by_display_name(display_name: str | None) -> dict[str, Any] | None:
    rendered = str(display_name or "").strip()
    if not rendered:
        return None
    user = users_store.get_user_by_display_name(rendered)
    return user if isinstance(user, dict) and user else None


def get_user_by_principal_id(principal_id: str | None) -> dict[str, Any] | None:
    rendered = str(principal_id or "").strip()
    if not rendered:
        return None
    user = users_store.get_user(rendered)
    return user if isinstance(user, dict) and user else None


def get_active_admin_user() -> dict[str, Any] | None:
    user = users_store.get_active_admin_user()
    return user if isinstance(user, dict) and user else None


def upsert_user(record: dict[str, Any]) -> str:
    return users_store.upsert_user(record)


def patch_user(user_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    return users_store.patch_user(user_id, updates)


def delete_user(user_id: str) -> bool:
    return users_store.delete_user(user_id)


def resolve_user_id(*, service_id: int | None, service_user_id: str | None) -> str | None:
    if service_id is None:
        return None
    rendered = str(service_user_id or "").strip()
    if not rendered:
        return None
    # TODO This function is actually finding the right user! make sure the others behave like this one. 
    resolved = resolver_store.resolve_user_id_by_service_user_id(
        service_id=int(service_id),
        service_user_id=rendered,
    )
    return resolved


def resolve_service_user_id(*, user_id: str | None, service_id: int | None) -> str | None:
    canonical_user_id = _validated_user_id(user_id)
    if canonical_user_id is None or service_id is None:
        return None
    return resolver_store.resolve_service_user_id(user_id=canonical_user_id, service_id=int(service_id))


def upsert_service_user_id(
    *,
    user_id: str,
    service_id: int,
    service_user_id: str,
    is_active: bool = True,
) -> str:
    canonical_user_id = _validated_user_id(user_id)
    if canonical_user_id is None:
        raise ValueError("missing_user_id")
    return resolver_store.upsert_service_resolver(
        user_id=canonical_user_id,
        service_id=int(service_id),
        service_user_id=str(service_user_id or "").strip(),
        is_active=is_active,
    )


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


def resolve_service_key(service_id: int | None) -> str | None:
    if service_id is None:
        return None
    try:
        service = get_service(int(service_id))
    except (TypeError, ValueError):
        return None
    if not isinstance(service, dict):
        return None
    value = str(service.get("service_key") or "").strip().lower()
    return value or None


def get_preferred_service_id(user_id: str | None) -> int | None:
    user = get_user(user_id)
    canonical_user_id = str((user or {}).get("user_id") or "").strip()
    if not canonical_user_id:
        return None
    preferred = str(get_user_preference(canonical_user_id, "preferred_communication_channel") or "").strip().lower()
    if not preferred:
        return None
    return resolve_service_id(preferred)


def resolve_delivery_target(*, user_id: str | None, service_id: int | None) -> str | None:
    return resolve_service_user_id(user_id=user_id, service_id=service_id)


def resolve_current_actor_user_id(state: dict[str, Any] | None) -> str | None:
    # TODO This function is candidate for removal
    payload = dict(state or {})
    for key in ("actor_person_id", "resolved_user_id", "session_user_id", "owner_id","user_id"):
        resolved = _validated_user_id(payload.get(key))
        if resolved:
            return resolved
    incoming_value = str(payload.get("incoming_user_id") or "").strip()
    resolved = _validated_user_id(incoming_value)
    if resolved:
        return resolved
    service_id = _resolve_state_service_id(payload)
    if service_id is None:
        return None
    return resolve_user_id(
        service_id=service_id,
        service_user_id=incoming_value or str(payload.get("service_user_id") or "").strip() or None,
    )


def _validated_user_id(value: object | None) -> str | None:
    rendered = str(value or "").strip()
    if not rendered:
        return None
    user = users_store.get_user(rendered)
    if not isinstance(user, dict) or not user:
        return None
    user_id = str(user.get("user_id") or "").strip()
    return user_id or None


def _resolve_state_service_id(payload: dict[str, Any]) -> int | None:
    candidates = (
        payload.get("service_id"),
        payload.get("service_key"),
        payload.get("origin_service_id"),
        payload.get("origin_channel"),
        payload.get("channel_type"),
        payload.get("channel"),
    )
    for candidate in candidates:
        resolved = resolve_service_id(str(candidate or "").strip() or None)
        if resolved is not None:
            return resolved
    return None
