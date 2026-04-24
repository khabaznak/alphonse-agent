from __future__ import annotations

from alphonse.agent.identity import get_user_by_display_name
from alphonse.agent.identity.service_resolvers import resolve_service_user_id
from alphonse.agent.identity.service_resolvers import resolve_service_id_by_channel_type
from alphonse.agent.identity.service_resolvers import resolve_user_id_by_service_user_id
from alphonse.agent.identity.service_resolvers import upsert_service_resolver
from alphonse.agent.nervous_system.services import TELEGRAM_SERVICE_ID

__all__ = [
    "resolve_service_id_by_channel_type",
    "resolve_service_user_id",
    "resolve_user_id_by_service_user_id",
    "upsert_service_resolver",
    "resolve_telegram_chat_id_for_user",
    "resolve_internal_user_by_telegram_id",
]


def resolve_telegram_chat_id_for_user(user_identifier: str) -> str | None:
    value = str(user_identifier or "").strip()
    if not value:
        return None
    if value.isdigit():
        mapped = resolve_user_id_by_service_user_id(
            service_id=TELEGRAM_SERVICE_ID,
            service_user_id=value,
        )
        if mapped:
            return value
    user = get_user_by_display_name(value)
    user_id = str((user or {}).get("user_id") or "").strip() if isinstance(user, dict) else value
    if user_id == value and user is None:
        user_id = value
    return resolve_service_user_id(user_id=user_id, service_id=TELEGRAM_SERVICE_ID)


def resolve_internal_user_by_telegram_id(telegram_id: str) -> str | None:
    return resolve_user_id_by_service_user_id(
        service_id=TELEGRAM_SERVICE_ID,
        service_user_id=telegram_id,
    )
