from __future__ import annotations

from alphonse.agent.identity.service_resolvers import resolve_internal_user_by_telegram_id
from alphonse.agent.identity.service_resolvers import resolve_service_user_id
from alphonse.agent.identity.service_resolvers import resolve_telegram_chat_id_for_user
from alphonse.agent.identity.service_resolvers import resolve_user_id_by_service_user_id
from alphonse.agent.identity.service_resolvers import upsert_service_resolver

__all__ = [
    "resolve_service_user_id",
    "resolve_user_id_by_service_user_id",
    "upsert_service_resolver",
    "resolve_telegram_chat_id_for_user",
    "resolve_internal_user_by_telegram_id",
]
