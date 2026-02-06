from __future__ import annotations

import logging
from typing import Any

from alphonse.agent.cognition.preferences import store as pref_store

logger = logging.getLogger(__name__)


def set_display_name(conversation_key: str, name: str) -> None:
    pref_store.set_preference_for_conversation(conversation_key, "display_name", name)
    logger.info(
        "identity display_name set key=%s value=%s",
        conversation_key,
        name,
    )


def get_display_name(conversation_key: str) -> str | None:
    value = pref_store.get_preference_for_conversation(conversation_key, "display_name")
    logger.info(
        "identity display_name get key=%s value=%s",
        conversation_key,
        value,
    )
    return value if isinstance(value, str) else None


def get_locale(conversation_key: str) -> str | None:
    value = pref_store.get_preference_for_conversation(conversation_key, "locale")
    return value if isinstance(value, str) else None


def set_locale(conversation_key: str, locale: str) -> None:
    pref_store.set_preference_for_conversation(conversation_key, "locale", locale)
    logger.info(
        "identity locale set key=%s value=%s",
        conversation_key,
        locale,
    )
