from __future__ import annotations

from alphonse.agent.observability.log_manager import get_component_logger

logger = get_component_logger("cognition.preferences.conversation_profile")


def set_display_name(conversation_key: str, name: str) -> None:
    logger.info(
        "conversation display_name ignored key=%s value=%s runtime_only=true",
        conversation_key,
        name,
    )


def get_display_name(conversation_key: str) -> str | None:
    logger.info("conversation display_name get ignored key=%s runtime_only=true", conversation_key)
    return None


def get_locale(conversation_key: str) -> str | None:
    logger.info("conversation locale get ignored key=%s runtime_only=true", conversation_key)
    return None


def set_locale(conversation_key: str, locale: str) -> None:
    logger.info(
        "conversation locale ignored key=%s value=%s runtime_only=true",
        conversation_key,
        locale,
    )
