from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.actions.session_context import IncomingContext, as_optional_str, build_session_key
from alphonse.agent.cognition.preferences.store import (
    get_or_create_principal_for_channel,
    resolve_preference_with_precedence,
)
from alphonse.agent.identity import profile as identity_profile
from alphonse.config import settings

logger = logging.getLogger(__name__)


def ensure_conversation_locale(
    *,
    conversation_key: str,
    stored_state: dict[str, Any],
    incoming: IncomingContext,
) -> None:
    if stored_state.get("locale"):
        return
    channel_locale = resolve_channel_locale_context(incoming)
    if channel_locale:
        stored_state["locale"] = channel_locale
        return
    existing = identity_profile.get_locale(conversation_key)
    if existing:
        stored_state["locale"] = existing
        return
    stored_state["locale"] = settings.get_default_locale()


def build_cortex_state(
    *,
    stored_state: dict[str, Any],
    incoming: IncomingContext,
    correlation_id: str,
    payload: dict[str, Any],
    normalized: object | None,
) -> dict[str, Any]:
    try:
        from alphonse.agent.nervous_system.migrate import apply_schema
        from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path

        apply_schema(resolve_nervous_system_db_path())
    except Exception:
        pass

    principal_id = principal_id_for_incoming(incoming)
    effective_locale = settings.get_default_locale()
    effective_tone = settings.get_tone()
    effective_address = settings.get_address_style()
    timezone_name = settings.get_timezone()
    if principal_id:
        timezone_name = resolve_preference_with_precedence(
            key="timezone",
            default=timezone_name,
            channel_principal_id=principal_id,
        )
        effective_locale = resolve_preference_with_precedence(
            key="locale",
            default=settings.get_default_locale(),
            channel_principal_id=principal_id,
        )
        effective_tone = resolve_preference_with_precedence(
            key="tone",
            default=settings.get_tone(),
            channel_principal_id=principal_id,
        )
        effective_address = resolve_preference_with_precedence(
            key="address_style",
            default=settings.get_address_style(),
            channel_principal_id=principal_id,
        )
    logger.info(
        "HandleIncomingMessageAction principal channel=%s channel_id=%s principal_id=%s locale=%s tone=%s address=%s",
        incoming.channel_type,
        incoming.address,
        principal_id,
        effective_locale,
        effective_tone,
        effective_address,
    )
    planning_mode = payload.get("planning_mode") if isinstance(payload, dict) else None
    autonomy_level = payload.get("autonomy_level") if isinstance(payload, dict) else None

    session_key = build_session_key(incoming)
    state_locale = stored_state.get("locale")
    if not state_locale:
        state_locale = resolve_channel_locale_context(incoming)
    if not state_locale:
        state_locale = identity_profile.get_locale(session_key)
    if not state_locale:
        state_locale = effective_locale

    incoming_user_id = as_optional_str(getattr(normalized, "user_id", None))
    incoming_user_name = as_optional_str(getattr(normalized, "user_name", None))
    incoming_meta = getattr(normalized, "metadata", {}) if normalized is not None else {}
    if not isinstance(incoming_meta, dict):
        incoming_meta = {}

    pending_interaction, ability_state = sanitize_interaction_state(stored_state)
    return {
        "chat_id": incoming.address or incoming.channel_type,
        "channel_type": incoming.channel_type,
        "channel_target": incoming.address or incoming.channel_type,
        "conversation_key": session_key,
        "incoming_raw_message": incoming_meta.get("raw")
        if isinstance(incoming_meta.get("raw"), dict)
        else None,
        "actor_person_id": incoming.person_id,
        "incoming_user_id": incoming_user_id,
        "incoming_user_name": incoming_user_name,
        "incoming_reply_to_user_id": as_optional_str(incoming_meta.get("reply_to_user")),
        "incoming_reply_to_user_name": as_optional_str(incoming_meta.get("reply_to_user_name")),
        "slots": stored_state.get("slots_collected") or {},
        "intent": stored_state.get("last_intent"),
        "locale": state_locale,
        "tone": effective_tone,
        "address_style": effective_address,
        "autonomy_level": autonomy_level or stored_state.get("autonomy_level"),
        "planning_mode": planning_mode or stored_state.get("planning_mode"),
        "intent_category": stored_state.get("intent_category"),
        "routing_rationale": stored_state.get("routing_rationale"),
        "routing_needs_clarification": stored_state.get("routing_needs_clarification"),
        "pending_interaction": pending_interaction,
        "ability_state": ability_state,
        "slot_machine": stored_state.get("slot_machine"),
        "correlation_id": correlation_id,
        "timezone": timezone_name,
    }


def sanitize_interaction_state(
    stored_state: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    pending = stored_state.get("pending_interaction")
    ability_state = stored_state.get("ability_state")
    if pending is None:
        return None, ability_state if isinstance(ability_state, dict) else None
    if not isinstance(pending, dict):
        raise TypeError("pending_interaction must be a dict when present")
    _validate_pending_interaction_shape(pending)
    if is_pending_interaction_expired(pending):
        logger.info("HandleIncomingMessageAction clearing expired pending_interaction")
        return None, ability_state if isinstance(ability_state, dict) else None
    return pending, ability_state if isinstance(ability_state, dict) else None


def is_pending_interaction_expired(pending: dict[str, Any]) -> bool:
    expires_at = pending.get("expires_at")
    if expires_at is None:
        return False
    if not isinstance(expires_at, str) or not expires_at.strip():
        raise ValueError("pending_interaction.expires_at must be a non-empty ISO timestamp when present")
    expires_dt = datetime.fromisoformat(expires_at.strip())
    if expires_dt.tzinfo is None:
        expires_dt = expires_dt.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) >= expires_dt


def resolve_channel_locale_context(incoming: IncomingContext) -> str | None:
    principal_id = principal_id_for_incoming(incoming)
    if not principal_id:
        return None
    value = resolve_preference_with_precedence(
        key="locale",
        default=None,
        channel_principal_id=principal_id,
    )
    return value if isinstance(value, str) else None


def principal_id_for_incoming(incoming: IncomingContext) -> str | None:
    if incoming.channel_type and (incoming.address or incoming.channel_type):
        channel_id = str(incoming.address or incoming.channel_type)
        return get_or_create_principal_for_channel(
            str(incoming.channel_type),
            channel_id,
        )
    return None


def outgoing_locale(cognition_state: dict[str, Any] | None) -> str:
    if isinstance(cognition_state, dict):
        locale = cognition_state.get("locale")
        if isinstance(locale, str) and locale.strip():
            return locale.strip()
    return settings.get_default_locale()


def _validate_pending_interaction_shape(pending: dict[str, Any]) -> None:
    interaction_type = pending.get("type")
    key = pending.get("key")
    if not isinstance(interaction_type, str) or not interaction_type.strip():
        raise ValueError("pending_interaction.type must be a non-empty string")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("pending_interaction.key must be a non-empty string")
