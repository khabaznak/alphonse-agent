from __future__ import annotations

from typing import Any

from alphonse.agent.actions.session_context import IncomingContext
from alphonse.agent.actions.state_context import principal_id_for_incoming
from alphonse.agent.cognition.preferences.store import resolve_preference_with_precedence
from alphonse.agent.nervous_system import users as users_store
from alphonse.agent.services.audio_dispatch import extract_tts_transcript
from alphonse.config import settings


def resolve_session_timezone(incoming: IncomingContext) -> str:
    principal_id = principal_id_for_incoming(incoming)
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
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    principal_id = principal_id_for_incoming(incoming)
    if principal_id:
        try:
            principal_user = users_store.get_user_by_principal_id(principal_id)
        except Exception:
            principal_user = None
        if isinstance(principal_user, dict):
            db_user_id = str(principal_user.get("user_id") or "").strip()
            if db_user_id:
                return db_user_id

    resolved_name = resolve_display_name(payload=payload, metadata=metadata)
    if resolved_name:
        try:
            matched_user = users_store.get_user_by_display_name(resolved_name)
        except Exception:
            matched_user = None
        if isinstance(matched_user, dict):
            db_user_id = str(matched_user.get("user_id") or "").strip()
            if db_user_id:
                return db_user_id
        if str(incoming.channel_type or "").strip().lower() == "telegram":
            return f"name:{resolved_name.lower()}"

    candidates = [
        incoming.person_id,
        metadata.get("person_id"),
        payload.get("person_id"),
        payload.get("user_id"),
        payload.get("from_user"),
        metadata.get("user_id"),
        metadata.get("from_user"),
        nested_get(payload, "metadata", "raw", "user_id"),
        nested_get(payload, "metadata", "raw", "from_user"),
        nested_get(payload, "metadata", "raw", "metadata", "user_id"),
    ]
    for candidate in candidates:
        rendered = str(candidate or "").strip()
        if rendered:
            return rendered
    if resolved_name:
        return f"name:{resolved_name.lower()}"
    chat_id = str(payload.get("chat_id") or "").strip()
    if chat_id:
        return chat_id
    if incoming.address:
        return f"{incoming.channel_type}:{incoming.address}"
    return f"{incoming.channel_type}:anonymous"


def resolve_display_name(*, payload: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    candidates = [
        payload.get("user_name"),
        payload.get("from_user_name"),
        metadata.get("user_name"),
        metadata.get("from_user_name"),
        nested_get(payload, "provider_event", "message", "from", "first_name"),
        nested_get(payload, "provider_event", "message", "from", "username"),
        nested_get(payload, "provider_event", "message", "chat", "first_name"),
        nested_get(payload, "metadata", "raw", "user_name"),
        nested_get(payload, "metadata", "raw", "from_user_name"),
        nested_get(payload, "metadata", "raw", "metadata", "user_name"),
        nested_get(payload, "metadata", "raw", "metadata", "from_user_name"),
    ]
    for candidate in candidates:
        rendered = str(candidate or "").strip()
        if rendered:
            return rendered
    return None


def nested_get(payload: dict[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def resolve_assistant_session_message(*, reply_text: str, plans: list[Any]) -> str:
    reply_line = str(reply_text or "").strip()
    transcript = extract_tts_transcript(plans)
    if not transcript:
        return reply_line
    if not reply_line:
        return f"[TTS transcript] {transcript}"
    if _normalized_text(reply_line) == _normalized_text(transcript):
        return f"[TTS transcript] {reply_line}"
    return f"{reply_line} [TTS transcript: {transcript}]"


def _normalized_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())
