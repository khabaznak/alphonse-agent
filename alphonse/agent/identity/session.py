from __future__ import annotations

from typing import Any

from alphonse.agent import identity
from alphonse.agent.actions.session_context import IncomingContext
from alphonse.agent.actions.state_context import principal_id_for_incoming
from alphonse.agent.cognition.preferences.store import resolve_preference_with_precedence
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
    return identity.resolve_session_user_id(incoming=incoming, payload=payload)


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
