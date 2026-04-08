from __future__ import annotations

from alphonse.agent.identity.session import resolve_session_timezone
from alphonse.agent.identity.session import resolve_session_user_id
from alphonse.agent.services.audio_dispatch import extract_tts_transcript


def resolve_assistant_session_message(*, reply_text: str, plans: list[object]) -> str:
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

__all__ = [
    "resolve_session_timezone",
    "resolve_session_user_id",
    "resolve_assistant_session_message",
]
