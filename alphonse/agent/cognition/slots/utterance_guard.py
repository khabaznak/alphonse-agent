from __future__ import annotations

import re


_GREETING = [
    r"\b(hi|hello|hey|hola|buenas|buenos dias|buenas tardes|buenas noches)\b",
]
_HELP = [r"\b(help|ayuda)\b"]
_META = [
    r"\b(what can you do|what else can you do|capabilities|que puedes hacer|qué puedes hacer|que sabes hacer|qué sabes hacer)\b",
]
_STATUS = [r"\b(status|estado)\b"]
_CANCEL = [r"\b(cancel|stop|olvida|olvidalo|olvídalo|cancelar|parar|detener)\b"]
_IDENTITY = [
    r"\b(quien eres|quién eres|who are you|what is your name|como te llamas|cómo te llamas)\b",
]
_USER_IDENTITY = [
    r"\b(quien soy yo|quién soy yo|como me llamo|cómo me llamo|what is my name|do you know my name)\b",
]
_RESUME = [
    r"\b(continuar|continua|seguir|retomar|resume)\b",
]


def detect_core_intent(text: str) -> str | None:
    normalized = _normalize(text)
    if _matches_any(normalized, _CANCEL):
        return "cancel"
    if _matches_any(normalized, _GREETING):
        return "greeting"
    if _matches_any(normalized, _HELP):
        return "help"
    if _matches_any(normalized, _META):
        return "meta.capabilities"
    if _matches_any(normalized, _STATUS):
        return "get_status"
    if _matches_any(normalized, _USER_IDENTITY):
        return "identity.query_user_name"
    if _matches_any(normalized, _IDENTITY):
        return "identity_question"
    return None


def is_core_conversational_utterance(text: str) -> bool:
    return detect_core_intent(text) is not None


def is_resume_utterance(text: str) -> bool:
    return _matches_any(_normalize(text), _RESUME)


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())
