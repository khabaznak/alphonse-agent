from __future__ import annotations

_RESUME_KEYWORDS = {"continuar", "continua", "seguir", "retomar", "resume"}


def detect_core_intent(text: str) -> str | None:
    _ = text
    return None


def is_core_conversational_utterance(text: str) -> bool:
    return detect_core_intent(text) is not None


def is_resume_utterance(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().lower().split())
    return any(keyword in normalized for keyword in _RESUME_KEYWORDS)


def _core_intent_priority(spec) -> int:
    _ = spec
    return 10
