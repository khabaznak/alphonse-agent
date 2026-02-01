from __future__ import annotations

import re
from typing import Any


def triage_gap_text(text: str) -> dict[str, Any]:
    normalized = _normalize(text)
    if not normalized:
        return _fallback()

    if _matches_greeting(normalized):
        return {
            "category": "intent_missing",
            "suggested_intent": "greeting",
            "confidence": 0.9,
        }
    if _matches_meta_gaps(normalized):
        return {
            "category": "intent_missing",
            "suggested_intent": "meta.gaps_list",
            "confidence": 0.9,
        }
    if _matches_meta_capabilities(normalized):
        return {
            "category": "intent_missing",
            "suggested_intent": "meta.capabilities",
            "confidence": 0.9,
        }
    if _matches_reminder_list(normalized):
        return {
            "category": "intent_missing",
            "suggested_intent": "timed_signals.list",
            "confidence": 0.9,
        }

    return _fallback()


def detect_language(text: str) -> str:
    normalized = _normalize(text)
    if not normalized:
        return "en"
    if re.search(r"[¿¡áéíóúñ]", normalized):
        return "es"
    for token in (
        "hola",
        "buenos",
        "buenas",
        "que",
        "qué",
        "como",
        "cómo",
        "recuerdame",
        "recordatorios",
        "ayuda",
    ):
        if token in normalized:
            return "es"
    return "en"


def _normalize(text: str) -> str:
    return str(text or "").strip().lower()


def _matches_greeting(text: str) -> bool:
    return bool(
        re.search(
            r"\b(hi|hello|hey|good morning|good afternoon|good evening|hola|buenos dias|buenos días|buenas)\b",
            text,
        )
    )


def _matches_meta_gaps(text: str) -> bool:
    return bool(re.search(r"\b(gaps\??|gap list|gaps list|lista de brechas|brechas)\b", text))


def _matches_meta_capabilities(text: str) -> bool:
    return bool(
        re.search(
            r"\b(what else can you do|what can you do|capabilities|qué más puedes hacer|que mas puedes hacer|que puedes hacer)\b",
            text,
        )
    )


def _matches_reminder_list(text: str) -> bool:
    return bool(
        re.search(
            r"\b(what reminders do you have|reminders scheduled|list reminders|qué recordatorios tienes|recordatorios programados)\b",
            text,
        )
    )


def _fallback() -> dict[str, Any]:
    return {
        "category": "intent_missing",
        "suggested_intent": None,
        "confidence": 0.2,
    }
