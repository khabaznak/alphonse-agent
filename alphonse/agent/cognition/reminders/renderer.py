from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def render_reminder(payload: dict[str, Any], prefs: dict[str, Any] | None = None) -> str:
    prefs = prefs or {}
    raw = _extract_raw_text(payload)
    cleaned = _normalize_text(raw)
    locale = _resolve_locale(payload, cleaned, prefs)
    prefix = "🕒 Recordatorio: " if locale.startswith("es") else "🕒 Reminder: "
    body = _ensure_terminal_punctuation(cleaned)
    rendered = f"{prefix}{body}"
    logger.info(
        "ReminderRenderer locale=%s raw_len=%s cleaned_len=%s",
        locale,
        len(raw or ""),
        len(cleaned),
    )
    return rendered


def _extract_raw_text(payload: dict[str, Any]) -> str:
    if isinstance(payload, dict):
        raw = payload.get("reminder_text_raw") or payload.get("message")
        if raw:
            return str(raw).strip()
    return ""


def _normalize_text(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return "recordatorio"
    cleaned = cleaned[0].lower() + cleaned[1:] if cleaned else cleaned
    cleaned = _safe_light_corrections(cleaned)
    return cleaned


def _safe_light_corrections(text: str) -> str:
    replacements = {
        r"\brecuarda\b": "recuerda",
        r"\brecuerda\b": "recuerda",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _resolve_locale(payload: dict[str, Any], text: str, prefs: dict[str, Any]) -> str:
    hint = payload.get("locale_hint") if isinstance(payload, dict) else None
    if isinstance(hint, str) and hint.strip():
        return hint
    pref_locale = prefs.get("locale") if isinstance(prefs, dict) else None
    if isinstance(pref_locale, str) and pref_locale.strip():
        return pref_locale
    lowered = text.lower()
    if any(token in lowered for token in ("recuérd", "recuerda", "mañana", "hoy", "bañar", "bañ")):
        return "es-MX"
    return "en-US"


def _ensure_terminal_punctuation(text: str) -> str:
    if not text:
        return ""
    if text[-1] in ".!?":
        return text
    return f"{text}."
