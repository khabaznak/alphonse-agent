from __future__ import annotations

import re
import unicodedata
from typing import Any


def build_intent_evidence(text: str) -> dict[str, Any]:
    reminder_verbs = _extract_reminder_verbs(text)
    time_hints = _extract_time_hints(text)
    quoted_spans = _extract_quoted_spans(text)
    score = _score_evidence(reminder_verbs, time_hints, quoted_spans)
    return {
        "reminder_verbs": reminder_verbs,
        "time_hints": time_hints,
        "quoted_spans": quoted_spans,
        "score": score,
    }


def pairing_command_intent(text: str) -> str | None:
    lowered = text.lower().strip()
    if lowered.startswith(("/approve", "approve")):
        return "pair.approve"
    if lowered.startswith(("/deny", "deny")):
        return "pair.deny"
    return None


def extract_preference_updates(text: str) -> list[dict[str, str]]:
    lowered = text.lower()
    normalized = _normalize_pref_text(text)
    updates: list[dict[str, str]] = []

    if _matches_any(
        normalized,
        [
            r"\bhablame\s+de\s+tu\b",
            r"\bhableme\s+de\s+tu\b",
            r"\bhable\s+de\s+tu\b",
            r"\bhabla\s+de\s+tu\b",
            r"\btutea(me|nos)\b",
        ],
    ):
        updates.append({"key": "address_style", "value": "tu"})

    if _matches_any(
        normalized,
        [
            r"\bhablame\s+de\s+usted\b",
            r"\bhableme\s+de\s+usted\b",
            r"\bhabla\s+de\s+usted\b",
            r"\bhable\s+de\s+usted\b",
            r"\btratame\s+de\s+usted\b",
        ],
    ):
        updates.append({"key": "address_style", "value": "usted"})

    if _matches_any(
        lowered,
        [
            r"\bspeak\s+english\b",
            r"\bin\s+english\b",
            r"\benglish\b",
            r"\bhabla\s+en\s+ingl[eé]s\b",
            r"\bhabla\s+ingl[eé]s\b",
        ],
    ):
        updates.append({"key": "locale", "value": "en-US"})

    if _matches_any(
        normalized,
        [
            r"\bhabla\s+en\s+espanol\b",
            r"\bhabla\s+espanol\b",
            r"\bhablemos\s+en\s+espanol\b",
            r"\bahora\s+hablemos\s+en\s+espanol\b",
            r"\ben\s+espanol\b",
            r"\bespanol\b",
        ],
    ):
        updates.append({"key": "locale", "value": "es-MX"})

    if _matches_any(
        lowered,
        [
            r"\bs[eé]\s+m[aá]s\s+formal\b",
            r"\bm[aá]s\s+formal\b",
            r"\bbe\s+more\s+formal\b",
            r"\bmore\s+formal\b",
        ],
    ):
        updates.append({"key": "tone", "value": "formal"})

    if _matches_any(
        lowered,
        [
            r"\bs[eé]\s+m[aá]s\s+casual\b",
            r"\bm[aá]s\s+casual\b",
            r"\bm[aá]s\s+relajado\b",
            r"\bm[aá]s\s+informal\b",
            r"\bbe\s+more\s+casual\b",
            r"\bmore\s+casual\b",
            r"\bbe\s+more\s+friendly\b",
        ],
    ):
        updates.append({"key": "tone", "value": "friendly"})

    return _dedupe_preference_updates(updates)


def _extract_reminder_verbs(text: str) -> list[str]:
    lowered = text.lower()
    tokens = [
        "remind",
        "reminder",
        "recuérdame",
        "recuerdame",
        "recuérdale",
        "recuerdale",
        "recordatorio",
        "recordar",
    ]
    return [token for token in tokens if token in lowered]


def _extract_time_hints(text: str) -> list[str]:
    lowered = text.lower()
    hints = []
    for token in (
        "hoy",
        "mañana",
        "manana",
        "tonight",
        "today",
        "tomorrow",
        "cada",
        "daily",
    ):
        if token in lowered:
            hints.append(token)
    for match in re.finditer(r"\b\d{1,2}(:\d{2})?\b", lowered):
        hints.append(match.group(0))
    return hints


def _extract_quoted_spans(text: str) -> list[str]:
    matches = re.findall(r"“([^”]+)”|\"([^\"]+)\"|'([^']+)'", text)
    spans: list[str] = []
    for match in matches:
        for item in match:
            if item:
                spans.append(item)
    return spans


def _score_evidence(
    reminder_verbs: list[str], time_hints: list[str], quoted_spans: list[str]
) -> float:
    score = 0.0
    if reminder_verbs:
        score += 0.5
    if time_hints:
        score += 0.4
    if quoted_spans:
        score += 0.2
    return min(score, 1.0)


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _normalize_pref_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def _dedupe_preference_updates(updates: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: dict[str, dict[str, str]] = {}
    for update in updates:
        key = update.get("key")
        value = update.get("value")
        if not key or value is None:
            continue
        deduped[key] = {"key": key, "value": value}
    return list(deduped.values())
