from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any

import dateparser

from alphonse.config import settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntentResult:
    intent: str
    confidence: float


def classify_intent(text: str, llm_client: object | None = None) -> IntentResult:
    from alphonse.agent.cognition.intent_catalog import get_catalog_service
    from alphonse.agent.cognition.intent_detector_llm import IntentDetectorLLM

    service = get_catalog_service()
    detector = IntentDetectorLLM(service)
    detection = detector.detect(text, llm_client=llm_client)
    if not detection:
        return IntentResult(intent="unknown", confidence=0.0)
    return IntentResult(intent=detection.intent_name, confidence=detection.confidence)


def extract_reminder_text(text: str) -> str | None:
    patterns = [
        r"recu[eé]rdame\s+(?P<msg>.+)",
        r"recu[eé]rda\s+(?P<msg>.+)",
        r"remind me to\s+(?P<msg>.+)",
        r"set a reminder to\s+(?P<msg>.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _strip_time_phrases(match.group("msg").strip())
    return None


def parse_trigger_time(text: str, timezone: str) -> str | None:
    tz_name = timezone or settings.get_timezone()
    now = datetime.now(tz=ZoneInfo(tz_name))
    logger.info(
        "intent parse_trigger_time input=%s timezone=%s now=%s",
        text,
        tz_name,
        now.isoformat(),
    )
    relative = _parse_relative_minutes(text, now)
    if relative:
        logger.info(
            "intent parse_trigger_time matched=relative expression=%s trigger_at=%s",
            relative[0],
            relative[1],
        )
        return relative[1]
    explicit = _parse_explicit_time(text, now)
    if explicit:
        logger.info(
            "intent parse_trigger_time matched=explicit expression=%s trigger_at=%s",
            explicit[0],
            explicit[1],
        )
        return explicit[1]
    parsed = dateparser.parse(
        text,
        settings={
            "TIMEZONE": tz_name,
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "future",
        },
    )
    if parsed:
        logger.info(
            "intent parse_trigger_time matched=dateparser trigger_at=%s",
            parsed.isoformat(),
        )
        return parsed.isoformat()
    logger.info("intent parse_trigger_time matched=none")
    return None


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


def _pairing_command_intent(text: str) -> str | None:
    lowered = text.lower().strip()
    if lowered.startswith(("/approve", "approve")):
        return "pair.approve"
    if lowered.startswith(("/deny", "deny")):
        return "pair.deny"
    return None


def pairing_command_intent(text: str) -> str | None:
    return _pairing_command_intent(text)


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


def _contains_reminder_intent(text: str) -> bool:
    verbs = _extract_reminder_verbs(text)
    return bool(verbs)


def _parse_relative_minutes(text: str, now: datetime) -> tuple[str, str] | None:
    match = re.search(
        r"\b(en|in)\s+(\d+)\s*(min|minuto|minutos|minutes?)\b", text, re.IGNORECASE
    )
    if not match:
        return None
    minutes = int(match.group(2))
    expression = match.group(0)
    return expression, (now + timedelta(minutes=minutes)).isoformat()


def _parse_explicit_time(text: str, now: datetime) -> tuple[str, str] | None:
    match = re.search(
        r"\b(a las|at)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text, re.IGNORECASE
    )
    if not match:
        return None
    hour = int(match.group(2))
    minute = int(match.group(3) or 0)
    meridiem = (match.group(4) or "").lower()
    if meridiem == "pm" and hour < 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    expression = match.group(0)
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate < now:
        candidate = candidate + timedelta(days=1)
    return expression, candidate.isoformat()


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


def _strip_time_phrases(text: str) -> str:
    return re.sub(r"\b(en|in|a las|at)\b.*", "", text, flags=re.IGNORECASE).strip()


def contains_reminder_intent(text: str) -> bool:
    return _contains_reminder_intent(text)
