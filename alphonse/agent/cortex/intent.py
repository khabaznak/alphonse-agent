from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import dateparser

from alphonse.agent.cognition.providers.ollama import OllamaClient


INTENTS = {
    "schedule_reminder",
    "get_status",
    "help",
    "identity_question",
    "greeting",
    "unknown",
}


@dataclass(frozen=True)
class IntentResult:
    intent: str
    confidence: float


def classify_intent(text: str, llm_client: OllamaClient | None) -> IntentResult:
    heuristic = _heuristic_intent(text)
    if heuristic.intent != "unknown":
        return heuristic
    if llm_client is None:
        return IntentResult(intent="unknown", confidence=0.2)
    prompt = (
        "Classify the intent into one of: schedule_reminder, get_status, help, "
        "identity_question, greeting, unknown. Return JSON {\"intent\":...,\"confidence\":0-1}."
    )
    try:
        raw = llm_client.complete(system_prompt=prompt, user_prompt=text)
    except Exception:
        return IntentResult(intent="unknown", confidence=0.2)
    data = _parse_json(str(raw)) or {}
    intent = str(data.get("intent") or "unknown").strip()
    if intent not in INTENTS:
        intent = "unknown"
    confidence = _as_float(data.get("confidence"), default=0.4)
    return IntentResult(intent=intent, confidence=confidence)


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
    relative = _parse_relative_minutes(text, timezone)
    if relative:
        return relative
    explicit = _parse_explicit_time(text, timezone)
    if explicit:
        return explicit
    parsed = dateparser.parse(
        text,
        settings={
            "TIMEZONE": timezone,
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "future",
        },
    )
    if parsed:
        return parsed.isoformat()
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


def _heuristic_intent(text: str) -> IntentResult:
    lowered = text.lower().strip()
    if any(token in lowered for token in ("status", "estado")):
        return IntentResult(intent="get_status", confidence=0.7)
    if any(token in lowered for token in ("ayuda", "help", "what can you do", "que puedes")):
        return IntentResult(intent="help", confidence=0.7)
    if any(token in lowered for token in ("quien eres", "quién eres", "quien soy yo", "quién soy yo")):
        return IntentResult(intent="identity_question", confidence=0.7)
    if lowered.startswith(("hola", "hello", "hi", "buenas", "buenos", "hey")):
        return IntentResult(intent="greeting", confidence=0.6)
    if _contains_reminder_intent(lowered):
        return IntentResult(intent="schedule_reminder", confidence=0.6)
    return IntentResult(intent="unknown", confidence=0.2)


def _contains_reminder_intent(text: str) -> bool:
    verbs = _extract_reminder_verbs(text)
    return bool(verbs)


def _parse_relative_minutes(text: str, timezone: str) -> str | None:
    match = re.search(r"\b(en|in)\s+(\d+)\s*(min|minuto|minutos|minutes?)\b", text, re.IGNORECASE)
    if not match:
        return None
    minutes = int(match.group(2))
    now = datetime.now().astimezone()
    return (now + timedelta(minutes=minutes)).isoformat()


def _parse_explicit_time(text: str, timezone: str) -> str | None:
    match = re.search(r"\b(a las|at)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text, re.IGNORECASE)
    if not match:
        return None
    hour = int(match.group(2))
    minute = int(match.group(3) or 0)
    meridiem = (match.group(4) or "").lower()
    if meridiem == "pm" and hour < 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    now = datetime.now().astimezone()
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate < now:
        candidate = candidate + timedelta(days=1)
    return candidate.isoformat()


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
    for token in ("hoy", "mañana", "manana", "tonight", "today", "tomorrow", "cada", "daily"):
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


def _score_evidence(reminder_verbs: list[str], time_hints: list[str], quoted_spans: list[str]) -> float:
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


def _parse_json(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _as_float(value: object | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
