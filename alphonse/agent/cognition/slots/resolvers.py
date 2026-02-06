from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, Protocol


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    value: Any | None = None
    confidence: float | None = None
    error: str | None = None
    normalized: str | None = None


class Resolver(Protocol):
    def parse(self, text: str, context: dict[str, Any]) -> ParseResult: ...


class ResolverRegistry:
    def __init__(self) -> None:
        self._resolvers: dict[str, Resolver] = {}

    def register(self, slot_type: str, resolver: Resolver) -> None:
        self._resolvers[slot_type] = resolver

    def get(self, slot_type: str) -> Resolver | None:
        return self._resolvers.get(slot_type)


class StringResolver:
    def parse(self, text: str, context: dict[str, Any]) -> ParseResult:
        value = str(text or "").strip()
        if not value:
            return ParseResult(ok=False, error="empty")
        return ParseResult(ok=True, value=value, confidence=0.9, normalized=value)


class TimeExpressionResolver:
    def parse(self, text: str, context: dict[str, Any]) -> ParseResult:
        normalized = _normalize_text(text)
        normalized = _replace_number_words(normalized)
        timezone_name = str(context.get("timezone") or "UTC")
        now = context.get("now")
        if not isinstance(now, datetime):
            now = datetime.now(tz=ZoneInfo(timezone_name))
        relative = _parse_relative_time(normalized, now)
        if relative:
            return ParseResult(ok=True, value=relative, confidence=0.8, normalized=normalized)
        explicit = _parse_clock_time(normalized, now)
        if explicit:
            return ParseResult(ok=True, value=explicit, confidence=0.8, normalized=normalized)
        return ParseResult(ok=False, error="unparsed_time", normalized=normalized)


class GeoExpressionResolver:
    def parse(self, text: str, context: dict[str, Any]) -> ParseResult:
        lowered = _normalize_text(text)
        if any(phrase in lowered for phrase in ("llegar a casa", "al llegar a casa", "arrive home", "when i get home")):
            return ParseResult(
                ok=True,
                value={
                    "kind": "geo",
                    "geofence": "home",
                    "event": "enter",
                    "status": "stub_needs_location_setup",
                },
                confidence=0.6,
                normalized=lowered,
            )
        return ParseResult(ok=False, error="unparsed_geo", normalized=lowered)


def build_default_registry() -> ResolverRegistry:
    registry = ResolverRegistry()
    registry.register("string", StringResolver())
    registry.register("time_expression", TimeExpressionResolver())
    registry.register("geo_expression", GeoExpressionResolver())
    return registry


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


_EN_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
}

_ES_NUMBERS = {
    "uno": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciseis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
    "treinta": 30,
    "cuarenta": 40,
    "cincuenta": 50,
}


def _replace_number_words(text: str) -> str:
    tokens = text.split()
    replaced: list[str] = []
    for token in tokens:
        if token in _EN_NUMBERS:
            replaced.append(str(_EN_NUMBERS[token]))
            continue
        if token in _ES_NUMBERS:
            replaced.append(str(_ES_NUMBERS[token]))
            continue
        replaced.append(token)
    return " ".join(replaced)


def _parse_relative_time(text: str, now: datetime) -> dict[str, Any] | None:
    match = re.search(
        r"\b(en|in)\s+(\d+)\s*(min|minutos|minutes?|hr|hora|horas|hours?)\b",
        text,
    )
    if not match:
        match = re.search(r"\b(\d+)\s*(min|minutos|minutes?)\b", text)
    if not match:
        return None
    if match.lastindex and match.lastindex >= 3:
        amount = int(match.group(2))
        unit = match.group(3) or ""
    else:
        amount = int(match.group(1))
        unit = match.group(2) or ""
    unit = unit or ""
    if unit.startswith("h") or "hora" in unit:
        delta = timedelta(hours=amount)
    else:
        delta = timedelta(minutes=amount)
    trigger_at = (now + delta).isoformat()
    return {"kind": "trigger_at", "trigger_at": trigger_at}


def _parse_clock_time(text: str, now: datetime) -> dict[str, Any] | None:
    match = re.search(
        r"\b(?:a\s+las|at)?\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
        text,
    )
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = (match.group(3) or "").lower()
    if meridiem == "pm" and hour < 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate = candidate + timedelta(days=1)
    return {"kind": "trigger_at", "trigger_at": candidate.isoformat()}
