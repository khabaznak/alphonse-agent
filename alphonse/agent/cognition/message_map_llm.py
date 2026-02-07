from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


Confidence = Literal["low", "medium", "high"]
IntentHint = Literal[
    "social_only",
    "single_action",
    "multi_action",
    "question_only",
    "command_only",
    "mixed",
    "other",
]


@dataclass(frozen=True)
class SocialFragment:
    is_greeting: bool = False
    is_farewell: bool = False
    is_thanks: bool = False
    text: str | None = None


@dataclass(frozen=True)
class Constraints:
    times: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ActionFragment:
    verb: str
    object: str | None = None
    target: str | None = None
    details: str | None = None
    priority: Literal["low", "normal", "high"] = "normal"


@dataclass(frozen=True)
class MessageMap:
    language: str
    social: SocialFragment
    actions: list[ActionFragment]
    entities: list[str]
    constraints: Constraints
    questions: list[str]
    commands: list[str]
    raw_intent_hint: IntentHint
    confidence: Confidence


@dataclass(frozen=True)
class MessageMapResult:
    raw_json: dict[str, Any]
    message_map: MessageMap
    parse_ok: bool
    errors: list[str]
    latency_ms: int


_SYSTEM_PROMPT = (
    "You are a message dissection engine for a personal AI assistant. "
    "Your job is to convert the user message into a compact JSON map that downstream "
    "code can route and plan from. Output MUST be valid JSON only. No markdown. No explanations."
)

_USER_PROMPT_TEMPLATE = """Dissect the message into these components:

Return a JSON object with exactly these keys:

"language": string (e.g., "en", "es", "mixed")

"social": object with:

"is_greeting": boolean

"is_farewell": boolean

"is_thanks": boolean

"text": string | null

"actions": array of objects, each with:

"verb": string

"object": string | null

"target": string | null

"details": string | null

"priority": "low" | "normal" | "high"

"entities": array of strings

"constraints": object with:

"times": array of strings

"numbers": array of strings

"locations": array of strings

"questions": array of strings

"commands": array of strings

"raw_intent_hint": string (one of: "social_only", "single_action", "multi_action", "question_only", "command_only", "mixed", "other")

"confidence": "low" | "medium" | "high"

Rules:

Do NOT choose intents from any catalog. Just dissect structure.

If the message contains multiple imperatives, create multiple "actions".

Keep strings short. Prefer null over empty strings for missing fields.

If unsure, set confidence lower but still output valid JSON.

Message:
<<<
{MESSAGE_TEXT}
"""


def dissect_message(
    text: str,
    *,
    llm_client: object | None,
    timeout_s: float = 30.0,
) -> MessageMapResult:
    _ = timeout_s
    started = time.perf_counter()
    if llm_client is None:
        fallback = _fallback_message_map(text)
        return MessageMapResult(
            raw_json={},
            message_map=fallback,
            parse_ok=False,
            errors=["llm_missing"],
            latency_ms=_latency_ms(started),
        )
    raw_content = ""
    errors: list[str] = []
    payload: dict[str, Any] = {}
    try:
        raw_content = str(
            llm_client.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=_USER_PROMPT_TEMPLATE.replace("{MESSAGE_TEXT}", text),
            )
        )
    except Exception as exc:
        errors.append(f"llm_error:{exc}")
        fallback = _fallback_message_map(text)
        return MessageMapResult(
            raw_json={},
            message_map=fallback,
            parse_ok=False,
            errors=errors,
            latency_ms=_latency_ms(started),
        )

    parsed, parse_errors = _parse_json_salvage(raw_content)
    errors.extend(parse_errors)
    if parsed is None:
        fallback = _fallback_message_map(text)
        return MessageMapResult(
            raw_json={},
            message_map=fallback,
            parse_ok=False,
            errors=errors,
            latency_ms=_latency_ms(started),
        )
    payload = parsed
    mapped = _coerce_message_map(payload, text=text)
    return MessageMapResult(
        raw_json=payload,
        message_map=mapped,
        parse_ok=True,
        errors=errors,
        latency_ms=_latency_ms(started),
    )


def _parse_json_salvage(raw: str) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed, errors
        errors.append("strict_parse_not_object")
    except json.JSONDecodeError as exc:
        errors.append(f"strict_parse_failed:{exc}")
    salvaged = _extract_first_json_object(candidate)
    if salvaged is None:
        errors.append("salvage_no_object_block")
        return None, errors
    sanitized = re.sub(r",\s*([}\]])", r"\1", salvaged)
    try:
        parsed = json.loads(sanitized)
    except json.JSONDecodeError as exc:
        errors.append(f"salvage_parse_failed:{exc}")
        return None, errors
    if not isinstance(parsed, dict):
        errors.append("salvage_parse_not_object")
        return None, errors
    return parsed, errors


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _coerce_message_map(payload: dict[str, Any], *, text: str) -> MessageMap:
    social_raw = payload.get("social") if isinstance(payload.get("social"), dict) else {}
    constraints_raw = (
        payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {}
    )
    actions_raw = payload.get("actions") if isinstance(payload.get("actions"), list) else []

    actions: list[ActionFragment] = []
    for item in actions_raw:
        if not isinstance(item, dict):
            continue
        priority = str(item.get("priority") or "normal").lower()
        if priority not in {"low", "normal", "high"}:
            priority = "normal"
        actions.append(
            ActionFragment(
                verb=str(item.get("verb") or "").strip(),
                object=_none_if_empty(item.get("object")),
                target=_none_if_empty(item.get("target")),
                details=_none_if_empty(item.get("details")),
                priority=priority,  # type: ignore[arg-type]
            )
        )

    raw_hint = str(payload.get("raw_intent_hint") or "other")
    if len(actions) >= 2:
        raw_hint = "multi_action"
    if raw_hint not in {
        "social_only",
        "single_action",
        "multi_action",
        "question_only",
        "command_only",
        "mixed",
        "other",
    }:
        raw_hint = "other"

    confidence = str(payload.get("confidence") or "low").lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"

    return MessageMap(
        language=str(payload.get("language") or "unknown"),
        social=SocialFragment(
            is_greeting=bool(social_raw.get("is_greeting", False)),
            is_farewell=bool(social_raw.get("is_farewell", False)),
            is_thanks=bool(social_raw.get("is_thanks", False)),
            text=_none_if_empty(social_raw.get("text")),
        ),
        actions=actions,
        entities=_to_string_list(payload.get("entities")),
        constraints=Constraints(
            times=_to_string_list(constraints_raw.get("times")),
            numbers=_to_string_list(constraints_raw.get("numbers")),
            locations=_to_string_list(constraints_raw.get("locations")),
        ),
        questions=_to_string_list(payload.get("questions")),
        commands=_to_string_list(payload.get("commands")),
        raw_intent_hint=raw_hint,  # type: ignore[arg-type]
        confidence=confidence,  # type: ignore[arg-type]
    )


def _fallback_message_map(text: str) -> MessageMap:
    normalized = " ".join(text.strip().lower().split())
    greeting_roots = (
        "hi",
        "hello",
        "hey",
        "hola",
        "buenos dias",
        "buenas tardes",
        "good morning",
        "good evening",
    )
    is_greeting = any(root in normalized for root in greeting_roots)
    commands: list[str] = []
    actions: list[ActionFragment] = []
    questions: list[str] = []
    times: list[str] = []

    if normalized.startswith("/approve") or normalized.startswith("approve"):
        commands.append("approve")
    if normalized.startswith("/deny") or normalized.startswith("deny"):
        commands.append("deny")
    if "remind" in normalized or "recu" in normalized:
        actions.append(ActionFragment(verb="remind", details=text, priority="normal"))
    if "?" in text or normalized.startswith(("what", "who", "how", "que", "qué", "como", "cómo")):
        questions.append(text)
    time_match = re.search(r"\b(\d+\s*(?:min|mins|minutes|hora|horas|h))\b", normalized)
    if time_match:
        times.append(time_match.group(1))

    return MessageMap(
        language="unknown",
        social=SocialFragment(is_greeting=is_greeting, text=text if is_greeting else None),
        actions=actions,
        entities=[],
        constraints=Constraints(times=times),
        questions=questions,
        commands=commands,
        raw_intent_hint="other",
        confidence="low",
    )


def _to_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    for item in value:
        item_str = str(item).strip()
        if item_str:
            output.append(item_str)
    return output


def _none_if_empty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _latency_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
