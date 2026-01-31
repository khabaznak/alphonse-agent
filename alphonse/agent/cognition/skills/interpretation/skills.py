from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any

from dateutil.parser import isoparse, parse as parse_datetime
from dateutil.rrule import rrulestr

from alphonse.agent.runtime import get_runtime
from alphonse.agent.cognition.providers.ollama import OllamaClient
from alphonse.agent.cognition.skills.interpretation.models import MessageEvent, RoutingDecision
from alphonse.agent.cognition.skills.interpretation.registry import SkillRegistry
from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
from alphonse.agent.core.settings_store import get_timezone

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillExecutor:
    registry: SkillRegistry
    llm_client: OllamaClient
    clarification_prompt: str = "Could you clarify what you need?"

    def respond(self, decision: RoutingDecision, message: MessageEvent) -> str:
        if decision.needs_clarification:
            return decision.clarifying_question or self.clarification_prompt

        if decision.skill == "system.status":
            return self._status_response()
        if decision.skill == "system.joke":
            return self._joke_response()
        if decision.skill == "system.help":
            return self._format_help()
        if decision.skill == "schedule.timed_signal":
            return self._schedule_timed_signal(decision, message)
        if decision.skill == "conversation.echo":
            return f"Echo: {message.text}"

        return f"Echo: {message.text}"

    def _status_response(self) -> str:
        runtime = get_runtime().snapshot()
        uptime = max(0, int(float(runtime.get("uptime_seconds", 0))))
        last_signal = runtime.get("last_signal", {}) or {}
        last_signal_type = last_signal.get("type")
        last_signal_at = last_signal.get("ts")
        system_prompt = (
            "You are Alphonse, a calm and restrained domestic presence.\n"
            "Summarize the current status if requested.\n"
            "Do not suggest actions.\n"
            "Do not speculate.\n"
            "Keep it under 2 sentences."
        )
        prompt = (
            f"{system_prompt}\n\n"
            "Runtime snapshot:\n"
            f"- Uptime: {uptime}s\n"
            f"- Last signal type: {last_signal_type}\n"
            f"- Last signal at: {last_signal_at}\n"
        )
        try:
            content = self.llm_client.complete(system_prompt=system_prompt, user_prompt=prompt)
            if content:
                return str(content).strip()
        except Exception as exc:
            logger.warning("Ollama status call failed: %s", exc)

        return (
            "Alphonse status: "
            f"uptime {uptime}s, "
            f"last signal {last_signal_type} at {last_signal_at}."
        )

    def _joke_response(self) -> str:
        system_prompt = (
            "You are Alphonse, a calm and restrained domestic presence.\n"
            "Tell a short, gentle joke in one sentence.\n"
            "Avoid sarcasm or insults.\n"
            "Keep it under 2 sentences."
        )
        prompt = "Please provide a short, kind joke."
        try:
            content = self.llm_client.complete(system_prompt=system_prompt, user_prompt=prompt)
            if content:
                return str(content).strip()
        except Exception as exc:
            logger.warning("Ollama joke call failed: %s", exc)

        return "Here is a gentle joke: Why did the scarecrow get promoted? Because he was outstanding in his field."

    def _format_help(self) -> str:
        lines = ["Available commands:"]
        for skill in sorted(self.registry.list_skills(), key=lambda item: item.key):
            if skill.key == "conversation.echo":
                continue
            aliases = [alias for alias in skill.aliases if alias and not alias.startswith("/")]
            alias_text = ", ".join(sorted(set(aliases)))
            if alias_text:
                lines.append(f"- {skill.key}: {alias_text}")
            else:
                lines.append(f"- {skill.key}")
        return "\n".join(lines)

    def _schedule_timed_signal(self, decision: RoutingDecision, message: MessageEvent) -> str:
        try:
            record = _build_timed_signal_record(decision, message)
        except ValueError as exc:
            return f"Unable to schedule: {exc}"

        try:
            _insert_timed_signal(record)
        except Exception as exc:
            logger.warning("Failed to insert timed signal: %s", exc)
            return "Unable to schedule the timed signal right now."

        when_text = record["next_trigger_at"] or record["trigger_at"]
        return f"Scheduled {record['signal_type']} for {when_text}."


def build_ollama_client() -> OllamaClient:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("LOCAL_LLM_MODEL", "mistral:7b-instruct")
    timeout_seconds = _parse_float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS"), default=240.0)
    return OllamaClient(
        base_url=base_url,
        model=model,
        timeout=timeout_seconds,
    )


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _build_timed_signal_record(decision: RoutingDecision, message: MessageEvent) -> dict[str, Any]:
    args = decision.args
    signal_type = _normalize_signal_type(args.get("signal_type"))
    payload = _normalize_payload(args.get("payload"))
    user_name = _as_optional_str(payload.get("user_name")) if isinstance(payload, dict) else None
    if not user_name:
        user_name = _as_optional_str(args.get("user_name"))
        if user_name and isinstance(payload, dict):
            payload["user_name"] = user_name
    if not user_name:
        metadata_name = _as_optional_str(message.metadata.get("user_name"))
        if metadata_name and isinstance(payload, dict):
            payload["user_name"] = metadata_name

    if isinstance(payload, dict) and not payload.get("message"):
        extracted = _extract_message_text(message.text)
        if extracted:
            payload["message"] = extracted
    if isinstance(payload, dict) and not payload.get("reminder_text_raw"):
        payload["reminder_text_raw"] = payload.get("message") or _extract_message_text(message.text) or message.text
    if isinstance(payload, dict) and not payload.get("chat_id"):
        payload["chat_id"] = _as_optional_str(args.get("target")) or _as_optional_str(message.metadata.get("target"))
    if isinstance(payload, dict) and not payload.get("origin_channel"):
        payload["origin_channel"] = _as_optional_str(args.get("origin")) or message.channel
    if isinstance(payload, dict) and not payload.get("created_at"):
        payload["created_at"] = datetime.now(timezone.utc).isoformat()

    tz_name = _normalize_timezone(args.get("timezone"))
    trigger_at_raw = args.get("trigger_at")
    trigger_at = _resolve_trigger_at(trigger_at_raw, message.text, tz_name)
    trigger_at_utc = trigger_at.astimezone(timezone.utc)

    rrule_raw = args.get("rrule")
    rrule_value = str(rrule_raw).strip() if isinstance(rrule_raw, str) and rrule_raw.strip() else None
    next_trigger_at = None
    if rrule_value:
        next_trigger_at = _next_occurrence(rrule_value, trigger_at, tz_name)
        if next_trigger_at is None:
            raise ValueError("rrule has no future occurrences")

    return {
        "id": str(uuid.uuid4()),
        "trigger_at": trigger_at_utc.isoformat(),
        "next_trigger_at": next_trigger_at.isoformat() if next_trigger_at else None,
        "rrule": rrule_value,
        "timezone": tz_name,
        "status": "pending",
        "fired_at": None,
        "attempt_count": 0,
        "attempts": 0,
        "last_error": None,
        "signal_type": signal_type,
        "payload": payload,
        "target": _as_optional_str(args.get("target")) or _as_optional_str(message.metadata.get("target")),
        "origin": _as_optional_str(args.get("origin")) or message.channel,
        "correlation_id": _as_optional_str(args.get("correlation_id")),
    }


def _insert_timed_signal(record: dict[str, Any]) -> None:
    db_path = resolve_nervous_system_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO timed_signals
              (id, trigger_at, next_trigger_at, rrule, timezone, status, fired_at, attempt_count, attempts, last_error, signal_type, payload, target, origin, correlation_id)
            VALUES
              (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record["trigger_at"],
                record["next_trigger_at"],
                record["rrule"],
                record["timezone"],
                record["status"],
                record["fired_at"],
                record["attempt_count"],
                record.get("attempts", record["attempt_count"]),
                record.get("last_error"),
                record["signal_type"],
                json.dumps(record["payload"]),
                record["target"],
                record["origin"],
                record["correlation_id"],
            ),
        )
        conn.commit()


def _normalize_signal_type(value: object | None) -> str:
    allowed = {"reminder", "execute", "routine", "custom"}
    if not value:
        return "custom"
    signal_type = str(value).strip().lower()
    if signal_type in allowed:
        return signal_type
    tokens = _tokenize_signal_type(signal_type)
    for token in tokens:
        if token in allowed:
            return token
    raise ValueError("signal_type must be reminder, execute, routine, or custom")


def _normalize_payload(value: object | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            return {"message": raw}
        if isinstance(parsed, dict):
            return parsed
        return {"message": raw}
    raise ValueError("payload must be an object")


def _tokenize_signal_type(value: str) -> list[str]:
    tokens: list[str] = []
    current = []
    for char in value:
        if char.isalpha():
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _normalize_timezone(value: object | None) -> str:
    if isinstance(value, str) and value.strip():
        tz_name = value.strip()
    else:
        tz_name = get_timezone()
    try:
        ZoneInfo(tz_name)
    except Exception as exc:
        raise ValueError(f"invalid timezone: {tz_name}") from exc
    return tz_name


def _parse_datetime(value: str, tz_name: str) -> datetime:
    tzinfo = ZoneInfo(tz_name)
    try:
        parsed = isoparse(value)
    except ValueError:
        fallback = datetime.now(tz=tzinfo)
        parsed = parse_datetime(value, default=fallback, fuzzy=True)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tzinfo)
    return parsed


def _resolve_trigger_at(value: object | None, text: str, tz_name: str) -> datetime:
    if isinstance(value, str) and value.strip():
        return _parse_datetime(value, tz_name)

    if _has_time_hint(text):
        return _parse_datetime(text, tz_name)

    raise ValueError("trigger_at is required")


def _next_occurrence(rrule_value: str, start: datetime, tz_name: str) -> datetime | None:
    tzinfo = ZoneInfo(tz_name)
    dtstart = start.astimezone(tzinfo)
    rule = rrulestr(rrule_value, dtstart=dtstart)
    candidate = rule.after(datetime.now(tz=tzinfo), inc=True)
    if candidate is None:
        return None
    if candidate.tzinfo is None:
        candidate = candidate.replace(tzinfo=tzinfo)
    return candidate.astimezone(timezone.utc)


def _as_optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _has_time_hint(text: str) -> bool:
    lowered = text.lower()
    if any(token in lowered for token in ("today", "tomorrow", "tonight")):
        return True
    return bool(re.search(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", lowered))


def _extract_message_text(text: str) -> str | None:
    lowered = text.lower()
    if "to" in lowered:
        parts = text.split("to", 1)
        if len(parts) == 2:
            candidate = parts[1].strip()
            candidate = re.sub(r"\bat\b\s+.+", "", candidate, flags=re.IGNORECASE).strip()
            return candidate or None
    return None
