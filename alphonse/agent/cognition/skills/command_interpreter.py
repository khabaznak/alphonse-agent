from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any

from alphonse.agent.cognition.providers.ollama import OllamaClient
from alphonse.agent.cognition.skills.command_plans import (
    CommandPlan,
    CreateReminderPlan,
    GreetingPlan,
    ReminderDelivery,
    ReminderMessage,
    ReminderSchedule,
    SendMessagePlan,
    UnknownPlan,
    parse_plan,
)


@dataclass(frozen=True)
class CommandInterpreterContext:
    created_by: str | None
    source: str
    correlation_id: str
    timezone: str
    language: str = "es"


class CommandInterpreterSkill:
    def __init__(self, llm_client: OllamaClient) -> None:
        self._llm_client = llm_client

    def interpret(self, text: str, context: CommandInterpreterContext) -> CommandPlan | None:
        candidate = self._interpret_with_llm(text, context)
        if candidate is None:
            if _is_greeting(text):
                return self._greeting_plan(text, context)
            return None
        normalized = self._merge_base_fields(candidate, text, context)
        try:
            plan = parse_plan(normalized)
        except Exception:
            return self._unknown_plan(text, context, question=None)
        if isinstance(plan, CreateReminderPlan) and not _should_accept_reminder(plan):
            return self._unknown_plan(text, context, question=None)
        if (isinstance(plan, GreetingPlan) or _is_greeting(text)) and not _should_accept_reminder(plan):
            return self._greeting_plan(text, context)
        return plan

    def _interpret_with_llm(
        self, text: str, context: CommandInterpreterContext
    ) -> dict[str, Any] | None:
        prompt = self._build_prompt(text, context)
        try:
            content = self._llm_client.complete(system_prompt=prompt, user_prompt=text)
        except Exception:
            return None
        return _parse_json(str(content))

    def _merge_base_fields(
        self, raw: dict[str, Any], text: str, context: CommandInterpreterContext
    ) -> dict[str, Any]:
        kind = raw.get("kind")
        evidence = _build_intent_evidence(text)
        base = {
            "plan_id": str(uuid.uuid4()),
            "version": 1,
            "created_at": datetime.utcnow().isoformat(),
            "created_by": context.created_by,
            "source": context.source,
            "correlation_id": context.correlation_id,
            "original_text": text,
            "kind": kind,
        }
        if kind == "create_reminder":
            schedule = raw.get("schedule") if isinstance(raw.get("schedule"), dict) else {}
            message = raw.get("message") if isinstance(raw.get("message"), dict) else {}
            delivery = raw.get("delivery") if isinstance(raw.get("delivery"), dict) else {}
            base.update(
                {
                    "target_person_id": raw.get("target_person_id"),
                    "schedule": {
                        "timezone": schedule.get("timezone") or context.timezone,
                        "trigger_at": schedule.get("trigger_at"),
                        "rrule": schedule.get("rrule"),
                    },
                    "message": {
                        "language": message.get("language") or context.language,
                        "text": message.get("text") or text,
                    },
                    "delivery": {
                        "preferred_channel_type": delivery.get("preferred_channel_type")
                        or context.source,
                        "priority": delivery.get("priority") or "normal",
                    },
                    "intent_evidence": evidence,
                }
            )
            return base
        if kind == "send_message":
            message = raw.get("message") if isinstance(raw.get("message"), dict) else {}
            base.update(
                {
                    "target_person_id": raw.get("target_person_id"),
                    "message": {
                        "language": message.get("language") or context.language,
                        "text": message.get("text") or text,
                    },
                    "channel_type": raw.get("channel_type") or context.source,
                }
            )
            return base
        if kind == "greeting":
            base.update({"language": context.language})
            return base
        if kind == "unknown":
            base.update({"question": raw.get("question")})
            return base
        return {
            **base,
            "kind": "unknown",
            "question": None,
        }

    def _build_prompt(self, text: str, context: CommandInterpreterContext) -> str:
        return (
            "You are Alphonse. Output strict JSON only. "
            "Choose exactly one kind: create_reminder, send_message, unknown. "
            "If unknown, include a short clarifying question. "
            "Schema examples:\n"
            "{\"kind\":\"create_reminder\",\"target_person_id\":null,"
            "\"schedule\":{\"timezone\":\"UTC\",\"trigger_at\":\"2026-01-31T08:00:00Z\",\"rrule\":null},"
            "\"message\":{\"language\":\"es\",\"text\":\"take medicine\"},"
            "\"delivery\":{\"preferred_channel_type\":\"telegram\",\"priority\":\"normal\"}}\n"
            "{\"kind\":\"send_message\",\"target_person_id\":null,"
            "\"message\":{\"language\":\"es\",\"text\":\"hello\"},\"channel_type\":\"telegram\"}\n"
            "{\"kind\":\"unknown\",\"question\":\"Que deseas que haga con ese mensaje?\"}\n"
            f"Timezone: {context.timezone}. Language: {context.language}."
        )

    def _unknown_plan(self, text: str, context: CommandInterpreterContext, question: str | None) -> UnknownPlan:
        return UnknownPlan(
            plan_id=str(uuid.uuid4()),
            kind="unknown",
            version=1,
            created_at=datetime.utcnow().isoformat(),
            created_by=context.created_by,
            source=context.source,
            correlation_id=context.correlation_id,
            original_text=text,
            question=question,
        )

    def _greeting_plan(self, text: str, context: CommandInterpreterContext) -> GreetingPlan:
        return GreetingPlan(
            plan_id=str(uuid.uuid4()),
            kind="greeting",
            version=1,
            created_at=datetime.utcnow().isoformat(),
            created_by=context.created_by,
            source=context.source,
            correlation_id=context.correlation_id,
            original_text=text,
            language=context.language,
        )


def build_default_command_interpreter() -> CommandInterpreterSkill:
    return CommandInterpreterSkill(_build_ollama_client())


def _build_ollama_client() -> OllamaClient:
    from alphonse.agent.cognition.skills.interpretation.skills import build_ollama_client

    return build_ollama_client()


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
    if isinstance(parsed, dict):
        return parsed
    return None


def _should_accept_reminder(plan: CreateReminderPlan | GreetingPlan) -> bool:
    if not isinstance(plan, CreateReminderPlan):
        return False
    evidence = plan.intent_evidence
    if evidence.score >= 0.6:
        return True
    if evidence.reminder_verbs or evidence.time_hints:
        return True
    return False


def _build_intent_evidence(text: str) -> dict[str, Any]:
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


def _extract_reminder_verbs(text: str) -> list[str]:
    lowered = text.lower()
    matches = []
    for token in ("remind", "reminder", "recordar", "recuerda", "recuerdame", "recuérdame", "recordatorio"):
        if token in lowered:
            matches.append(token)
    return matches


def _extract_time_hints(text: str) -> list[str]:
    lowered = text.lower()
    hints = []
    for token in ("today", "tomorrow", "tonight", "hoy", "manana", "mañana", "cada", "daily"):
        if token in lowered:
            hints.append(token)
    for match in re.finditer(r"\b\d{1,2}(:\d{2})?\b", lowered):
        hints.append(match.group(0))
    return hints


def _extract_quoted_spans(text: str) -> list[str]:
    matches = re.findall(r"“([^”]+)”|\"([^\"]+)\"|'([^']+)'", text)
    spans = []
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


def _is_greeting(text: str) -> bool:
    lowered = text.lower().strip()
    return any(
        lowered.startswith(token)
        for token in ("hola", "hello", "hi", "buenos", "buenas", "hey")
    )
