from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from alphonse.agent.cognition.providers.ollama import OllamaClient
from alphonse.agent.cognition.skills.command_plans import (
    Actor,
    ActorChannel,
    CommandPlan,
    CreateReminderPayload,
    CreateReminderPlan,
    GreetingPayload,
    GreetingPlan,
    IntentEvidence,
    ReminderDelivery,
    ReminderMessage,
    ReminderSchedule,
    SendMessagePayload,
    SendMessagePlan,
    TargetRef,
    UnknownPayload,
    UnknownPlan,
    parse_command_plan,
    PersonRef,
)
from alphonse.agent.cognition.skills.plan_registry import list_enabled_plan_specs


@dataclass(frozen=True)
class CommandInterpreterContext:
    actor_person_id: str | None
    channel_type: str
    channel_target: str
    source: str
    correlation_id: str
    timezone: str
    language: str = "es"


class CommandInterpreterSkill:
    def __init__(self, llm_client: OllamaClient) -> None:
        self._llm_client = llm_client

    def interpret(self, text: str, context: CommandInterpreterContext) -> CommandPlan:
        candidate = self._interpret_with_llm(text, context)
        if candidate is None:
            if _is_greeting(text):
                return self._greeting_plan(text, context)
            return self._unknown_plan(text, context, reason="llm_unavailable")

        plan_kind = str(candidate.get("plan_kind") or candidate.get("kind") or "unknown").strip()
        plan_version = int(candidate.get("plan_version") or 1)
        payload = candidate.get("payload") if isinstance(candidate.get("payload"), dict) else {}
        intent_confidence = _as_float(candidate.get("intent_confidence"), default=0.5)
        requires_confirmation = bool(candidate.get("requires_confirmation", False))
        questions = candidate.get("questions") if isinstance(candidate.get("questions"), list) else []

        if _is_greeting(text):
            return self._greeting_plan(text, context)

        if not _is_enabled_plan(plan_kind, plan_version):
            return self._unknown_plan(text, context, reason="unsupported_kind")

        evidence = _build_intent_evidence(text)
        base = _build_base_plan(
            plan_kind=plan_kind,
            plan_version=plan_version,
            context=context,
            intent_confidence=intent_confidence,
            requires_confirmation=requires_confirmation,
            questions=questions,
            evidence=evidence,
            original_text=text,
        )

        plan_data = _merge_payload(plan_kind, payload, text, context, evidence, base)
        try:
            return parse_command_plan(plan_data)
        except Exception:
            return self._unknown_plan(text, context, reason="invalid_plan")

    def _interpret_with_llm(self, text: str, context: CommandInterpreterContext) -> dict[str, Any] | None:
        prompt = self._build_prompt(context)
        try:
            content = self._llm_client.complete(system_prompt=prompt, user_prompt=text)
        except Exception:
            return None
        return _parse_json(str(content))

    def _build_prompt(self, context: CommandInterpreterContext) -> str:
        specs = list_enabled_plan_specs()
        lines = ["Available plan kinds:"]
        for spec in specs:
            example = spec.example or "{}"
            lines.append(f"- {spec.plan_kind}@{spec.plan_version}: example {example}")
        return (
            "You are Alphonse. Output strict JSON only. "
            "Return one plan with fields: plan_kind, plan_version, payload, intent_confidence, "
            "requires_confirmation, questions. No prose.\n"
            + "\n".join(lines)
        )

    def _unknown_plan(self, text: str, context: CommandInterpreterContext, reason: str) -> UnknownPlan:
        base = _build_base_plan(
            plan_kind="unknown",
            plan_version=1,
            context=context,
            intent_confidence=0.2,
            requires_confirmation=False,
            questions=["¿Puedes aclarar qué necesitas?"],
            evidence=_build_intent_evidence(text),
            original_text=text,
        )
        payload = UnknownPayload(user_text=text, reason=reason, suggestions=[])
        data = {**base, "payload": payload.model_dump()}
        return UnknownPlan.model_validate(data)

    def _greeting_plan(self, text: str, context: CommandInterpreterContext) -> GreetingPlan:
        base = _build_base_plan(
            plan_kind="greeting",
            plan_version=1,
            context=context,
            intent_confidence=0.3,
            requires_confirmation=False,
            questions=[],
            evidence=_build_intent_evidence(text),
            original_text=text,
        )
        payload = GreetingPayload(language=context.language, text=None)
        data = {**base, "payload": payload.model_dump()}
        return GreetingPlan.model_validate(data)


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


def _build_base_plan(
    *,
    plan_kind: str,
    plan_version: int,
    context: CommandInterpreterContext,
    intent_confidence: float,
    requires_confirmation: bool,
    questions: list[str],
    evidence: dict[str, Any],
    original_text: str,
) -> dict[str, Any]:
    actor = Actor(
        person_id=context.actor_person_id,
        channel=ActorChannel(type=context.channel_type, target=context.channel_target),
    )
    return {
        "plan_kind": plan_kind,
        "plan_version": plan_version,
        "plan_id": str(uuid.uuid4()),
        "correlation_id": context.correlation_id,
        "created_at": datetime.utcnow().isoformat(),
        "source": context.source,
        "actor": actor.model_dump(),
        "intent_confidence": intent_confidence,
        "requires_confirmation": requires_confirmation,
        "questions": questions,
        "intent_evidence": evidence,
        "payload": {},
        "metadata": None,
        "original_text": original_text,
    }


def _merge_payload(
    plan_kind: str,
    payload: dict[str, Any],
    text: str,
    context: CommandInterpreterContext,
    evidence: dict[str, Any],
    base: dict[str, Any],
) -> dict[str, Any]:
    if plan_kind == "create_reminder":
        schedule = payload.get("schedule") if isinstance(payload.get("schedule"), dict) else {}
        message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
        delivery = payload.get("delivery") if isinstance(payload.get("delivery"), dict) else {}
        target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
        person_ref = target.get("person_ref") if isinstance(target.get("person_ref"), dict) else {}
        data = CreateReminderPayload(
            target=TargetRef(
                person_ref=PersonRef(
                    kind=str(person_ref.get("kind") or "person_id"),
                    id=person_ref.get("id"),
                    name=person_ref.get("name"),
                )
            ),
            schedule=ReminderSchedule(
                timezone=schedule.get("timezone") or context.timezone,
                trigger_at=schedule.get("trigger_at"),
                rrule=schedule.get("rrule"),
                time_of_day=schedule.get("time_of_day"),
            ),
            message=ReminderMessage(
                language=message.get("language") or context.language,
                text=message.get("text") or text,
            ),
            delivery=ReminderDelivery(
                channel_type=delivery.get("channel_type") or context.channel_type,
                priority=delivery.get("priority"),
            )
            if delivery
            else None,
            idempotency_key=payload.get("idempotency_key"),
        )
        return {**base, "payload": data.model_dump(), "intent_evidence": evidence}
    if plan_kind == "send_message":
        target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
        person_ref = target.get("person_ref") if isinstance(target.get("person_ref"), dict) else {}
        message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
        delivery = payload.get("delivery") if isinstance(payload.get("delivery"), dict) else {}
        data = SendMessagePayload(
            target=TargetRef(
                person_ref=PersonRef(
                    kind=str(person_ref.get("kind") or "person_id"),
                    id=person_ref.get("id"),
                    name=person_ref.get("name"),
                )
            ),
            message=ReminderMessage(
                language=message.get("language") or context.language,
                text=message.get("text") or text,
            ),
            delivery=ReminderDelivery(
                channel_type=delivery.get("channel_type") or context.channel_type,
                priority=delivery.get("priority"),
            )
            if delivery
            else None,
        )
        return {**base, "payload": data.model_dump(), "intent_evidence": evidence}
    if plan_kind == "greeting":
        data = GreetingPayload(language=context.language, text=None)
        return {**base, "payload": data.model_dump(), "intent_evidence": evidence}
    if plan_kind == "unknown":
        data = UnknownPayload(user_text=text, reason="unspecified", suggestions=[])
        return {**base, "payload": data.model_dump(), "intent_evidence": evidence}
    return {**base, "plan_kind": "unknown"}


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
    matches: list[str] = []
    for token in (
        "remind",
        "reminder",
        "recordar",
        "recuerda",
        "recuerdale",
        "recuérdale",
        "recuerdame",
        "recuérdame",
        "recordatorio",
    ):
        if token in lowered:
            matches.append(token)
    return matches


def _extract_time_hints(text: str) -> list[str]:
    lowered = text.lower()
    hints: list[str] = []
    for token in ("today", "tomorrow", "tonight", "hoy", "manana", "mañana", "cada", "daily"):
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


def _as_float(value: object | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_greeting(text: str) -> bool:
    lowered = text.lower().strip()
    return any(
        lowered.startswith(token)
        for token in ("hola", "hello", "hi", "buenos", "buenas", "hey")
    )


def _is_enabled_plan(plan_kind: str, plan_version: int) -> bool:
    specs = list_enabled_plan_specs()
    return any(spec.plan_kind == plan_kind and spec.plan_version == plan_version for spec in specs)
