from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from alphonse.agent.cognition.slots.resolvers import ResolverRegistry, ParseResult
from alphonse.agent.cognition.slots.slot_fsm import (
    SlotMachine,
    apply_input,
    create_machine,
    serialize_machine,
)
from alphonse.agent.cognition.slots.utterance_guard import (
    is_core_conversational_utterance,
    is_resume_utterance,
)
from alphonse.agent.cognition.plans import CortexPlan, PlanType
from alphonse.agent.core.settings_store import get_timezone


@dataclass(frozen=True)
class SlotFillResult:
    intent_name: str
    slots: dict[str, Any]
    missing_slots: list[str] = field(default_factory=list)
    response_key: str | None = None
    response_vars: dict[str, Any] | None = None
    slot_machine: dict[str, Any] | None = None
    plans: list[dict[str, Any]] = field(default_factory=list)
    needs_clarification: bool = False
    abort_flow: bool = False


def fill_slots(
    intent: Any,
    *,
    text: str,
    slot_guesses: dict[str, Any],
    registry: ResolverRegistry,
    context: dict[str, Any],
    existing_slots: dict[str, Any] | None = None,
    machine: SlotMachine | None = None,
    max_attempts: int = 3,
) -> SlotFillResult:
    slots: dict[str, Any] = dict(existing_slots or {})
    missing: list[str] = []

    if machine:
        if machine.expires_at and machine.expires_at < datetime.now(timezone.utc).isoformat():
            return SlotFillResult(
                intent_name=intent.intent_name,
                slots=slots,
                response_key="clarify.slot_abort",
                needs_clarification=True,
                abort_flow=True,
            )
        if machine.paused_at and not is_resume_utterance(text):
            return SlotFillResult(
                intent_name=intent.intent_name,
                slots=slots,
                response_key=None,
                slot_machine=serialize_machine(machine),
                needs_clarification=False,
            )
        if machine.paused_at and is_resume_utterance(text):
            machine.paused_at = None
            prompt_key = _prompt_for_slot(intent, machine.slot_name) or "clarify.intent"
            return SlotFillResult(
                intent_name=intent.intent_name,
                slots=slots,
                response_key=prompt_key,
                slot_machine=serialize_machine(machine),
                needs_clarification=True,
            )
        machine, parsed = apply_input(machine, text, registry)
        if parsed.ok:
            slots[machine.slot_name] = parsed.value
        else:
            if machine.attempts >= max_attempts:
                return SlotFillResult(
                    intent_name=intent.intent_name,
                    slots=slots,
                    missing_slots=[machine.slot_name],
                    response_key="clarify.slot_abort",
                    slot_machine=None,
                    needs_clarification=True,
                    abort_flow=True,
                )
            prompt_key = _prompt_for_slot(intent, machine.slot_name)
            return SlotFillResult(
                intent_name=intent.intent_name,
                slots=slots,
                missing_slots=[machine.slot_name],
                response_key=prompt_key or "clarify.intent",
                slot_machine=serialize_machine(machine),
                needs_clarification=True,
            )

    for spec in intent.required_slots + intent.optional_slots:
        guess = slot_guesses.get(spec.name)
        if guess is None:
            continue
        parsed = _parse_guess(spec, guess, registry, context)
        if parsed.ok:
            slots[spec.name] = parsed.value

    if _geo_stub_present(slots):
        return SlotFillResult(
            intent_name=intent.intent_name,
            slots=slots,
            response_key="clarify.trigger_geo.stub_setup",
            needs_clarification=True,
        )

    missing = _missing_required(intent, slots)
    if missing:
        next_slot = _next_missing_slot(intent, missing)
        if next_slot:
            machine = create_machine(
                next_slot.name,
                next_slot.type,
                context,
            )
            return SlotFillResult(
                intent_name=intent.intent_name,
                slots=slots,
                missing_slots=missing,
                response_key=next_slot.prompt_key,
                slot_machine=serialize_machine(machine),
                needs_clarification=True,
            )

    plans = _build_plans(intent, slots, context)
    return SlotFillResult(
        intent_name=intent.intent_name,
        slots=slots,
        plans=plans,
    )


def _parse_guess(
    spec: Any,
    guess: Any,
    registry: ResolverRegistry,
    context: dict[str, Any],
) -> ParseResult:
    if spec.semantic_text:
        text_value = str(guess or "").strip()
        if spec.min_length and len(text_value) < spec.min_length:
            return ParseResult(ok=False, error="too_short")
        if spec.reject_if_core_conversational and is_core_conversational_utterance(text_value):
            return ParseResult(ok=False, error="core_conversational")
    resolver = registry.get(spec.type)
    if not resolver:
        return ParseResult(ok=False, error="missing_resolver")
    return resolver.parse(str(guess), context)


def _missing_required(intent: Any, slots: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for spec in intent.required_slots:
        if spec.name not in slots:
            missing.append(spec.name)
    return missing


def _next_missing_slot(intent: Any, missing: list[str]) -> Any | None:
    for spec in intent.required_slots:
        if spec.name in missing and spec.critical:
            return spec
    if missing:
        for spec in intent.required_slots:
            if spec.name in missing:
                return spec
    return None


def _prompt_for_slot(intent: Any, slot_name: str) -> str | None:
    for spec in intent.required_slots + intent.optional_slots:
        if spec.name == slot_name:
            return spec.prompt_key
    return None


def _geo_stub_present(slots: dict[str, Any]) -> bool:
    value = slots.get("trigger_geo")
    if isinstance(value, dict) and value.get("status") == "stub_needs_location_setup":
        return True
    return False


def _build_plans(
    intent: Any, slots: dict[str, Any], context: dict[str, Any]
) -> list[dict[str, Any]]:
    if intent.handler == "timed_signals.create":
        reminder_text = slots.get("reminder_text")
        trigger = slots.get("trigger_time")
        trigger_at = None
        if isinstance(trigger, dict) and trigger.get("kind") == "trigger_at":
            trigger_at = trigger.get("trigger_at")
        if not reminder_text or not trigger_at:
            return []
        plan = CortexPlan(
            plan_type=PlanType.SCHEDULE_TIMED_SIGNAL,
            target=str(context.get("channel_target") or context.get("chat_id") or ""),
            channels=None,
            payload={
                "signal_type": "reminder",
                "trigger_at": str(trigger_at),
                "timezone": str(context.get("timezone") or get_timezone()),
                "reminder_text": str(reminder_text),
                "reminder_text_raw": str(reminder_text),
                "origin": str(context.get("channel_type") or "system"),
                "chat_id": str(context.get("channel_target") or context.get("chat_id") or ""),
                "origin_channel": str(context.get("channel_type") or "system"),
                "locale_hint": context.get("locale"),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "correlation_id": str(context.get("correlation_id") or context.get("chat_id") or ""),
            },
        )
        return [plan.model_dump()]
    if intent.handler == "timed_signals.list":
        plan = CortexPlan(
            plan_type=PlanType.QUERY_STATUS,
            target=str(context.get("channel_target") or context.get("chat_id") or ""),
            channels=[str(context.get("channel_type") or "telegram")],
            payload={
                "include": ["timed_signals"],
                "limit": 10,
                "locale": context.get("locale"),
            },
        )
        return [plan.model_dump()]
    return []
