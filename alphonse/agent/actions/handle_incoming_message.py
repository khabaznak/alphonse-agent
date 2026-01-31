from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.actions.plan_executors import EXECUTOR_MAP, ExecutorContext
from alphonse.agent.cognition.skills.command_interpreter import (
    CommandInterpreterContext,
    build_default_command_interpreter,
)
from alphonse.agent.cognition.skills.command_plans import (
    CommandPlan,
    CreateReminderPlan,
    GreetingPlan,
    SendMessagePlan,
    UnknownPlan,
    parse_command_plan,
    IntentEvidence,
)
from alphonse.agent.cognition.skills.plan_registry import get_plan_spec
from alphonse.agent.cognition.skills.plan_validation import validate_plan
from alphonse.agent.cognition.skills.plan_interpreter import infer_trigger_at
from alphonse.agent.core.settings_store import get_timezone
from alphonse.agent.identity import store as identity_store
from alphonse.agent.nervous_system.pending_store import create_pending_plan, get_pending_plan, update_pending_status
from alphonse.agent.nervous_system.plan_instances_store import (
    insert_plan_instance,
    update_plan_instance_status,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IncomingContext:
    channel_type: str
    address: str | None
    person_id: str | None
    correlation_id: str
    update_id: str | None


class HandleIncomingMessageAction(Action):
    key = "handle_incoming_message"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        payload = getattr(signal, "payload", {}) if signal else {}
        text = str(payload.get("text", "")).strip()
        correlation_id = getattr(signal, "correlation_id", None) if signal else None
        if not correlation_id and isinstance(payload, dict):
            correlation_id = payload.get("correlation_id")
        correlation_id = str(correlation_id or uuid.uuid4())

        incoming = _build_incoming_context(payload, signal, correlation_id)
        logger.info(
            "HandleIncomingMessageAction start channel=%s person=%s text=%s",
            incoming.channel_type,
            incoming.person_id,
            _snippet(text),
        )
        if not text:
            return _message_result("No te escuché bien. ¿Puedes repetir?", incoming)

        pending = get_pending_plan(incoming.person_id, incoming.channel_type)
        if pending:
            return _continue_pending_plan(pending, text, incoming)

        return _interpret_new_message(text, incoming)


def _build_incoming_context(payload: dict, signal: object | None, correlation_id: str) -> IncomingContext:
    origin = payload.get("origin") or getattr(signal, "source", None) or "system"
    channel_type = str(origin)
    address = _resolve_address(channel_type, payload)
    person_id = _resolve_person_id(payload, channel_type, address)
    update_id = payload.get("update_id") if isinstance(payload, dict) else None
    return IncomingContext(
        channel_type=channel_type,
        address=address,
        person_id=person_id,
        correlation_id=correlation_id,
        update_id=str(update_id) if update_id is not None else None,
    )


def _resolve_address(channel_type: str, payload: dict) -> str | None:
    if channel_type == "telegram":
        chat_id = payload.get("chat_id")
        return str(chat_id) if chat_id is not None else None
    if channel_type == "cli":
        return "cli"
    if channel_type == "api":
        return "api"
    target = payload.get("target")
    return str(target) if target is not None else None


def _resolve_person_id(payload: dict, channel_type: str, address: str | None) -> str | None:
    person_id = payload.get("person_id")
    if person_id:
        return str(person_id)
    if channel_type and address:
        person = identity_store.resolve_person_by_channel(channel_type, address)
        if person:
            return str(person.get("person_id"))
    return None


def _interpret_new_message(text: str, incoming: IncomingContext) -> ActionResult:
    timezone = get_timezone()
    interpreter = build_default_command_interpreter()
    plan = interpreter.interpret(
        text,
        CommandInterpreterContext(
            actor_person_id=incoming.person_id,
            channel_type=incoming.channel_type,
            channel_target=incoming.address or incoming.channel_type,
            source=incoming.channel_type,
            correlation_id=incoming.correlation_id,
            timezone=timezone,
        ),
    )

    plan = _apply_idempotency(plan, incoming)
    plan = _apply_confirmation_rules(plan)
    plan = _enforce_side_effect_evidence(plan)

    spec = get_plan_spec(plan.plan_kind, plan.plan_version)
    if not spec:
        plan = _unknown_plan_from(plan, reason="plan_spec_missing")
        spec = get_plan_spec(plan.plan_kind, plan.plan_version)
    validation_errors = _validate_against_spec(plan, spec)
    if validation_errors:
        logger.warning("Plan validation failed: %s", "; ".join(validation_errors))
        plan = _unknown_plan_from(plan, reason="schema_validation")
        spec = get_plan_spec(plan.plan_kind, plan.plan_version)

    status = "pending_confirmation" if plan.requires_confirmation else "ready"
    insert_plan_instance(plan, status)

    if plan.requires_confirmation:
        create_pending_plan(
            {
                "pending_id": str(uuid.uuid4()),
                "person_id": incoming.person_id,
                "channel_type": incoming.channel_type,
                "correlation_id": incoming.correlation_id,
                "plan_json": plan.model_dump(),
                "status": "pending",
                "expires_at": _expires_at(),
            }
        )
        question = plan.questions[0] if plan.questions else "¿Puedes aclarar qué necesitas?"
        update_plan_instance_status(plan.plan_id, "pending_confirmation")
        return _message_result(question, incoming)

    try:
        message = _execute_plan(plan, incoming, spec)
    except Exception as exc:
        update_plan_instance_status(plan.plan_id, "failed")
        if "insufficient_intent_evidence" in str(exc):
            return _message_result("¿Puedes aclarar qué necesitas?", incoming)
        return _message_result(f"No pude completar eso: {exc}", incoming)

    update_plan_instance_status(plan.plan_id, "executed")
    return _message_result(message, incoming)


def _continue_pending_plan(pending: dict[str, Any], text: str, incoming: IncomingContext) -> ActionResult:
    if _is_cancel(text):
        update_pending_status(str(pending.get("pending_id")), "cancelled")
        return _message_result("De acuerdo, cancelado.", incoming)

    raw_plan = pending.get("plan_json")
    plan_data = _parse_plan_json(raw_plan)
    if plan_data is None:
        update_pending_status(str(pending.get("pending_id")), "cancelled")
        return _message_result("Esa solicitud ya no es válida. Intentemos de nuevo.", incoming)

    try:
        plan = parse_command_plan(plan_data)
    except ValueError:
        update_pending_status(str(pending.get("pending_id")), "cancelled")
        return _message_result("Esa solicitud ya no es válida. Intentemos de nuevo.", incoming)

    timezone = get_timezone()
    if isinstance(plan, CreateReminderPlan) and not plan.payload.schedule.trigger_at:
        trigger_at = infer_trigger_at(text, timezone)
        if not trigger_at:
            return _message_result("¿A qué hora debo programarlo?", incoming)
        plan = plan.model_copy(
            update={
                "payload": plan.payload.model_copy(
                    update={"schedule": plan.payload.schedule.model_copy(update={"trigger_at": trigger_at})}
                ),
                "requires_confirmation": False,
                "questions": [],
            }
        )

    update_pending_status(str(pending.get("pending_id")), "confirmed")
    plan = _apply_confirmation_rules(plan)
    plan = _enforce_side_effect_evidence(plan)
    spec = get_plan_spec(plan.plan_kind, plan.plan_version)
    validation_errors = _validate_against_spec(plan, spec)
    if validation_errors:
        plan = _unknown_plan_from(plan, reason="schema_validation")
        spec = get_plan_spec(plan.plan_kind, plan.plan_version)
    try:
        message = _execute_plan(plan, incoming, spec)
    except Exception as exc:
        update_plan_instance_status(plan.plan_id, "failed")
        if "insufficient_intent_evidence" in str(exc):
            return _message_result("¿Puedes aclarar qué necesitas?", incoming)
        return _message_result(f"No pude completar eso: {exc}", incoming)
    update_plan_instance_status(plan.plan_id, "executed")
    return _message_result(message, incoming)


def _execute_plan(plan: CommandPlan, incoming: IncomingContext, spec: object | None) -> str:
    if not spec or not getattr(spec, "executor_key", None):
        raise ValueError("Plan executor missing")
    executor = EXECUTOR_MAP.get(spec.executor_key)
    if not executor:
        raise ValueError(f"Unknown executor: {spec.executor_key}")
    return executor(plan, ExecutorContext(actor_person_id=incoming.person_id))


def _validate_against_spec(plan: CommandPlan, spec: object | None) -> list[str]:
    if not spec or not getattr(spec, "json_schema", None):
        return ["plan_spec_missing"]
    return validate_plan(spec.json_schema, plan.model_dump())


def _apply_idempotency(plan: CommandPlan, incoming: IncomingContext) -> CommandPlan:
    if isinstance(plan, CreateReminderPlan) and not plan.payload.idempotency_key and incoming.update_id:
        return plan.model_copy(
            update={"payload": plan.payload.model_copy(update={"idempotency_key": f"telegram:{incoming.update_id}"})}
        )
    return plan


def _apply_confirmation_rules(plan: CommandPlan) -> CommandPlan:
    if isinstance(plan, CreateReminderPlan):
        if not plan.payload.schedule.trigger_at and not plan.payload.schedule.rrule:
            return plan.model_copy(
                update={
                    "requires_confirmation": True,
                    "questions": ["¿A qué hora debo programarlo?"],
                }
            )
    if isinstance(plan, SendMessagePlan):
        target_id = plan.payload.target.person_ref.id or plan.payload.target.person_ref.name
        if target_id and plan.actor.person_id and target_id != plan.actor.person_id:
            if plan.intent_confidence < 0.6:
                return plan.model_copy(
                    update={
                        "requires_confirmation": True,
                        "questions": ["¿Confirmas que debo enviar ese mensaje?"],
                    }
                )
    return plan


def _enforce_side_effect_evidence(plan: CommandPlan) -> CommandPlan:
    if isinstance(plan, (CreateReminderPlan, SendMessagePlan)) and not _evidence_sufficient(plan.intent_evidence):
        return _unknown_plan_from(plan, reason="insufficient_evidence")
    return plan


def _unknown_plan_from(plan: CommandPlan, reason: str) -> UnknownPlan:
    questions = ["¿Puedes aclarar qué necesitas?"]
    return UnknownPlan(
        plan_kind="unknown",
        plan_version=1,
        plan_id=plan.plan_id,
        correlation_id=plan.correlation_id,
        created_at=plan.created_at,
        source=plan.source,
        actor=plan.actor,
        intent_confidence=plan.intent_confidence,
        requires_confirmation=False,
        questions=questions,
        intent_evidence=plan.intent_evidence,
        payload={"user_text": plan.original_text or "", "reason": reason, "suggestions": []},
        metadata=plan.metadata,
        original_text=plan.original_text,
    )


def _evidence_sufficient(evidence: IntentEvidence) -> bool:
    if evidence.score >= 0.6:
        return True
    if evidence.reminder_verbs or evidence.time_hints or evidence.quoted_spans:
        return True
    return False


def _message_result(message: str, incoming: IncomingContext) -> ActionResult:
    logger.info(
        "HandleIncomingMessageAction response channel=%s message=%s",
        incoming.channel_type,
        _snippet(message),
    )
    payload: dict[str, Any] = {
        "message": message,
        "origin": incoming.channel_type,
        "channel_hint": incoming.channel_type,
        "correlation_id": incoming.correlation_id,
        "audience": _audience_for(incoming.person_id),
    }
    if incoming.address:
        payload["target"] = incoming.address
    if incoming.channel_type == "telegram" and incoming.address:
        payload["direct_reply"] = {
            "channel_type": "telegram",
            "target": incoming.address,
            "text": message,
            "correlation_id": incoming.correlation_id,
        }
    return ActionResult(
        intention_key="MESSAGE_READY",
        payload=payload,
        urgency="normal",
    )


def _audience_for(person_id: str | None) -> dict[str, str]:
    if person_id:
        return {"kind": "person", "id": person_id}
    return {"kind": "system", "id": "system"}


def _parse_plan_json(raw_plan: object | None) -> dict[str, Any] | None:
    if raw_plan is None:
        return None
    if isinstance(raw_plan, dict):
        return raw_plan
    if isinstance(raw_plan, str):
        try:
            parsed = json.loads(raw_plan)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _is_cancel(text: str) -> bool:
    return text.strip().lower() in {"no", "n", "cancel", "stop"}


def _expires_at() -> str:
    expires = datetime.now(tz=timezone.utc) + timedelta(hours=1)
    return expires.isoformat()


def _snippet(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."
