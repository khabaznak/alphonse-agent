from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.cognition.skills.command_plans import (
    CommandPlan,
    CreateReminderPlan,
    SendMessagePlan,
    parse_plan,
)
from alphonse.agent.cognition.skills.plan_interpreter import infer_trigger_at, interpret_plan
from alphonse.agent.core.settings_store import get_timezone
from alphonse.agent.identity import store as identity_store
from alphonse.agent.nervous_system.pending_store import create_pending_plan, get_pending_plan, update_pending_status
from alphonse.agent.nervous_system.timed_commands import insert_timed_signal_from_plan


@dataclass(frozen=True)
class IncomingContext:
    channel_type: str
    address: str | None
    person_id: str | None
    correlation_id: str


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
        if not text:
            return _message_result(
                "I did not catch that. Could you rephrase?",
                incoming,
            )

        pending = get_pending_plan(incoming.person_id, incoming.channel_type)
        if pending:
            return _continue_pending_plan(pending, text, incoming)

        return _interpret_new_message(text, incoming)


def _build_incoming_context(payload: dict, signal: object | None, correlation_id: str) -> IncomingContext:
    origin = payload.get("origin") or getattr(signal, "source", None) or "system"
    channel_type = str(origin)
    address = _resolve_address(channel_type, payload)
    person_id = _resolve_person_id(payload, channel_type, address)
    return IncomingContext(
        channel_type=channel_type,
        address=address,
        person_id=person_id,
        correlation_id=correlation_id,
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
    created_by = incoming.person_id or _channel_identity(incoming)
    plan = interpret_plan(
        text=text,
        created_by=created_by,
        source=incoming.channel_type,
        correlation_id=incoming.correlation_id,
        timezone=timezone,
    )
    if plan is None:
        return _message_result(
            "I can help set reminders. Try 'remind me to water plants at 6pm'.",
            incoming,
        )
    return _handle_plan(plan, incoming)


def _handle_plan(plan: CommandPlan, incoming: IncomingContext) -> ActionResult:
    if isinstance(plan, CreateReminderPlan):
        if not plan.schedule.trigger_at:
            pending_id = str(uuid.uuid4())
            create_pending_plan(
                {
                    "pending_id": pending_id,
                    "person_id": incoming.person_id,
                    "channel_type": incoming.channel_type,
                    "correlation_id": incoming.correlation_id,
                    "plan_json": plan.model_dump(),
                    "status": "pending",
                    "expires_at": _expires_at(),
                }
            )
            return _message_result("When should I set the reminder?", incoming)
        insert_timed_signal_from_plan(plan)
        return _message_result(
            f"Scheduled reminder for {plan.schedule.trigger_at}.",
            incoming,
        )
    if isinstance(plan, SendMessagePlan):
        return _message_result(plan.message.text, incoming)
    return _message_result("I could not process that request.", incoming)


def _continue_pending_plan(pending: dict[str, Any], text: str, incoming: IncomingContext) -> ActionResult:
    if _is_cancel(text):
        update_pending_status(str(pending.get("pending_id")), "cancelled")
        return _message_result("Okay, cancelled.", incoming)

    raw_plan = pending.get("plan_json")
    plan_data = _parse_plan_json(raw_plan)
    if plan_data is None:
        update_pending_status(str(pending.get("pending_id")), "cancelled")
        return _message_result("Pending request was invalid. Please try again.", incoming)

    try:
        plan = parse_plan(plan_data)
    except ValueError:
        update_pending_status(str(pending.get("pending_id")), "cancelled")
        return _message_result("Pending request was invalid. Please try again.", incoming)

    timezone = get_timezone()
    if isinstance(plan, CreateReminderPlan) and not plan.schedule.trigger_at:
        trigger_at = infer_trigger_at(text, timezone)
        if not trigger_at:
            return _message_result("What time should I use for the reminder?", incoming)
        plan = plan.model_copy(update={"schedule": plan.schedule.model_copy(update={"trigger_at": trigger_at})})
        update_pending_status(str(pending.get("pending_id")), "confirmed")
        insert_timed_signal_from_plan(plan)
        return _message_result(
            f"Scheduled reminder for {plan.schedule.trigger_at}.",
            incoming,
        )

    if not _is_confirm(text):
        return _message_result("Please confirm with yes or no.", incoming)

    update_pending_status(str(pending.get("pending_id")), "confirmed")
    return _handle_plan(plan, incoming)


def _message_result(message: str, incoming: IncomingContext) -> ActionResult:
    payload: dict[str, Any] = {
        "message": message,
        "origin": incoming.channel_type,
        "channel_hint": incoming.channel_type,
        "correlation_id": incoming.correlation_id,
        "audience": _audience_for(incoming.person_id),
    }
    if incoming.address:
        payload["target"] = incoming.address
    return ActionResult(
        intention_key="MESSAGE_READY",
        payload=payload,
        urgency="normal",
    )


def _audience_for(person_id: str | None) -> dict[str, str]:
    if person_id:
        return {"kind": "person", "id": person_id}
    return {"kind": "system", "id": "system"}


def _channel_identity(incoming: IncomingContext) -> str:
    if incoming.address:
        return f"{incoming.channel_type}:{incoming.address}"
    return incoming.channel_type


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


def _is_confirm(text: str) -> bool:
    return text.strip().lower() in {"yes", "y", "confirm", "ok", "sure"}


def _is_cancel(text: str) -> bool:
    return text.strip().lower() in {"no", "n", "cancel", "stop"}


def _expires_at() -> str:
    expires = datetime.now(tz=timezone.utc) + timedelta(hours=1)
    return expires.isoformat()
