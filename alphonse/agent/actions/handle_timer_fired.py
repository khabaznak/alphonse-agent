from __future__ import annotations

import logging
from typing import Any

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.cognition.skills.command_plans import CreateReminderPlan, parse_command_plan


logger = logging.getLogger(__name__)


class HandleTimerFiredAction(Action):
    key = "handle_timer_fired"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        payload = getattr(signal, "payload", {}) if signal else {}
        logger.info(
            "HandleTimerFiredAction invoked signal_id=%s timed_signal_id=%s correlation_id=%s",
            getattr(signal, "id", None),
            payload.get("timed_signal_id"),
            getattr(signal, "correlation_id", None),
        )
        plan = _extract_plan(payload)
        message = _message_from_plan(plan, payload)
        correlation_id = _correlation_id(payload, signal, plan)
        channel_hint = _channel_hint(plan, payload)
        target = _target_address(payload)
        audience = _audience(plan, payload)

        result_payload: dict[str, Any] = {
            "message": message,
            "origin": "timer",
            "channel_hint": channel_hint,
            "correlation_id": correlation_id,
            "audience": audience,
        }
        if target:
            result_payload["target"] = target

        return ActionResult(
            intention_key="MESSAGE_READY",
            payload=result_payload,
            urgency="normal",
        )


def _extract_plan(payload: dict[str, Any]) -> CreateReminderPlan | None:
    plan_data = None
    signal_payload = payload.get("payload") if isinstance(payload, dict) else None
    if isinstance(signal_payload, dict):
        plan_data = signal_payload.get("plan")
    if not isinstance(plan_data, dict):
        return None
    try:
        plan = parse_command_plan(plan_data)
    except ValueError:
        return None
    if isinstance(plan, CreateReminderPlan):
        return plan
    return None


def _message_from_plan(plan: CreateReminderPlan | None, payload: dict[str, Any]) -> str:
    if plan:
        return plan.payload.message.text
    signal_payload = payload.get("payload") if isinstance(payload, dict) else None
    if isinstance(signal_payload, dict):
        message = signal_payload.get("message")
        if message:
            return str(message)
    return "Reminder: scheduled task is due."


def _channel_hint(plan: CreateReminderPlan | None, payload: dict[str, Any]) -> str:
    if plan and plan.payload.delivery and plan.payload.delivery.channel_type:
        return str(plan.payload.delivery.channel_type)
    signal_payload = payload.get("payload") if isinstance(payload, dict) else None
    if isinstance(signal_payload, dict):
        origin = signal_payload.get("origin") or payload.get("origin")
        if origin:
            return str(origin)
    return "system"


def _target_address(payload: dict[str, Any]) -> str | None:
    target = payload.get("target") if isinstance(payload, dict) else None
    if target is None:
        return None
    return str(target)


def _audience(plan: CreateReminderPlan | None, payload: dict[str, Any]) -> dict[str, str]:
    if plan and plan.payload.target.person_ref.id:
        return {"kind": "person", "id": str(plan.payload.target.person_ref.id)}
    signal_payload = payload.get("payload") if isinstance(payload, dict) else None
    if isinstance(signal_payload, dict):
        person_id = signal_payload.get("person_id")
        if person_id:
            return {"kind": "person", "id": str(person_id)}
    return {"kind": "system", "id": "system"}


def _correlation_id(payload: dict[str, Any], signal: object | None, plan: CreateReminderPlan | None) -> str | None:
    if plan:
        return plan.correlation_id
    if isinstance(payload, dict):
        cid = payload.get("correlation_id")
        if cid:
            return str(cid)
    return getattr(signal, "correlation_id", None) if signal else None
