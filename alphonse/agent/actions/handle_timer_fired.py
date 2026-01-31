from __future__ import annotations

import logging
from typing import Any

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.cognition.skills.command_plans import (
    CreateReminderPlan,
    parse_command_plan,
)
from alphonse.agent.cognition.reminders.renderer import render_reminder
from alphonse.agent.cognition.plan_executor import PlanExecutionContext, PlanExecutor
from alphonse.agent.cognition.plans import CortexPlan, PlanType
from alphonse.agent.policy.engine import PolicyEngine
from alphonse.config import settings


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
        inner = _payload_from_signal(payload)
        reminder_payload = _build_reminder_payload(plan, payload, inner)
        rendered = render_reminder(reminder_payload, prefs=None)
        locale = _locale_from_payload(reminder_payload)
        correlation_id = _correlation_id(payload, signal, plan)
        target = reminder_payload.get("chat_id") or _target_address(payload)
        channels = ["telegram"]

        logger.info(
            "HandleTimerFiredAction render message_len=%s channels=%s",
            len(rendered),
            ",".join(channels),
        )

        if target:
            communicate_plan = CortexPlan(
                plan_type=PlanType.COMMUNICATE,
                target=str(target),
                channels=channels,
                payload={
                    "message": rendered,
                    "style": "friendly",
                    "locale": locale,
                },
            )
            policy = PolicyEngine()
            approved = policy.approve([communicate_plan], context)
            executor = PlanExecutor()
            exec_context = PlanExecutionContext(
                channel_type=str(reminder_payload.get("origin_channel") or "telegram"),
                channel_target=str(target),
                actor_person_id=_actor_person_id(reminder_payload),
                correlation_id=str(correlation_id or ""),
            )
            if approved:
                executor.execute(approved, context, exec_context)
        else:
            logger.warning("HandleTimerFiredAction missing target; skipping dispatch")

        return ActionResult(intention_key="NOOP", payload={}, urgency=None)


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


def _payload_from_signal(payload: dict[str, Any]) -> dict[str, Any]:
    signal_payload = payload.get("payload") if isinstance(payload, dict) else None
    return signal_payload if isinstance(signal_payload, dict) else {}


def _build_reminder_payload(
    plan: CreateReminderPlan | None,
    payload: dict[str, Any],
    inner: dict[str, Any],
) -> dict[str, Any]:
    if plan:
        return {
            "reminder_text_raw": plan.payload.message.text,
            "chat_id": plan.actor.channel.target
            if plan.actor and plan.actor.channel
            else None,
            "origin_channel": plan.source,
            "locale_hint": plan.payload.message.language,
            "created_at": plan.created_at,
            "trigger_at": plan.payload.schedule.trigger_at,
        }
    reminder_text_raw = inner.get("reminder_text_raw") or inner.get("message")
    return {
        "reminder_text_raw": reminder_text_raw,
        "chat_id": inner.get("chat_id") or inner.get("target"),
        "origin_channel": inner.get("origin_channel")
        or inner.get("origin")
        or payload.get("origin"),
        "locale_hint": inner.get("locale_hint"),
        "created_at": inner.get("created_at"),
        "person_id": inner.get("person_id"),
        "trigger_at": inner.get("trigger_at"),
        "user_name": inner.get("user_name"),
    }


def _target_address(payload: dict[str, Any]) -> str | None:
    target = payload.get("target") if isinstance(payload, dict) else None
    if target is None:
        return None
    return str(target)


def _correlation_id(
    payload: dict[str, Any], signal: object | None, plan: CreateReminderPlan | None
) -> str | None:
    if plan:
        return plan.correlation_id
    if isinstance(payload, dict):
        cid = payload.get("correlation_id")
        if cid:
            return str(cid)
    return getattr(signal, "correlation_id", None) if signal else None


def _locale_from_payload(payload: dict[str, Any]) -> str:
    hint = payload.get("locale_hint")
    if isinstance(hint, str) and hint.strip():
        return hint
    raw = payload.get("reminder_text_raw") or ""
    if any(
        token in str(raw).lower()
        for token in ("recuérd", "recuerda", "bañar", "mañana", "hoy")
    ):
        return "es-MX"
    if any(
        token in str(raw).lower() for token in ("remind", "tomorrow", "today", "please")
    ):
        return "en-US"
    return settings.get_default_locale()


def _actor_person_id(payload: dict[str, Any]) -> str | None:
    person_id = payload.get("person_id")
    if person_id:
        return str(person_id)
    return None
