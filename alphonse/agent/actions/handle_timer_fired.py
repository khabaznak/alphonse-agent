from __future__ import annotations

import logging
from typing import Any

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.cognition.reminders.renderer import render_reminder
from alphonse.agent.actions.handle_daily_report import dispatch_daily_report
from alphonse.agent.cognition.plan_executor import PlanExecutionContext, PlanExecutor
from alphonse.agent.cognition.plans import CortexPlan, PlanType
from alphonse.agent.policy.engine import PolicyEngine
from alphonse.agent.services.job_runner import JobRunner
from alphonse.agent.services.job_store import JobStore
from alphonse.agent.services.scratchpad_service import ScratchpadService
from alphonse.agent.cognition.preferences.store import (
    get_or_create_principal_for_channel,
    resolve_preference_with_precedence,
)
from alphonse.agent.nervous_system.senses.bus import Signal as BusSignal
from alphonse.agent.tools.registry import build_default_tool_registry
from alphonse.config import settings


logger = logging.getLogger(__name__)


class HandleTimerFiredAction(Action):
    key = "handle_timer_fired"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        payload = getattr(signal, "payload", {}) if signal else {}
        signal_type = payload.get("signal_type")
        mind_layer = str(payload.get("mind_layer") or "subconscious").strip().lower()
        dispatch_mode = str(payload.get("dispatch_mode") or "deterministic").strip().lower()
        logger.info(
            "HandleTimerFiredAction invoked signal_id=%s timed_signal_id=%s correlation_id=%s",
            getattr(signal, "id", None),
            payload.get("timed_signal_id"),
            getattr(signal, "correlation_id", None),
        )
        if signal_type == "daily_report":
            dispatch_daily_report(context, payload)
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)
        if signal_type == "job_trigger":
            self._handle_job_trigger(context=context, payload=payload, signal=signal)
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)
        if dispatch_mode == "graph" or mind_layer == "conscious":
            self._dispatch_conscious_timed_signal(context=context, payload=payload, signal=signal)
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)
        inner = _payload_from_signal(payload)
        reminder_payload = _build_reminder_payload(payload, inner)
        rendered = render_reminder(reminder_payload, prefs=None)
        locale = _locale_from_payload(reminder_payload)
        correlation_id = _correlation_id(payload, signal)
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

    def _dispatch_conscious_timed_signal(self, *, context: dict, payload: dict[str, Any], signal: Any) -> None:
        bus = context.get("ctx")
        if not hasattr(bus, "emit"):
            logger.warning("HandleTimerFiredAction conscious dispatch skipped reason=no_bus")
            return
        inner = _payload_from_signal(payload)
        message_text = str(
            inner.get("agent_internal_prompt")
            or inner.get("prompt_text")
            or inner.get("message")
            or payload.get("signal_type")
            or "You just remembered something important."
        ).strip()
        target = str(inner.get("chat_id") or payload.get("target") or inner.get("delivery_target") or "").strip()
        user_id = str(inner.get("person_id") or target or "").strip()
        bus.emit(
            BusSignal(
                type="api.message_received",
                payload={
                    "text": message_text,
                    "channel": "api",
                    "origin": "api",
                    "target": target or user_id,
                    "user_id": user_id or target,
                    "metadata": {
                        "timed_signal": payload,
                        "channel_hint": "api",
                    },
                },
                source="timer",
                correlation_id=_correlation_id(payload, signal),
            )
        )

    def _handle_job_trigger(self, *, context: dict, payload: dict[str, Any], signal: Any) -> None:
        inner = _payload_from_signal(payload)
        job_id = str(inner.get("job_id") or payload.get("job_id") or "").strip()
        user_id = str(inner.get("user_id") or payload.get("target") or "").strip()
        if not job_id or not user_id:
            logger.warning("HandleTimerFiredAction missing job trigger ids job_id=%s user_id=%s", job_id, user_id)
            return
        bus = context.get("ctx")
        def _emit_brain_event(event: dict[str, Any]) -> None:
            if not hasattr(bus, "emit"):
                return
            message_text = _render_job_event_message(event)
            bus.emit(
                BusSignal(
                    type="api.message_received",
                    payload={
                        "text": message_text,
                        "channel": "api",
                        "origin": "api",
                        "target": str(user_id),
                        "user_id": str(user_id),
                        "metadata": {
                            "job_event": event,
                            "channel_hint": "api",
                        },
                    },
                    source="job_runner",
                    correlation_id=str(event.get("job_id") or job_id),
                )
            )
        runner = JobRunner(
            job_store=JobStore(),
            scratchpad_service=ScratchpadService(),
            tool_registry=build_default_tool_registry(),
            brain_event_sink=_emit_brain_event if hasattr(bus, "emit") else None,
            tick_seconds=45,
        )
        try:
            runner.run_job_now(user_id=user_id, job_id=job_id)
            logger.info("HandleTimerFiredAction job_trigger executed job_id=%s user_id=%s", job_id, user_id)
        except Exception as exc:
            logger.exception(
                "HandleTimerFiredAction job_trigger failed job_id=%s user_id=%s error=%s",
                job_id,
                user_id,
                exc,
            )


def _render_job_event_message(event: dict[str, Any]) -> str:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    internal = str(payload.get("agent_internal_prompt") or "").strip()
    if internal:
        return internal
    prompt_text = str(payload.get("prompt_text") or "").strip()
    if prompt_text:
        return prompt_text
    message = str(event.get("message") or "").strip()
    if message:
        return message
    job_name = str(event.get("job_name") or "scheduled job").strip()
    return f"You just remembered a scheduled task: {job_name}."


def _payload_from_signal(payload: dict[str, Any]) -> dict[str, Any]:
    signal_payload = payload.get("payload") if isinstance(payload, dict) else None
    return signal_payload if isinstance(signal_payload, dict) else {}


def _build_reminder_payload(
    payload: dict[str, Any],
    inner: dict[str, Any],
) -> dict[str, Any]:
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
    payload: dict[str, Any], signal: object | None
) -> str | None:
    if isinstance(payload, dict):
        cid = payload.get("correlation_id")
        if cid:
            return str(cid)
    return getattr(signal, "correlation_id", None) if signal else None


def _locale_from_payload(payload: dict[str, Any]) -> str:
    channel_type = payload.get("origin_channel")
    channel_id = payload.get("chat_id")
    if channel_type and channel_id:
        principal_id = get_or_create_principal_for_channel(
            str(channel_type),
            str(channel_id),
        )
        if principal_id:
            return resolve_preference_with_precedence(
                key="locale",
                default=settings.get_default_locale(),
                channel_principal_id=principal_id,
            )
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
