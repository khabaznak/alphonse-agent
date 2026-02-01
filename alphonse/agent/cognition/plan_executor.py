from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from alphonse.agent.actions.models import ActionResult
from alphonse.agent.cognition.narration.coordinator import (
    DeliveryCoordinator,
    build_default_coordinator,
)
from alphonse.agent.cognition.plans import (
    CommunicatePayload,
    CortexPlan,
    PlanType,
    QueryStatusPayload,
    ScheduleTimedSignalPayload,
    UpdatePreferencesPayload,
)
from alphonse.agent.cognition.preferences.store import (
    get_or_create_principal_for_channel,
    set_preference,
)
from alphonse.agent.extremities.api_extremity import ApiExtremity
from alphonse.agent.extremities.cli_extremity import CliExtremity
from alphonse.agent.extremities.registry import ExtremityRegistry
from alphonse.agent.extremities.scheduler_extremity import schedule_reminder
from alphonse.agent.extremities.telegram_notification import (
    TelegramNotificationExtremity,
)
from alphonse.agent.policy.engine import PolicyDecision, PolicyEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlanExecutionContext:
    channel_type: str
    channel_target: str | None
    actor_person_id: str | None
    correlation_id: str


class PlanExecutor:
    def __init__(
        self,
        *,
        coordinator: DeliveryCoordinator | None = None,
        extremities: ExtremityRegistry | None = None,
        policy_engine: PolicyEngine | None = None,
    ) -> None:
        self._coordinator = coordinator or build_default_coordinator()
        self._extremities = extremities or _build_default_extremities()
        self._policy = policy_engine or PolicyEngine()

    def execute(
        self, plans: list[CortexPlan], context: dict, exec_context: PlanExecutionContext
    ) -> None:
        for plan in plans:
            try:
                self._execute_plan(plan, context, exec_context)
            except Exception as exc:
                logger.exception(
                    "executor dispatch failed plan_id=%s plan_type=%s error=%s",
                    plan.plan_id,
                    plan.plan_type,
                    exc,
                )
                self._dispatch_error(exec_context, context)

    def _execute_plan(
        self, plan: CortexPlan, context: dict, exec_context: PlanExecutionContext
    ) -> None:
        decision = self._policy.approve_plan(plan, exec_context)
        if not decision.allowed:
            logger.warning(
                "executor policy denied plan_id=%s plan_type=%s reason=%s",
                plan.plan_id,
                plan.plan_type,
                decision.reason,
            )
            self._dispatch_policy_rejection(decision, context, exec_context, plan)
            return
        if plan.plan_type == PlanType.COMMUNICATE:
            payload = CommunicatePayload.model_validate(plan.payload)
            self._execute_communicate(plan, payload, context, exec_context)
            return
        if plan.plan_type == PlanType.SCHEDULE_TIMED_SIGNAL:
            payload = ScheduleTimedSignalPayload.model_validate(plan.payload)
            self._execute_schedule(plan, payload, context, exec_context)
            return
        if plan.plan_type == PlanType.QUERY_STATUS:
            QueryStatusPayload.model_validate(plan.payload)
            logger.info(
                "executor dispatch plan_id=%s plan_type=%s outcome=ignored",
                plan.plan_id,
                plan.plan_type,
            )
            return
        if plan.plan_type == PlanType.UPDATE_PREFERENCES:
            payload = UpdatePreferencesPayload.model_validate(plan.payload)
            self._execute_update_preferences(plan, payload)
            return
        logger.info(
            "executor dispatch plan_id=%s plan_type=%s outcome=noop",
            plan.plan_id,
            plan.plan_type,
        )

    def _execute_communicate(
        self,
        plan: CortexPlan,
        payload: CommunicatePayload,
        context: dict,
        exec_context: PlanExecutionContext,
    ) -> None:
        channels = plan.channels or [exec_context.channel_type]
        target = plan.target or exec_context.channel_target
        for channel in channels:
            self._dispatch_message(
                channel=channel,
                target=target,
                message=payload.message,
                context=context,
                exec_context=exec_context,
                plan=plan,
            )

    def _execute_schedule(
        self,
        plan: CortexPlan,
        payload: ScheduleTimedSignalPayload,
        context: dict,
        exec_context: PlanExecutionContext,
    ) -> None:
        logger.info(
            "executor dispatch plan_id=%s plan_type=%s tool=scheduler trigger_at=%s",
            plan.plan_id,
            plan.plan_type,
            payload.trigger_at,
        )
        result = schedule_reminder(
            reminder_text=payload.reminder_text,
            trigger_time=payload.trigger_at,
            chat_id=payload.chat_id,
            channel_type=payload.origin,
            actor_person_id=exec_context.actor_person_id,
            intent_evidence={},
            correlation_id=payload.correlation_id,
        )
        logger.info(
            "executor dispatch plan_id=%s plan_type=%s outcome=scheduled schedule_id=%s",
            plan.plan_id,
            plan.plan_type,
            result,
        )
        confirmation = f"Programé el recordatorio para {payload.trigger_at}."
        self._dispatch_message(
            channel=payload.origin,
            target=payload.chat_id,
            message=confirmation,
            context=context,
            exec_context=exec_context,
            plan=plan,
        )

    def _dispatch_message(
        self,
        *,
        channel: str,
        target: str | None,
        message: str,
        context: dict,
        exec_context: PlanExecutionContext,
        plan: CortexPlan,
    ) -> None:
        if channel in {"telegram", "api"} and not target:
            logger.warning(
                "executor dispatch skipped plan_id=%s channel=%s reason=missing_target",
                plan.plan_id,
                channel,
            )
            return
        logger.info(
            "executor dispatch plan_id=%s plan_type=%s channel=%s target=%s",
            plan.plan_id,
            plan.plan_type,
            channel,
            target or "none",
        )
        payload = _message_payload(message, channel, target, exec_context)
        action = ActionResult(
            intention_key="MESSAGE_READY", payload=payload, urgency="normal"
        )
        delivery = self._coordinator.deliver(action, context)
        if delivery:
            self._extremities.dispatch(delivery, None)

    def _execute_update_preferences(
        self,
        plan: CortexPlan,
        payload: UpdatePreferencesPayload,
    ) -> None:
        principal = payload.principal
        principal_id = get_or_create_principal_for_channel(
            principal.channel_type,
            principal.channel_id,
        )
        if not principal_id:
            logger.warning(
                "executor dispatch plan_id=%s plan_type=%s outcome=missing_principal",
                plan.plan_id,
                plan.plan_type,
            )
            return
        for update in payload.updates:
            set_preference(principal_id, update.key, update.value, source="user")
        logger.info(
            "executor dispatch plan_id=%s plan_type=%s outcome=preferences_updated count=%s",
            plan.plan_id,
            plan.plan_type,
            len(payload.updates),
        )

    def _dispatch_error(
        self, exec_context: PlanExecutionContext, context: dict
    ) -> None:
        payload = _message_payload(
            "Lo siento, tuve un problema al procesar la solicitud.",
            exec_context.channel_type,
            exec_context.channel_target,
            exec_context,
        )
        action = ActionResult(
            intention_key="MESSAGE_READY", payload=payload, urgency="normal"
        )
        delivery = self._coordinator.deliver(action, context)
        if delivery:
            self._extremities.dispatch(delivery, None)

    def _dispatch_policy_rejection(
        self,
        decision: PolicyDecision,
        context: dict,
        exec_context: PlanExecutionContext,
        plan: CortexPlan,
    ) -> None:
        if (
            exec_context.channel_type in {"telegram", "cli", "api"}
            and not exec_context.channel_target
        ):
            logger.warning(
                "executor policy denial could not notify plan_id=%s reason=missing_target",
                plan.plan_id,
            )
            return
        message = "No estoy autorizado para programar ese recordatorio."
        payload = _message_payload(
            message,
            exec_context.channel_type,
            exec_context.channel_target,
            exec_context,
        )
        action = ActionResult(
            intention_key="MESSAGE_READY", payload=payload, urgency="normal"
        )
        delivery = self._coordinator.deliver(action, context)
        if delivery:
            self._extremities.dispatch(delivery, None)


def _message_payload(
    message: str,
    channel: str,
    target: str | None,
    exec_context: PlanExecutionContext,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "message": message,
        "origin": channel,
        "channel_hint": channel,
        "correlation_id": exec_context.correlation_id,
        "audience": _audience_for(exec_context.actor_person_id),
    }
    if target:
        payload["target"] = target
    return payload


def _audience_for(person_id: str | None) -> dict[str, str]:
    if person_id:
        return {"kind": "person", "id": person_id}
    return {"kind": "system", "id": "system"}


def _build_default_extremities() -> ExtremityRegistry:
    registry = ExtremityRegistry()
    registry.register(TelegramNotificationExtremity())
    registry.register(ApiExtremity())
    registry.register(CliExtremity())
    return registry
