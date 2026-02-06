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
    CapabilityGapPayload,
    LanArmPayload,
    PairingDecisionPayload,
)
from alphonse.agent.cognition.localization import render_message
from alphonse.agent.cognition.status_summaries import (
    summarize_gaps,
    summarize_timed_signals,
)
from alphonse.agent.cognition.preferences.store import (
    get_or_create_principal_for_channel,
    get_with_fallback,
    set_preference,
)
from alphonse.agent.nervous_system.capability_gaps import insert_gap
from alphonse.agent.lan.store import arm_device, disarm_device, get_latest_paired_device, get_paired_device
from alphonse.agent.lan.pairing_store import approve_pairing, deny_pairing, get_pairing_request
from alphonse.config import settings
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
            payload = QueryStatusPayload.model_validate(plan.payload)
            self._execute_query_status(plan, payload, context, exec_context)
            return
        if plan.plan_type == PlanType.UPDATE_PREFERENCES:
            payload = UpdatePreferencesPayload.model_validate(plan.payload)
            self._execute_update_preferences(plan, payload)
            return
        if plan.plan_type == PlanType.CAPABILITY_GAP:
            payload = CapabilityGapPayload.model_validate(plan.payload)
            self._execute_capability_gap(payload)
            return
        if plan.plan_type in {PlanType.LAN_ARM, PlanType.LAN_DISARM}:
            payload = LanArmPayload.model_validate(plan.payload)
            self._execute_lan_arm(plan, payload, context, exec_context)
            return
        if plan.plan_type in {PlanType.PAIR_APPROVE, PlanType.PAIR_DENY}:
            payload = PairingDecisionPayload.model_validate(plan.payload)
            self._execute_pairing_decision(plan, payload, context, exec_context)
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
        confirmation = self._render_schedule_confirmation(payload)
        self._dispatch_message(
            channel=payload.origin,
            target=payload.chat_id,
            message=confirmation,
            context=context,
            exec_context=exec_context,
            plan=plan,
        )

    def _execute_query_status(
        self,
        plan: CortexPlan,
        payload: QueryStatusPayload,
        context: dict,
        exec_context: PlanExecutionContext,
    ) -> None:
        logger.info(
            "executor dispatch plan_id=%s plan_type=%s include=%s",
            plan.plan_id,
            plan.plan_type,
            ",".join(payload.include),
        )
        parts: list[str] = []
        locale = payload.locale or "en-US"
        limit = payload.limit or 10
        if "timed_signals" in payload.include:
            parts.append(summarize_timed_signals(locale, limit=limit))
        if "gaps_summary" in payload.include:
            parts.append(summarize_gaps(locale, limit=limit))
        if not parts:
            logger.info(
                "executor dispatch plan_id=%s plan_type=%s outcome=ignored",
                plan.plan_id,
                plan.plan_type,
            )
            return
        message = "\n\n".join(parts)
        channels = plan.channels or [exec_context.channel_type]
        target = plan.target or exec_context.channel_target
        for channel in channels:
            self._dispatch_message(
                channel=channel,
                target=target,
                message=message,
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
            "executor dispatch plan_id=%s plan_type=%s channel=%s target=%s locale=%s",
            plan.plan_id,
            plan.plan_type,
            channel,
            target or "none",
            plan.payload.get("locale") if isinstance(plan.payload, dict) else None,
        )
        payload = _message_payload(message, channel, target, exec_context)
        if isinstance(plan.payload, dict) and plan.payload.get("locale"):
            payload["locale"] = plan.payload.get("locale")
        action = ActionResult(
            intention_key="MESSAGE_READY", payload=payload, urgency="normal"
        )
        delivery = self._coordinator.deliver(action, context)
        if delivery:
            self._extremities.dispatch(delivery, None)

    def _render_schedule_confirmation(self, payload: ScheduleTimedSignalPayload) -> str:
        principal_id = get_or_create_principal_for_channel(
            str(payload.origin),
            str(payload.chat_id),
        )
        locale = payload.locale_hint or settings.get_default_locale()
        address_style = settings.get_address_style()
        tone = settings.get_tone()
        if principal_id:
            locale = get_with_fallback(principal_id, "locale", locale)
            address_style = get_with_fallback(
                principal_id, "address_style", address_style
            )
            tone = get_with_fallback(principal_id, "tone", tone)
        logger.info(
            "executor schedule confirmation locale=%s address=%s",
            locale,
            address_style,
        )
        return render_message(
            "ack.reminder_scheduled",
            locale,
            {"address_style": address_style, "tone": tone},
        )

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

    def _execute_lan_arm(
        self,
        plan: CortexPlan,
        payload: LanArmPayload,
        context: dict,
        exec_context: PlanExecutionContext,
    ) -> None:
        device = get_paired_device(payload.device_id) if payload.device_id else get_latest_paired_device()
        if not device:
            message = _arm_message("not_found", payload.locale)
            self._dispatch_message(
                channel=exec_context.channel_type,
                target=exec_context.channel_target,
                message=message,
                context=context,
                exec_context=exec_context,
                plan=plan,
            )
            return
        if plan.plan_type == PlanType.LAN_ARM:
            arm_device(device.device_id, armed_by=str(exec_context.channel_target or "telegram"))
            status = _arm_message("armed", payload.locale)
        else:
            disarm_device(device.device_id)
            status = _arm_message("disarmed", payload.locale)
        name = device.device_name or "Unnamed device"
        message = f"{status} {name} ({device.device_id})"
        self._dispatch_message(
            channel=exec_context.channel_type,
            target=exec_context.channel_target,
            message=message,
            context=context,
            exec_context=exec_context,
            plan=plan,
        )

    def _execute_pairing_decision(
        self,
        plan: CortexPlan,
        payload: PairingDecisionPayload,
        context: dict,
        exec_context: PlanExecutionContext,
    ) -> None:
        request = get_pairing_request(payload.pairing_id)
        if not request:
            message = "Pairing not found."
        elif request.status != "pending":
            message = f"Pairing already {request.status}."
        elif plan.plan_type == PlanType.PAIR_APPROVE:
            if not payload.otp:
                message = "Missing OTP."
            else:
                approved = approve_pairing(payload.pairing_id, payload.otp, exec_context.channel_type)
                message = "✅ Pairing approved." if approved else "Invalid OTP or expired."
        else:
            denied = deny_pairing(payload.pairing_id, exec_context.channel_type)
            message = "Pairing denied." if denied else "Pairing already resolved."
        self._dispatch_message(
            channel=exec_context.channel_type,
            target=exec_context.channel_target,
            message=message,
            context=context,
            exec_context=exec_context,
            plan=plan,
        )

    def _execute_capability_gap(self, payload: CapabilityGapPayload) -> None:
        record = payload.model_dump()
        insert_gap(record)
        logger.info(
            "executor dispatch plan_type=%s outcome=gap_written reason=%s",
            PlanType.CAPABILITY_GAP,
            payload.reason,
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
        self._record_policy_gap(decision, exec_context, plan)
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

    def _record_policy_gap(
        self,
        decision: PolicyDecision,
        exec_context: PlanExecutionContext,
        plan: CortexPlan,
    ) -> None:
        channel_type = exec_context.channel_type
        channel_id = exec_context.channel_target
        principal_id = None
        if channel_type and channel_id:
            principal_id = get_or_create_principal_for_channel(channel_type, channel_id)
        user_text = None
        if isinstance(plan.payload, dict):
            user_text = (
                plan.payload.get("reminder_text_raw")
                or plan.payload.get("reminder_text")
                or plan.payload.get("message")
            )
        insert_gap(
            {
                "user_text": str(user_text or ""),
                "reason": "policy_denied",
                "status": "open",
                "intent": str(plan.plan_type),
                "confidence": None,
                "missing_slots": None,
                "principal_type": "channel_chat",
                "principal_id": principal_id,
                "channel_type": channel_type,
                "channel_id": channel_id,
                "correlation_id": exec_context.correlation_id,
                "metadata": {"policy_reason": decision.reason},
            }
        )


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


def _arm_message(kind: str, locale: str | None) -> str:
    language = "es" if str(locale or "").lower().startswith("es") else "en"
    if kind == "not_found":
        return "No paired devices found." if language == "en" else "No se encontraron dispositivos emparejados."
    if kind == "disarmed":
        return "🔒 Disarmed device" if language == "en" else "🔒 Dispositivo desarmado"
    return "✅ Armed device" if language == "en" else "✅ Dispositivo armado"
