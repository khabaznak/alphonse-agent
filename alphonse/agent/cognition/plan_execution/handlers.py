from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from alphonse.agent.cognition.plans import CortexPlan
from alphonse.agent.cognition.preferences.store import (
    get_or_create_principal_for_channel,
    set_preference,
)
from alphonse.agent.cognition.status_summaries import summarize_gaps, summarize_timed_signals
from alphonse.agent.lan.pairing_store import approve_pairing, deny_pairing, get_pairing_request
from alphonse.agent.lan.store import arm_device, disarm_device, get_latest_paired_device, get_paired_device
from alphonse.agent.nervous_system.capability_gaps import insert_gap
from alphonse.agent.tools.scheduler_tool import SchedulerTool


DispatchMessageFn = Callable[..., None]


class CommunicatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str
    style: str | None = None
    locale: str | None = None


class ScheduleTimedSignalPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    signal_type: str
    trigger_at: str
    timezone: str
    reminder_text: str
    reminder_text_raw: str | None = None
    origin: str
    chat_id: str
    origin_channel: str | None = None
    locale_hint: str | None = None
    created_at: str | None = None
    correlation_id: str


class QueryStatusPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include: list[str]
    limit: int | None = None
    locale: str | None = None


class PreferencePrincipal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    channel_type: str
    channel_id: str


class PreferenceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str
    value: Any


class UpdatePreferencesPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    principal: PreferencePrincipal
    updates: list[PreferenceUpdate]


class CapabilityGapPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_text: str
    reason: str
    status: str = "open"
    intent: str | None = None
    confidence: float | None = None
    missing_slots: list[str] | None = None
    principal_type: str | None = None
    principal_id: str | None = None
    channel_type: str | None = None
    channel_id: str | None = None
    correlation_id: str | None = None
    metadata: dict[str, Any] | None = None


class LanArmPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device_id: str | None = None
    locale: str | None = None


class PairingDecisionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pairing_id: str
    otp: str | None = None
    locale: str | None = None


class PlanningPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan: dict[str, Any]
    mode: str
    autonomy_level: float
def execute_communicate(
    *,
    plan: CortexPlan,
    payload: CommunicatePayload,
    context: dict[str, Any],
    exec_context: Any,
    dispatch_message: DispatchMessageFn,
) -> None:
    channels = plan.channels or [exec_context.channel_type]
    target = plan.target or exec_context.channel_target
    for channel in channels:
        dispatch_message(
            channel=channel,
            target=target,
            message=payload.message,
            context=context,
            exec_context=exec_context,
            plan=plan,
        )


def execute_plan(
    *,
    logger: logging.Logger,
    plan: CortexPlan,
    context: dict[str, Any],
    exec_context: Any,
    dispatch_message: DispatchMessageFn,
) -> bool:
    tool = _tool_key(plan)
    params = _plan_params(plan)
    if tool == "communicate":
        payload = CommunicatePayload.model_validate(params)
        execute_communicate(
            plan=plan,
            payload=payload,
            context=context,
            exec_context=exec_context,
            dispatch_message=dispatch_message,
        )
        return True
    if tool == "schedule_timed_signal":
        payload = ScheduleTimedSignalPayload.model_validate(params)
        execute_schedule(
            logger=logger,
            plan=plan,
            payload=payload,
            context=context,
            exec_context=exec_context,
            dispatch_message=dispatch_message,
        )
        return True
    if tool == "query_status":
        payload = QueryStatusPayload.model_validate(params)
        execute_query_status(
            logger=logger,
            plan=plan,
            payload=payload,
            context=context,
            exec_context=exec_context,
            dispatch_message=dispatch_message,
        )
        return True
    if tool == "update_preferences":
        payload = UpdatePreferencesPayload.model_validate(params)
        execute_update_preferences(
            logger=logger,
            plan=plan,
            payload=payload,
        )
        return True
    if tool == "capability_gap":
        payload = CapabilityGapPayload.model_validate(params)
        execute_capability_gap(
            logger=logger,
            payload=payload,
        )
        return True
    if tool == "planning":
        payload = PlanningPayload.model_validate(params)
        execute_planning(
            logger=logger,
            payload=payload,
        )
        return True
    if tool in {"lan_arm", "lan_disarm"}:
        payload = LanArmPayload.model_validate(params)
        execute_lan_arm(
            plan=plan,
            payload=payload,
            context=context,
            exec_context=exec_context,
            dispatch_message=dispatch_message,
        )
        return True
    if tool in {"pair_approve", "pair_deny"}:
        payload = PairingDecisionPayload.model_validate(params)
        execute_pairing_decision(
            plan=plan,
            payload=payload,
            context=context,
            exec_context=exec_context,
            dispatch_message=dispatch_message,
        )
        return True
    return False


def execute_schedule(
    *,
    logger: logging.Logger,
    plan: CortexPlan,
    payload: ScheduleTimedSignalPayload,
    context: dict[str, Any],
    exec_context: Any,
    dispatch_message: DispatchMessageFn,
) -> None:
    now_utc = datetime.now(tz=timezone.utc)
    trigger_at = _parse_iso_datetime(payload.trigger_at)
    eta_seconds = None
    if trigger_at is not None:
        eta_seconds = int((trigger_at - now_utc).total_seconds())
    logger.info(
        "executor dispatch plan_id=%s tool=%s trigger_at=%s now_utc=%s eta_seconds=%s",
        plan.plan_id,
        _tool_key(plan),
        payload.trigger_at,
        now_utc.isoformat(),
        eta_seconds if eta_seconds is not None else "unknown",
    )
    schedule_payload = {
        "prompt": payload.reminder_text,
        "message": payload.reminder_text,
        "reminder_text_raw": payload.reminder_text,
        "person_id": exec_context.actor_person_id or payload.chat_id,
        "chat_id": payload.chat_id,
        "origin_channel": payload.origin,
        "locale_hint": payload.locale_hint,
        "intent_evidence": {},
    }
    result = SchedulerTool().schedule_event(
        trigger_time=payload.trigger_at,
        timezone_name=payload.timezone,
        payload=schedule_payload,
        target=str(exec_context.actor_person_id or payload.chat_id),
        origin=payload.origin,
        correlation_id=payload.correlation_id,
    )
    logger.info(
        "executor dispatch plan_id=%s tool=%s outcome=scheduled schedule_id=%s",
        plan.plan_id,
        _tool_key(plan),
        result,
    )
    confirmation = "ack.reminder_scheduled"
    dispatch_message(
        channel=payload.origin,
        target=payload.chat_id,
        message=confirmation,
        context=context,
        exec_context=exec_context,
        plan=plan,
    )


def execute_query_status(
    *,
    logger: logging.Logger,
    plan: CortexPlan,
    payload: QueryStatusPayload,
    context: dict[str, Any],
    exec_context: Any,
    dispatch_message: DispatchMessageFn,
) -> None:
    logger.info(
        "executor dispatch plan_id=%s tool=%s include=%s",
        plan.plan_id,
        _tool_key(plan),
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
            "executor dispatch plan_id=%s tool=%s outcome=ignored",
            plan.plan_id,
            _tool_key(plan),
        )
        return
    message = "\n\n".join(parts)
    channels = plan.channels or [exec_context.channel_type]
    target = plan.target or exec_context.channel_target
    for channel in channels:
        dispatch_message(
            channel=channel,
            target=target,
            message=message,
            context=context,
            exec_context=exec_context,
            plan=plan,
        )


def execute_update_preferences(
    *,
    logger: logging.Logger,
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
            "executor dispatch plan_id=%s tool=%s outcome=missing_principal",
            plan.plan_id,
            _tool_key(plan),
        )
        return
    for update in payload.updates:
        set_preference(principal_id, update.key, update.value, source="user")
    logger.info(
        "executor dispatch plan_id=%s tool=%s outcome=preferences_updated count=%s",
        plan.plan_id,
        _tool_key(plan),
        len(payload.updates),
    )


def execute_lan_arm(
    *,
    plan: CortexPlan,
    payload: LanArmPayload,
    context: dict[str, Any],
    exec_context: Any,
    dispatch_message: DispatchMessageFn,
) -> None:
    device = get_paired_device(payload.device_id) if payload.device_id else get_latest_paired_device()
    if not device:
        message = "lan.device.not_found"
        dispatch_message(
            channel=exec_context.channel_type,
            target=exec_context.channel_target,
            message=message,
            context=context,
            exec_context=exec_context,
            plan=plan,
        )
        return
    if _tool_key(plan) == "lan_arm":
        arm_device(device.device_id, armed_by=str(exec_context.channel_target or "telegram"))
        key = "lan.device.armed"
    else:
        disarm_device(device.device_id)
        key = "lan.device.disarmed"
    _ = {"device_name": device.device_name or "Unnamed device", "device_id": device.device_id}
    dispatch_message(
        channel=exec_context.channel_type,
        target=exec_context.channel_target,
        message=key,
        context=context,
        exec_context=exec_context,
        plan=plan,
    )


def execute_pairing_decision(
    *,
    plan: CortexPlan,
    payload: PairingDecisionPayload,
    context: dict[str, Any],
    exec_context: Any,
    dispatch_message: DispatchMessageFn,
) -> None:
    request = get_pairing_request(payload.pairing_id)
    if not request:
        key = "pairing.not_found"
        variables: dict[str, Any] = {}
    elif request.status != "pending":
        key = "pairing.already_resolved"
        variables = {"status": request.status}
    elif _tool_key(plan) == "pair_approve":
        if not payload.otp:
            key = "pairing.missing_otp"
            variables = {}
        else:
            approved = approve_pairing(payload.pairing_id, payload.otp, exec_context.channel_type)
            key = "pairing.approved" if approved else "pairing.invalid_otp"
            variables = {}
    else:
        denied = deny_pairing(payload.pairing_id, exec_context.channel_type)
        key = "pairing.denied" if denied else "pairing.already_resolved"
        variables = {}
    _ = variables
    dispatch_message(
        channel=exec_context.channel_type,
        target=exec_context.channel_target,
        message=key,
        context=context,
        exec_context=exec_context,
        plan=plan,
    )


def execute_capability_gap(
    *,
    logger: logging.Logger,
    payload: CapabilityGapPayload,
) -> None:
    record = payload.model_dump()
    insert_gap(record)
    logger.info(
        "executor dispatch tool=%s outcome=gap_written reason=%s",
        "capability_gap",
        payload.reason,
    )


def execute_planning(
    *,
    logger: logging.Logger,
    payload: PlanningPayload,
) -> None:
    logger.info(
        "executor dispatch tool=%s outcome=planning_scaffold mode=%s autonomy=%.2f",
        "planning",
        payload.mode,
        payload.autonomy_level,
    )


def _parse_iso_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _tool_key(plan: CortexPlan) -> str:
    key = str(getattr(plan, "tool", "") or "").strip().lower()
    return key


def _plan_params(plan: CortexPlan) -> dict[str, Any]:
    params = getattr(plan, "parameters", None)
    if isinstance(params, dict) and params:
        return dict(params)
    payload = getattr(plan, "payload", None)
    if isinstance(payload, dict):
        return dict(payload)
    return {}
