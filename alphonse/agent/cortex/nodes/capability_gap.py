from __future__ import annotations

from typing import Any, Callable

from alphonse.agent.cognition.plans import CortexPlan, PlanType


def has_capability_gap_plan(state: dict[str, Any]) -> bool:
    plans = state.get("plans")
    if not isinstance(plans, list):
        return False
    return any(
        isinstance(item, dict)
        and str(item.get("plan_type") or "") == PlanType.CAPABILITY_GAP.value
        for item in plans
    )


def build_gap_plan(
    *,
    state: dict[str, Any],
    reason: str,
    missing_slots: list[str] | None = None,
    get_or_create_principal_for_channel: Callable[[str, str], str],
) -> dict[str, Any]:
    channel_type = state.get("channel_type")
    channel_id = state.get("channel_target") or state.get("chat_id")
    principal_id = None
    if channel_type and channel_id:
        principal_id = get_or_create_principal_for_channel(
            str(channel_type), str(channel_id)
        )
    plan = CortexPlan(
        plan_type=PlanType.CAPABILITY_GAP,
        payload={
            "user_text": str(state.get("last_user_message") or ""),
            "reason": reason,
            "status": "open",
            "intent": str(state.get("intent") or ""),
            "confidence": state.get("intent_confidence"),
            "missing_slots": missing_slots,
            "principal_type": "channel_chat",
            "principal_id": principal_id,
            "channel_type": str(channel_type) if channel_type else None,
            "channel_id": str(channel_id) if channel_id else None,
            "correlation_id": state.get("correlation_id"),
            "metadata": {
                "intent_evidence": state.get("intent_evidence"),
            },
        },
    )
    return plan.model_dump()


def run_capability_gap_tool(
    *,
    state: dict[str, Any],
    llm_client: Any,
    reason: str,
    missing_slots: list[str] | None,
    append_existing_plans: bool,
    emit_transition_event: Callable[[dict[str, Any], str, dict[str, Any] | None], None],
    logger_info: Callable[[str, Any, Any, str], None],
    build_capability_gap_apology: Callable[..., str],
    get_or_create_principal_for_channel: Callable[[str, str], str],
) -> dict[str, Any]:
    emit_transition_event(state, "failed", {"reason": reason})
    logger_info(
        "cortex tool capability_gap.create chat_id=%s correlation_id=%s reason=%s",
        state.get("chat_id"),
        state.get("correlation_id"),
        reason,
    )
    plan = build_gap_plan(
        state=state,
        reason=reason,
        missing_slots=missing_slots,
        get_or_create_principal_for_channel=get_or_create_principal_for_channel,
    )
    plans = list(state.get("plans") or []) if append_existing_plans else []
    plans.append(plan)
    apology = build_capability_gap_apology(
        state=state,
        llm_client=llm_client,
        reason=reason,
        missing_slots=missing_slots,
    )
    payload: dict[str, Any] = {
        "plans": plans,
        "ability_state": {},
        "pending_interaction": None,
        "events": state.get("events") or [],
    }
    if apology:
        payload["response_text"] = apology
    return payload
