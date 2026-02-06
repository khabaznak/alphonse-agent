from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from alphonse.agent.cognition.plans import CortexPlan, CortexResult, PlanType
from alphonse.agent.cognition.capability_gaps.triage import detect_language
from alphonse.agent.cognition.localization import render_message
from alphonse.agent.cognition.status_summaries import summarize_capabilities
from alphonse.agent.cognition.providers.ollama import OllamaClient
from alphonse.agent.cognition.planning import PlanningMode, parse_planning_mode
from alphonse.agent.cognition.planning_engine import propose_plan
from alphonse.agent.cognition.preferences.store import (
    get_or_create_principal_for_channel,
    get_with_fallback,
)
from alphonse.agent.core.settings_store import get_timezone
from alphonse.agent.cortex.intent import (
    IntentResult,
    build_intent_evidence,
    classify_intent,
    extract_preference_updates,
    extract_reminder_text,
    parse_trigger_time,
    pairing_command_intent,
)
from alphonse.config import settings

logger = logging.getLogger(__name__)


class CortexState(TypedDict, total=False):
    chat_id: str
    channel_type: str
    channel_target: str
    actor_person_id: str | None
    incoming_text: str
    last_user_message: str
    intent: str | None
    intent_confidence: float
    slots: dict[str, Any]
    missing_slots: list[str]
    pending_intent: str | None
    messages: list[dict[str, str]]
    response_text: str | None
    response_key: str | None
    response_vars: dict[str, Any] | None
    timezone: str
    intent_evidence: dict[str, Any]
    correlation_id: str | None
    plans: list[dict[str, Any]]
    last_intent: str | None
    locale: str | None
    autonomy_level: float | None
    planning_mode: str | None


def build_cortex_graph(llm_client: OllamaClient | None = None) -> StateGraph:
    graph = StateGraph(CortexState)
    graph.add_node("ingest_node", _ingest_node)
    graph.add_node("intent_node", _intent_node(llm_client))
    graph.add_node("slot_fill_node", _slot_fill_node)
    graph.add_node("clarify_node", _clarify_node)
    graph.add_node("plan_node", _plan_node)
    graph.add_node("respond_node", _respond_node)

    graph.set_entry_point("ingest_node")
    graph.add_edge("ingest_node", "intent_node")
    graph.add_conditional_edges(
        "intent_node",
        _route_after_intent,
        {
            "schedule_reminder": "slot_fill_node",
            "update_preferences": "plan_node",
            "respond": "respond_node",
        },
    )
    graph.add_conditional_edges(
        "slot_fill_node",
        _route_after_slots,
        {
            "clarify": "clarify_node",
            "plan": "plan_node",
        },
    )
    graph.add_edge("clarify_node", "respond_node")
    graph.add_edge("plan_node", "respond_node")
    graph.add_edge("respond_node", END)
    return graph


def invoke_cortex(
    state: dict[str, Any], text: str, *, llm_client: OllamaClient | None = None
) -> CortexResult:
    graph = build_cortex_graph(llm_client).compile()
    result_state = graph.invoke({**state, "incoming_text": text})
    plans = [
        CortexPlan.model_validate(plan) for plan in result_state.get("plans") or []
    ]
    response_text = result_state.get("response_text")
    if not response_text and result_state.get("response_key"):
        response_text = render_message(
            str(result_state.get("response_key")),
            _locale_for_state(result_state),
            result_state.get("response_vars") or {},
        )
    return CortexResult(
        reply_text=response_text,
        plans=plans,
        cognition_state=_build_cognition_state(result_state),
        meta=_build_meta(result_state),
    )


def _ingest_node(state: CortexState) -> dict[str, Any]:
    text = state.get("incoming_text", "").strip()
    messages = list(state.get("messages") or [])
    if text:
        messages.append({"role": "user", "content": text})
    messages = messages[-8:]
    locale = _locale_for_text(text)
    logger.info("cortex ingest chat_id=%s text=%s", state.get("chat_id"), text)
    return {
        "last_user_message": text,
        "messages": messages,
        "timezone": state.get("timezone") or get_timezone(),
        "response_text": None,
        "response_key": None,
        "response_vars": None,
        "plans": [],
        "locale": locale,
    }


def _intent_node(llm_client: OllamaClient | None):
    def _node(state: CortexState) -> dict[str, Any]:
        pairing = pairing_command_intent(state.get("last_user_message", ""))
        if pairing is not None:
            evidence = build_intent_evidence(state.get("last_user_message", ""))
            logger.info(
                "cortex intent chat_id=%s intent=%s confidence=%.2f",
                state.get("chat_id"),
                pairing,
                0.9,
            )
            return {
                "intent": pairing,
                "intent_confidence": 0.9,
                "intent_evidence": evidence,
                "last_intent": pairing,
                "pending_intent": None,
                "missing_slots": [],
                "slots": {},
            }
        pending = state.get("pending_intent")
        missing = state.get("missing_slots") or []
        if pending and missing:
            intent = pending
            confidence = state.get("intent_confidence", 0.6)
        else:
            result: IntentResult = classify_intent(
                state.get("last_user_message", ""), llm_client
            )
            intent = result.intent
            confidence = result.confidence
        evidence = build_intent_evidence(state.get("last_user_message", ""))
        logger.info(
            "cortex intent chat_id=%s intent=%s confidence=%.2f",
            state.get("chat_id"),
            intent,
            confidence,
        )
        return {
            "intent": intent,
            "intent_confidence": confidence,
            "intent_evidence": evidence,
            "last_intent": intent,
        }

    return _node


def _slot_fill_node(state: CortexState) -> dict[str, Any]:
    if state.get("intent") != "schedule_reminder":
        return {}
    if state.get("pending_intent") != "schedule_reminder":
        slots: dict[str, Any] = {}
    else:
        slots = dict(state.get("slots") or {})
    text = state.get("last_user_message", "")
    reminder_text = slots.get("reminder_text")
    trigger_time = slots.get("trigger_time")
    if not reminder_text:
        reminder_text = extract_reminder_text(text) or reminder_text
    if not reminder_text and state.get("pending_intent") == "schedule_reminder":
        reminder_text = text or reminder_text
    if not trigger_time:
        trigger_time = parse_trigger_time(text, state.get("timezone") or get_timezone())
    slots["reminder_text"] = reminder_text
    slots["trigger_time"] = trigger_time
    missing = []
    if not reminder_text:
        missing.append("reminder_text")
    if not trigger_time:
        missing.append("trigger_time")
    pending_intent = "schedule_reminder" if missing else None
    logger.info(
        "cortex slots chat_id=%s missing=%s",
        state.get("chat_id"),
        ",".join(missing) or "none",
    )
    return {
        "slots": slots,
        "missing_slots": missing,
        "pending_intent": pending_intent,
    }


def _clarify_node(state: CortexState) -> dict[str, Any]:
    missing = state.get("missing_slots") or []
    if "reminder_text" in missing:
        response_key = "clarify.reminder_text"
    else:
        response_key = "clarify.trigger_time"
    plan = _build_gap_plan(
        state,
        reason="missing_slots",
        missing_slots=missing,
    )
    plans = list(state.get("plans") or [])
    plans.append(plan.model_dump())
    return {"response_key": response_key, "plans": plans}


def _plan_node(state: CortexState) -> dict[str, Any]:
    if state.get("intent") == "update_preferences":
        updates = extract_preference_updates(state.get("last_user_message", ""))
        if not updates:
            return {"response_key": "preference.missing"}
        channel_type = state.get("channel_type")
        channel_id = state.get("channel_target") or state.get("chat_id")
        if not channel_type or not channel_id:
            return {"response_key": "preference.no_channel"}
        plan = CortexPlan(
            plan_type=PlanType.UPDATE_PREFERENCES,
            payload={
                "principal": {
                    "channel_type": str(channel_type),
                    "channel_id": str(channel_id),
                },
                "updates": updates,
            },
        )
        plans = list(state.get("plans") or [])
        _attach_planning_scaffold(
            state,
            intent="update_preferences",
            draft_steps=[
                "Apply the requested preference updates.",
                "Confirm which preferences were updated.",
            ],
            acceptance_criteria=[
                "All requested preferences are updated for the principal.",
            ],
            plans=plans,
        )
        plans.append(plan.model_dump())
        logger.info(
            "cortex plans chat_id=%s plan_type=%s plan_id=%s",
            state.get("chat_id"),
            plan.plan_type,
            plan.plan_id,
        )
        return {
            "plans": plans,
            "response_key": "ack.preference_updated",
            "response_vars": {"updates": updates},
        }
    slots = state.get("slots") or {}
    reminder_text = slots.get("reminder_text")
    trigger_time = slots.get("trigger_time")
    if not reminder_text or not trigger_time:
        return {"response_key": "generic.unknown"}
    plan = CortexPlan(
        plan_type=PlanType.SCHEDULE_TIMED_SIGNAL,
        target=str(state.get("channel_target") or state.get("chat_id") or ""),
        channels=None,
        payload={
            "signal_type": "reminder",
            "trigger_at": str(trigger_time),
            "timezone": str(state.get("timezone") or get_timezone()),
            "reminder_text": str(reminder_text),
            "reminder_text_raw": str(reminder_text),
            "origin": str(state.get("channel_type") or "system"),
            "chat_id": str(state.get("channel_target") or state.get("chat_id") or ""),
            "origin_channel": str(state.get("channel_type") or "system"),
            "locale_hint": _preferred_locale_hint(state),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "correlation_id": str(
                state.get("correlation_id") or state.get("chat_id") or ""
            ),
        },
    )
    plans = list(state.get("plans") or [])
    _attach_planning_scaffold(
        state,
        intent="schedule_reminder",
        draft_steps=[
            "Confirm the reminder message and time.",
            "Schedule the reminder.",
            "Send a confirmation.",
        ],
        acceptance_criteria=[
            "Reminder is scheduled at the agreed time.",
            "User receives a confirmation message.",
        ],
        plans=plans,
    )
    plans.append(plan.model_dump())
    logger.info(
        "cortex plans chat_id=%s plan_type=%s plan_id=%s",
        state.get("chat_id"),
        plan.plan_type,
        plan.plan_id,
    )
    return {
        "plans": plans,
        "pending_intent": None,
        "missing_slots": [],
    }


def _respond_node(state: CortexState) -> dict[str, Any]:
    if state.get("response_text") or state.get("response_key"):
        return {}
    intent = state.get("intent")
    if intent in {"schedule_reminder", "greeting"} and state.get("plans"):
        return {}
    if intent == "get_status":
        response_key = "status"
    elif intent == "help":
        response_key = "help"
    elif intent == "identity_question":
        response_key = "identity"
    elif intent == "greeting":
        response_key = "greeting"
    elif intent == "meta.capabilities":
        return {"response_text": summarize_capabilities(_locale_for_state(state))}
    elif intent == "meta.gaps_list":
        plan = CortexPlan(
            plan_type=PlanType.QUERY_STATUS,
            target=str(state.get("channel_target") or state.get("chat_id") or ""),
            channels=[str(state.get("channel_type") or "telegram")],
            payload={
                "include": ["gaps_summary"],
                "limit": 5,
                "locale": _locale_for_state(state),
            },
        )
        plans = list(state.get("plans") or [])
        plans.append(plan.model_dump())
        logger.info(
            "cortex plans chat_id=%s plan_type=%s plan_id=%s",
            state.get("chat_id"),
            plan.plan_type,
            plan.plan_id,
        )
        return {"plans": plans}
    elif intent == "timed_signals.list":
        plan = CortexPlan(
            plan_type=PlanType.QUERY_STATUS,
            target=str(state.get("channel_target") or state.get("chat_id") or ""),
            channels=[str(state.get("channel_type") or "telegram")],
            payload={
                "include": ["timed_signals"],
                "limit": 10,
                "locale": _locale_for_state(state),
            },
        )
        plans = list(state.get("plans") or [])
        plans.append(plan.model_dump())
        logger.info(
            "cortex plans chat_id=%s plan_type=%s plan_id=%s",
            state.get("chat_id"),
            plan.plan_type,
            plan.plan_id,
        )
        return {"plans": plans}
    elif intent in {"lan.arm", "lan.disarm"}:
        device_id = _extract_device_id(state.get("last_user_message", ""))
        plan = CortexPlan(
            plan_type=PlanType.LAN_ARM if intent == "lan.arm" else PlanType.LAN_DISARM,
            target=str(state.get("channel_target") or state.get("chat_id") or ""),
            channels=[str(state.get("channel_type") or "telegram")],
            payload={
                "device_id": device_id,
                "locale": _locale_for_state(state),
            },
        )
        plans = list(state.get("plans") or [])
        plans.append(plan.model_dump())
        logger.info(
            "cortex plans chat_id=%s plan_type=%s plan_id=%s",
            state.get("chat_id"),
            plan.plan_type,
            plan.plan_id,
        )
        return {"plans": plans}
    elif intent in {"pair.approve", "pair.deny"}:
        pairing_id, otp = _extract_pairing_decision(state.get("last_user_message", ""))
        if not pairing_id:
            return {"response_key": "generic.unknown"}
        plan = CortexPlan(
            plan_type=PlanType.PAIR_APPROVE if intent == "pair.approve" else PlanType.PAIR_DENY,
            target=str(state.get("channel_target") or state.get("chat_id") or ""),
            channels=[str(state.get("channel_type") or "telegram")],
            payload={
                "pairing_id": pairing_id,
                "otp": otp,
                "locale": _locale_for_state(state),
            },
        )
        plans = list(state.get("plans") or [])
        plans.append(plan.model_dump())
        logger.info(
            "cortex plans chat_id=%s plan_type=%s plan_id=%s",
            state.get("chat_id"),
            plan.plan_type,
            plan.plan_id,
        )
        return {"plans": plans}
    else:
        response_key = "generic.unknown"
    result: dict[str, Any] = {"response_key": response_key}
    if response_key == "generic.unknown":
        reason = "unknown_intent" if intent == "unknown" else "no_tool"
        plan = _build_gap_plan(state, reason=reason)
        plans = list(state.get("plans") or [])
        plans.append(plan.model_dump())
        result["plans"] = plans
    return result


def _route_after_intent(state: CortexState) -> str:
    if state.get("intent") == "schedule_reminder":
        return "schedule_reminder"
    if state.get("intent") == "update_preferences":
        return "update_preferences"
    return "respond"


def _route_after_slots(state: CortexState) -> str:
    missing = state.get("missing_slots") or []
    return "clarify" if missing else "plan"


def _preferred_locale_hint(state: CortexState) -> str | None:
    locale = state.get("locale")
    if isinstance(locale, str) and locale:
        return locale
    channel_type = state.get("channel_type")
    channel_id = state.get("channel_target") or state.get("chat_id")
    if not channel_type or not channel_id:
        return None
    principal_id = get_or_create_principal_for_channel(
        str(channel_type), str(channel_id)
    )
    if not principal_id:
        return None
    return get_with_fallback(principal_id, "locale", settings.get_default_locale())


def _build_gap_plan(
    state: CortexState,
    *,
    reason: str,
    missing_slots: list[str] | None = None,
) -> CortexPlan:
    channel_type = state.get("channel_type")
    channel_id = state.get("channel_target") or state.get("chat_id")
    principal_id = None
    if channel_type and channel_id:
        principal_id = get_or_create_principal_for_channel(
            str(channel_type), str(channel_id)
        )
    return CortexPlan(
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


def _attach_planning_scaffold(
    state: CortexState,
    *,
    intent: str,
    draft_steps: list[str],
    acceptance_criteria: list[str],
    plans: list[dict[str, Any]],
) -> None:
    requested_mode = _planning_mode_request(state)
    proposal = propose_plan(
        intent=intent,
        autonomy_level=state.get("autonomy_level"),
        requested_mode=requested_mode,
        draft_steps=draft_steps,
        acceptance_criteria=acceptance_criteria,
    )
    scaffold = CortexPlan(
        plan_type=PlanType.PLANNING,
        payload={
            "plan": proposal.plan.model_dump(),
            "mode": proposal.plan.planning_mode.value,
            "autonomy_level": proposal.plan.autonomy_level,
        },
    )
    plans.append(scaffold.model_dump())
    logger.info(
        "cortex planning scaffold chat_id=%s mode=%s autonomy=%.2f plan_id=%s",
        state.get("chat_id"),
        proposal.plan.planning_mode.value,
        proposal.plan.autonomy_level,
        scaffold.plan_id,
    )


def _planning_mode_request(state: CortexState) -> PlanningMode | None:
    raw = state.get("planning_mode")
    if not raw:
        return None
    if isinstance(raw, PlanningMode):
        return raw
    return parse_planning_mode(str(raw))


def _build_cognition_state(state: CortexState) -> dict[str, Any]:
    return {
        "pending_intent": state.get("pending_intent"),
        "slots_collected": state.get("slots") or {},
        "missing_slots": state.get("missing_slots") or [],
        "last_intent": state.get("intent"),
        "locale": state.get("locale"),
        "autonomy_level": state.get("autonomy_level"),
        "planning_mode": state.get("planning_mode"),
        "last_updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _build_meta(state: CortexState) -> dict[str, Any]:
    return {
        "intent": state.get("intent"),
        "intent_confidence": state.get("intent_confidence"),
        "correlation_id": state.get("correlation_id"),
        "chat_id": state.get("chat_id"),
        "response_key": state.get("response_key"),
        "response_vars": state.get("response_vars"),
        "autonomy_level": state.get("autonomy_level"),
        "planning_mode": state.get("planning_mode"),
    }


def _extract_device_id(text: str) -> str | None:
    candidate = re.search(r"\bdevice\s+([A-Za-z0-9_-]{4,})\b", text, re.IGNORECASE)
    if candidate:
        return candidate.group(1)
    return None


def _extract_pairing_decision(text: str) -> tuple[str | None, str | None]:
    parts = text.strip().split()
    if not parts:
        return None, None
    if parts[0].startswith("/"):
        parts = parts[1:]
    if not parts:
        return None, None
    pairing_id = parts[0]
    otp = parts[1] if len(parts) > 1 else None
    return pairing_id, otp


def _locale_for_text(text: str) -> str:
    language = detect_language(text)
    return "es-MX" if language == "es" else "en-US"


def _locale_for_state(state: CortexState) -> str:
    locale = state.get("locale")
    if isinstance(locale, str) and locale:
        return locale
    return "en-US"
