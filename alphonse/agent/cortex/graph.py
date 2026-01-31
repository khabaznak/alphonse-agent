from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from alphonse.agent.cognition.providers.ollama import OllamaClient
from alphonse.agent.core.settings_store import get_timezone
from alphonse.agent.cortex.intent import (
    IntentResult,
    build_intent_evidence,
    classify_intent,
    extract_reminder_text,
    parse_trigger_time,
)
from alphonse.agent.extremities.scheduler_extremity import schedule_reminder

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
    timezone: str
    intent_evidence: dict[str, Any]


def build_cortex_graph(llm_client: OllamaClient | None = None) -> StateGraph:
    graph = StateGraph(CortexState)
    graph.add_node("ingest", _ingest_node)
    graph.add_node("intent", _intent_node(llm_client))
    graph.add_node("slot_fill", _slot_fill_node)
    graph.add_node("clarify", _clarify_node)
    graph.add_node("execute", _execute_node)
    graph.add_node("respond", _respond_node)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "intent")
    graph.add_conditional_edges(
        "intent",
        _route_after_intent,
        {
            "schedule_reminder": "slot_fill",
            "respond": "respond",
        },
    )
    graph.add_conditional_edges(
        "slot_fill",
        _route_after_slots,
        {
            "clarify": "clarify",
            "execute": "execute",
        },
    )
    graph.add_edge("clarify", "respond")
    graph.add_edge("execute", "respond")
    graph.add_edge("respond", END)
    return graph


def invoke_cortex(state: dict[str, Any], text: str, *, llm_client: OllamaClient | None = None) -> dict[str, Any]:
    graph = build_cortex_graph(llm_client).compile()
    return graph.invoke({**state, "incoming_text": text})


def _ingest_node(state: CortexState) -> dict[str, Any]:
    text = state.get("incoming_text", "").strip()
    messages = list(state.get("messages") or [])
    if text:
        messages.append({"role": "user", "content": text})
    messages = messages[-8:]
    logger.info("cortex ingest chat_id=%s text=%s", state.get("chat_id"), text)
    return {
        "last_user_message": text,
        "messages": messages,
        "timezone": state.get("timezone") or get_timezone(),
        "response_text": None,
    }


def _intent_node(llm_client: OllamaClient | None):
    def _node(state: CortexState) -> dict[str, Any]:
        pending = state.get("pending_intent")
        missing = state.get("missing_slots") or []
        if pending and missing:
            intent = pending
            confidence = state.get("intent_confidence", 0.6)
        else:
            result: IntentResult = classify_intent(state.get("last_user_message", ""), llm_client)
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
        }

    return _node


def _slot_fill_node(state: CortexState) -> dict[str, Any]:
    if state.get("intent") != "schedule_reminder":
        return {}
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
        response = "¿Qué debo recordarte? Por ejemplo: \"llamar a mamá\"."
    else:
        response = "¿Cuándo debo recordarlo? Ejemplo: \"en 10 min\" o \"a las 7pm\"."
    return {"response_text": response}


def _execute_node(state: CortexState) -> dict[str, Any]:
    slots = state.get("slots") or {}
    reminder_text = slots.get("reminder_text")
    trigger_time = slots.get("trigger_time")
    if not reminder_text or not trigger_time:
        return {"response_text": "¿Puedes aclarar qué necesitas?"}
    result = schedule_reminder(
        reminder_text=str(reminder_text),
        trigger_time=str(trigger_time),
        chat_id=str(state.get("channel_target") or state.get("chat_id")),
        channel_type=str(state.get("channel_type")),
        actor_person_id=state.get("actor_person_id"),
        intent_evidence=state.get("intent_evidence") or {},
        correlation_id=str(state.get("chat_id")),
    )
    logger.info("cortex execute reminder status=%s", result)
    return {
        "response_text": f"Programé el recordatorio para {trigger_time}.",
        "pending_intent": None,
        "missing_slots": [],
    }


def _respond_node(state: CortexState) -> dict[str, Any]:
    if state.get("response_text"):
        return {}
    intent = state.get("intent")
    if intent == "get_status":
        response = "Estoy activo y listo para recordatorios."
    elif intent == "help":
        response = "Puedo programar recordatorios. Ejemplo: \"Recuérdame tomar agua en 10 min\"."
    elif intent == "identity_question":
        response = "Soy Alphonse, tu asistente. Solo conozco este chat autorizado."
    elif intent == "greeting":
        response = "¡Hola! ¿En qué te ayudo?"
    else:
        response = "Puedo programar recordatorios. Prueba: \"Recuérdame X en 10 min\"."
    return {"response_text": response}


def _route_after_intent(state: CortexState) -> str:
    return "schedule_reminder" if state.get("intent") == "schedule_reminder" else "respond"


def _route_after_slots(state: CortexState) -> str:
    missing = state.get("missing_slots") or []
    return "clarify" if missing else "execute"
