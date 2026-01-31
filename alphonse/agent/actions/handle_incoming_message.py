from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.cortex.graph import invoke_cortex
from alphonse.agent.cortex.state_store import load_state, save_state
from alphonse.agent.cognition.skills.interpretation.skills import build_ollama_client
from alphonse.agent.identity import store as identity_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IncomingContext:
    channel_type: str
    address: str | None
    person_id: str | None
    correlation_id: str
    update_id: str | None


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
        logger.info(
            "HandleIncomingMessageAction start channel=%s person=%s text=%s",
            incoming.channel_type,
            incoming.person_id,
            _snippet(text),
        )
        if not text:
            return _message_result("No te escuché bien. ¿Puedes repetir?", incoming)

        state = load_state(incoming.address or incoming.channel_type) or {}
        state.update(
            {
                "chat_id": incoming.address or incoming.channel_type,
                "channel_type": incoming.channel_type,
                "channel_target": incoming.address or incoming.channel_type,
                "actor_person_id": incoming.person_id,
            }
        )
        llm_client = _build_llm_client()
        result_state = invoke_cortex(state, text, llm_client=llm_client)
        response_text = result_state.get("response_text") or "¿En qué puedo ayudarte?"
        save_state(incoming.address or incoming.channel_type, _persisted_state(result_state))
        return _message_result(str(response_text), incoming)


def _build_incoming_context(payload: dict, signal: object | None, correlation_id: str) -> IncomingContext:
    origin = payload.get("origin") or getattr(signal, "source", None) or "system"
    channel_type = str(origin)
    address = _resolve_address(channel_type, payload)
    person_id = _resolve_person_id(payload, channel_type, address)
    update_id = payload.get("update_id") if isinstance(payload, dict) else None
    return IncomingContext(
        channel_type=channel_type,
        address=address,
        person_id=person_id,
        correlation_id=correlation_id,
        update_id=str(update_id) if update_id is not None else None,
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


def _persisted_state(state: dict[str, Any]) -> dict[str, Any]:
    persisted = dict(state)
    persisted.pop("incoming_text", None)
    persisted.pop("response_text", None)
    return persisted


def _message_result(message: str, incoming: IncomingContext) -> ActionResult:
    logger.info(
        "HandleIncomingMessageAction response channel=%s message=%s",
        incoming.channel_type,
        _snippet(message),
    )
    payload: dict[str, Any] = {
        "message": message,
        "origin": incoming.channel_type,
        "channel_hint": incoming.channel_type,
        "correlation_id": incoming.correlation_id,
        "audience": _audience_for(incoming.person_id),
    }
    if incoming.address:
        payload["target"] = incoming.address
    if incoming.channel_type == "telegram" and incoming.address:
        payload["direct_reply"] = {
            "channel_type": "telegram",
            "target": incoming.address,
            "text": message,
            "correlation_id": incoming.correlation_id,
        }
    return ActionResult(
        intention_key="MESSAGE_READY",
        payload=payload,
        urgency="normal",
    )


def _audience_for(person_id: str | None) -> dict[str, str]:
    if person_id:
        return {"kind": "person", "id": person_id}
    return {"kind": "system", "id": "system"}


def _snippet(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."


def _build_llm_client():
    try:
        return build_ollama_client()
    except Exception:
        return None
