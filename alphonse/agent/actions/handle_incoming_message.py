from __future__ import annotations

import json
import logging
import uuid

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.actions.session_context import IncomingContext
from alphonse.agent.actions.session_context import build_session_key
from alphonse.agent.actions.state_context import build_cortex_state
from alphonse.agent.actions.state_context import ensure_conversation_locale
from alphonse.agent.actions.state_context import outgoing_locale
from alphonse.agent.actions.transitions import emit_agent_transitions_from_meta
from alphonse.agent.cognition.plan_executor import PlanExecutionContext, PlanExecutor
from alphonse.agent.cognition.plans import CortexPlan, PlanType
from alphonse.agent.cognition.providers.factory import build_llm_client
from alphonse.agent.cortex.graph import CortexGraph
from alphonse.agent.cortex.state_store import load_state, save_state
from alphonse.agent.identity import store as identity_store
from alphonse.agent.io import get_io_registry

logger = logging.getLogger(__name__)
_CORTEX_GRAPH = CortexGraph()


class HandleIncomingMessageAction(Action):
    key = "handle_incoming_message"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        payload = getattr(signal, "payload", {}) if signal else {}
        correlation_id = getattr(signal, "correlation_id", None) if signal else None
        if not correlation_id and isinstance(payload, dict):
            correlation_id = payload.get("correlation_id")
        correlation_id = str(correlation_id or uuid.uuid4())

        incoming = _build_incoming_context_from_payload(payload, signal, correlation_id=correlation_id)
        provider_payload = payload.get("provider_event") if isinstance(payload.get("provider_event"), dict) else payload
        packed_input = _pack_raw_provider_event_markdown(
            channel_type=incoming.channel_type,
            payload=provider_payload,
            correlation_id=correlation_id,
        )
        logger.info(
            "HandleIncomingMessageAction start channel=%s person=%s text=%s",
            incoming.channel_type,
            incoming.person_id,
            _text_log_snippet(str(payload.get("text") or "")) if isinstance(payload, dict) else "",
        )

        session_key = build_session_key(incoming)
        logger.info(
            "HandleIncomingMessageAction session_key=%s channel=%s address=%s",
            session_key,
            incoming.channel_type,
            incoming.address,
        )

        stored_state = load_state(session_key) or {}
        ensure_conversation_locale(
            conversation_key=session_key,
            stored_state=stored_state,
            incoming=incoming,
        )
        state = build_cortex_state(
            stored_state=stored_state,
            incoming=incoming,
            correlation_id=correlation_id,
            payload=payload,
            normalized=None,
        )
        has_live_transition_sink = _attach_transition_sink(state, incoming)

        try:
            llm_client = build_llm_client()
        except Exception:
            logger.exception("HandleIncomingMessageAction failed to build llm client")
            llm_client = None
        try:
            result = _CORTEX_GRAPH.invoke(state, packed_input, llm_client=llm_client)
        except Exception:
            logger.exception(
                "HandleIncomingMessageAction cortex_invoke_failed channel=%s target=%s correlation_id=%s",
                incoming.channel_type,
                incoming.address,
                incoming.correlation_id,
            )
            raise

        if not has_live_transition_sink:
            emit_agent_transitions_from_meta(
                incoming=incoming,
                meta=result.meta if isinstance(result.meta, dict) else {},
                emit_transition=lambda ctx, phase: _emit_channel_transition(ctx, phase),
                skip_phases=set(),
            )
        logger.info(
            "HandleIncomingMessageAction cortex_result reply_len=%s plans=%s correlation_id=%s",
            len(str(result.reply_text or "")),
            len(result.plans or []),
            incoming.correlation_id,
        )
        if result.plans:
            logger.info(
                "HandleIncomingMessageAction cortex_plans correlation_id=%s plan_types=%s",
                incoming.correlation_id,
                [str(getattr(plan, "plan_type", "unknown")) for plan in result.plans],
            )

        save_state(session_key, result.cognition_state)
        if isinstance(result.cognition_state, dict):
            logger.info(
                "HandleIncomingMessageAction saved_state pending_interaction=%s",
                result.cognition_state.get("pending_interaction"),
            )

        executor = PlanExecutor()
        exec_context = PlanExecutionContext(
            channel_type=incoming.channel_type,
            channel_target=incoming.address,
            actor_person_id=incoming.person_id,
            correlation_id=incoming.correlation_id,
        )
        if result.reply_text:
            locale = outgoing_locale(result.cognition_state)
            reply_plan = CortexPlan(
                plan_type=PlanType.COMMUNICATE,
                payload={
                    "message": str(result.reply_text),
                    "locale": locale,
                },
            )
            executor.execute([reply_plan], context, exec_context)
        if result.plans:
            executor.execute(result.plans, context, exec_context)

        logger.info(
            "HandleIncomingMessageAction response channel=%s message=noop",
            incoming.channel_type,
        )
        return ActionResult(intention_key="NOOP", payload={}, urgency=None)


def _text_log_snippet(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."


def _pack_raw_provider_event_markdown(*, channel_type: str, payload: dict[str, object], correlation_id: str) -> str:
    return (
        "# Incoming Provider Event\n"
        f"- channel: {channel_type}\n"
        f"- correlation_id: {correlation_id}\n\n"
        "## RAW JSON\n"
        "```json\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        "```\n"
    )


def _build_incoming_context_from_payload(
    payload: object,
    signal: object | None,
    *,
    correlation_id: str,
) -> IncomingContext:
    if not isinstance(payload, dict):
        raise TypeError("incoming signal payload must be a dict")
    channel_type = payload.get("channel") or payload.get("origin")
    if not channel_type and signal is not None:
        channel_type = getattr(signal, "source", None)
    if channel_type == "api" and payload.get("channel"):
        channel_type = payload.get("channel")
    if not channel_type:
        raise ValueError("incoming payload is missing channel/origin")
    address = payload.get("target") or payload.get("chat_id") or payload.get("channel_target")
    if address is None:
        address = str(channel_type)
    metadata = payload.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    person_id = metadata_dict.get("person_id") or payload.get("person_id")
    if not person_id and channel_type and address:
        person = identity_store.resolve_person_by_channel(str(channel_type), str(address))
        if person:
            person_id = person.get("person_id")
    update_id = metadata_dict.get("update_id") or payload.get("update_id")
    message_id = metadata_dict.get("message_id") or payload.get("message_id")
    return IncomingContext(
        channel_type=str(channel_type),
        address=str(address),
        person_id=str(person_id) if person_id is not None else None,
        correlation_id=correlation_id,
        update_id=str(update_id) if update_id is not None else None,
        message_id=str(message_id) if message_id is not None else None,
    )


def _emit_channel_transition(incoming: IncomingContext, phase: str) -> None:
    phase_value = str(phase or "").strip().lower()
    if not phase_value:
        return
    registry = get_io_registry()
    adapter = registry.get_extremity(incoming.channel_type)
    if adapter is None:
        return
    emit_fn = getattr(adapter, "emit_transition", None)
    if not callable(emit_fn):
        return
    try:
        emit_fn(
            channel_target=incoming.address,
            phase=phase_value,
            correlation_id=incoming.correlation_id,
            message_id=incoming.message_id,
        )
    except Exception:
        logger.exception(
            "HandleIncomingMessageAction transition emit failed channel=%s phase=%s",
            incoming.channel_type,
            phase_value,
        )


def _emit_channel_transition_event(incoming: IncomingContext, event: dict[str, object]) -> None:
    if not isinstance(event, dict):
        return
    registry = get_io_registry()
    adapter = registry.get_extremity(incoming.channel_type)
    if adapter is None:
        return
    emit_event_fn = getattr(adapter, "emit_transition_event", None)
    if callable(emit_event_fn):
        try:
            emit_event_fn(
                channel_target=incoming.address,
                event=event,
                correlation_id=incoming.correlation_id,
                message_id=incoming.message_id,
            )
            return
        except Exception:
            logger.exception(
                "HandleIncomingMessageAction transition event emit failed channel=%s",
                incoming.channel_type,
            )
            return
    phase = str(event.get("phase") or "").strip().lower()
    if phase:
        _emit_channel_transition(incoming, phase)


def _attach_transition_sink(state: dict[str, object], incoming: IncomingContext) -> bool:
    registry = get_io_registry()
    adapter = registry.get_extremity(incoming.channel_type)
    if adapter is None:
        return False
    emit_event_fn = getattr(adapter, "emit_transition_event", None)
    emit_phase_fn = getattr(adapter, "emit_transition", None)
    if not callable(emit_event_fn) and not callable(emit_phase_fn):
        return False
    state["_transition_sink"] = lambda event: _emit_channel_transition_event(incoming, event)
    return True
