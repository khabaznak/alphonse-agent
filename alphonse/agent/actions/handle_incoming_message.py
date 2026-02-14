from __future__ import annotations

import logging
import uuid

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.actions.session_context import IncomingContext
from alphonse.agent.actions.session_context import build_incoming_context_from_normalized
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

        normalized = _normalize_incoming_payload(payload, signal)

        raw_text = getattr(normalized, "text", None)
        if not isinstance(raw_text, str):
            raise TypeError("normalized incoming text must be a string")
        text = raw_text.strip()
        incoming = build_incoming_context_from_normalized(normalized, correlation_id)
        logger.info(
            "HandleIncomingMessageAction start channel=%s person=%s text=%s",
            incoming.channel_type,
            incoming.person_id,
            _text_log_snippet(text),
        )
        if not text:
            raise ValueError("normalized incoming text must be non-empty")

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
            normalized=normalized,
        )

        try:
            llm_client = build_llm_client()
        except Exception:
            logger.exception("HandleIncomingMessageAction failed to build llm client")
            llm_client = None
        try:
            result = _CORTEX_GRAPH.invoke(state, text, llm_client=llm_client)
        except Exception:
            logger.exception(
                "HandleIncomingMessageAction cortex_invoke_failed channel=%s target=%s correlation_id=%s",
                incoming.channel_type,
                incoming.address,
                incoming.correlation_id,
            )
            raise

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


def _normalize_incoming_payload(payload: object, signal: object | None) -> object:
    if not isinstance(payload, dict):
        raise TypeError("incoming signal payload must be a dict")
    channel_type = payload.get("channel") or payload.get("origin")
    if not channel_type and signal is not None:
        channel_type = getattr(signal, "source", None)
    if channel_type == "api" and payload.get("channel"):
        channel_type = payload.get("channel")
    if not channel_type:
        raise ValueError("incoming payload is missing channel/origin")

    registry = get_io_registry()
    adapter = registry.get_sense(str(channel_type))
    if not adapter:
        raise LookupError(f"no sense adapter registered for channel={channel_type}")
    try:
        return adapter.normalize(payload)
    except Exception:
        logger.exception("Failed to normalize payload for channel=%s", channel_type)
        raise


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
