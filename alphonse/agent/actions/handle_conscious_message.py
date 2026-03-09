from __future__ import annotations

import uuid

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.conscious_message_context_adapter import (
    build_incoming_context_from_envelope,
)
from alphonse.agent.actions.conscious_message_handler import IncomingMessageEnvelope
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.actions.session_context import build_session_key
from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.services.pdca_ingress import enqueue_pdca_slice
from alphonse.agent.services.session_identity_resolution import resolve_session_timezone
from alphonse.agent.services.session_identity_resolution import resolve_session_user_id
from alphonse.agent.session.day_state import resolve_day_session

logger = get_component_logger("actions.handle_conscious_message")


class HandleConsciousMessageAction(Action):
    key = "handle_conscious_message"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        raw_payload = getattr(signal, "payload", {}) if signal else {}
        if not isinstance(raw_payload, dict):
            raise ValueError("invalid_envelope: payload must be an object")

        envelope = IncomingMessageEnvelope.from_payload(raw_payload)
        correlation_id = envelope.correlation_id or getattr(signal, "correlation_id", None) or str(uuid.uuid4())
        payload = envelope.runtime_payload()
        incoming = build_incoming_context_from_envelope(
            envelope=envelope,
            correlation_id=str(correlation_id),
        )

        session_key = build_session_key(incoming)
        session_user_id = resolve_session_user_id(incoming=incoming, payload=payload)
        day_session = resolve_day_session(
            user_id=session_user_id,
            channel=incoming.channel_type,
            timezone_name=resolve_session_timezone(incoming),
        )

        task_id = enqueue_pdca_slice(
            context=context,
            incoming=incoming,
            state={
                "conversation_key": session_key,
                "chat_id": session_key,
                "channel_type": incoming.channel_type,
                "channel_target": incoming.address,
                "actor_person_id": incoming.person_id,
                "locale": envelope.context.get("locale"),
                "timezone": envelope.context.get("timezone"),
            },
            session_key=session_key,
            session_user_id=session_user_id,
            day_session=day_session,
            payload=payload,
            correlation_id=str(correlation_id),
        )
        logger.info(
            "HandleConsciousMessageAction enqueued task_id=%s channel=%s target=%s",
            task_id,
            incoming.channel_type,
            incoming.address,
        )
        return ActionResult(
            intention_key="NOOP",
            payload={"task_id": task_id},
            urgency=None,
        )
