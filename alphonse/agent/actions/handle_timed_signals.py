from __future__ import annotations

import logging

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.nervous_system.senses.bus import Signal as BusSignal


logger = logging.getLogger(__name__)


class HandleTimedSignalsAction(Action):
    key = "handle_timed_signals"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        payload = getattr(signal, "payload", {}) if signal else {}
        inner = _payload_from_signal(payload)
        mind_layer = str(payload.get("mind_layer") or inner.get("mind_layer") or "subconscious").strip().lower()
        prompt = _extract_prompt_text(inner=inner)
        target = str(inner.get("chat_id") or payload.get("target") or inner.get("delivery_target") or "").strip()
        user_id = str(inner.get("person_id") or target or "").strip()
        channel_type = str(inner.get("origin_channel") or inner.get("origin") or payload.get("origin") or "api").strip()
        logger.info(
            "HandleTimedSignalsAction invoked signal_id=%s timed_signal_id=%s correlation_id=%s mind_layer=%s",
            getattr(signal, "id", None),
            payload.get("timed_signal_id"),
            getattr(signal, "correlation_id", None),
            mind_layer,
        )
        bus = context.get("ctx")
        if not hasattr(bus, "emit"):
            logger.warning("HandleTimedSignalsAction route skipped reason=no_bus")
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)

        routed_signal_type = (
            "api.message_received" if mind_layer == "conscious" else "timed_signal.subconscious_prompt"
        )
        bus.emit(
            BusSignal(
                type=routed_signal_type,
                payload={
                    "text": prompt,
                    "channel": channel_type,
                    "origin": channel_type,
                    "target": target or user_id,
                    "user_id": user_id or target,
                    "metadata": {
                        "timed_signal": payload,
                        "mind_layer": mind_layer,
                        "channel_hint": channel_type,
                    },
                },
                source="timer",
                correlation_id=_correlation_id(payload, signal),
            )
        )
        logger.info("HandleTimedSignalsAction routed signal_type=%s", routed_signal_type)
        return ActionResult(intention_key="NOOP", payload={}, urgency=None)


def _payload_from_signal(payload: dict) -> dict:
    signal_payload = payload.get("payload") if isinstance(payload, dict) else None
    return signal_payload if isinstance(signal_payload, dict) else {}


def _correlation_id(payload: dict, signal: object | None) -> str | None:
    if isinstance(payload, dict):
        cid = payload.get("correlation_id")
        if cid:
            return str(cid)
    return getattr(signal, "correlation_id", None) if signal else None


def _extract_prompt_text(*, inner: dict) -> str:
    text = str(
        inner.get("prompt")
        or inner.get("agent_internal_prompt")
        or inner.get("prompt_text")
        or inner.get("message_text")
        or inner.get("message")
        or inner.get("reminder_text_raw")
        or "You just remembered something important."
    ).strip()
    return text or "You just remembered something important."
