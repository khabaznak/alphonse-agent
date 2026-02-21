from __future__ import annotations

import logging
from typing import Any

from alphonse.agent.actions.models import ActionResult
from alphonse.agent.cognition.narration.outbound_narration_orchestrator import DeliveryCoordinator
from alphonse.agent.io import NormalizedOutboundMessage, get_io_registry


class CommunicationDispatcher:
    def __init__(self, *, coordinator: DeliveryCoordinator, logger: logging.Logger) -> None:
        self._coordinator = coordinator
        self._logger = logger

    def dispatch_step_message(
        self,
        *,
        channel: str,
        target: str | None,
        message: str,
        context: dict[str, Any],
        exec_context: Any,
        plan: Any,
    ) -> None:
        plan_id = str(getattr(plan, "plan_id", "") or "")
        step = str(getattr(plan, "tool", "") or "unknown")
        payload_dict = getattr(plan, "payload", {}) if hasattr(plan, "payload") else {}
        if channel in {"telegram", "api"} and not target:
            self._logger.warning(
                "executor dispatch skipped plan_id=%s channel=%s reason=missing_target",
                plan_id,
                channel,
            )
            return
        self._logger.info(
            "executor dispatch plan_id=%s step=%s channel=%s target=%s locale=%s",
            plan_id,
            step,
            channel,
            target or "none",
            payload_dict.get("locale") if isinstance(payload_dict, dict) else None,
        )
        payload = _message_payload(message, channel, target, exec_context)
        if isinstance(payload_dict, dict) and payload_dict.get("locale"):
            payload["locale"] = payload_dict.get("locale")
        action = ActionResult(intention_key="MESSAGE_READY", payload=payload, urgency="normal")
        delivery = self._coordinator.deliver(action, context)
        if delivery:
            self._deliver_normalized(delivery)

    def dispatch_execution_error(
        self,
        *,
        exec_context: Any,
        context: dict[str, Any],
        failure: dict[str, Any] | None = None,
    ) -> None:
        message = "error.execution_failed"
        payload = _message_payload(
            message,
            str(getattr(exec_context, "channel_type", "") or ""),
            str(getattr(exec_context, "channel_target", "") or "") or None,
            exec_context,
        )
        if isinstance(failure, dict):
            payload["error"] = failure
        action = ActionResult(intention_key="MESSAGE_READY", payload=payload, urgency="normal")
        delivery = self._coordinator.deliver(action, context)
        if delivery:
            self._deliver_normalized(delivery)

    def _deliver_normalized(self, delivery: NormalizedOutboundMessage) -> None:
        registry = get_io_registry()
        adapter = registry.get_extremity(delivery.channel_type)
        if not adapter:
            return
        adapter.deliver(delivery)


def _message_payload(
    message: str,
    channel: str,
    target: str | None,
    exec_context: Any,
) -> dict[str, Any]:
    actor_person_id = str(getattr(exec_context, "actor_person_id", "") or "").strip()
    audience = {"kind": "person", "id": actor_person_id} if actor_person_id else {"kind": "system", "id": "system"}
    payload: dict[str, Any] = {
        "message": message,
        "origin": channel,
        "channel_hint": channel,
        "correlation_id": str(getattr(exec_context, "correlation_id", "") or ""),
        "audience": audience,
    }
    if target:
        payload["target"] = target
    return payload
