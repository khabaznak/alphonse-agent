from __future__ import annotations

from dataclasses import dataclass
import os
import traceback

from alphonse.agent.actions.registry import ActionRegistry
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.actions.system_reminder import SystemReminderAction
from alphonse.agent.actions.handle_incoming_message import HandleIncomingMessageAction
from alphonse.agent.actions.handle_status import HandleStatusAction
from alphonse.agent.actions.handle_timed_signals import HandleTimedSignalsAction
from alphonse.agent.actions.handle_action_failure import HandleActionFailure
from alphonse.agent.actions.handle_timer_fired import HandleTimerFiredAction
from alphonse.agent.extremities.notification import NotificationExtremity
from alphonse.agent.extremities.telegram_notification import TelegramNotificationExtremity
from alphonse.agent.extremities.api_extremity import ApiExtremity
from alphonse.agent.extremities.cli_extremity import CliExtremity
from alphonse.agent.extremities.registry import ExtremityRegistry
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.agent.nervous_system.trace_store import write_trace
from alphonse.agent.cognition.narration.coordinator import build_default_coordinator, DeliveryCoordinator
from alphonse.agent.io import get_io_registry, NormalizedOutboundMessage


@dataclass
class IntentPipeline:
    actions: ActionRegistry
    extremities: ExtremityRegistry
    bus: Bus
    coordinator: DeliveryCoordinator

    def handle(self, action_key: str | None, context: dict) -> None:
        if not action_key:
            return
        factory = self.actions.get(action_key)
        if not factory:
            return
        action = factory(context)
        try:
            result = action.execute(context)
            if result.intention_key == "MESSAGE_READY":
                delivery = self.coordinator.deliver(result, context)
                if delivery:
                    self._deliver_normalized(delivery, result)
            else:
                if result.intention_key.startswith("NOTIFY_"):
                    self._deliver_notify(result)
                else:
                    self.extremities.dispatch(result, None)
            self._emit_outcome(result, context, success=True, error=None)
        except Exception as exc:
            self._emit_outcome(None, context, success=False, error=exc)

    def _emit_outcome(
        self,
        result: ActionResult | None,
        context: dict,
        *,
        success: bool,
        error: Exception | None,
    ) -> None:
        payload = _outcome_signal_payload(context, result, success, error)
        if _should_emit_outcome(context):
            _emit_outcome_signal(self.bus, payload, success)
        _emit_trace(context, success, error)

    def _deliver_normalized(self, delivery: ActionResult, original: ActionResult) -> None:
        normalized = _build_normalized_outbound(delivery, original)
        if not normalized:
            return
        registry = get_io_registry()
        adapter = registry.get_extremity(normalized.channel_type)
        if not adapter:
            return
        adapter.deliver(normalized)

    def _deliver_notify(self, result: ActionResult) -> None:
        normalized = _build_normalized_from_notify(result)
        if not normalized:
            return
        registry = get_io_registry()
        adapter = registry.get_extremity(normalized.channel_type)
        if not adapter:
            return
        adapter.deliver(normalized)


def build_default_pipeline() -> IntentPipeline:
    raise RuntimeError("build_default_pipeline requires a Bus instance")


def build_default_pipeline_with_bus(bus: Bus) -> IntentPipeline:
    actions = ActionRegistry()
    actions.register("system_reminder", lambda _: SystemReminderAction())
    actions.register("handle_incoming_message", lambda _: HandleIncomingMessageAction())
    actions.register("handle_status", lambda _: HandleStatusAction())
    actions.register("handle_timed_signals", lambda _: HandleTimedSignalsAction())
    actions.register("handle_action_failure", lambda _: HandleActionFailure())
    actions.register("handle_timer_fired", lambda _: HandleTimerFiredAction())
    extremities = ExtremityRegistry()
    extremities.register(NotificationExtremity())
    extremities.register(TelegramNotificationExtremity())
    extremities.register(ApiExtremity())
    extremities.register(CliExtremity())
    return IntentPipeline(
        actions=actions,
        extremities=extremities,
        bus=bus,
        coordinator=build_default_coordinator(),
    )


def _channel_from_intention(intention_key: str) -> str | None:
    mapping = {
        "NOTIFY_TELEGRAM": "telegram",
        "NOTIFY_CLI": "cli",
        "NOTIFY_API": "api",
    }
    return mapping.get(intention_key)


def _default_target_for_channel(channel_type: str) -> str | None:
    if channel_type == "cli":
        return "cli"
    if channel_type == "api":
        return "api"
    return None


def _audience_from_payload(payload: dict) -> dict[str, str]:
    audience = payload.get("audience")
    if isinstance(audience, dict):
        kind = audience.get("kind")
        ident = audience.get("id")
        if isinstance(kind, str) and isinstance(ident, str):
            return {"kind": kind, "id": ident}
    return {"kind": "system", "id": "system"}


def _build_normalized_outbound(
    delivery: ActionResult, original: ActionResult
) -> NormalizedOutboundMessage | None:
    channel_type = _channel_from_intention(delivery.intention_key)
    if not channel_type:
        return None
    payload = delivery.payload or {}
    message = payload.get("message")
    if not message:
        return None
    channel_target = (
        str(payload.get("chat_id"))
        if channel_type == "telegram" and payload.get("chat_id") is not None
        else _default_target_for_channel(channel_type)
    )
    return NormalizedOutboundMessage(
        message=str(message),
        channel_type=channel_type,
        channel_target=channel_target,
        audience=_audience_from_payload(original.payload or {}),
        correlation_id=str(payload.get("correlation_id") or original.payload.get("correlation_id") or ""),
        metadata={"data": payload.get("data")},
    )


def _build_normalized_from_notify(
    result: ActionResult,
) -> NormalizedOutboundMessage | None:
    channel_type = _channel_from_intention(result.intention_key)
    if not channel_type:
        return None
    payload = result.payload or {}
    message = payload.get("message")
    if not message:
        return None
    channel_target = (
        str(payload.get("chat_id"))
        if channel_type == "telegram" and payload.get("chat_id") is not None
        else _default_target_for_channel(channel_type)
    )
    return NormalizedOutboundMessage(
        message=str(message),
        channel_type=channel_type,
        channel_target=channel_target,
        audience=_audience_from_payload(payload),
        correlation_id=str(payload.get("correlation_id") or ""),
        metadata={"data": payload.get("data")},
    )


def _extract_context_payload(context: dict) -> dict:
    signal = context.get("signal")
    outcome = context.get("outcome")
    state = context.get("state")
    state_before = getattr(state, "key", None) or getattr(state, "id", None)
    state_after = getattr(outcome, "next_state_key", None) or getattr(outcome, "next_state_id", None)
    correlation_id = getattr(signal, "correlation_id", None) if signal else None
    if not correlation_id and signal and isinstance(getattr(signal, "payload", None), dict):
        correlation_id = signal.payload.get("correlation_id")
    return {
        "state_before": state_before,
        "state_after": state_after,
        "signal_type": getattr(signal, "type", None),
        "transition_id": getattr(outcome, "transition_id", None),
        "action_key": getattr(outcome, "action_key", None),
        "correlation_id": correlation_id,
        "depth": _extract_depth(signal),
    }


def _outcome_signal_payload(context: dict, result: ActionResult | None, success: bool, error: Exception | None) -> dict:
    payload = _extract_context_payload(context)
    payload["result"] = "success" if success else "failure"
    payload["depth"] = payload.get("depth", 0) + 1
    if result is not None:
        payload["output"] = result.payload
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = str(error)
        payload["stack"] = "".join(traceback.format_exception(error))
    return payload


def _outcome_signal_type(success: bool) -> str:
    return "action.succeeded" if success else "action.failed"


def _trace_payload(context: dict, success: bool, error: Exception | None) -> dict:
    payload = _extract_context_payload(context)
    payload["result"] = "success" if success else "failure"
    payload["error_summary"] = None if error is None else str(error)
    return payload


def _emit_outcome_signal(bus: Bus, payload: dict, success: bool) -> None:
    bus.emit(
        Signal(
            type=_outcome_signal_type(success),
            payload=payload,
            source="intent_pipeline",
            correlation_id=payload.get("correlation_id"),
        )
    )


def _emit_trace(context: dict, success: bool, error: Exception | None) -> None:
    write_trace(_trace_payload(context, success, error))


def _extract_depth(signal: object | None) -> int:
    if not signal:
        return 0
    payload = getattr(signal, "payload", {})
    if isinstance(payload, dict):
        try:
            return int(payload.get("depth", 0))
        except (TypeError, ValueError):
            return 0
    return 0


def _should_emit_outcome(context: dict) -> bool:
    max_depth_raw = os.getenv("MAX_TRANSITION_DEPTH", "5")
    try:
        max_depth = int(max_depth_raw)
    except ValueError:
        max_depth = 5
    payload = _extract_context_payload(context)
    depth = int(payload.get("depth", 0))
    return depth < max_depth
