from __future__ import annotations

from dataclasses import dataclass
import os
import traceback

from alphonse.agent import identity
from alphonse.agent.actions.conscious_message_handler import build_incoming_message_envelope
from alphonse.agent.actions.registry import ActionRegistry
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.actions.handle_conscious_message import HandleConsciousMessageAction
from alphonse.agent.actions.handle_pdca_dispatch_kick import HandlePdcaDispatchKickAction
from alphonse.agent.actions.handle_pdca_failure_notice import HandlePdcaFailureNoticeAction
from alphonse.agent.actions.handle_pdca_slice_request import HandlePdcaSliceRequestAction
from alphonse.agent.actions.handle_timed_signals import HandleTimedSignalsAction
from alphonse.agent.actions.shutdown import ShutdownAction
from alphonse.agent.observability.log_manager import get_log_manager
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.agent.nervous_system.services import TELEGRAM_SERVICE_ID
from alphonse.agent.nervous_system.trace_store import write_trace
from alphonse.agent.cognition.narration.outbound_narration_orchestrator import (
    build_default_coordinator,
    DeliveryCoordinator,
)
from alphonse.agent.io import get_io_registry, NormalizedOutboundMessage

_LOG = get_log_manager()
_RUNTIME_FAILURE_SIGNAL = "sense.runtime.message.user.received"


@dataclass
class IntentPipeline:
    actions: ActionRegistry
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
                    self._deliver_normalized(delivery)
            ## self._emit_outcome(result, context, success=True, error=None)
        except Exception as exc:
            self._escalate_subconscious_failure(action_key=action_key, context=context, error=exc)
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

    def _deliver_normalized(self, delivery: NormalizedOutboundMessage) -> None:
        registry = get_io_registry()
        adapter = registry.get_extremity(delivery.channel_type)
        if not adapter:
            return
        adapter.deliver(delivery)

    def _escalate_subconscious_failure(self, *, action_key: str, context: dict, error: Exception) -> None:
        signal = context.get("signal")
        signal_type = str(getattr(signal, "type", "") or "").strip()
        # PDCA/conscious flows already surface failures through dedicated lifecycles.
        if signal_type.startswith("pdca.") or action_key in {
            "handle_conscious_message",
            "handle_pdca_failure_notice",
        }:
            return
        admin_target = _resolve_admin_telegram_target()
        if not admin_target:
            _LOG.emit(
                level="warning",
                event="runtime.failure.escalation_skipped",
                component="cognition.intentions.intent_pipeline",
                correlation_id=_extract_correlation_id(context),
                payload={
                    "reason": "admin_target_unresolved",
                    "action_key": action_key,
                    "signal_type": signal_type or None,
                    "error_type": type(error).__name__,
                },
            )
            return
        correlation_id = _extract_correlation_id(context)
        text = (
            "Runtime deterministic action failed. "
            f"action={action_key} signal={signal_type or 'unknown'} "
            f"error={type(error).__name__}: {str(error or '').strip()[:280]}"
        )
        envelope = build_incoming_message_envelope(
            message_id=f"runtime-failure:{correlation_id or 'unknown'}",
            channel_type="telegram",
            channel_target=admin_target,
            provider="runtime",
            text=text,
            correlation_id=correlation_id,
            actor_external_user_id="runtime",
            actor_display_name="Alphonse Runtime",
            metadata={
                "message_kind": "runtime_failure_escalation",
                "failure": {
                    "action_key": action_key,
                    "signal_type": signal_type or None,
                    "error_type": type(error).__name__,
                },
            },
        )
        self.bus.emit(
            Signal(
                type=_RUNTIME_FAILURE_SIGNAL,
                payload=envelope,
                source="system",
                correlation_id=correlation_id,
            )
        )
        _LOG.emit(
            event="runtime.failure.escalated_to_admin",
            component="cognition.intentions.intent_pipeline",
            correlation_id=correlation_id,
            payload={
                "action_key": action_key,
                "signal_type": signal_type or None,
                "admin_channel_type": "telegram",
                "admin_target": admin_target,
                "error_type": type(error).__name__,
            },
        )


def build_default_pipeline() -> IntentPipeline:
    raise RuntimeError("build_default_pipeline requires a Bus instance")


def build_default_pipeline_with_bus(bus: Bus) -> IntentPipeline:
    actions = ActionRegistry()
    actions.register("handle_conscious_message", lambda _: HandleConsciousMessageAction())
    actions.register("handle_pdca_dispatch_kick", lambda _: HandlePdcaDispatchKickAction())
    actions.register("handle_pdca_failure_notice", lambda _: HandlePdcaFailureNoticeAction())
    actions.register("handle_pdca_slice_request", lambda _: HandlePdcaSliceRequestAction())
    actions.register("handle_timed_dispatch", lambda _: HandleTimedSignalsAction())
    actions.register("shutdown", lambda _: ShutdownAction())
    return IntentPipeline(
        actions=actions,
        bus=bus,
        coordinator=build_default_coordinator(),
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


def _resolve_admin_telegram_target() -> str | None:
    admin = identity.get_active_admin_user()
    if not isinstance(admin, dict):
        return None
    return identity.resolve_service_user_id(
        user_id=str(admin.get("user_id") or "").strip() or None,
        service_id=TELEGRAM_SERVICE_ID,
    )


def _extract_correlation_id(context: dict) -> str | None:
    signal = context.get("signal")
    cid = str(getattr(signal, "correlation_id", "") or "").strip()
    if cid:
        return cid
    payload = getattr(signal, "payload", {}) if signal else {}
    if isinstance(payload, dict):
        rendered = str(payload.get("correlation_id") or "").strip()
        if rendered:
            return rendered
    return None
