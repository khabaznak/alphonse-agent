from __future__ import annotations

from dataclasses import dataclass

from alphonse.agent import identity
from alphonse.agent.actions.conscious_message_handler import build_incoming_message_envelope
from alphonse.agent.actions.handle_conscious_message import HandleConsciousMessageAction
from alphonse.agent.actions.handle_pdca_dispatch_kick import HandlePdcaDispatchKickAction
from alphonse.agent.actions.handle_pdca_failure_notice import HandlePdcaFailureNoticeAction
from alphonse.agent.actions.handle_pdca_slice_request import HandlePdcaSliceRequestAction
from alphonse.agent.actions.handle_timed_signals import HandleTimedSignalsAction
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.actions.registry import ActionRegistry
from alphonse.agent.actions.shutdown import ShutdownAction
from alphonse.agent.cognition.narration.outbound_narration_orchestrator import (
    DeliveryCoordinator,
    build_default_coordinator,
)
from alphonse.agent.io import NormalizedOutboundMessage, get_io_registry
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.agent.nervous_system.services import TELEGRAM_SERVICE_ID
from alphonse.agent.observability.log_manager import get_log_manager

_LOG = get_log_manager()
_RUNTIME_FAILURE_SIGNAL = "sense.runtime.message.user.received"


@dataclass
class ActionExecutionRuntime:
    actions: ActionRegistry
    bus: Bus | None = None
    coordinator: DeliveryCoordinator | None = None

    def execute(self, action_key: str | None, context: dict) -> ActionResult | None:
        if not action_key:
            return None
        factory = self.actions.get(action_key)
        if not factory:
            return None
        action = factory(context)
        try:
            result = action.execute(context)
        except Exception as exc:
            self._escalate_failure(action_key=action_key, context=context, error=exc)
            raise
        if result.delivers_message:
            self._deliver_result(result=result, context=context)
        return result

    def _deliver_result(self, *, result: ActionResult, context: dict) -> None:
        coordinator = self.coordinator
        if coordinator is None:
            return
        delivery = coordinator.deliver(result, context)
        if delivery:
            _deliver_normalized(delivery)

    def _escalate_failure(self, *, action_key: str, context: dict, error: Exception) -> None:
        bus = self.bus
        if bus is None:
            return
        signal = context.get("signal")
        signal_type = str(getattr(signal, "type", "") or "").strip()
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
                component="actions.runtime",
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
        bus.emit(
            Signal(
                type=_RUNTIME_FAILURE_SIGNAL,
                payload=envelope,
                source="system",
                correlation_id=correlation_id,
            )
        )
        _LOG.emit(
            event="runtime.failure.escalated_to_admin",
            component="actions.runtime",
            correlation_id=correlation_id,
            payload={
                "action_key": action_key,
                "signal_type": signal_type or None,
                "admin_channel_type": "telegram",
                "admin_target": admin_target,
                "error_type": type(error).__name__,
            },
        )


def build_default_action_registry() -> ActionRegistry:
    actions = ActionRegistry()
    actions.register("handle_conscious_message", lambda _: HandleConsciousMessageAction())
    actions.register("handle_pdca_dispatch_kick", lambda _: HandlePdcaDispatchKickAction())
    actions.register("handle_pdca_failure_notice", lambda _: HandlePdcaFailureNoticeAction())
    actions.register("handle_pdca_slice_request", lambda _: HandlePdcaSliceRequestAction())
    actions.register("handle_timed_dispatch", lambda _: HandleTimedSignalsAction())
    actions.register("shutdown", lambda _: ShutdownAction())
    return actions


def build_action_runtime(*, bus: Bus | None = None) -> ActionExecutionRuntime:
    return ActionExecutionRuntime(
        actions=build_default_action_registry(),
        bus=bus,
        coordinator=build_default_coordinator(),
    )


def _deliver_normalized(delivery: NormalizedOutboundMessage) -> None:
    registry = get_io_registry()
    adapter = registry.get_extremity(delivery.channel_type)
    if not adapter:
        return
    adapter.deliver(delivery)


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
