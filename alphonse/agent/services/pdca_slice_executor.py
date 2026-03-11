from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alphonse.agent.actions.handle_pdca_slice_request import HandlePdcaSliceRequestAction
from alphonse.agent.nervous_system.pdca_queue_store import (
    append_pdca_event,
    get_pdca_task,
    update_pdca_task_status,
)
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.agent.observability.log_manager import get_component_logger

logger = get_component_logger("services.pdca_slice_executor")


@dataclass(frozen=True)
class _SignalStub:
    type: str
    payload: dict[str, Any]
    correlation_id: str | None
    source: str


class PdcaSliceExecutor:
    def __init__(self, *, bus: Bus) -> None:
        self._bus = bus
        self._legacy_action = HandlePdcaSliceRequestAction()

    def execute_task(
        self,
        *,
        task_id: str,
        correlation_id: str | None,
        signal_type: str = "pdca.slice.requested",
        source: str = "pdca_queue_runner",
    ) -> None:
        task = get_pdca_task(task_id)
        if not isinstance(task, dict):
            return
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        if bool(metadata.get("cancel_requested")):
            self._cancel_task(task_id=task_id, correlation_id=correlation_id)
            return

        self._legacy_action.execute(
            {
                "ctx": self._bus,
                "signal": _SignalStub(
                    type=signal_type,
                    payload={"task_id": task_id, "correlation_id": correlation_id},
                    correlation_id=correlation_id,
                    source=source,
                ),
            }
        )

    def _cancel_task(self, *, task_id: str, correlation_id: str | None) -> None:
        update_pdca_task_status(task_id=task_id, status="failed", last_error="cancelled_by_user")
        append_pdca_event(
            task_id=task_id,
            event_type="slice.cancelled",
            payload={"reason": "cancel_requested"},
            correlation_id=correlation_id,
        )
        self._bus.emit(
            Signal(
                type="pdca.task_cancelled",
                payload={"task_id": task_id, "correlation_id": correlation_id},
                source="pdca_executor",
                correlation_id=correlation_id,
            )
        )
        logger.info("PdcaSliceExecutor cancelled task_id=%s", task_id)
