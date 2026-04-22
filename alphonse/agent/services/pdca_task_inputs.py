from __future__ import annotations

from alphonse.agent.services.pdca_runtime import get_pdca_runtime
from alphonse.agent.observability.log_manager import get_log_manager

_LOG = get_log_manager()


def consume_task_inputs_for_check(*, task_id: str, correlation_id: str | None = None) -> list[dict[str, Any]]:
    consumed = get_pdca_runtime().consume_inputs_for_check(task_id=task_id)
    if consumed:
        _LOG.emit(
            event="pdca.input.dequeued",
            component="services.pdca_task_inputs",
            correlation_id=correlation_id,
            payload={"task_id": task_id, "count": len(consumed)},
        )
    return consumed
