"""Bounded, payload-free timing for selected V3 orchestration boundaries."""
from __future__ import annotations

from contextlib import contextmanager
from time import monotonic


@contextmanager
def measured_span(context, task, operation):
    """Record inclusive wall time, without recording exceptions or task content.

    Durations include any nested provider/tool calls. The sink is observational:
    a broken sink must not change execution or hide the original failure.
    """
    if operation not in {"phase_planning", "tactical_composition"}:
        raise ValueError("unknown_measurement_operation")
    start = monotonic()
    status = "returned"
    try:
        yield
    except BaseException:
        status = "raised"
        raise
    finally:
        event = {"event_type": "v3_measurement", "task_id": task.task_id,
                 "operation": operation, "status": status,
                 "duration_ms": max(0, round((monotonic() - start) * 1000))}
        try:
            context.emit_telemetry(event)
        except Exception:
            pass
