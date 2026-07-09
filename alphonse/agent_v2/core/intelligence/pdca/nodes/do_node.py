"""Do node for the v2 PDCA graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.intelligence.task_state import TaskState

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import CoreLoopContext


def do_node(task: TaskState, context: CoreLoopContext | None = None) -> TaskState:
    """Execute the planned work for the current PDCA pass."""
    planned_call = task.get_next_planned_call()
    if planned_call is None:
        task.append_update("Do found no unexecuted planned tool call.")
        return task

    if context is not None:
        tool_name = str(planned_call.get("tool_name") or planned_call.get("tool_id") or "tool").strip()
        context.emit_activity(
            phase=ImprovementPhase.DO,
            label="working",
            message=f"Running {tool_name}.",
        )

    call_id = str(planned_call.get("id") or "").strip()
    if context is None or context.tools is None:
        task.record_plan_call_exception(call_id, "Tool registry is not available.")
        task.metadata["do_executed_since_last_act"] = True
        task.append_update("Do could not execute planned tool call because no tool registry is available.")
        return task

    tool_id = str(planned_call.get("tool_id") or "").strip()
    arguments = planned_call.get("arguments")
    if not isinstance(arguments, dict):
        task.record_plan_call_exception(call_id, "Planned tool call arguments must be a JSON object.")
        task.metadata["do_executed_since_last_act"] = True
        task.append_update("Do recorded an invalid planned tool call argument shape.")
        return task

    try:
        result: Any = context.tools.execute(tool_id, dict(arguments))
    except Exception as exc:
        task.record_plan_call_exception(call_id, exc)
        task.metadata["do_executed_since_last_act"] = True
        task.append_update(f"Do recorded exception from planned tool call: {call_id}.")
        return task

    task.record_plan_call_success(call_id, result)
    task.metadata["do_executed_since_last_act"] = True
    task.append_update(f"Do executed planned tool call: {call_id}.")
    return task
