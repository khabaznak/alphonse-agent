"""Do node for the v2 PDCA graph."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.tools.registry.native.ask_question import ASK_QUESTION_TOOL_ID

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
        context.record_memory_event(task, "Tool Call", {"tool_id": tool_id, "tool_name": tool_name, "arguments": arguments})
        context.emit_ui_event(
            "tool_call_started",
            {
                "tool_call_id": call_id,
                "tool_name": tool_name,
                "tool_id": tool_id,
                "arguments": dict(arguments),
            },
        )
        result: Any = _execute_tool(context, task, tool_id, dict(arguments))
    except Exception as exc:
        context.record_memory_event(task, "Tool Result", {"status": "exception", "error": f"{type(exc).__name__}: {exc}"})
        context.emit_ui_event(
            "tool_call_result",
            {
                "tool_call_id": call_id,
                "tool_name": tool_name,
                "tool_id": tool_id,
                "status": "exception",
                "exception": f"{type(exc).__name__}: {exc}",
            },
        )
        task.record_plan_call_exception(call_id, exc)
        context.emit_activity(
            phase=ImprovementPhase.DO,
            label="tool failed",
            message=f"{tool_name} returned an error.",
        )
        task.metadata["do_executed_since_last_act"] = True
        task.append_update(f"Do recorded exception from planned tool call: {call_id}.")
        return task

    if tool_id == ASK_QUESTION_TOOL_ID or _result_waits_for_answer(result):
        context.record_memory_event(task, "Tool Result", {"status": "waiting", "result": result})
        context.emit_ui_event(
            "tool_call_result",
            {
                "tool_call_id": call_id,
                "tool_name": tool_name,
                "tool_id": tool_id,
                "status": "waiting",
                "result": result,
            },
        )
        task.record_plan_call_waiting(call_id, result)
        context.emit_activity(
            phase=ImprovementPhase.DO,
            label="waiting",
            message=f"{tool_name} needs user input.",
        )
        task.metadata["do_executed_since_last_act"] = True
        task.metadata["task_parked"] = True
        task.status = "waiting_user"
        task.append_update(f"Do parked task waiting for question answer: {call_id}.")
        return task

    context.emit_ui_event(
        "tool_call_result",
        {
            "tool_call_id": call_id,
            "tool_name": tool_name,
            "tool_id": tool_id,
            "status": "success",
            "result": result,
        },
    )
    context.record_memory_event(task, "Tool Result", {"status": "success", "result": result})
    if tool_id == "native.respond" and isinstance(result, dict):
        context.record_memory_event(task, "Conversation", f"- Alphonse: {str(result.get('message') or '')}")
    task.record_plan_call_success(call_id, result)
    context.emit_activity(
        phase=ImprovementPhase.DO,
        label="tool completed",
        message=f"{tool_name} completed.",
    )
    if _is_silent_successful_bash(tool_id, result):
        task.metadata["pending_silent_bash_confirmation"] = {
            "tool_call_id": call_id,
            "command": str(arguments.get("command") or ""),
            "internal_state": str(planned_call.get("internal_state") or ""),
        }
        task.append_update("Do requires a native.respond confirmation for the silent Bash action.")
    elif tool_id == "native.respond" and task.metadata.get("pending_silent_bash_confirmation"):
        task.metadata.pop("pending_silent_bash_confirmation", None)
        task.append_update("Do recorded the required user confirmation for the silent Bash action.")
    task.metadata["do_executed_since_last_act"] = True
    task.append_update(f"Do executed planned tool call: {call_id}.")
    return task


def _execute_tool(context: CoreLoopContext, task: TaskState, tool_id: str, arguments: dict[str, Any]) -> Any:
    execute = context.tools.execute
    signature = inspect.signature(execute)
    if "execution_context" in signature.parameters:
        return execute(tool_id, arguments, execution_context=context.tool_execution_context(task))
    return execute(tool_id, arguments)


def _result_waits_for_answer(result: Any) -> bool:
    return isinstance(result, dict) and result.get("waiting_for_answer") is True


def _is_silent_successful_bash(tool_id: str, result: Any) -> bool:
    if tool_id != "native.bash" or not isinstance(result, dict):
        return False
    if result.get("timed_out") is True:
        return False
    try:
        if int(result.get("exit_code", 0)) != 0:
            return False
    except (TypeError, ValueError):
        return False
    return not str(result.get("stdout") or "").strip() and not str(result.get("stderr") or "").strip()
