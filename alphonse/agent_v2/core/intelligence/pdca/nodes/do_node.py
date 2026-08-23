"""Do node for the v2 PDCA graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.tools.registry.native.ask_question import ASK_QUESTION_TOOL_ID
from alphonse.agent_v2.core.tools.invocation import ToolInvocationService

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
    if str(planned_call.get("execution_mode") or "direct").strip().lower() == "program":
        return _execute_program(task, context, planned_call, call_id)
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
        result: Any = ToolInvocationService(context=context, task=task).execute_or_raise(tool_id, dict(arguments))
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


def _execute_program(task: TaskState, context: CoreLoopContext | None, planned_call: dict[str, Any], call_id: str) -> TaskState:
    if context is None or context.tools is None or context.program_runner is None:
        task.record_plan_call_exception(call_id, "Program execution is not available.")
        task.metadata["do_executed_since_last_act"] = True
        task.append_update("Do could not execute program mode because the program runner is unavailable.")
        return task
    program = planned_call.get("program") if isinstance(planned_call.get("program"), dict) else {}
    source = str(program.get("source") or "").strip()
    tool_name = "program"
    context.emit_activity(phase=ImprovementPhase.DO, label="working", message="Running a bounded program phase.")
    context.emit_ui_event("program_execution_started", {"tool_call_id": call_id, "source_language": "python"})
    try:
        outcome = context.program_runner.run(source=source, invocation_service=ToolInvocationService(context=context, task=task))
    except Exception as exc:
        outcome = {"status": "failed", "error": {"code": type(exc).__name__, "message": str(exc)}, "tool_calls": []}
    child_calls = outcome.get("tool_calls") if isinstance(outcome.get("tool_calls"), list) else []
    result = {"program_result": outcome.get("program_result"), "tool_calls": child_calls}
    status = str(outcome.get("status") or "failed")
    if status == "waiting":
        task.record_plan_call_waiting(call_id, result)
        task.metadata["do_executed_since_last_act"] = True
        task.metadata["task_parked"] = True
        task.status = "waiting_user"
        context.emit_ui_event("program_execution_finished", {"tool_call_id": call_id, "status": "waiting", "result": result})
        task.append_update("Do parked task after a program tool call requested human input.")
        return task
    if status != "success":
        error = outcome.get("error") if isinstance(outcome.get("error"), dict) else {}
        task.record_plan_call_exception(call_id, str(error.get("message") or "Program execution failed."))
        task.metadata["do_executed_since_last_act"] = True
        context.emit_ui_event("program_execution_finished", {"tool_call_id": call_id, "status": "failed", "error": error, "tool_calls": child_calls})
        task.append_update("Do recorded a failed program execution.")
        return task
    task.record_plan_call_success(call_id, result)
    task.metadata["do_executed_since_last_act"] = True
    context.emit_ui_event("program_execution_finished", {"tool_call_id": call_id, "status": "success", "result": result})
    context.emit_activity(phase=ImprovementPhase.DO, label="tool completed", message=f"{tool_name} completed.")
    task.append_update(f"Do executed program call: {call_id}.")
    return task


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
