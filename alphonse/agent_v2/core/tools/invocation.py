"""Shared, task-scoped execution boundary for direct and programmatic tools."""

from __future__ import annotations

import inspect
from time import monotonic
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.telemetry import ToolTelemetryEvent, json_chars, safe_telemetry_value

if TYPE_CHECKING:
    from alphonse.agent_v2.core.intelligence.task_state import TaskState


class ToolInvocationService:
    """Execute registered tools once and project a JSON-safe result to callers."""

    def __init__(self, *, context: CoreLoopContext, task: TaskState) -> None:
        self.context = context
        self.task = task

    def execute_or_raise(self, tool_id: str, arguments: dict[str, Any]) -> Any:
        started = monotonic()
        try:
            if self.context.tools is None:
                raise RuntimeError("tool_registry_unavailable")
            descriptor = _descriptor_for(self.context.tools, tool_id)
            # Older test/adapter registries can execute by opaque id without
            # implementing descriptor lookup. The concrete V2 registry still
            # enforces enabled tools during execution.
            if descriptor is not None:
                self._validate(descriptor.argument_schema, arguments)
            execute = self.context.tools.execute
            signature = inspect.signature(execute)
            if "execution_context" in signature.parameters:
                result = execute(tool_id, dict(arguments), execution_context=self.context.tool_execution_context(self.task))
            else:
                result = execute(tool_id, dict(arguments))
        except Exception as exc:
            self._emit_tool_telemetry(tool_id, arguments, None, started, "failed", f"{type(exc).__name__}: {exc}")
            raise
        self._emit_tool_telemetry(tool_id, arguments, result, started, "success", "")
        return result

    def _emit_tool_telemetry(
        self, tool_id: str, arguments: dict[str, Any], result: Any, started: float, status: str, error: str
    ) -> None:
        self.context.emit_telemetry(ToolTelemetryEvent(
            tool_id=tool_id,
            task_id=str(self.task.task_id or ""),
            project_id=self.task.project_id,
            argument_chars=json_chars(arguments),
            result_chars=json_chars(result) if result is not None else 0,
            duration_ms=max(0, round((monotonic() - started) * 1000)),
            status=status,
            error=str(safe_telemetry_value(error)),
            metadata=_tool_result_metadata(result),
        ).to_dict())

    def invoke(self, tool_id: str, arguments: dict[str, Any], *, call_id: str | None = None, parallel: bool = False) -> dict[str, Any]:
        call_id = str(call_id or f"program-call-{uuid4()}")
        descriptor = _descriptor_for(self.context.tools, tool_id) if self.context.tools is not None else None
        if descriptor is None:
            return _failure(call_id, tool_id, "tool_not_found", "The requested tool is not enabled.")
        if parallel and not descriptor.read_only:
            return _failure(call_id, tool_id, "tool_not_parallel_safe", "This tool may only be called sequentially in program mode.")

        self.context.emit_ui_event("tool_call_started", {"tool_call_id": call_id, "tool_id": tool_id, "tool_name": descriptor.name, "arguments": dict(arguments), "programmatic": True})
        self.context.record_memory_event(self.task, "Tool Call", {"tool_id": tool_id, "tool_name": descriptor.name, "arguments": arguments, "programmatic": True})
        try:
            result = self.execute_or_raise(tool_id, arguments)
        except Exception as exc:
            outcome = _failure(call_id, tool_id, type(exc).__name__, str(exc))
        else:
            waiting = isinstance(result, dict) and result.get("waiting_for_answer") is True
            bash_error = _bash_result_error(tool_id, result)
            tool_exception = result.get("exception") if isinstance(result, dict) else None
            if isinstance(tool_exception, dict) and tool_exception:
                code = str(tool_exception.get("code") or "tool_reported_failure")
                message = str(tool_exception.get("message") or code)
                outcome = _failure(call_id, tool_id, code, message)
                outcome["result"] = result
            elif bash_error:
                outcome = _failure(call_id, tool_id, "command_failed", bash_error)
                outcome["result"] = result
            else:
                outcome = {
                    "call_id": call_id,
                    "tool_id": tool_id,
                    "status": "waiting" if waiting else "success",
                    "result": result,
                    "error": None,
                }
        self.context.emit_ui_event("tool_call_result", {"tool_call_id": call_id, "tool_id": tool_id, "tool_name": descriptor.name, **outcome, "programmatic": True})
        self.context.record_memory_event(self.task, "Tool Result", {**outcome, "programmatic": True})
        return outcome

    @staticmethod
    def _validate(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
        if not schema:
            return
        try:
            Draft202012Validator(schema).validate(arguments)
        except ValidationError as exc:
            raise ValueError(f"tool_arguments_invalid: {exc.message}") from exc


def _failure(call_id: str, tool_id: str, code: str, message: str) -> dict[str, Any]:
    return {"call_id": call_id, "tool_id": tool_id, "status": "failed", "result": None, "error": {"code": code, "message": message}}


def _descriptor_for(registry: Any, tool_id: str) -> Any | None:
    get = getattr(registry, "get", None)
    if callable(get):
        descriptor = get(tool_id)
        if descriptor is not None:
            return descriptor
    listing = getattr(registry, "list", None)
    if callable(listing):
        for descriptor in listing() or ():
            if str(getattr(descriptor, "tool_id", "")) == tool_id or str(getattr(descriptor, "name", "")) == tool_id:
                return descriptor
    return None


def _bash_result_error(tool_id: str, result: Any) -> str:
    if tool_id != "native.bash" or not isinstance(result, dict):
        return ""
    if result.get("timed_out") is True:
        return str(result.get("stderr") or "Bash command timed out.").strip()
    try:
        exit_code = int(result.get("exit_code", 0))
    except (TypeError, ValueError):
        return "Bash returned an invalid exit code."
    if exit_code == 0:
        return ""
    detail = str(result.get("stderr") or result.get("stdout") or "").strip()
    return f"Bash exited with code {exit_code}." + (f" {detail}" if detail else "")


def _tool_result_metadata(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    paths = result.get("affected_paths")
    if not isinstance(paths, (list, tuple)):
        paths = [result["path"]] if isinstance(result.get("path"), str) else []
    return {
        "affected_paths": [str(path) for path in paths[:25]],
        "waiting_for_answer": result.get("waiting_for_answer") is True,
    }
