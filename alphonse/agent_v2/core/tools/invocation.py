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
        descriptor = _descriptor_for(self.context.tools, tool_id) if self.context.tools is not None else None
        sensitive = bool(descriptor and descriptor.metadata.get("sensitive_content") is True)
        try:
            _check_browser_effect_guard(self.task, descriptor, tool_id)
            retirement = _retirement_reason(tool_id, arguments)
            if retirement:
                raise ValueError(retirement)
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
            self._emit_tool_telemetry(tool_id, arguments, None, started, "failed", "sensitive_tool_failed" if sensitive else f"{type(exc).__name__}: {exc}", sensitive=sensitive)
            if sensitive:
                raise RuntimeError("sensitive_tool_failed") from None
            raise
        self._emit_tool_telemetry(tool_id, arguments, result, started, "success", "", sensitive=sensitive)
        return result

    def _emit_tool_telemetry(
        self, tool_id: str, arguments: dict[str, Any], result: Any, started: float, status: str, error: str, *, sensitive: bool = False
    ) -> None:
        self.context.emit_telemetry(ToolTelemetryEvent(
            tool_id=tool_id,
            task_id=str(self.task.task_id or ""),
            project_id=self.task.project_id,
            argument_chars=0 if sensitive else json_chars(arguments),
            result_chars=0 if sensitive else (json_chars(result) if result is not None else 0),
            duration_ms=max(0, round((monotonic() - started) * 1000)),
            status=status,
            error=str(safe_telemetry_value(error)),
            metadata={} if sensitive else _tool_result_metadata(result),
        ).to_dict())

    def invoke(self, tool_id: str, arguments: dict[str, Any], *, call_id: str | None = None, parallel: bool = False) -> dict[str, Any]:
        call_id = str(call_id or f"program-call-{uuid4()}")
        retirement = _retirement_reason(tool_id, arguments)
        if retirement:
            return _failure(call_id, tool_id, "legacy_artifact_execution_retired", retirement)
        descriptor = _descriptor_for(self.context.tools, tool_id) if self.context.tools is not None else None
        if descriptor is None:
            return _failure(call_id, tool_id, "tool_not_found", "The requested tool is not enabled.")
        try:
            _check_browser_effect_guard(self.task, descriptor, tool_id)
        except PermissionError:
            return _failure(call_id, tool_id, "browser_reconciliation_required", "Browser effects require reconciliation before another mutation.")
        sensitive = descriptor.metadata.get("sensitive_content") is True
        if parallel and not descriptor.read_only:
            return _failure(call_id, tool_id, "tool_not_parallel_safe", "This tool may only be called sequentially in program mode.")

        actor = str(self.task.user or "").strip()
        # Tool arguments and results may contain private household data. A UI
        # event without an actor is broadcast to all Desktop users, so omit it
        # rather than projecting an unscoped payload.
        recorded_arguments = {"redacted": True} if sensitive else dict(arguments)
        if actor:
            self.context.emit_ui_event("tool_call_started", {"user": actor, "task_id": str(self.task.task_id or ""), "tool_call_id": call_id, "tool_id": tool_id, "tool_name": descriptor.name, "arguments": recorded_arguments, "programmatic": True})
        self.context.record_memory_event(self.task, "Tool Call", {"tool_id": tool_id, "tool_name": descriptor.name, "arguments": recorded_arguments, "programmatic": True})
        try:
            result = self.execute_or_raise(tool_id, arguments)
        except Exception as exc:
            outcome = _failure(call_id, tool_id, "sensitive_tool_failed" if sensitive else type(exc).__name__, "sensitive_tool_failed" if sensitive else str(exc))
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
        if sensitive:
            outcome = {"call_id": call_id, "tool_id": tool_id, "status": outcome["status"],
                       "result": {"redacted": True} if outcome["status"] != "failed" else None,
                       "error": {"code": "sensitive_tool_failed", "message": "sensitive_tool_failed"} if outcome["status"] == "failed" else None}
        event_outcome = {"call_id": call_id, "tool_id": tool_id, "status": outcome["status"], "redacted": True} if sensitive else outcome
        if actor:
            self.context.emit_ui_event("tool_call_result", {"user": actor, "task_id": str(self.task.task_id or ""), "tool_call_id": call_id, "tool_id": tool_id, "tool_name": descriptor.name, **event_outcome, "programmatic": True})
        self.context.record_memory_event(self.task, "Tool Result", {**event_outcome, "programmatic": True})
        return outcome

    @staticmethod
    def _validate(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
        if not schema:
            return
        try:
            Draft202012Validator(schema).validate(arguments)
        except ValidationError as exc:
            raise ValueError(f"tool_arguments_invalid: {exc.message}") from exc


def _check_browser_effect_guard(task: Any, descriptor: Any, tool_id: str) -> None:
    from alphonse.agent_v2.core.intelligence.v3 import browser_workflow
    if browser_workflow.KEY not in task.metadata:
        if browser_workflow._legacy_browser_state(task):
            raise PermissionError("browser_reconciliation_required")
        return
    try:
        state = browser_workflow.load(task)
    except browser_workflow.BrowserWorkflowError:
        raise PermissionError("browser_reconciliation_required") from None
    if not browser_workflow.unresolved(state):
        return
    recovery = {"native.browser_workflow_reacquire", "native.browser_workflow_observe",
                "native.browser_workflow_observe_result", "native.browser_workflow_reconcile",
                "native.browser_workflow_verify_submission", "native.browser_workflow_request_value"}
    if tool_id not in recovery and (descriptor is None or not descriptor.read_only):
        raise PermissionError("browser_reconciliation_required")

def _retirement_reason(tool_id: str, arguments: dict[str, Any]) -> str:
    if (tool_id.startswith("artifact.") or tool_id in {"native.skill_artifact_register", "native.artifact_metadata_update"}
            or (tool_id in {"native.bash", "bash"} and "artifact_id" in arguments)):
        return "legacy_artifact_execution_retired: review legacy_skill_migration and replan with ordinary skill commands; no command was dispatched"
    return ""


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
    if result.get("verification_failed") is True:
        return "Independent read-back failed; the action may have taken effect and must be reconciled before retrying."
    if result.get("cancelled") is True:
        return "Bash command cancelled; reconcile any partial effects before retrying."
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
