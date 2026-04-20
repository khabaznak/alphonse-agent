from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Literal, TypedDict

from alphonse.agent.cognition.memory import record_after_tool_call
from alphonse.agent.cognition.memory import record_plan_step_completion
from alphonse.agent.cortex.task_mode.plan import PlannerOutput
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.cortex.transitions import emit_presence_transition_event
from alphonse.agent.tools.base import ensure_tool_result
from alphonse.agent.tools.registry import ToolRegistry
from alphonse.agent.tools.registry import build_default_tool_registry

_PLANNER_INTENT_MAX_LENGTH = 160

class DoResult(TypedDict):
    task_record: TaskRecord
    provenance: Literal["do"]


def execute_step_node_impl(
    task_record: TaskRecord,
    planner_output: PlannerOutput,
    *,
    logger: Any,
    log_task_event: Any,
) -> DoResult:
    tool_call = _require_canonical_tool_call(planner_output)
    planner_intent = str(planner_output.get("planner_intent") or "").strip()[:_PLANNER_INTENT_MAX_LENGTH]
    _emit_planner_intent_progress(
        task_record=task_record,
        proposal=tool_call,
        planner_intent=planner_intent,
    )
    result = _execute_tool_call(
        task_record=task_record,
        tool_name=str(tool_call["tool_name"]),
        args=dict(tool_call["args"]),
    )
    output_payload = _serialize_result(result.get("output"))
    exception_payload = _serialize_result(result.get("exception"))
    _append_tool_call_history_entry(
        task_record=task_record,
        tool_name=str(tool_call["tool_name"]),
        args=dict(tool_call["args"]),
        output=output_payload,
        exception=exception_payload,
    )
    _merge_stable_facts(task_record=task_record, tool_name=str(tool_call["tool_name"]), output=output_payload)
    _append_memory_fact(
        task_record=task_record,
        tool_name=str(tool_call["tool_name"]),
        output=output_payload,
        exception=exception_payload,
    )
    _record_execution_hooks(
        task_record=task_record,
        tool_name=str(tool_call["tool_name"]),
        args=dict(tool_call["args"]),
        result=result,
        planner_output=planner_output,
    )
    _log_execution(
        task_record=task_record,
        tool_name=str(tool_call["tool_name"]),
        planner_intent=planner_intent,
        logger=logger,
        log_task_event=log_task_event,
    )
    return {
        "task_record": task_record,
        "provenance": "do",
    }


def _require_canonical_tool_call(planner_output: PlannerOutput) -> dict[str, Any]:
    if not isinstance(planner_output, dict):
        raise ValueError("execute_step.invalid_planner_output: planner_output must be an object")
    tool_call = planner_output.get("tool_call")
    if not isinstance(tool_call, dict):
        raise ValueError("execute_step.invalid_planner_output: missing tool_call")
    kind = str(tool_call.get("kind") or "").strip()
    tool_name = str(tool_call.get("tool_name") or "").strip()
    args = tool_call.get("args")
    if kind != "call_tool":
        raise ValueError("execute_step.invalid_planner_output: invalid tool_call.kind")
    if not tool_name:
        raise ValueError("execute_step.invalid_planner_output: missing tool_call.tool_name")
    if not isinstance(args, dict):
        raise ValueError("execute_step.invalid_planner_output: invalid tool_call.args")
    return {
        "kind": "call_tool",
        "tool_name": tool_name,
        "args": dict(args),
    }


def _emit_planner_intent_progress(
    *,
    task_record: TaskRecord,
    proposal: dict[str, Any],
    planner_intent: str,
) -> None:
    hint = str(planner_intent or "").strip()
    emit_presence_transition_event(
        {
            "events": [],
            "correlation_id": task_record.correlation_id or None,
        },
        event_family="presence.progress",
        phase="thinking",
        detail={
            "tool": str(proposal.get("tool_name") or ""),
            "intention": "planning_next_step",
            "text": hint[:_PLANNER_INTENT_MAX_LENGTH] if hint else "",
        },
    )


def _execute_tool_call(
    *,
    task_record: TaskRecord,
    tool_name: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    definition = _tool_registry().get(tool_name)
    if definition is None:
        raise RuntimeError(f"tool_not_found:{tool_name}")
    execute = getattr(definition.executor, "execute", None)
    if not callable(execute):
        raise RuntimeError(f"tool_not_executable:{tool_name}")
    try:
        call_args = dict(args or {})
        signature = inspect.signature(execute)
        accepts_state = "state" in signature.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
        )
        if accepts_state:
            call_args["state"] = {
                "correlation_id": task_record.correlation_id or None,
                "task_record": task_record,
            }
        raw_result = definition.invoke(call_args)
    except Exception as exc:
        as_payload = getattr(exc, "as_payload", None)
        if callable(as_payload):
            error_payload = as_payload()
            if not isinstance(error_payload, dict):
                error_payload = {"code": "tool_execution_exception", "message": str(exc)}
        else:
            error_payload = {
                "code": "tool_execution_exception",
                "message": str(exc) or type(exc).__name__,
                "details": {"exception_type": type(exc).__name__},
            }
        return {
            "output": None,
            "exception": _coerce_error(error_payload),
            "metadata": {"tool": tool_name},
        }
    return _coerce_tool_result(tool_name=tool_name, raw_result=raw_result)


def _append_tool_call_history_entry(
    *,
    task_record: TaskRecord,
    tool_name: str,
    args: dict[str, Any],
    output: Any,
    exception: Any,
) -> None:
    redacted_args = _redact_sensitive(dict(args or {}))
    line = (
        f"{tool_name} "
        f"args={_compact_json(redacted_args)} "
        f"output={_compact_json(output)} "
        f"exception={_compact_json(exception)}"
    )
    task_record.append_tool_call_history_entry(line)


def _merge_stable_facts(*, task_record: TaskRecord, tool_name: str, output: Any) -> None:
    if not isinstance(output, dict):
        return
    if tool_name == "get_user_details":
        for key in (
            "actor_person_id",
            "user_id",
            "resolved_user_id",
            "incoming_user_id",
            "channel_type",
            "channel_target",
            "service_id",
            "service_key",
            "chat_id",
            "conversation_key",
        ):
            if key in output:
                task_record.append_fact(f"{key}: {_compact_json(output.get(key))}")
    elif tool_name == "get_my_settings":
        for key in ("locale", "tone", "address_style", "timezone", "channel_type"):
            if key in output:
                task_record.append_fact(f"{key}: {_compact_json(output.get(key))}")
    elif tool_name == "users.search":
        task_record.append_fact(f"users.search.last_result: {_compact_json(output)}")


def _append_memory_fact(
    *,
    task_record: TaskRecord,
    tool_name: str,
    output: Any,
    exception: Any,
) -> None:
    if not str(tool_name or "").strip().startswith("memory."):
        return
    task_record.append_memory_fact(
        f"{tool_name} output={_compact_json(output)} exception={_compact_json(exception)}"
    )


def _record_execution_hooks(
    *,
    task_record: TaskRecord,
    tool_name: str,
    args: dict[str, Any],
    result: dict[str, Any],
    planner_output: PlannerOutput,
) -> None:
    state = {
        "correlation_id": task_record.correlation_id or None,
        "task_record": task_record,
    }
    current = {"status": "failed" if _has_exception_payload(result.get("exception")) else "executed"}
    record_after_tool_call(        
        task_record=task_record,
        current=current,
        tool_name=tool_name,
        args=dict(args or {}),
        result=result,
        correlation_id=task_record.correlation_id or None,
    )
    record_plan_step_completion(       
        task_record=task_record,
        current=current,
        proposal=planner_output.get("tool_call") if isinstance(planner_output.get("tool_call"), dict) else {},
        correlation_id=task_record.correlation_id or None,
    )


def _log_execution(
    *,
    task_record: TaskRecord,
    tool_name: str,
    planner_intent: str,
    logger: Any,
    log_task_event: Any,
) -> None:
    logger.info(
        "task_mode do executed tool=%s task_id=%s",
        tool_name,
        str(task_record.task_id or ""),
    )
    log_task_event(
        logger=logger,
        state={
            "correlation_id": task_record.correlation_id or None,
            "channel_type": None,
            "actor_person_id": task_record.user_id,
        },
        node="execute_step_node",
        event="graph.execute_step.completed",
        task_record=task_record,
        cycle_index=0,
        tool_name=tool_name,
        planner_intent=planner_intent,
    )


@lru_cache(maxsize=1)
def _tool_registry() -> ToolRegistry:
    return build_default_tool_registry()


def _coerce_tool_result(*, tool_name: str, raw_result: Any) -> dict[str, Any]:
    return ensure_tool_result(tool_key=tool_name, value=raw_result)


def _coerce_error(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        code = str(value.get("code") or "tool_execution_error")
        message = str(value.get("message") or code)
        retryable = bool(value.get("retryable", False))
        details = value.get("details")
        if not isinstance(details, dict):
            details = {}
        return {"code": code, "message": message, "retryable": retryable, "details": details}
    if isinstance(value, str):
        text = value.strip() or "tool_execution_error"
        return {"message": text}
    return {"message": "Tool execution failed"}


def _has_exception_payload(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        if str(value.get("message") or "").strip():
            return True
        if str(value.get("code") or "").strip():
            return True
        return bool(value)
    return True


def _serialize_result(result: Any) -> Any:
    if isinstance(result, (dict, list, str, int, float, bool)) or result is None:
        return result
    return str(result)


def _redact_sensitive(value: Any) -> Any:
    secret_tokens = ("password", "pass", "token", "api_key", "secret", "authorization")
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key or "")
            lowered = key_text.lower()
            if any(token in lowered for token in secret_tokens):
                out[key_text] = "[redacted]"
            else:
                out[key_text] = _redact_sensitive(item)
        return out
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


def _compact_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)[:500]
    except Exception:
        return str(value)[:500]
