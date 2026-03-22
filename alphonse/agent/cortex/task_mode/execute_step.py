from __future__ import annotations

import inspect
import logging
from datetime import datetime, timezone
from typing import Any, Callable

from alphonse.agent.cognition.memory import record_after_tool_call
from alphonse.agent.cognition.memory import record_plan_step_completion
from alphonse.agent.cortex.transitions import emit_presence_transition_event
from alphonse.agent.tools.base import ensure_tool_result


def execute_step_node_impl(
    state: dict[str, Any],
    *,
    tool_registry: Any,
    task_state_with_defaults: Callable[[dict[str, Any]], dict[str, Any]],
    correlation_id: Callable[[dict[str, Any]], str | None],
    current_step: Callable[[dict[str, Any]], dict[str, Any] | None],
    append_trace_event: Callable[[dict[str, Any], dict[str, Any]], None],
    logger: logging.Logger,
    log_task_event: Callable[..., None],
) -> dict[str, Any]:
    _ = (logger, log_task_event)
    task_state = task_state_with_defaults(state)
    task_state["pdca_phase"] = "do"
    task_state["check_provenance"] = "do"
    corr = correlation_id(state)
    current = current_step(task_state)
    proposal, proposal_error, planner_intent = _proposal_from_pending_plan_raw(task_state=task_state, current=current)
    if isinstance(proposal_error, dict):
        _record_planner_output_error(
            task_state=task_state,
            current=current,
            corr=corr,
            error_payload=proposal_error,
            append_trace_event=append_trace_event,
        )
        return {"task_state": task_state}
    if not isinstance(proposal, dict):
        _record_planner_output_error(
            task_state=task_state,
            current=current,
            corr=corr,
            error_payload={
                "code": "invalid_planner_output",
                "raw_error": "no_executable_tool_call",
                "details": {},
            },
            append_trace_event=append_trace_event,
        )
        return {"task_state": task_state}

    kind = str(proposal.get("kind") or "").strip()
    if kind != "call_tool":
        _record_planner_output_error(
            task_state=task_state,
            current=current,
            corr=corr,
            error_payload={
                "code": "invalid_planner_output",
                "raw_error": f"unsupported_proposal_kind:{kind or 'unknown'}",
                "details": {"proposal_kind": kind},
            },
            append_trace_event=append_trace_event,
        )
        return {"task_state": task_state}
    _emit_planner_intent_progress(
        state=state,
        task_state=task_state,
        proposal=proposal,
        planner_intent=planner_intent,
    )
    return _execute_call_tool_step(
        state=state,
        task_state=task_state,
        current=current,
        proposal=proposal,
        corr=corr,
        tool_registry=tool_registry,
        append_trace_event=append_trace_event,
    )


def _record_planner_output_error(
    *,
    task_state: dict[str, Any],
    current: dict[str, Any] | None,
    corr: str | None,
    error_payload: dict[str, Any],
    append_trace_event: Callable[[dict[str, Any], dict[str, Any]], None],
) -> None:
    step_id = str((current or {}).get("step_id") or "")
    task_state["status"] = "running"
    error_code = str(error_payload.get("code") or "invalid_planner_output")
    raw_error = str(error_payload.get("raw_error") or error_code)
    details = error_payload.get("details") if isinstance(error_payload.get("details"), dict) else {}
    facts = task_state.get("facts")
    fact_bucket = dict(facts) if isinstance(facts, dict) else {}
    fact_entry = _build_mission_fact_entry(
        step_id=step_id,
        tool_name="planner_output",
        args={},
        output=None,
        exception={
            "code": error_code,
            "raw_error": raw_error,
            "details": details,
        },
    )
    fact_bucket[step_id or f"result_{len(fact_bucket) + 1}"] = fact_entry
    task_state["facts"] = fact_bucket
    append_trace_event(
        task_state,
        {
            "type": "planner_output_invalid",
            "summary": f"Planner output invalid: {raw_error}.",
            "correlation_id": corr,
        },
    )


def _proposal_from_pending_plan_raw(
    *,
    task_state: dict[str, Any],
    current: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    existing = (current or {}).get("proposal") if isinstance(current, dict) else None
    if isinstance(existing, dict):
        canonical = _coerce_canonical_tool_call(existing)
        if isinstance(canonical, dict):
            return canonical, None, ""
        return None, {
            "code": "invalid_planner_output",
            "raw_error": "current_step_proposal_non_canonical",
            "details": {},
        }, ""
    raw = task_state.get("pending_plan_raw")
    if isinstance(raw, dict):
        tool_call = raw.get("tool_call")
        if isinstance(tool_call, dict):
            canonical = _coerce_canonical_tool_call(tool_call)
            if isinstance(canonical, dict):
                planner_intent = _extract_planner_intent(raw)
                task_state["current_plan_step"] = {
                    "step_id": str((current or {}).get("step_id") or ""),
                    "tool_call": dict(canonical),
                }
                if isinstance(current, dict):
                    current["proposal"] = dict(canonical)
                task_state["pending_plan_raw"] = None
                return dict(canonical), None, planner_intent
        task_state["current_plan_step"] = {
            "step_id": str((current or {}).get("step_id") or ""),
            "tool_call": None,
        }
        task_state["pending_plan_raw"] = None
    elif raw is not None:
        task_state["pending_plan_raw"] = None
    return None, {
        "code": "invalid_planner_output",
        "raw_error": "pending_plan_raw_non_canonical",
        "details": {},
    }, ""


def _coerce_canonical_tool_call(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        kind = str(raw.get("kind") or "").strip()
        tool_name = str(raw.get("tool_name") or "").strip()
        args = raw.get("args")
        if kind == "call_tool" and tool_name and isinstance(args, dict):
            return {"kind": "call_tool", "tool_name": tool_name, "args": dict(args)}
    return None


def _extract_planner_intent(raw: dict[str, Any]) -> str:
    value = raw.get("planner_intent")
    if not isinstance(value, str):
        return ""
    text = value.strip()
    return text[:160] if text else ""


def _emit_planner_intent_progress(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    proposal: dict[str, Any],
    planner_intent: str,
) -> None:
    hint = str(planner_intent or "").strip()
    emit_presence_transition_event(
        state,
        event_family="presence.progress",
        phase="thinking",
        detail={
            "cycle": int(task_state.get("cycle_index") or 0) + 1,
            "tool": str(proposal.get("tool_name") or ""),
            "intention": "planning_next_step",
            "text": hint[:160] if hint else "",
        },
    )


def _execute_call_tool_step(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    current: dict[str, Any] | None,
    proposal: dict[str, Any],
    corr: str | None,
    tool_registry: Any,
    append_trace_event: Callable[[dict[str, Any], dict[str, Any]], None],
) -> dict[str, Any]:
    tool_name = str(proposal.get("tool_name") or "").strip()
    args = proposal.get("args")
    params = dict(args) if isinstance(args, dict) else {}
    step_id = str((current or {}).get("step_id") or "")
    result = _execute_tool_call(
        state=state,
        tool_registry=tool_registry,
        tool_name=tool_name,
        args=params,
    )
    facts = task_state.get("facts")
    fact_bucket = dict(facts) if isinstance(facts, dict) else {}
    output_payload = _serialize_result(result.get("output")) if isinstance(result, dict) else None
    exception_payload = _serialize_result(result.get("exception")) if isinstance(result, dict) else {"message": "unknown_tool_error"}
    fact_entry = _build_mission_fact_entry(
        step_id=step_id,
        tool_name=tool_name,
        args=params,
        output=output_payload,
        exception=exception_payload,
    )
    fact_bucket[step_id or f"result_{len(fact_bucket) + 1}"] = fact_entry
    task_state["facts"] = fact_bucket
    has_exception = _has_exception_payload(exception_payload)
    task_state["status"] = "running"
    if isinstance(current, dict):
        current["status"] = "failed" if has_exception else "executed"
        current["failure_retryable"] = _exception_retryable(exception_payload)
    append_trace_event(
        task_state,
        {
            "type": "tool_executed",
            "summary": f"Executed tool {tool_name} with exception={'yes' if has_exception else 'no'}.",
            "correlation_id": corr,
        },
    )
    record_after_tool_call(
        state=state,
        task_state=task_state,
        current=current,
        tool_name=tool_name,
        args=params,
        result=result if isinstance(result, dict) else {"output": None, "exception": {"message": "unknown_tool_error"}, "metadata": {}},
        correlation_id=corr,
    )
    record_plan_step_completion(
        state=state,
        task_state=task_state,
        current=current,
        proposal=proposal,
        correlation_id=corr,
    )
    return {"task_state": task_state}


def _build_mission_fact_entry(
    *,
    step_id: str,
    tool_name: str,
    args: dict[str, Any],
    output: Any,
    exception: Any,
) -> dict[str, Any]:
    redacted_args = _redact_sensitive(dict(args or {}))
    return {
        "step_id": step_id or None,
        "tool_name": tool_name,
        "params": redacted_args,
        "output": _serialize_result(output),
        "exception": _serialize_result(exception),
        # Compatibility aliases during migration.
        "tool": tool_name,
        "args": redacted_args,
        "result": {
            "output": _serialize_result(output),
            "exception": _serialize_result(exception),
        },
        "internal": False,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def _execute_tool_call(
    *,
    state: dict[str, Any],
    tool_registry: Any,
    tool_name: str,
    args: dict[str, Any],
) -> Any:
    definition = tool_registry.get(tool_name) if hasattr(tool_registry, "get") else None
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
            call_args["state"] = state
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


def _exception_retryable(value: Any) -> bool | None:
    if not isinstance(value, dict):
        return None
    if "retryable" not in value:
        return None
    return bool(value.get("retryable"))


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
