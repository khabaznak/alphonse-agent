from __future__ import annotations

import inspect
import logging
from datetime import datetime, timezone
from typing import Any, Callable

from alphonse.agent.cognition.memory import record_after_tool_call
from alphonse.agent.cognition.memory import record_plan_step_completion
from alphonse.agent.cortex.llm_output.json_parse import parse_json_object
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
    task_state = task_state_with_defaults(state)
    task_state["pdca_phase"] = "do"
    task_state["check_provenance"] = "do"
    corr = correlation_id(state)
    current = current_step(task_state)
    proposal, proposal_error = _proposal_from_pending_plan_raw(task_state=task_state, current=current)
    if isinstance(proposal_error, dict):
        _record_planner_output_error(
            state=state,
            task_state=task_state,
            current=current,
            corr=corr,
            error_payload=proposal_error,
            append_trace_event=append_trace_event,
            logger=logger,
            log_task_event=log_task_event,
        )
        return {"task_state": task_state}
    if not isinstance(proposal, dict):
        _record_planner_output_error(
            state=state,
            task_state=task_state,
            current=current,
            corr=corr,
            error_payload={
                "code": "invalid_planner_output",
                "raw_error": "no_executable_tool_call",
                "details": {},
            },
            append_trace_event=append_trace_event,
            logger=logger,
            log_task_event=log_task_event,
        )
        return {"task_state": task_state}
    task_state["planner_error_last"] = None

    kind = str(proposal.get("kind") or "").strip()
    if kind != "call_tool":
        _record_planner_output_error(
            state=state,
            task_state=task_state,
            current=current,
            corr=corr,
            error_payload={
                "code": "invalid_planner_output",
                "raw_error": f"unsupported_proposal_kind:{kind or 'unknown'}",
                "details": {"proposal_kind": kind},
            },
            append_trace_event=append_trace_event,
            logger=logger,
            log_task_event=log_task_event,
        )
        return {"task_state": task_state}
    return _execute_call_tool_step(
        state=state,
        task_state=task_state,
        current=current,
        proposal=proposal,
        corr=corr,
        tool_registry=tool_registry,
        append_trace_event=append_trace_event,
        logger=logger,
        log_task_event=log_task_event,
    )


def _record_planner_output_error(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    current: dict[str, Any] | None,
    corr: str | None,
    error_payload: dict[str, Any],
    append_trace_event: Callable[[dict[str, Any], dict[str, Any]], None],
    logger: logging.Logger,
    log_task_event: Callable[..., None],
) -> None:
    step_id = str((current or {}).get("step_id") or "")
    if isinstance(current, dict):
        current["status"] = "failed"
    task_state["status"] = "running"
    error_code = str(error_payload.get("code") or "invalid_planner_output")
    raw_error = str(error_payload.get("raw_error") or error_code)
    details = error_payload.get("details") if isinstance(error_payload.get("details"), dict) else {}
    task_state["planner_error_last"] = {"code": error_code, "raw_error": raw_error, "details": details}
    facts = task_state.get("facts")
    fact_bucket = dict(facts) if isinstance(facts, dict) else {}
    fact_bucket[step_id or f"result_{len(fact_bucket) + 1}"] = {
        "tool": "planner_output",
        "result": {
            "status": "failed",
            "error": {
                "code": error_code,
                "raw_error": raw_error,
                "details": details,
            },
        },
    }
    task_state["facts"] = fact_bucket
    append_trace_event(
        task_state,
        {
            "type": "planner_output_invalid",
            "summary": f"Planner output invalid: {raw_error}.",
            "correlation_id": corr,
        },
    )
    logger.info(
        "task_mode execute planner_output_invalid correlation_id=%s step_id=%s error_code=%s",
        corr,
        step_id,
        error_code,
    )
    log_task_event(
        logger=logger,
        state=state,
        task_state=task_state,
        node="execute_step_node",
        event="graph.plan_output.invalid",
        level="warning",
        step_id=step_id,
        error_code=error_code,
    )


def _proposal_from_pending_plan_raw(
    *,
    task_state: dict[str, Any],
    current: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    existing = (current or {}).get("proposal") if isinstance(current, dict) else None
    if isinstance(existing, dict):
        return dict(existing), None
    raw = task_state.get("pending_plan_raw")
    normalized = _normalize_raw_plan_artifact(raw)
    if isinstance(normalized, dict):
        tool_call = normalized.get("tool_call")
        if not isinstance(tool_call, dict):
            preview = str(raw)[:280] if raw is not None else ""
            return None, {
                "code": "invalid_planner_output",
                "raw_error": "no_parseable_tool_call",
                "details": {"raw_output_preview": preview},
            }
        task_state["current_plan_step"] = {
            "step_id": str((current or {}).get("step_id") or ""),
            "tool_call": dict(tool_call),
        }
        if isinstance(current, dict):
            current["proposal"] = dict(tool_call)
        task_state["pending_plan_raw"] = None
        return dict(tool_call), None
    preview = str(raw)[:280] if raw is not None else ""
    return None, {
        "code": "invalid_planner_output",
        "raw_error": "malformed_or_unparseable_tool_call",
        "details": {"raw_output_preview": preview},
    }


def _normalize_raw_plan_artifact(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        tool_call = raw.get("tool_call")
        if isinstance(tool_call, dict):
            normalized_tool_call = _normalize_raw_plan_candidate(tool_call)
            if isinstance(normalized_tool_call, dict):
                return {"tool_call": normalized_tool_call}
        normalized_candidate = _normalize_raw_plan_candidate(raw)
        if isinstance(normalized_candidate, dict):
            return {"tool_call": normalized_candidate}
        content = raw.get("content")
        if isinstance(content, str) and content.strip():
            parsed = parse_json_object(content)
            return _normalize_raw_plan_artifact(parsed)
        return None
    if isinstance(raw, str) and raw.strip():
        parsed = parse_json_object(raw)
        return _normalize_raw_plan_artifact(parsed)
    return None


def _normalize_raw_plan_candidate(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        tool_calls = raw.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                name = str(call.get("name") or "").strip()
                if not name:
                    continue
                arguments = call.get("arguments")
                args = dict(arguments) if isinstance(arguments, dict) else {}
                return {"kind": "call_tool", "tool_name": name, "args": args}
        kind = str(raw.get("kind") or "").strip()
        tool_name = str(raw.get("tool_name") or "").strip()
        args = raw.get("args")
        if kind == "call_tool" and tool_name and isinstance(args, dict):
            return {"kind": "call_tool", "tool_name": tool_name, "args": dict(args)}
        content = raw.get("content")
        if isinstance(content, str) and content.strip():
            parsed = parse_json_object(content)
            return _normalize_raw_plan_candidate(parsed)
        return None
    if isinstance(raw, str) and raw.strip():
        parsed = parse_json_object(raw)
        return _normalize_raw_plan_candidate(parsed)
    return None


def _execute_call_tool_step(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    current: dict[str, Any] | None,
    proposal: dict[str, Any],
    corr: str | None,
    tool_registry: Any,
    append_trace_event: Callable[[dict[str, Any], dict[str, Any]], None],
    logger: logging.Logger,
    log_task_event: Callable[..., None],
) -> dict[str, Any]:
    tool_name = str(proposal.get("tool_name") or "").strip()
    args = proposal.get("args")
    params = dict(args) if isinstance(args, dict) else {}
    step_id = str((current or {}).get("step_id") or "")
    try:
        result = _execute_tool_call(
            state=state,
            tool_registry=tool_registry,
            tool_name=tool_name,
            args=params,
        )
        facts = task_state.get("facts")
        fact_bucket = dict(facts) if isinstance(facts, dict) else {}
        result_payload = _serialize_result(result)
        fact_entry = _build_mission_fact_entry(
            step_id=step_id,
            tool_name=tool_name,
            args=params,
            result=result_payload,
        )
        fact_bucket[step_id or f"result_{len(fact_bucket) + 1}"] = fact_entry
        task_state["facts"] = fact_bucket
        result_status = str((result or {}).get("status") or "").strip().lower() if isinstance(result, dict) else ""
        task_state["status"] = "running"
        if isinstance(current, dict):
            current["status"] = "executed"
            current.pop("failure_retryable", None)
            current.pop("failure_error_code", None)
        append_trace_event(
            task_state,
            {
                "type": "tool_executed",
                "summary": f"Executed tool {tool_name} with reported_status={result_status or 'unknown'}.",
                "correlation_id": corr,
            },
        )
        logger.info(
            "task_mode execute tool_executed correlation_id=%s step_id=%s tool=%s reported_status=%s",
            corr,
            step_id,
            tool_name,
            result_status or "unknown",
        )
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="execute_step_node",
            event="graph.tool.executed",
            step_id=step_id,
            tool=tool_name,
            reported_status=result_status or "unknown",
        )
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="execute_step_node",
            event="graph.do.mission_step_executed",
            step_id=step_id,
            tool=tool_name,
            status=result_status or "unknown",
        )
        record_after_tool_call(
            state=state,
            task_state=task_state,
            current=current,
            tool_name=tool_name,
            args=params,
            result=result if isinstance(result, dict) else {"status": "unknown"},
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
    except Exception as exc:
        task_state["status"] = "running"
        if isinstance(current, dict):
            current["status"] = "executed"
            current.pop("failure_retryable", None)
            current.pop("failure_error_code", None)
        result = {
            "status": "failed",
            "result": None,
            "error": {
                "code": "tool_execution_exception",
                "message": str(exc),
                "details": {"type": type(exc).__name__},
            },
        }
        facts = task_state.get("facts")
        fact_bucket = dict(facts) if isinstance(facts, dict) else {}
        fact_entry = _build_mission_fact_entry(
            step_id=step_id,
            tool_name=tool_name,
            args=params,
            result=_serialize_result(result),
        )
        fact_bucket[step_id or f"result_{len(fact_bucket) + 1}"] = fact_entry
        task_state["facts"] = fact_bucket
        append_trace_event(
            task_state,
            {
                "type": "tool_executed",
                "summary": f"Executed tool {tool_name} with reported_status=failed.",
                "correlation_id": corr,
            },
        )
        logger.info(
            "task_mode execute tool_executed correlation_id=%s step_id=%s tool=%s reported_status=failed error=%s",
            corr,
            step_id,
            tool_name,
            type(exc).__name__,
        )
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="execute_step_node",
            event="graph.tool.executed",
            step_id=step_id,
            tool=tool_name,
            reported_status="failed",
            error_code="tool_execution_exception",
        )
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="execute_step_node",
            event="graph.do.mission_step_executed",
            step_id=step_id,
            tool=tool_name,
            status="failed",
        )
        record_after_tool_call(
            state=state,
            task_state=task_state,
            current=current,
            tool_name=tool_name,
            args=params,
            result=result,
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
    result: Any,
) -> dict[str, Any]:
    payload = result if isinstance(result, dict) else {}
    status = str(payload.get("status") or "").strip().lower() if isinstance(payload, dict) else ""
    result_payload = payload.get("result") if isinstance(payload, dict) else None
    error_payload = payload.get("error") if isinstance(payload, dict) else None
    return {
        "step_id": step_id or None,
        "tool": tool_name,
        "args": _redact_sensitive(dict(args or {})),
        "status": status or "unknown",
        "result": payload,
        "result_payload": _serialize_result(result_payload),
        "error": _serialize_result(error_payload),
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
    tool = tool_registry.get(tool_name) if hasattr(tool_registry, "get") else None
    if tool is None:
        raise RuntimeError(f"tool_not_found:{tool_name}")
    execute = getattr(tool, "execute", None)
    if not callable(execute):
        raise RuntimeError(f"tool_not_executable:{tool_name}")
    try:
        call_args = dict(args or {})
        signature = inspect.signature(execute)
        accepts_state = "state" in signature.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
        )
        if accepts_state:
            raw_result = execute(**call_args, state=state)
        else:
            raw_result = execute(**call_args)
    except TypeError as exc:
        return {
            "status": "failed",
            "result": None,
            "error": {
                "code": "invalid_tool_arguments",
                "message": str(exc) or "invalid tool arguments",
                "retryable": False,
                "details": {
                    "tool": tool_name,
                    "args_keys": sorted(list(dict(args or {}).keys())),
                },
            },
            "metadata": {"tool": tool_name},
        }
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
                "retryable": True,
                "details": {"exception_type": type(exc).__name__},
            }
        return {
            "status": "failed",
            "result": None,
            "error": _coerce_error(error_payload),
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
        return {"code": text, "message": text, "retryable": False, "details": {}}
    return {"code": "tool_execution_error", "message": "Tool execution failed", "retryable": False, "details": {}}


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
