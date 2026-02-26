from __future__ import annotations

import inspect
import logging
from typing import Any, Callable

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
    corr = correlation_id(state)
    current = current_step(task_state)
    proposal = (current or {}).get("proposal") if isinstance(current, dict) else None
    if not isinstance(proposal, dict):
        logger.info(
            "task_mode execute skipped correlation_id=%s reason=no_proposal",
            corr,
        )
        return {"task_state": task_state}

    kind = str(proposal.get("kind") or "").strip()
    handlers: dict[str, Callable[[], dict[str, Any]]] = {
        "ask_user": lambda: _execute_ask_user_step(
            state=state,
            task_state=task_state,
            current=current,
            proposal=proposal,
            corr=corr,
            append_trace_event=append_trace_event,
            logger=logger,
            log_task_event=log_task_event,
        ),
        "finish": lambda: _execute_finish_step(
            state=state,
            task_state=task_state,
            current=current,
            proposal=proposal,
            corr=corr,
            append_trace_event=append_trace_event,
            logger=logger,
            log_task_event=log_task_event,
        ),
        "call_tool": lambda: _execute_call_tool_step(
            state=state,
            task_state=task_state,
            current=current,
            proposal=proposal,
            corr=corr,
            tool_registry=tool_registry,
            append_trace_event=append_trace_event,
            logger=logger,
            log_task_event=log_task_event,
        ),
    }
    handler = handlers.get(kind)
    if handler is None:
        return {"task_state": task_state}
    return handler()


def _execute_ask_user_step(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    current: dict[str, Any] | None,
    proposal: dict[str, Any],
    corr: str | None,
    append_trace_event: Callable[[dict[str, Any], dict[str, Any]], None],
    logger: logging.Logger,
    log_task_event: Callable[..., None],
) -> dict[str, Any]:
    question = str(proposal.get("question") or "").strip()
    if not question:
        task_state["status"] = "failed"
        if isinstance(current, dict):
            current["status"] = "failed"
        append_trace_event(
            task_state,
            {
                "type": "validation_failed",
                "summary": "ask_user proposal reached execute without a question.",
                "correlation_id": corr,
            },
        )
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="execute_step_node",
            event="graph.step.ask_user_invalid",
            level="warning",
            step_id=str((current or {}).get("step_id") or ""),
            reason="missing_question",
        )
        return {"task_state": task_state}
    task_state["status"] = "waiting_user"
    task_state["next_user_question"] = question
    if isinstance(current, dict):
        current["status"] = "executed"
    append_trace_event(
        task_state,
        {
            "type": "status_changed",
            "summary": "Status changed to waiting_user by ask_user proposal.",
            "correlation_id": corr,
        },
    )
    logger.info(
        "task_mode execute ask_user correlation_id=%s step_id=%s question=%s",
        corr,
        str((current or {}).get("step_id") or ""),
        question,
    )
    log_task_event(
        logger=logger,
        state=state,
        task_state=task_state,
        node="execute_step_node",
        event="graph.step.ask_user",
        step_id=str((current or {}).get("step_id") or ""),
    )
    return {"task_state": task_state}


def _execute_finish_step(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    current: dict[str, Any] | None,
    proposal: dict[str, Any],
    corr: str | None,
    append_trace_event: Callable[[dict[str, Any], dict[str, Any]], None],
    logger: logging.Logger,
    log_task_event: Callable[..., None],
) -> dict[str, Any]:
    final_text = str(proposal.get("final_text") or "").strip()
    task_state["status"] = "done"
    task_state["outcome"] = {
        "kind": "task_completed",
        "final_text": final_text,
    }
    if isinstance(current, dict):
        current["status"] = "executed"
    append_trace_event(
        task_state,
        {
            "type": "status_changed",
            "summary": "Status changed to done by finish proposal.",
            "correlation_id": corr,
        },
    )
    logger.info(
        "task_mode execute finish correlation_id=%s step_id=%s",
        corr,
        str((current or {}).get("step_id") or ""),
    )
    log_task_event(
        logger=logger,
        state=state,
        task_state=task_state,
        node="execute_step_node",
        event="graph.task.completed",
        step_id=str((current or {}).get("step_id") or ""),
    )
    return {"task_state": task_state}


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
    send_after_search = _is_send_after_user_search(tool_name=tool_name, task_state=task_state)
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
        fact_entry = {
            "tool": tool_name,
            "result": result_payload,
        }
        fact_bucket[step_id or f"result_{len(fact_bucket) + 1}"] = fact_entry
        task_state["facts"] = fact_bucket
        result_status = ""
        if isinstance(result, dict):
            result_status = str(result.get("status") or "").strip().lower()
        if result_status == "failed":
            task_state["status"] = "running"
            if isinstance(current, dict):
                current["status"] = "failed"
            raw_error = (result or {}).get("error") if isinstance(result, dict) else None
            if isinstance(raw_error, dict):
                error_code = str(raw_error.get("code") or "tool_failed")
                error_message = str(raw_error.get("message") or "").strip()
            else:
                error_code = str(raw_error or "tool_failed")
                error_message = ""
            append_trace_event(
                task_state,
                {
                    "type": "tool_failed",
                    "summary": (
                        f"Tool {tool_name} reported failure: {error_code}"
                        + (f" ({error_message})." if error_message else ".")
                    ),
                    "correlation_id": corr,
                },
            )
            logger.info(
                "task_mode execute tool_failed_reported correlation_id=%s step_id=%s tool=%s error_code=%s error_message=%s",
                corr,
                step_id,
                tool_name,
                error_code,
                error_message,
            )
            if send_after_search:
                logger.info(
                    "task_mode metric user_search_to_sendMessage correlation_id=%s outcome=failed error_code=%s",
                    corr,
                    error_code,
                )
            log_task_event(
                logger=logger,
                state=state,
                task_state=task_state,
                node="execute_step_node",
                event="graph.tool.failed",
                level="warning",
                step_id=step_id,
                tool=tool_name,
                error_code=error_code,
                error_message=error_message,
            )
            return {"task_state": task_state}
        derived_outcome = _derive_tool_outcome_from_result(
            tool_name=tool_name,
            result=result,
            state=state,
        )
        if isinstance(derived_outcome, dict):
            task_state["outcome"] = derived_outcome
        task_state["status"] = "running"
        if isinstance(current, dict):
            current["status"] = "executed"
        append_trace_event(
            task_state,
            {
                "type": "tool_executed",
                "summary": f"Executed tool {tool_name}.",
                "correlation_id": corr,
            },
        )
        logger.info(
            "task_mode execute tool_ok correlation_id=%s step_id=%s tool=%s",
            corr,
            step_id,
            tool_name,
        )
        if send_after_search:
            logger.info(
                "task_mode metric user_search_to_sendMessage correlation_id=%s outcome=succeeded",
                corr,
            )
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="execute_step_node",
            event="graph.tool.succeeded",
            step_id=step_id,
            tool=tool_name,
        )
        return {"task_state": task_state}
    except Exception as exc:
        task_state["status"] = "failed"
        if isinstance(current, dict):
            current["status"] = "failed"
        append_trace_event(
            task_state,
            {
                "type": "tool_failed",
                "summary": f"Tool {tool_name} failed: {type(exc).__name__}.",
                "correlation_id": corr,
            },
        )
        logger.info(
            "task_mode execute tool_failed correlation_id=%s step_id=%s tool=%s error=%s",
            corr,
            step_id,
            tool_name,
            type(exc).__name__,
        )
        if send_after_search:
            logger.info(
                "task_mode metric user_search_to_sendMessage correlation_id=%s outcome=exception error=%s",
                corr,
                type(exc).__name__,
            )
        return {"task_state": task_state}


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


def _derive_tool_outcome_from_result(
    *,
    tool_name: str,
    result: Any,
    state: dict[str, Any],
) -> dict[str, Any] | None:
    if tool_name not in {"create_reminder", "createReminder"} or not isinstance(result, dict):
        return None
    payload = result
    status = str(result.get("status") or "").strip().lower()
    if status in {"ok", "executed"} and isinstance(result.get("result"), dict):
        payload = dict(result.get("result") or {})
    reminder_id = str(payload.get("reminder_id") or "").strip()
    fire_at = str(payload.get("fire_at") or "").strip()
    if not reminder_id or not fire_at:
        return None
    message = str(payload.get("message") or "").strip()
    for_whom = str(payload.get("for_whom") or state.get("channel_target") or "").strip()
    if not for_whom:
        for_whom = str(payload.get("delivery_target") or "").strip()
    return {
        "kind": "reminder_created",
        "evidence": {
            "reminder_id": reminder_id,
            "fire_at": fire_at,
            "message": message,
            "for_whom": for_whom,
        },
    }


def _is_send_after_user_search(*, tool_name: str, task_state: dict[str, Any]) -> bool:
    if str(tool_name or "").strip() not in {"send_message", "sendMessage"}:
        return False
    facts = task_state.get("facts")
    if not isinstance(facts, dict):
        return False
    for _, entry in reversed(list(facts.items())):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("tool") or "").strip() == "user_search":
            return True
    return False
