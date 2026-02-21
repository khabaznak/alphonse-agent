from __future__ import annotations

import logging
import re
from typing import Any, Callable


def execute_step_node_impl(
    state: dict[str, Any],
    *,
    tool_registry: Any,
    task_state_with_defaults: Callable[[dict[str, Any]], dict[str, Any]],
    correlation_id: Callable[[dict[str, Any]], str | None],
    current_step: Callable[[dict[str, Any]], dict[str, Any] | None],
    append_trace_event: Callable[[dict[str, Any], dict[str, Any]], None],
    serialize_result: Callable[[Any], Any],
    execute_tool_call: Callable[..., Any],
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
    if kind == "ask_user":
        question = str(proposal.get("question") or "What task should I run?").strip()
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

    if kind == "finish":
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

    if kind != "call_tool":
        return {"task_state": task_state}

    tool_name = str(proposal.get("tool_name") or "").strip()
    args = proposal.get("args")
    params = dict(args) if isinstance(args, dict) else {}
    step_id = str((current or {}).get("step_id") or "")
    try:
        result = execute_tool_call(
            state=state,
            tool_registry=tool_registry,
            tool_name=tool_name,
            args=params,
        )
        result = _maybe_retry_repaired_reminder_call(
            state=state,
            task_state=task_state,
            tool_registry=tool_registry,
            tool_name=tool_name,
            args=params,
            result=result,
            correlation_id=correlation_id,
            execute_tool_call=execute_tool_call,
            logger=logger,
        )
        facts = task_state.get("facts")
        fact_bucket = dict(facts) if isinstance(facts, dict) else {}
        result_payload = serialize_result(result)
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
        return {"task_state": task_state}


def _maybe_retry_repaired_reminder_call(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    tool_registry: Any,
    tool_name: str,
    args: dict[str, Any],
    result: Any,
    correlation_id: Callable[[dict[str, Any]], str | None],
    execute_tool_call: Callable[..., Any],
    logger: logging.Logger,
) -> Any:
    if tool_name != "createReminder" or not isinstance(result, dict):
        return result
    if str(result.get("status") or "").strip().lower() != "failed":
        return result
    error = result.get("error")
    if not isinstance(error, dict):
        return result
    error_code = str(error.get("code") or "").strip().lower()
    if error_code not in {"missing_time", "missing_message", "missing_for_whom"}:
        return result
    repaired_args = _repair_create_reminder_args(
        state=state,
        task_state=task_state,
        args=args,
        error_code=error_code,
    )
    if repaired_args == args:
        return result
    logger.info(
        "task_mode execute retry_repaired_reminder correlation_id=%s error_code=%s",
        correlation_id(state),
        error_code,
    )
    return execute_tool_call(
        state=state,
        tool_registry=tool_registry,
        tool_name=tool_name,
        args=repaired_args,
    )


def _repair_create_reminder_args(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    args: dict[str, Any],
    error_code: str,
) -> dict[str, Any]:
    repaired = dict(args)
    needs_whom = error_code == "missing_for_whom" or not _arg_value(repaired, "ForWhom", "for_whom", "To")
    needs_time = error_code == "missing_time" or not _arg_value(repaired, "Time", "time")
    needs_message = error_code == "missing_message" or not _arg_value(repaired, "Message", "message")

    if needs_whom:
        for_whom = str(state.get("channel_target") or "").strip() or "me"
        _set_arg_value(args=repaired, canonical_key="ForWhom", alias_keys=["for_whom", "To"], value=for_whom)
    if needs_time:
        inferred_time = _infer_reminder_time_expression(state=state, task_state=task_state)
        if inferred_time:
            _set_arg_value(args=repaired, canonical_key="Time", alias_keys=["time"], value=inferred_time)
    if needs_message:
        inferred_message = _infer_reminder_message(state=state)
        if inferred_message:
            _set_arg_value(args=repaired, canonical_key="Message", alias_keys=["message"], value=inferred_message)
    return repaired


def _arg_value(args: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(args.get(key) or "").strip()
        if value:
            return value
    return ""


def _set_arg_value(*, canonical_key: str, alias_keys: list[str], value: str, args: dict[str, Any]) -> None:
    for key in (canonical_key, *alias_keys):
        if key in args:
            args[key] = value
            return
    args[canonical_key] = value


def _infer_reminder_time_expression(*, state: dict[str, Any], task_state: dict[str, Any]) -> str:
    candidates = [
        str(state.get("last_user_message") or "").strip(),
        str(task_state.get("goal") or "").strip(),
    ]
    for text in candidates:
        if not text:
            continue
        matched = _extract_time_expression(text)
        if matched:
            return matched
    return ""


def _extract_time_expression(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    patterns = [
        r"\bin\s+\d+\s*(?:s|sec|secs|second|seconds|min|mins|minute|minutes|hour|hours|day|days|week|weeks)\b",
        r"\bin\s+an?\s*(?:hour|day|week)\b",
        r"\btomorrow(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?\b",
        r"\btoday(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?\b",
        r"\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
    ]
    for pattern in patterns:
        matched = re.search(pattern, value, flags=re.IGNORECASE)
        if matched:
            return matched.group(0).strip()
    return ""


def _infer_reminder_message(*, state: dict[str, Any]) -> str:
    user_text = _extract_primary_user_text(state)
    if user_text:
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', user_text)
        for a, b in quoted:
            candidate = str(a or b).strip()
            if _is_viable_reminder_message(candidate):
                return candidate
        if _is_viable_reminder_message(user_text):
            return user_text
    locale = str(state.get("locale") or "").strip().lower()
    if locale.startswith("es"):
        return "Recordatorio"
    return "Reminder"


def _extract_primary_user_text(state: dict[str, Any]) -> str:
    candidates: list[str] = []
    incoming = state.get("incoming_raw_message")
    if isinstance(incoming, dict):
        direct_text = str(incoming.get("text") or "").strip()
        if direct_text:
            candidates.append(direct_text)
        provider_event = incoming.get("provider_event")
        if isinstance(provider_event, dict):
            provider_message = provider_event.get("message")
            if isinstance(provider_message, dict):
                provider_text = str(provider_message.get("text") or "").strip()
                if provider_text:
                    candidates.append(provider_text)
    last_user_message = str(state.get("last_user_message") or "").strip()
    if last_user_message:
        candidates.append(_extract_text_from_packed_input(last_user_message))
        candidates.append(last_user_message)
    for candidate in candidates:
        normalized = str(candidate or "").strip()
        if normalized:
            return normalized
    return ""


def _extract_text_from_packed_input(text: str) -> str:
    rendered = str(text or "")
    line_match = re.search(r"^- text:\s*(.+)$", rendered, flags=re.IGNORECASE | re.MULTILINE)
    if line_match:
        return str(line_match.group(1) or "").strip()
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", rendered, flags=re.DOTALL | re.IGNORECASE)
    if json_match:
        block = str(json_match.group(1) or "")
        message_match = re.search(r'"text"\s*:\s*"([^"]+)"', block)
        if message_match:
            return str(message_match.group(1) or "").strip()
    return str(text or "").strip()


def _is_viable_reminder_message(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    lower = candidate.lower()
    if lower in {"update_id", "message_id", "chat_id", "correlation_id", "signal_id", "id"}:
        return False
    if re.fullmatch(r"[a-z_]+_id", lower):
        return False
    return True
