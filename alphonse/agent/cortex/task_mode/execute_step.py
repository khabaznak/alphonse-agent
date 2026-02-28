from __future__ import annotations

import inspect
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Callable

from alphonse.agent.cognition.memory import append_episode
from alphonse.agent.cognition.memory import put_artifact
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
    _record_step_completion_memory(
        state=state,
        task_state=task_state,
        current=current,
        proposal=proposal,
        correlation_id=corr,
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
    _record_step_completion_memory(
        state=state,
        task_state=task_state,
        current=current,
        proposal=proposal,
        correlation_id=corr,
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
    terminal_context = _terminal_command_context(tool_name=tool_name, args=params)
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
            args_preview = _safe_args_preview(params)
            raw_error = (result or {}).get("error") if isinstance(result, dict) else None
            if isinstance(raw_error, dict):
                error_code = str(raw_error.get("code") or "tool_failed")
                error_message = str(raw_error.get("message") or "").strip()
                error_details = raw_error.get("details")
            else:
                error_code = str(raw_error or "tool_failed")
                error_message = ""
                error_details = None
            failure_context = _tool_failure_context(
                tool_name=tool_name,
                result=result if isinstance(result, dict) else {},
                error_details=error_details if isinstance(error_details, dict) else {},
            )
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
                "task_mode execute tool_failed_reported correlation_id=%s step_id=%s tool=%s error_code=%s error_message=%s args_preview=%s",
                corr,
                step_id,
                tool_name,
                error_code,
                error_message,
                args_preview,
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
                args_preview=args_preview,
                **terminal_context,
                **failure_context,
            )
            _record_after_tool_call_memory(
                state=state,
                task_state=task_state,
                current=current,
                tool_name=tool_name,
                args=params,
                result=result if isinstance(result, dict) else {"status": "failed"},
                correlation_id=corr,
            )
            _record_step_completion_memory(
                state=state,
                task_state=task_state,
                current=current,
                proposal=proposal,
                correlation_id=corr,
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
            **terminal_context,
        )
        _record_after_tool_call_memory(
            state=state,
            task_state=task_state,
            current=current,
            tool_name=tool_name,
            args=params,
            result=result if isinstance(result, dict) else {"status": "ok"},
            correlation_id=corr,
        )
        _record_step_completion_memory(
            state=state,
            task_state=task_state,
            current=current,
            proposal=proposal,
            correlation_id=corr,
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
        _record_after_tool_call_memory(
            state=state,
            task_state=task_state,
            current=current,
            tool_name=tool_name,
            args=params,
            result={
                "status": "failed",
                "error": {"code": "tool_execution_exception", "message": str(exc), "details": {"type": type(exc).__name__}},
            },
            correlation_id=corr,
        )
        _record_step_completion_memory(
            state=state,
            task_state=task_state,
            current=current,
            proposal=proposal,
            correlation_id=corr,
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


def _terminal_command_context(*, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    normalized = str(tool_name or "").strip().lower()
    if normalized not in {"terminal_sync", "terminal_async", "ssh_terminal"}:
        return {}
    command = str(args.get("command") or "").strip()
    cwd = str(args.get("cwd") or "").strip()
    return {
        "terminal_command": command,
        "terminal_cwd": cwd,
    }


def _tool_failure_context(*, tool_name: str, result: dict[str, Any], error_details: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if str(tool_name or "").strip() != "mcp_call":
        return context
    result_payload = result.get("result") if isinstance(result.get("result"), dict) else {}
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    mcp_command = str(
        error_details.get("mcp_command")
        or metadata.get("mcp_command")
        or ""
    ).strip()
    stderr_preview = str(
        error_details.get("stderr_preview")
        or result_payload.get("stderr")
        or ""
    ).strip()
    stdout_preview = str(
        error_details.get("stdout_preview")
        or result_payload.get("stdout")
        or ""
    ).strip()
    if mcp_command:
        context["mcp_command"] = mcp_command
    if stderr_preview:
        context["stderr_preview"] = stderr_preview[:600]
    if stdout_preview:
        context["stdout_preview"] = stdout_preview[:400]
    return context


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


def _safe_args_preview(args: dict[str, Any]) -> str:
    redacted = _redact_sensitive(args)
    try:
        rendered = json.dumps(redacted, ensure_ascii=False, sort_keys=True)
    except Exception:
        rendered = str(redacted)
    return rendered[:800]


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


def _record_after_tool_call_memory(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    current: dict[str, Any] | None,
    tool_name: str,
    args: dict[str, Any],
    result: dict[str, Any],
    correlation_id: str | None,
) -> None:
    if not _memory_hooks_enabled():
        return
    if str(tool_name or "").strip() in {"search_episodes", "get_mission", "list_active_missions", "get_workspace_pointer"}:
        return
    user_id = _memory_user_id(state)
    mission_id = _memory_mission_id(task_state=task_state, correlation_id=correlation_id)
    artifacts = _memory_artifacts_from_result(
        mission_id=mission_id,
        tool_name=tool_name,
        result=result,
    )
    status = str((result or {}).get("status") or "").strip().lower()
    payload = {
        "intent": str(task_state.get("goal") or "").strip() or "task execution",
        "action": f"{tool_name}({_safe_args_preview(args)})",
        "result": f"status={status or 'unknown'}",
        "step_id": str((current or {}).get("step_id") or "").strip() or None,
        "next": _memory_next_hint(task_state=task_state, current=current),
    }
    _memory_append_episode(
        user_id=user_id,
        mission_id=mission_id,
        event_type="after_tool_call",
        payload=payload,
        artifacts=artifacts,
    )


def _record_step_completion_memory(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    current: dict[str, Any] | None,
    proposal: dict[str, Any],
    correlation_id: str | None,
) -> None:
    if not _memory_hooks_enabled():
        return
    user_id = _memory_user_id(state)
    mission_id = _memory_mission_id(task_state=task_state, correlation_id=correlation_id)
    payload = {
        "intent": str(task_state.get("goal") or "").strip() or "task execution",
        "step_id": str((current or {}).get("step_id") or "").strip() or None,
        "action": str(proposal.get("kind") or "").strip(),
        "result": f"step_status={str((current or {}).get('status') or '').strip() or 'unknown'}",
        "next": _memory_next_hint(task_state=task_state, current=current),
    }
    _memory_append_episode(
        user_id=user_id,
        mission_id=mission_id,
        event_type="plan_step_completed",
        payload=payload,
        artifacts=[],
    )


def _memory_hooks_enabled() -> bool:
    raw = str(os.getenv("ALPHONSE_MEMORY_HOOKS_ENABLED", "true")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _memory_user_id(state: dict[str, Any]) -> str:
    for key in ("incoming_user_id", "actor_person_id", "channel_target", "chat_id", "conversation_key"):
        value = str(state.get(key) or "").strip()
        if value:
            return value
    return "anonymous"


def _memory_mission_id(*, task_state: dict[str, Any], correlation_id: str | None) -> str:
    explicit = str(task_state.get("mission_id") or "").strip()
    if explicit:
        return explicit
    task_id = str(task_state.get("task_id") or "").strip()
    if task_id:
        return task_id
    corr = str(correlation_id or "").strip()
    if corr:
        return f"corr_{corr}"
    return "ad_hoc"


def _memory_next_hint(*, task_state: dict[str, Any], current: dict[str, Any] | None) -> str:
    status = str(task_state.get("status") or "").strip().lower()
    if status == "waiting_user":
        return str(task_state.get("next_user_question") or "waiting for user input").strip()
    if status == "done":
        return "mission complete"
    return f"continue from step {str((current or {}).get('step_id') or '').strip() or '?'}"


def _memory_append_episode(
    *,
    user_id: str,
    mission_id: str,
    event_type: str,
    payload: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> None:
    try:
        append_episode(
            user_id=user_id,
            mission_id=mission_id,
            event_type=event_type,
            payload=payload,
            artifacts=artifacts,
        )
    except Exception:
        return


def _memory_artifacts_from_result(
    *,
    mission_id: str,
    tool_name: str,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    max_items = max(1, int(str(os.getenv("ALPHONSE_MEMORY_MAX_ARTIFACTS_PER_EVENT") or "5").strip() or "5"))
    max_bytes = max(1024, int(str(os.getenv("ALPHONSE_MEMORY_MAX_ARTIFACT_SIZE_BYTES") or str(20 * 1024 * 1024)).strip() or str(20 * 1024 * 1024)))
    paths = _extract_path_candidates(result)
    seen: set[str] = set()
    for candidate in paths:
        if len(refs) >= max_items:
            break
        text = str(candidate or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        if text.startswith("http://") or text.startswith("https://"):
            refs.append({"url": text})
            continue
        file_path = Path(text)
        if not file_path.exists() or not file_path.is_file():
            continue
        try:
            size = file_path.stat().st_size
        except Exception:
            continue
        if size > max_bytes:
            refs.append({"path": str(file_path), "note": "artifact_too_large"})
            continue
        try:
            data = file_path.read_bytes()
            mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            artifact_ref = put_artifact(
                mission_id=mission_id,
                content=data,
                mime=mime,
                name_hint=file_path.name,
            )
            if isinstance(artifact_ref, dict):
                artifact_ref.setdefault("source_path", str(file_path))
                artifact_ref.setdefault("tool", tool_name)
                refs.append(artifact_ref)
        except Exception:
            refs.append({"path": str(file_path), "note": "artifact_copy_failed"})
            continue
    return refs


def _extract_path_candidates(value: Any) -> list[str]:
    out: list[str] = []

    def _walk(node: Any, parent_key: str = "") -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                _walk(item, str(key))
            return
        if isinstance(node, list):
            for item in node:
                _walk(item, parent_key)
            return
        if not isinstance(node, str):
            return
        text = node.strip()
        if not text:
            return
        key_hint = parent_key.lower()
        if any(token in key_hint for token in ("path", "file", "url", "artifact", "snapshot")):
            out.append(text)
            return
        if text.startswith("/") or text.startswith("./") or text.startswith("../"):
            out.append(text)
            return
        if text.startswith("http://") or text.startswith("https://"):
            out.append(text)

    _walk(value)
    return out
