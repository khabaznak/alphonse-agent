from __future__ import annotations

import logging
import re
import shlex
from typing import Any, Callable

from alphonse.agent.cognition.tool_schemas import planner_tool_schemas
from alphonse.agent.tools.mcp.registry import McpProfileRegistry


def validate_step_node_impl(
    state: dict[str, Any],
    *,
    tool_registry: Any,
    task_state_with_defaults: Callable[[dict[str, Any]], dict[str, Any]],
    correlation_id: Callable[[dict[str, Any]], str | None],
    task_plan: Callable[[dict[str, Any]], dict[str, Any]],
    current_step: Callable[[dict[str, Any]], dict[str, Any] | None],
    append_trace_event: Callable[[dict[str, Any], dict[str, Any]], None],
    logger: logging.Logger,
    log_task_event: Callable[..., None],
) -> dict[str, Any]:
    task_state = task_state_with_defaults(state)
    task_state["pdca_phase"] = "check"
    corr = correlation_id(state)
    _ = task_plan(task_state)
    current = current_step(task_state)
    proposal = (current or {}).get("proposal")
    validation = _validate_proposal(proposal=proposal, tool_registry=tool_registry)

    if validation["ok"]:
        task_state["last_validation_error"] = None
        if current is not None:
            current["status"] = "validated"
        logger.info(
            "task_mode validate passed correlation_id=%s step_id=%s kind=%s",
            corr,
            str((current or {}).get("step_id") or ""),
            str((proposal or {}).get("kind") if isinstance(proposal, dict) else ""),
        )
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="validate_step_node",
            event="graph.step.validated",
            step_id=str((current or {}).get("step_id") or ""),
            kind=str((proposal or {}).get("kind") if isinstance(proposal, dict) else ""),
        )
        return {"task_state": task_state}

    attempts = int(task_state.get("repair_attempts") or 0) + 1
    task_state["repair_attempts"] = attempts
    error = {
        "reason": validation.get("reason") or "validation_failed",
        "proposal": proposal,
    }
    task_state["last_validation_error"] = error
    if current is not None:
        current["status"] = "validation_failed"
    append_trace_event(
        task_state,
        {
            "type": "validation_failed",
            "summary": f"Validation failed: {error['reason']}.",
            "correlation_id": corr,
        },
    )
    logger.info(
        "task_mode validate failed correlation_id=%s step_id=%s reason=%s repair_attempts=%s",
        corr,
        str((current or {}).get("step_id") or ""),
        str(error.get("reason") or ""),
        attempts,
    )
    log_task_event(
        logger=logger,
        state=state,
        task_state=task_state,
        node="validate_step_node",
        event="graph.step.validation_failed",
        level="warning",
        step_id=str((current or {}).get("step_id") or ""),
        reason=str(error.get("reason") or ""),
        repair_attempts=attempts,
    )
    return {"task_state": task_state}


def _validate_proposal(*, proposal: Any, tool_registry: Any) -> dict[str, Any]:
    if not isinstance(proposal, dict):
        return {"ok": False, "executable": False, "reason": "missing_proposal"}
    kind = str(proposal.get("kind") or "").strip()
    if kind == "ask_user":
        question = str(proposal.get("question") or "").strip()
        if not question:
            return {"ok": False, "executable": False, "reason": "missing_question"}
        return {"ok": True, "executable": True, "reason": None}
    if kind == "finish":
        final_text = str(proposal.get("final_text") or "").strip()
        if not final_text:
            return {"ok": False, "executable": False, "reason": "missing_final_text"}
        return {"ok": True, "executable": True, "reason": None}
    if kind != "call_tool":
        return {"ok": False, "executable": False, "reason": "unknown_kind"}

    tool_name = str(proposal.get("tool_name") or "").strip()
    if not tool_name:
        return {"ok": False, "executable": False, "reason": "missing_tool_name"}
    if not _tool_exists(tool_registry, tool_name):
        return {"ok": False, "executable": False, "reason": f"tool_not_found:{tool_name}"}

    args = proposal.get("args")
    if args is not None and not isinstance(args, dict):
        return {"ok": False, "executable": False, "reason": "invalid_args_type"}

    required = _required_args_for_tool(tool_name)
    provided = set(dict(args or {}).keys())
    missing = [key for key in required if key not in provided]
    if missing:
        missing_keys = ",".join(missing)
        return {"ok": False, "executable": False, "reason": f"missing_required_args:{missing_keys}"}
    invalid_keys = _invalid_args_for_tool(tool_name=tool_name, args=dict(args or {}))
    if invalid_keys:
        keys = ",".join(invalid_keys)
        return {"ok": False, "executable": False, "reason": f"invalid_args_keys:{keys}"}
    mcp_reason = _validate_mcp_call_args(tool_name=tool_name, args=dict(args or {}))
    if mcp_reason:
        return {"ok": False, "executable": False, "reason": mcp_reason}
    terminal_command = _terminal_command_from_tool_call(tool_name=tool_name, args=dict(args or {}))
    if terminal_command and _is_mcp_like_terminal_command(terminal_command):
        return {
            "ok": False,
            "executable": False,
            "reason": "policy_violation:mcp_binaries_require_mcp_call",
        }
    return {"ok": True, "executable": True, "reason": None}


def _tool_exists(tool_registry: Any, tool_name: str) -> bool:
    if hasattr(tool_registry, "get"):
        return tool_registry.get(tool_name) is not None
    return False


def _required_args_for_tool(tool_name: str) -> list[str]:
    for schema in planner_tool_schemas():
        fn = schema.get("function")
        if not isinstance(fn, dict):
            continue
        if str(fn.get("name") or "") != tool_name:
            continue
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        required = params.get("required")
        if isinstance(required, list):
            return [str(item) for item in required if str(item)]
        return []
    return []


def _invalid_args_for_tool(*, tool_name: str, args: dict[str, Any]) -> list[str]:
    params = _tool_parameters_for_tool(tool_name)
    if not isinstance(params, dict):
        return []
    additional_properties = params.get("additionalProperties")
    if additional_properties is not False:
        return []
    props = params.get("properties")
    if not isinstance(props, dict):
        return []
    allowed = {str(k) for k in props.keys()}
    invalid = [key for key in args.keys() if str(key) not in allowed]
    return sorted([str(item) for item in invalid if str(item)])


def _tool_parameters_for_tool(tool_name: str) -> dict[str, Any] | None:
    for schema in planner_tool_schemas():
        fn = schema.get("function")
        if not isinstance(fn, dict):
            continue
        if str(fn.get("name") or "") != tool_name:
            continue
        params = fn.get("parameters")
        if isinstance(params, dict):
            return params
        return None
    return None


def _terminal_command_from_tool_call(*, tool_name: str, args: dict[str, Any]) -> str:
    normalized = str(tool_name or "").strip().lower()
    if normalized not in {"terminal_sync", "terminal_async", "ssh_terminal"}:
        return ""
    return str(args.get("command") or "").strip()


def _is_mcp_like_terminal_command(command: str) -> bool:
    raw = str(command or "").strip()
    if not raw:
        return False
    for segment in re.split(r"[|;&]+", raw):
        probe = _segment_binary_probe(segment)
        if not probe:
            continue
        if _is_mcp_binary_token(probe):
            return True
    return False


def _segment_binary_probe(segment: str) -> str:
    text = str(segment or "").strip()
    if not text:
        return ""
    try:
        parts = shlex.split(text)
    except ValueError:
        return ""
    if not parts:
        return ""

    head = str(parts[0]).strip()
    if head in {"command", "which"} and len(parts) >= 3 and str(parts[1]).strip() in {"-v", "-V"}:
        return str(parts[2]).strip()
    if head == "npx":
        for item in parts[1:]:
            token = str(item).strip()
            if token and not token.startswith("-"):
                return token
        return ""
    return head


def _is_mcp_binary_token(token: str) -> bool:
    value = str(token or "").strip().lower()
    if not value:
        return False
    basename = value.split("/")[-1]
    if basename in {"chrome-mcp", "chrome-devtools-mcp"}:
        return True
    if basename.startswith("mcp-server-"):
        return True
    if basename.endswith("-mcp") or basename.startswith("mcp-"):
        return True
    return False


def _validate_mcp_call_args(*, tool_name: str, args: dict[str, Any]) -> str:
    if str(tool_name or "").strip() != "mcp_call":
        return ""
    profile_key = str(args.get("profile") or "").strip()
    operation_key = str(args.get("operation") or "").strip()
    registry = McpProfileRegistry()
    profile = registry.get(profile_key)
    if profile is None:
        return f"unknown_mcp_profile:{profile_key or 'missing'}"
    if operation_key in profile.operations:
        return ""
    metadata = profile.metadata if isinstance(profile.metadata, dict) else {}
    supports_native = bool(metadata.get("native_tools")) or str(metadata.get("capability_model") or "").strip().lower() in {
        "interactive_browser_server",
        "native_mcp",
    }
    if not supports_native:
        return f"unknown_mcp_operation:{operation_key or 'missing'}"
    return ""
