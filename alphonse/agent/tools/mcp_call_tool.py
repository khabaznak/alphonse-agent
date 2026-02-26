from __future__ import annotations

from typing import Any

from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.tools.mcp_connector import McpConnector
from alphonse.agent.tools.mcp_connector import McpInvocationError
from alphonse.agent.tools.terminal_execute_tool import _allowed_roots
from alphonse.agent.tools.terminal_execute_tool import _normalize_cwd
from alphonse.agent.tools.terminal_execute_tool import _resolve_timeout
from alphonse.config import settings

logger = get_component_logger("tools.mcp_call_tool")


def normalize_mcp_call_invocation(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = dict(raw or {})
    mapped: list[str] = []
    ignored: list[str] = []

    profile = str(payload.get("profile") or "").strip()
    operation = str(payload.get("operation") or "").strip()
    arguments = payload.get("arguments")
    canonical_arguments = dict(arguments) if isinstance(arguments, dict) else {}
    if arguments is not None and not isinstance(arguments, dict):
        ignored.append("arguments(non_dict)")

    nested_args = payload.get("args")
    if isinstance(nested_args, dict):
        nested_profile = str(nested_args.get("profile") or "").strip()
        nested_operation = str(nested_args.get("operation") or "").strip()
        nested_arguments = nested_args.get("arguments")
        if not profile and nested_profile:
            profile = nested_profile
            mapped.append("args.profile->profile")
        if not operation and nested_operation:
            operation = nested_operation
            mapped.append("args.operation->operation")
        if isinstance(nested_arguments, dict):
            for key, value in nested_arguments.items():
                if key not in canonical_arguments:
                    canonical_arguments[key] = value
            mapped.append("args.arguments->arguments")
        for key, value in nested_args.items():
            if key in {"profile", "operation", "arguments"}:
                continue
            if key not in canonical_arguments:
                canonical_arguments[key] = value
        mapped.append("args->arguments")
    elif nested_args is not None:
        ignored.append("args(non_dict)")

    query = payload.get("query")
    if query is not None:
        if "query" not in canonical_arguments:
            canonical_arguments["query"] = query
            mapped.append("query->arguments.query")
        else:
            ignored.append("query(already_present)")

    if payload.get("mode") is not None:
        ignored.append("mode")

    for key in payload.keys():
        if key in {
            "profile",
            "operation",
            "arguments",
            "args",
            "query",
            "mode",
            "cwd",
            "timeout_seconds",
            "timeout_ms",
            "state",
        }:
            continue
        ignored.append(key)

    normalized = {
        "profile": profile,
        "operation": operation,
        "arguments": canonical_arguments,
        "cwd": str(payload.get("cwd") or "."),
        "timeout_seconds": payload.get("timeout_seconds"),
        "timeout_ms": payload.get("timeout_ms"),
    }
    report = {
        "normalized": bool(mapped or ignored),
        "mapped": mapped,
        "ignored": ignored,
    }
    return normalized, report


class McpCallTool:
    def __init__(self, connector: McpConnector | None = None) -> None:
        self._connector = connector or McpConnector()

    def execute(
        self,
        *,
        profile: str | None = None,
        operation: str | None = None,
        arguments: dict[str, Any] | None = None,
        cwd: str = ".",
        timeout_seconds: float | None = None,
        timeout_ms: float | int | None = None,
        state: dict[str, Any] | None = None,
        **legacy_kwargs: Any,
    ) -> dict[str, Any]:
        payload = {
            "profile": profile,
            "operation": operation,
            "arguments": arguments,
            "cwd": cwd,
            "timeout_seconds": timeout_seconds,
            "timeout_ms": timeout_ms,
            "state": state,
            **dict(legacy_kwargs or {}),
        }
        normalized, report = normalize_mcp_call_invocation(payload)
        profile_key = str(normalized.get("profile") or "").strip()
        operation_key = str(normalized.get("operation") or "").strip()
        normalized_arguments = normalized.get("arguments")
        canonical_arguments = (
            dict(normalized_arguments)
            if isinstance(normalized_arguments, dict)
            else {}
        )
        if report.get("normalized"):
            logger.warning(
                "mcp_call normalized invocation profile=%s operation=%s mapped=%s ignored=%s",
                profile_key,
                operation_key,
                list(report.get("mapped") or []),
                list(report.get("ignored") or []),
            )
        if not profile_key or not operation_key:
            return {
                "status": "failed",
                "result": None,
                "error": {
                    "code": "invalid_tool_arguments",
                    "message": "mcp_call requires `profile` and `operation` (with inputs under `arguments`).",
                    "retryable": False,
                    "details": {
                        "mapped": list(report.get("mapped") or []),
                        "ignored": list(report.get("ignored") or []),
                    },
                },
                "metadata": {"tool": "mcp_call"},
            }

        mode = settings.get_execution_mode()
        roots = _allowed_roots()
        if not roots:
            return {
                "status": "failed",
                "result": None,
                "error": {
                    "code": "sandbox_roots_not_configured",
                    "message": "No enabled sandbox directories found. Configure sandbox dirs in nerve-db.",
                },
                "metadata": {"mode": mode, "tool": "mcp_call"},
            }
        try:
            return self._connector.execute(
                profile_key=profile_key,
                operation_key=operation_key,
                arguments=canonical_arguments,
                cwd=_normalize_cwd(cwd=str(normalized.get("cwd") or "."), roots=roots),
                allowed_roots=roots,
                mode=mode,
                timeout_seconds=_resolve_timeout(
                    normalized.get("timeout_seconds"),
                    normalized.get("timeout_ms"),
                ),
            )
        except McpInvocationError as exc:
            payload = exc.as_payload()
            metadata = dict(payload.get("metadata") or {})
            metadata["mode"] = mode
            payload["metadata"] = metadata
            return payload
