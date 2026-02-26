from __future__ import annotations

from typing import Any

from alphonse.agent.tools.mcp_connector import McpConnector
from alphonse.agent.tools.mcp_connector import McpInvocationError
from alphonse.agent.tools.terminal_execute_tool import _allowed_roots
from alphonse.agent.tools.terminal_execute_tool import _normalize_cwd
from alphonse.agent.tools.terminal_execute_tool import _resolve_timeout
from alphonse.config import settings


class McpCallTool:
    def __init__(self, connector: McpConnector | None = None) -> None:
        self._connector = connector or McpConnector()

    def execute(
        self,
        *,
        profile: str,
        operation: str,
        arguments: dict[str, Any] | None = None,
        cwd: str = ".",
        timeout_seconds: float | None = None,
        timeout_ms: float | int | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = state
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
                profile_key=profile,
                operation_key=operation,
                arguments=dict(arguments or {}),
                cwd=_normalize_cwd(cwd=cwd, roots=roots),
                allowed_roots=roots,
                mode=mode,
                timeout_seconds=_resolve_timeout(timeout_seconds, timeout_ms),
            )
        except McpInvocationError as exc:
            payload = exc.as_payload()
            metadata = dict(payload.get("metadata") or {})
            metadata["mode"] = mode
            payload["metadata"] = metadata
            return payload

