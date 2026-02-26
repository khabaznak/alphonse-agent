from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any

from alphonse.agent.tools.mcp.registry import McpOperationProfile
from alphonse.agent.tools.mcp.registry import McpProfileRegistry
from alphonse.agent.tools.mcp.registry import McpServerProfile
from alphonse.agent.tools.terminal import TerminalTool


@dataclass(frozen=True)
class McpInvocationError(Exception):
    code: str
    message: str

    def as_payload(self) -> dict[str, Any]:
        return {
            "status": "failed",
            "result": None,
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": False,
                "details": {},
            },
            "metadata": {"tool": "mcp_call"},
        }


class McpConnector:
    def __init__(
        self,
        *,
        terminal: TerminalTool | None = None,
        profile_registry: McpProfileRegistry | None = None,
    ) -> None:
        self._terminal = terminal or TerminalTool()
        self._profiles = profile_registry or McpProfileRegistry()

    def execute(
        self,
        *,
        profile_key: str,
        operation_key: str,
        arguments: dict[str, Any],
        cwd: str,
        allowed_roots: list[str],
        mode: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        profile = self._profiles.get(profile_key)
        if profile is None:
            raise McpInvocationError("mcp_profile_not_found", f"Unknown MCP profile: {profile_key}")
        if str(mode or "").strip() not in set(profile.allowed_modes):
            raise McpInvocationError(
                "mcp_mode_not_allowed",
                f"MCP profile '{profile.key}' is not allowed in mode '{mode}'.",
            )
        operation = profile.operations.get(str(operation_key or "").strip())
        if operation is None:
            raise McpInvocationError(
                "mcp_operation_not_found",
                f"Unknown operation '{operation_key}' for MCP profile '{profile.key}'.",
            )
        command = self._build_command(
            profile=profile,
            operation=operation,
            arguments=arguments,
        )
        outcome = self._terminal.execute_with_policy(
            command=command,
            cwd=cwd,
            allowed_roots=allowed_roots,
            mode=mode,
            timeout_seconds=timeout_seconds,
        )
        metadata = dict(outcome.get("metadata") or {}) if isinstance(outcome, dict) else {}
        metadata["tool"] = "mcp_call"
        metadata["policy_envelope"] = {
            "execution_surface": "mcp",
            "profile": profile.key,
            "operation": operation.key,
            "mode": mode,
            "cwd": cwd,
        }
        metadata["mcp_profile"] = profile.key
        metadata["mcp_operation"] = operation.key
        metadata["mcp_command"] = command
        if not isinstance(outcome, dict):
            return {
                "status": "failed",
                "result": None,
                "error": {"code": "mcp_connector_invalid_result", "message": "MCP connector returned invalid result"},
                "metadata": metadata,
            }
        return {
            "status": str(outcome.get("status") or "failed"),
            "result": outcome.get("result"),
            "error": outcome.get("error"),
            "metadata": metadata,
        }

    def _build_command(
        self,
        *,
        profile: McpServerProfile,
        operation: McpOperationProfile,
        arguments: dict[str, Any],
    ) -> str:
        values = dict(arguments or {})
        for field in operation.required_args:
            if field not in values or str(values.get(field) or "").strip() == "":
                raise McpInvocationError("missing_required_args", f"Missing required MCP argument: {field}")
        quoted_args = {key: shlex.quote(str(value)) for key, value in values.items()}
        try:
            operation_segment = operation.command_template.format(**quoted_args).strip()
        except KeyError as exc:
            missing = str(exc).strip("'")
            raise McpInvocationError("missing_required_args", f"Missing required MCP argument: {missing}") from exc

        launcher = _launcher_expression(profile)
        shell_script = f"{launcher} {operation_segment}".strip()
        return f"sh -lc {shlex.quote(shell_script)}"


def _launcher_expression(profile: McpServerProfile) -> str:
    checks: list[str] = []
    branch_index = 0
    for binary in profile.binary_candidates:
        name = str(binary or "").strip()
        if not name:
            continue
        prefix = "if" if branch_index == 0 else "elif"
        checks.append(f"{prefix} command -v {shlex.quote(name)} >/dev/null 2>&1; then {shlex.quote(name)}")
        branch_index += 1
    fallback = "echo MCP_BINARY_NOT_FOUND; exit 127"
    package = str(profile.npx_package_fallback or "").strip()
    if package:
        fallback = f"npx -y {shlex.quote(package)}"
    if not checks:
        return fallback
    chain = " ; ".join(checks)
    return f"{chain} ; else {fallback} ; fi"
