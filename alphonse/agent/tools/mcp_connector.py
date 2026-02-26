from __future__ import annotations

import json
import os
import select
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any
from typing import Literal

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
        if operation is None and _supports_native_tools(profile):
            return self._execute_native(
                profile=profile,
                operation_key=str(operation_key or "").strip(),
                arguments=arguments,
                cwd=cwd,
                mode=mode,
                timeout_seconds=timeout_seconds,
            )
        if operation is None:
            raise McpInvocationError(
                "mcp_operation_not_found",
                f"Unknown operation '{operation_key}' for MCP profile '{profile.key}'.",
            )

        mismatch = _contract_mismatch(profile=profile, operation=operation)
        if mismatch:
            raise McpInvocationError("mcp_operation_contract_mismatch", mismatch)

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
        error_payload = outcome.get("error") if isinstance(outcome.get("error"), dict) else None
        result_payload = outcome.get("result") if isinstance(outcome.get("result"), dict) else {}
        if str(outcome.get("status") or "").strip().lower() == "failed" and isinstance(error_payload, dict):
            stderr_preview = str(result_payload.get("stderr") or "").strip()
            stdout_preview = str(result_payload.get("stdout") or "").strip()
            details = error_payload.get("details")
            merged_details = dict(details) if isinstance(details, dict) else {}
            if stderr_preview:
                merged_details["stderr_preview"] = stderr_preview[:600]
            if stdout_preview:
                merged_details["stdout_preview"] = stdout_preview[:400]
            merged_details["mcp_command"] = command
            error_payload = {
                **error_payload,
                "details": merged_details,
            }
        return {
            "status": str(outcome.get("status") or "failed"),
            "result": outcome.get("result"),
            "error": error_payload if error_payload is not None else outcome.get("error"),
            "metadata": metadata,
        }

    def _execute_native(
        self,
        *,
        profile: McpServerProfile,
        operation_key: str,
        arguments: dict[str, Any],
        cwd: str,
        mode: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        launcher_argv = _launcher_argv(profile)
        transport = _native_transport(profile)
        operation = str(operation_key or "").strip()
        if not operation:
            raise McpInvocationError("mcp_operation_not_found", f"Unknown operation '{operation_key}' for MCP profile '{profile.key}'.")

        try:
            with _McpJsonRpcStdioClient(
                launcher_argv=launcher_argv,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                transport=transport,
            ) as client:
                client.initialize()
                tools = client.list_tools()
                tool_names = [str(item.get("name") or "").strip() for item in tools if str(item.get("name") or "").strip()]

                if operation in {"list_tools", "discover_tools"}:
                    return {
                        "status": "ok",
                        "result": {
                            "tools": tools,
                            "tool_names": tool_names,
                            "count": len(tool_names),
                        },
                        "error": None,
                        "metadata": {
                            "tool": "mcp_call",
                            "policy_envelope": {
                                "execution_surface": "mcp_native",
                                "profile": profile.key,
                                "operation": operation,
                                "mode": mode,
                                "cwd": cwd,
                            },
                            "mcp_profile": profile.key,
                            "mcp_operation": operation,
                            "mcp_command": " ".join(shlex.quote(part) for part in launcher_argv),
                            "mcp_native": True,
                            "mcp_transport": transport,
                        },
                    }

                resolved_operation = _resolve_native_tool_name(operation=operation, available=tool_names)
                if not resolved_operation:
                    available = ", ".join(tool_names[:20])
                    raise McpInvocationError(
                        "mcp_operation_not_found",
                        f"Unknown operation '{operation}' for MCP profile '{profile.key}'. Available tools: {available or '(none)'}.",
                    )

                call_result = client.call_tool(name=resolved_operation, arguments=arguments)
                if isinstance(call_result, dict) and bool(call_result.get("isError")):
                    content = call_result.get("content")
                    message = "MCP tool reported an error"
                    if isinstance(content, list) and content:
                        first = content[0]
                        if isinstance(first, dict):
                            text = str(first.get("text") or "").strip()
                            if text:
                                message = text
                    raise McpInvocationError("mcp_tools_call_failed", message)
                return {
                    "status": "ok",
                    "result": call_result,
                    "error": None,
                    "metadata": {
                        "tool": "mcp_call",
                        "policy_envelope": {
                            "execution_surface": "mcp_native",
                            "profile": profile.key,
                            "operation": resolved_operation,
                            "mode": mode,
                            "cwd": cwd,
                        },
                        "mcp_profile": profile.key,
                        "mcp_operation": resolved_operation,
                        "mcp_requested_operation": operation,
                        "mcp_command": " ".join(shlex.quote(part) for part in launcher_argv),
                        "mcp_native": True,
                        "mcp_transport": transport,
                        "available_tools": tool_names,
                    },
                }
        except McpInvocationError:
            raise
        except TimeoutError as exc:
            raise McpInvocationError("mcp_timeout", str(exc)) from exc
        except Exception as exc:
            raise McpInvocationError("mcp_native_call_failed", str(exc)) from exc

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
        shell_script = (
            f"{launcher} ; "
            'if [ -z "$MCP_BIN" ]; then echo MCP_BINARY_NOT_FOUND; exit 127; fi ; '
            f'eval "$MCP_BIN {operation_segment}"'
        ).strip()
        return f"sh -lc {shlex.quote(shell_script)}"


class _McpJsonRpcStdioClient:
    def __init__(
        self,
        *,
        launcher_argv: list[str],
        cwd: str,
        timeout_seconds: float,
        transport: Literal["content_length", "ndjson"],
    ) -> None:
        self._launcher_argv = list(launcher_argv)
        self._cwd = cwd
        self._timeout_seconds = max(float(timeout_seconds), 1.0)
        self._transport: Literal["content_length", "ndjson"] = transport
        self._proc: subprocess.Popen[bytes] | None = None
        self._seq = 0
        self._buffer = bytearray()

    def __enter__(self) -> _McpJsonRpcStdioClient:
        self._proc = subprocess.Popen(
            self._launcher_argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self._cwd,
            env=dict(os.environ),
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except Exception:
                proc.kill()

    def initialize(self) -> None:
        versions = ("2024-11-05", "2024-10-07", "0.1.0")
        last_error = "initialize failed"
        for version in versions:
            payload = self._request(
                method="initialize",
                params={
                    "protocolVersion": version,
                    "clientInfo": {"name": "alphonse", "version": "1.0"},
                    "capabilities": {},
                },
            )
            if "error" in payload:
                last_error = str((payload.get("error") or {}).get("message") or last_error)
                continue
            self._notify("notifications/initialized", {})
            return
        raise McpInvocationError("mcp_initialize_failed", last_error)

    def list_tools(self) -> list[dict[str, Any]]:
        payload = self._request(method="tools/list", params={})
        if "error" in payload:
            message = str((payload.get("error") or {}).get("message") or "tools/list failed")
            raise McpInvocationError("mcp_tools_list_failed", message)
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        tools = result.get("tools") if isinstance(result.get("tools"), list) else []
        normalized: list[dict[str, Any]] = []
        for item in tools:
            if isinstance(item, dict):
                normalized.append(dict(item))
        return normalized

    def call_tool(self, *, name: str, arguments: dict[str, Any]) -> Any:
        payload = self._request(
            method="tools/call",
            params={
                "name": name,
                "arguments": dict(arguments or {}),
            },
        )
        if "error" in payload:
            message = str((payload.get("error") or {}).get("message") or "tools/call failed")
            raise McpInvocationError("mcp_tools_call_failed", message)
        return payload.get("result")

    def _request(self, *, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._seq += 1
        req_id = self._seq
        self._send_message({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        deadline = time.monotonic() + self._timeout_seconds
        while True:
            payload = self._read_message(deadline=deadline)
            if not isinstance(payload, dict):
                continue
            if payload.get("id") == req_id:
                return payload

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._send_message({"jsonrpc": "2.0", "method": method, "params": params})

    def _send_message(self, payload: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise McpInvocationError("mcp_transport_closed", "MCP transport is not available")
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if self._transport == "ndjson":
            proc.stdin.write(body + b"\n")
        else:
            header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
            proc.stdin.write(header + body)
        proc.stdin.flush()

    def _read_message(self, *, deadline: float) -> dict[str, Any]:
        if self._transport == "ndjson":
            return self._read_ndjson_message(deadline=deadline)
        return self._read_content_length_message(deadline=deadline)

    def _read_ndjson_message(self, *, deadline: float) -> dict[str, Any]:
        line = self._readline(deadline=deadline)
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            raise McpInvocationError("mcp_protocol_error", "Received empty MCP JSON-RPC line")
        try:
            parsed = json.loads(text)
        except Exception as exc:
            raise McpInvocationError("mcp_protocol_error", "Invalid NDJSON payload from MCP server") from exc
        return dict(parsed) if isinstance(parsed, dict) else {}

    def _read_content_length_message(self, *, deadline: float) -> dict[str, Any]:
        headers: dict[str, str] = {}
        while True:
            line = self._readline(deadline=deadline)
            if line == b"":
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                break
            if ":" not in text:
                continue
            key, value = text.split(":", 1)
            headers[key.strip().lower()] = value.strip()

        length_raw = headers.get("content-length")
        if not length_raw:
            raise McpInvocationError("mcp_protocol_error", "Missing Content-Length header")
        try:
            length = int(length_raw)
        except ValueError as exc:
            raise McpInvocationError("mcp_protocol_error", "Invalid Content-Length header") from exc
        body = self._read_exact(length=length, deadline=deadline)
        try:
            parsed = json.loads(body.decode("utf-8"))
        except Exception as exc:
            raise McpInvocationError("mcp_protocol_error", "Invalid JSON payload from MCP server") from exc
        return dict(parsed) if isinstance(parsed, dict) else {}

    def _readline(self, *, deadline: float) -> bytes:
        while True:
            idx = self._buffer.find(b"\n")
            if idx >= 0:
                out = bytes(self._buffer[: idx + 1])
                del self._buffer[: idx + 1]
                return out
            self._fill_buffer(deadline=deadline)

    def _read_exact(self, *, length: int, deadline: float) -> bytes:
        while len(self._buffer) < length:
            self._fill_buffer(deadline=deadline)
        out = bytes(self._buffer[:length])
        del self._buffer[:length]
        return out

    def _fill_buffer(self, *, deadline: float) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise McpInvocationError("mcp_transport_closed", "MCP transport is not available")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for MCP server response")
        ready, _, _ = select.select([proc.stdout], [], [], remaining)
        if not ready:
            raise TimeoutError("Timed out waiting for MCP server response")
        chunk = os.read(proc.stdout.fileno(), 65536)
        if not chunk:
            stderr_text = ""
            if proc.stderr is not None:
                try:
                    stderr_text = os.read(proc.stderr.fileno(), 8192).decode("utf-8", errors="replace").strip()
                except Exception:
                    stderr_text = ""
            raise McpInvocationError("mcp_transport_closed", f"MCP server closed the stream. {stderr_text}".strip())
        self._buffer.extend(chunk)


def _launcher_expression(profile: McpServerProfile) -> str:
    checks: list[str] = ['MCP_BIN=""']
    branch_open = False
    for binary in profile.binary_candidates:
        name = str(binary or "").strip()
        if not name:
            continue
        quoted = shlex.quote(name)
        if not branch_open:
            checks.append(f"if command -v {quoted} >/dev/null 2>&1; then MCP_BIN={quoted}")
            branch_open = True
        else:
            checks.append(f"elif command -v {quoted} >/dev/null 2>&1; then MCP_BIN={quoted}")
    if branch_open:
        checks.append("fi")
    package = str(profile.npx_package_fallback or "").strip()
    if package:
        checks.append(f'if [ -z "$MCP_BIN" ]; then MCP_BIN="npx -y {shlex.quote(package)}"; fi')
    return " ; ".join(checks).strip()


def _launcher_argv(profile: McpServerProfile) -> list[str]:
    metadata = profile.metadata if isinstance(profile.metadata, dict) else {}
    extra_args_raw = metadata.get("launcher_args")
    extra_args = [
        str(item).strip()
        for item in (extra_args_raw if isinstance(extra_args_raw, list) else [])
        if str(item).strip()
    ]
    for binary in profile.binary_candidates:
        name = str(binary or "").strip()
        if not name:
            continue
        resolved = shutil.which(name)
        if resolved:
            return [resolved, *extra_args]
    package = str(profile.npx_package_fallback or "").strip()
    if package:
        return ["npx", "-y", package, *extra_args]
    raise McpInvocationError(
        "mcp_binary_not_found",
        f"No MCP launcher found for profile '{profile.key}'.",
    )


def _native_transport(profile: McpServerProfile) -> Literal["content_length", "ndjson"]:
    metadata = profile.metadata if isinstance(profile.metadata, dict) else {}
    configured = str(metadata.get("mcp_transport") or "").strip().lower()
    if configured in {"ndjson", "content_length"}:
        return configured  # type: ignore[return-value]
    binaries = {str(item or "").strip().lower() for item in profile.binary_candidates}
    if "chrome-devtools-mcp" in binaries or "chrome-mcp" in binaries:
        return "ndjson"
    return "content_length"


def _supports_native_tools(profile: McpServerProfile) -> bool:
    metadata = profile.metadata if isinstance(profile.metadata, dict) else {}
    explicit = metadata.get("native_tools")
    if isinstance(explicit, bool):
        return explicit
    capability_model = str(metadata.get("capability_model") or "").strip().lower()
    return capability_model in {"interactive_browser_server", "native_mcp"}


def _resolve_native_tool_name(*, operation: str, available: list[str]) -> str | None:
    op = str(operation or "").strip()
    if not op:
        return None
    if op in available:
        return op

    normalized = {item.lower(): item for item in available}
    if op.lower() in normalized:
        return normalized[op.lower()]

    alias_map = {
        "web_search": ["search", "web_search", "search_web", "browser_search"],
    }
    for candidate in alias_map.get(op.lower(), []):
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    if op.lower() == "web_search":
        for name in available:
            low = name.lower()
            if "search" in low or "query" in low:
                return name

    return None


def _contract_mismatch(*, profile: McpServerProfile, operation: McpOperationProfile) -> str:
    binaries = {str(item or "").strip().lower() for item in profile.binary_candidates}
    template = str(operation.command_template or "").strip().lower()
    if operation.key == "web_search" and template.startswith("search ") and (
        "chrome-devtools-mcp" in binaries or "chrome-mcp" in binaries
    ):
        return (
            "Profile operation `web_search` uses one-shot command_template `search {query}` "
            "but chrome-devtools-mcp/chrome-mcp are MCP server launchers, not search CLIs."
        )
    return ""
