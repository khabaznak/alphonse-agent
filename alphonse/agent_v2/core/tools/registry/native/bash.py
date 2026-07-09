"""v2-native Bash tool."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.core import ToolExecutionContext
from alphonse.agent_v2.core.core import ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition

BASH_TOOL_ID = "native.bash"
BASH_TOOL_NAME = "bash"
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_TIMEOUT_SECONDS = 120.0
MAX_OUTPUT_CHARS = 12000

BASH_ARGUMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "command": {
            "type": "string",
            "description": "Shell command to execute with bash.",
        },
        "cwd": {
            "type": "string",
            "description": "Optional working directory.",
        },
        "timeout_seconds": {
            "type": "number",
            "description": "Optional timeout in seconds.",
        },
    },
    "required": ["command"],
}


def build_bash_tool_definition() -> ToolDefinition:
    """Build the native Bash tool definition."""
    descriptor = ToolDescriptor(
        tool_id=BASH_TOOL_ID,
        name=BASH_TOOL_NAME,
        kind=ToolKind.NATIVE,
        description="Execute a bounded local bash command and capture stdout/stderr.",
        argument_schema=dict(BASH_ARGUMENT_SCHEMA),
        capabilities=("shell", "local_execution"),
        tags=("native", "shell"),
    )
    return ToolDefinition(
        descriptor=descriptor,
        callable=execute_bash,
        argument_schema=dict(BASH_ARGUMENT_SCHEMA),
        enabled=True,
        accepts_context=True,
    )


def execute_bash(arguments: dict[str, Any], *, context: ToolExecutionContext | None = None) -> dict[str, Any]:
    """Execute a bash command and return a JSON-safe result."""
    command = str(arguments.get("command") or "").strip()
    if not command:
        raise ValueError("bash_command_required")

    bash_bin = shutil.which("bash") or "/bin/bash"
    if not Path(bash_bin).exists():
        raise RuntimeError("bash_executable_missing")

    cwd = _resolve_cwd(arguments.get("cwd"), context=context)
    timeout_seconds = _coerce_timeout(arguments.get("timeout_seconds"))

    try:
        completed = subprocess.run(
            [bash_bin, "-lc", command],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "exit_code": -1,
            "stdout": _truncate_output(_coerce_output(exc.stdout)),
            "stderr": _truncate_output(_coerce_output(exc.stderr) or f"Command timed out after {timeout_seconds:g} seconds."),
            "timed_out": True,
            "cwd": cwd,
        }

    return {
        "exit_code": int(completed.returncode),
        "stdout": _truncate_output(completed.stdout),
        "stderr": _truncate_output(completed.stderr),
        "timed_out": False,
        "cwd": cwd,
    }


def _resolve_cwd(raw_cwd: Any, *, context: ToolExecutionContext | None = None) -> str:
    if raw_cwd is None or str(raw_cwd).strip() == "":
        project_root = _project_root_from_context(context)
        if project_root:
            return project_root
        return str(Path.cwd())
    path = Path(str(raw_cwd)).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"bash_cwd_not_found: {path}")
    if not path.is_dir():
        raise ValueError(f"bash_cwd_not_directory: {path}")
    return str(path)


def _project_root_from_context(context: ToolExecutionContext | None) -> str:
    if context is None or context.project_store is None:
        return ""
    task = context.task
    project_id = str(getattr(task, "project_id", "") or "").strip()
    if not project_id:
        return ""
    get_project = getattr(context.project_store, "get_project", None)
    if get_project is None:
        return ""
    project = get_project(project_id, requester_user_id=getattr(task, "user", None))
    if project is None:
        return ""
    root = Path(str(project.root_path)).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return ""
    return str(root)


def _coerce_timeout(raw_timeout: Any) -> float:
    if raw_timeout is None or raw_timeout == "":
        return DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = float(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("bash_timeout_invalid") from exc
    if timeout <= 0:
        raise ValueError("bash_timeout_must_be_positive")
    return min(timeout, MAX_TIMEOUT_SECONDS)


def _coerce_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _truncate_output(value: Any) -> str:
    rendered = _coerce_output(value)
    if len(rendered) <= MAX_OUTPUT_CHARS:
        return rendered
    return f"{rendered[:MAX_OUTPUT_CHARS]}\n... [truncated]"
