"""v2-native Bash tool."""

from __future__ import annotations

import os
import selectors
import signal
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.core import ToolExecutionContext
from alphonse.agent_v2.core.core import ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition

BASH_TOOL_ID = "native.bash"
BASH_TOOL_NAME = "bash"
DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_TIMEOUT_SECONDS = 120.0
MAX_OUTPUT_CHARS = 12000
TERMINATION_GRACE_SECONDS = 0.25

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
            "minimum": 0.01,
            "maximum": MAX_TIMEOUT_SECONDS,
            "default": DEFAULT_TIMEOUT_SECONDS,
            "description": "Optional total timeout in seconds. Defaults to 10; use a longer explicit value only for expected long-running commands.",
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
        description=(
            "Execute one bounded Bash command on the local host and return stdout/stderr. "
            "Use for direct filesystem, process, build, test, or diagnostic work; do not use "
            "for multi-tool orchestration or aggregation of tool results."
        ),
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
    started_at = time.monotonic()

    try:
        process = subprocess.Popen(
            [bash_bin, "-lc", command],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stdout, stderr = _collect_output_until_exit(process, timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = _stop_process_group(process, exc)
        return {
            "exit_code": -1,
            "stdout": _truncate_output(stdout),
            "stderr": _truncate_output(stderr or f"Command timed out after {timeout_seconds:g} seconds."),
            "timed_out": True,
            "cwd": cwd,
            "timeout_seconds": timeout_seconds,
            "duration_ms": _duration_ms(started_at),
        }

    return {
        "exit_code": int(process.returncode),
        "stdout": _truncate_output(stdout),
        "stderr": _truncate_output(stderr),
        "timed_out": False,
        "cwd": cwd,
        "timeout_seconds": timeout_seconds,
        "duration_ms": _duration_ms(started_at),
    }


def _stop_process_group(
    process: subprocess.Popen[bytes],
    timeout_error: subprocess.TimeoutExpired,
) -> tuple[str, str]:
    """Stop the shell and every child that inherited its captured streams."""
    _signal_process_group(process.pid, signal.SIGTERM)
    try:
        stdout, stderr = process.communicate(timeout=TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        _signal_process_group(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    else:
        if _process_group_exists(process.pid):
            _signal_process_group(process.pid, signal.SIGKILL)
    return (
        _merge_output(timeout_error.stdout, stdout),
        _merge_output(timeout_error.stderr, stderr),
    )


def _collect_output_until_exit(
    process: subprocess.Popen[bytes],
    timeout_seconds: float,
) -> tuple[str, str]:
    """Read available output until the shell exits, without awaiting orphaned pipe writers."""
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    streams = {
        process.stdout: stdout_chunks,
        process.stderr: stderr_chunks,
    }
    deadline = time.monotonic() + timeout_seconds

    with selectors.DefaultSelector() as selector:
        for stream in streams:
            if stream is None:
                continue
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ)

        while True:
            exited = process.poll() is not None
            remaining = deadline - time.monotonic()
            if not exited and remaining <= 0:
                raise subprocess.TimeoutExpired(
                    process.args,
                    timeout_seconds,
                    output=b"".join(stdout_chunks),
                    stderr=b"".join(stderr_chunks),
                )

            wait_seconds = 0 if exited else min(remaining, 0.02)
            ready = selector.select(wait_seconds)
            for key, _ in ready:
                stream = key.fileobj
                chunks = streams[stream]
                while True:
                    try:
                        chunk = os.read(stream.fileno(), 65_536)
                    except BlockingIOError:
                        break
                    if not chunk:
                        selector.unregister(stream)
                        break
                    chunks.append(chunk)

            if exited:
                # The shell's own writes are buffered by now. Do not wait for EOF
                # from a background descendant that inherited either pipe.
                for stream, chunks in streams.items():
                    if stream is None:
                        continue
                    while True:
                        try:
                            chunk = os.read(stream.fileno(), 65_536)
                        except BlockingIOError:
                            break
                        if not chunk:
                            break
                        chunks.append(chunk)
                    stream.close()
                return (
                    _coerce_output(b"".join(stdout_chunks)),
                    _coerce_output(b"".join(stderr_chunks)),
                )


def _signal_process_group(process_group_id: int, requested_signal: signal.Signals) -> None:
    try:
        os.killpg(process_group_id, requested_signal)
    except ProcessLookupError:
        return


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
        return True
    except ProcessLookupError:
        return False


def _duration_ms(started_at: float) -> int:
    return max(0, round((time.monotonic() - started_at) * 1000))


def _merge_output(initial: Any, final: Any) -> str:
    return _coerce_output(initial) + _coerce_output(final)


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
