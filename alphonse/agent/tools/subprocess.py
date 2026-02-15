from __future__ import annotations

import os
import shlex
import subprocess
from typing import Any


class SubprocessTool:
    def execute(self, *, command: str, timeout_seconds: float | None = None) -> dict[str, Any]:
        if not _subprocess_enabled():
            return _failed("python_subprocess_disabled", retryable=False)

        raw_command = str(command or "").strip()
        if not raw_command:
            return _failed("command_required", retryable=False)
        try:
            argv = shlex.split(raw_command)
        except Exception:
            return _failed("command_parse_failed", retryable=False)
        if not argv:
            return _failed("command_required", retryable=False)
        if not _is_safe_executable(argv[0]):
            return _failed("command_not_allowed", retryable=False, executable=argv[0])
        timeout_value = _normalize_timeout(timeout_seconds)
        try:
            completed = subprocess.run(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=timeout_value,
            )
        except subprocess.TimeoutExpired:
            return _failed("subprocess_timeout", retryable=True)
        except Exception as exc:
            return _failed("subprocess_execution_failed", retryable=True, detail=str(exc))
        success = int(completed.returncode or 0) == 0
        if not success:
            return {
                "status": "failed",
                "error": "subprocess_non_zero_exit",
                "retryable": False,
                "exit_code": int(completed.returncode or 1),
                "stdout": _truncate_output(completed.stdout),
                "stderr": _truncate_output(completed.stderr),
            }
        return {
            "status": "ok",
            "exit_code": int(completed.returncode or 0),
            "stdout": _truncate_output(completed.stdout),
            "stderr": _truncate_output(completed.stderr),
        }

def _failed(error_code: str, *, retryable: bool, **kwargs: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "failed",
        "error": str(error_code or "subprocess_failed"),
        "retryable": bool(retryable),
    }
    payload.update(kwargs)
    return payload


def _subprocess_enabled() -> bool:
    value = str(os.getenv("ALPHONSE_ENABLE_PYTHON_SUBPROCESS") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _normalize_timeout(timeout_seconds: float | None) -> float:
    if timeout_seconds is None:
        return 30.0
    try:
        timeout = float(timeout_seconds)
    except (TypeError, ValueError):
        return 30.0
    return max(1.0, min(timeout, 120.0))


def _is_safe_executable(executable: str) -> bool:
    name = str(executable or "").strip().lower()
    return name in {"python", "python3", "pip", "pip3", "which"}


def _truncate_output(value: Any, *, limit: int = 4000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit]
