from __future__ import annotations

import logging
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TerminalExecutionResult:
    status: str
    stdout: str
    stderr: str
    exit_code: int | None


class TerminalTool:
    """Execute shell commands in a sandboxed directory."""

    _AUTO_APPROVE = {
        "ls",
        "pwd",
        "rg",
        "cat",
        "head",
        "tail",
        "stat",
    }
    _REQUIRE_APPROVAL = {
        "touch",
        "mkdir",
        "cp",
        "mv",
        "rm",
        "tee",
    }
    _REJECT = {
        "sudo",
    }
    _DISALLOWED_TOKENS = {"|", "||", "&&", ";", ">", ">>", "<"}

    def classify_command(self, command: str) -> str:
        cmd = str(command or "").strip()
        if not cmd:
            return "reject"
        try:
            parts = shlex.split(cmd)
        except ValueError:
            return "reject"
        if not parts:
            return "reject"
        if any(token in self._DISALLOWED_TOKENS for token in parts):
            return "reject"
        head = parts[0]
        if head in self._REJECT:
            return "reject"
        if head in self._AUTO_APPROVE:
            return "auto"
        if head in self._REQUIRE_APPROVAL:
            return "approval"
        return "approval"

    def execute(
        self,
        *,
        command: str,
        cwd: str,
        sandbox_path: str,
        timeout_seconds: float = 30.0,
    ) -> TerminalExecutionResult:
        cmd = str(command or "").strip()
        if not cmd:
            return TerminalExecutionResult(
                status="failed",
                stdout="",
                stderr="Command is empty",
                exit_code=None,
            )
        try:
            parts = shlex.split(cmd)
        except ValueError as exc:
            return TerminalExecutionResult(
                status="failed",
                stdout="",
                stderr=f"Invalid command: {exc}",
                exit_code=None,
            )
        if not parts:
            return TerminalExecutionResult(
                status="failed",
                stdout="",
                stderr="Command is empty",
                exit_code=None,
            )
        if any(token in self._DISALLOWED_TOKENS for token in parts):
            return TerminalExecutionResult(
                status="rejected",
                stdout="",
                stderr="Command contains disallowed operators",
                exit_code=None,
            )
        sandbox_root = Path(sandbox_path).expanduser().resolve()
        if not sandbox_root.exists():
            return TerminalExecutionResult(
                status="failed",
                stdout="",
                stderr="Sandbox path does not exist",
                exit_code=None,
            )
        resolved_cwd = _resolve_cwd(sandbox_root, cwd)
        if not resolved_cwd:
            return TerminalExecutionResult(
                status="failed",
                stdout="",
                stderr="Working directory is outside sandbox",
                exit_code=None,
            )
        if not resolved_cwd.exists():
            return TerminalExecutionResult(
                status="failed",
                stdout="",
                stderr="Working directory does not exist",
                exit_code=None,
            )
        try:
            completed = subprocess.run(
                parts,
                cwd=str(resolved_cwd),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                env=_scrub_env(os.environ),
            )
        except subprocess.TimeoutExpired:
            return TerminalExecutionResult(
                status="failed",
                stdout="",
                stderr=f"Command timed out after {timeout_seconds:.0f}s",
                exit_code=None,
            )
        except Exception as exc:
            logger.exception("Terminal command failed")
            return TerminalExecutionResult(
                status="failed",
                stdout="",
                stderr=f"Execution error: {exc}",
                exit_code=None,
            )
        status = "executed" if completed.returncode == 0 else "failed"
        return TerminalExecutionResult(
            status=status,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            exit_code=completed.returncode,
        )


def _resolve_cwd(sandbox_root: Path, cwd: str) -> Path | None:
    raw = str(cwd or ".").strip() or "."
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = sandbox_root / path
    try:
        resolved = path.resolve()
    except FileNotFoundError:
        resolved = path.absolute()
    root = sandbox_root.resolve()
    if resolved == root:
        return resolved
    root_str = f"{root}{os.sep}"
    if f"{resolved}".startswith(root_str):
        return resolved
    return None


def _scrub_env(env: dict[str, str]) -> dict[str, str]:
    safe = dict(env)
    for key in list(safe.keys()):
        if key.upper().endswith("_API_KEY") or key.upper().endswith("_TOKEN"):
            safe.pop(key, None)
    return safe
