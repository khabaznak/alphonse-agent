from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import uuid

from alphonse.agent.observability.log_manager import get_component_logger

logger = get_component_logger("tools.terminal")


@dataclass(frozen=True)
class TerminalExecutionResult:
    status: str
    stdout: str
    stderr: str
    exit_code: int | None


@dataclass(frozen=True)
class TerminalPolicyResult:
    allowed: bool
    reason: str
    mode: str
    command: str
    cwd: str
    policy_decision: str
    audit_id: str


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
    _READONLY_CMDS = {
        "ls",
        "pwd",
        "rg",
        "cat",
        "head",
        "tail",
        "stat",
        "find",
        "wc",
        "echo",
        "which",
    }
    _DEV_CMDS = _READONLY_CMDS | {
        "python",
        "python3",
        "pip",
        "pip3",
        "pytest",
        "git",
        "node",
        "npm",
        "pnpm",
        "yarn",
    }
    _BLOCK_ALWAYS = {"sudo", "su", "chmod", "chown", "mkfs", "shutdown", "reboot"}
    _WRITE_HINT_CMDS = {"rm", "mv", "cp", "mkdir", "touch", "tee", "sed"}
    _NETWORK_HINT_CMDS = {"curl", "wget", "nc", "ssh", "scp"}

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

    def execute_with_policy(
        self,
        *,
        command: str,
        cwd: str,
        allowed_roots: list[str],
        mode: str,
        timeout_seconds: float = 30.0,
    ) -> dict[str, object]:
        decision = self.evaluate_policy(
            command=command,
            cwd=cwd,
            allowed_roots=allowed_roots,
            mode=mode,
        )
        metadata = {
            "policy_decision": decision.policy_decision,
            "mode": decision.mode,
            "reason": decision.reason,
            "audit_id": decision.audit_id,
            "cwd": decision.cwd,
        }
        if not decision.allowed:
            return {
                "status": "failed",
                "result": None,
                "error": {"code": decision.reason, "message": f"Command blocked by policy: {decision.reason}"},
                "metadata": metadata,
            }
        try:
            argv = shlex.split(str(command or "").strip())
        except ValueError as exc:
            return {
                "status": "failed",
                "result": None,
                "error": {"code": "invalid_command", "message": str(exc)},
                "metadata": metadata,
            }
        resolved_cwd = Path(decision.cwd)
        try:
            completed = subprocess.run(
                argv,
                cwd=str(resolved_cwd),
                capture_output=True,
                text=True,
                timeout=max(1.0, min(float(timeout_seconds), 120.0)),
                check=False,
                env=_scrub_env(os.environ),
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "result": None,
                "error": {"code": "timeout", "message": "Command timed out"},
                "metadata": metadata,
            }
        except Exception as exc:
            return {
                "status": "failed",
                "result": None,
                "error": {"code": "execution_error", "message": str(exc)},
                "metadata": metadata,
            }
        if int(completed.returncode or 0) != 0:
            return {
                "status": "failed",
                "result": {
                    "exit_code": int(completed.returncode or 1),
                    "stdout": _truncate_text(completed.stdout),
                    "stderr": _truncate_text(completed.stderr),
                },
                "error": {"code": "subprocess_non_zero_exit", "message": "Command returned non-zero exit status"},
                "metadata": metadata,
            }
        return {
            "status": "ok",
            "result": {
                "exit_code": int(completed.returncode or 0),
                "stdout": _truncate_text(completed.stdout),
                "stderr": _truncate_text(completed.stderr),
            },
            "error": None,
            "metadata": metadata,
        }

    def evaluate_policy(
        self,
        *,
        command: str,
        cwd: str,
        allowed_roots: list[str],
        mode: str,
    ) -> TerminalPolicyResult:
        audit_id = f"term_{uuid.uuid4().hex[:12]}"
        cmd = str(command or "").strip()
        if not cmd:
            return _decision(False, "command_empty", mode, cmd, cwd, "reject", audit_id)
        try:
            parts = shlex.split(cmd)
        except ValueError:
            return _decision(False, "command_parse_failed", mode, cmd, cwd, "reject", audit_id)
        if not parts:
            return _decision(False, "command_empty", mode, cmd, cwd, "reject", audit_id)
        if any(token in self._DISALLOWED_TOKENS for token in parts):
            return _decision(False, "shell_operators_disallowed", mode, cmd, cwd, "reject", audit_id)
        head = str(parts[0]).strip().lower()
        if head in self._BLOCK_ALWAYS:
            return _decision(False, "command_blocked", mode, cmd, cwd, "reject", audit_id)

        resolved_roots = _resolve_allowed_roots(allowed_roots)
        resolved_cwd = _resolve_cwd_from_roots(cwd=cwd, allowed_roots=resolved_roots)
        if resolved_cwd is None:
            return _decision(False, "cwd_not_allowed", mode, cmd, cwd, "reject", audit_id)

        if not _paths_in_args_are_allowed(parts=parts, cwd=resolved_cwd, allowed_roots=resolved_roots):
            return _decision(False, "path_not_allowed", mode, cmd, str(resolved_cwd), "reject", audit_id)

        effective_mode = mode if mode in {"readonly", "dev", "ops"} else "readonly"
        if effective_mode == "readonly":
            if head not in self._READONLY_CMDS:
                return _decision(False, "mode_readonly_command_disallowed", effective_mode, cmd, str(resolved_cwd), "reject", audit_id)
            if _looks_like_write(parts, write_cmds=self._WRITE_HINT_CMDS):
                return _decision(False, "mode_readonly_write_disallowed", effective_mode, cmd, str(resolved_cwd), "reject", audit_id)
            if _looks_like_network(head, network_cmds=self._NETWORK_HINT_CMDS):
                return _decision(False, "mode_readonly_network_disallowed", effective_mode, cmd, str(resolved_cwd), "reject", audit_id)
            return _decision(True, "allowed_readonly", effective_mode, cmd, str(resolved_cwd), "auto", audit_id)
        if effective_mode == "dev":
            if head not in self._DEV_CMDS:
                return _decision(False, "mode_dev_command_disallowed", effective_mode, cmd, str(resolved_cwd), "reject", audit_id)
            if _looks_like_network(head, network_cmds=self._NETWORK_HINT_CMDS):
                return _decision(False, "mode_dev_network_disallowed", effective_mode, cmd, str(resolved_cwd), "reject", audit_id)
            return _decision(True, "allowed_dev", effective_mode, cmd, str(resolved_cwd), "auto", audit_id)
        return _decision(True, "allowed_ops", "ops", cmd, str(resolved_cwd), "approval", audit_id)


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


def _decision(
    allowed: bool,
    reason: str,
    mode: str,
    command: str,
    cwd: str,
    policy_decision: str,
    audit_id: str,
) -> TerminalPolicyResult:
    return TerminalPolicyResult(
        allowed=allowed,
        reason=reason,
        mode=mode,
        command=command,
        cwd=cwd,
        policy_decision=policy_decision,
        audit_id=audit_id,
    )


def _resolve_allowed_roots(roots: list[str]) -> list[Path]:
    items = roots if isinstance(roots, list) else []
    resolved: list[Path] = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        root = Path(text).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        resolved.append(root.resolve())
    if not resolved:
        resolved.append(Path.cwd().resolve())
    unique: list[Path] = []
    seen: set[str] = set()
    for value in resolved:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def _resolve_cwd_from_roots(*, cwd: str, allowed_roots: list[Path]) -> Path | None:
    raw = str(cwd or ".").strip() or "."
    candidate = Path(raw).expanduser()
    base = allowed_roots[0]
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve()
    for root in allowed_roots:
        if _is_subpath(resolved, root):
            return resolved
    return None


def _is_subpath(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except Exception:
        return False


def _paths_in_args_are_allowed(*, parts: list[str], cwd: Path, allowed_roots: list[Path]) -> bool:
    for token in parts[1:]:
        text = str(token or "").strip()
        if not text or text.startswith("-"):
            continue
        if "://" in text:
            continue
        maybe_path = text.startswith("/") or text.startswith(".") or "/" in text
        if not maybe_path:
            continue
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = cwd / candidate
        resolved = candidate.resolve()
        if not any(_is_subpath(resolved, root) for root in allowed_roots):
            return False
    return True


def _looks_like_write(parts: list[str], *, write_cmds: set[str]) -> bool:
    if not parts:
        return False
    head = str(parts[0]).strip().lower()
    if head in write_cmds:
        return True
    return any(token in {"--write", "--save", "--save-dev", "-w"} for token in parts[1:])


def _looks_like_network(head: str, *, network_cmds: set[str]) -> bool:
    return str(head or "").strip().lower() in network_cmds


def _truncate_text(value: str | None, limit: int = 4000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit]
