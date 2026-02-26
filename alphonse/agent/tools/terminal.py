from __future__ import annotations

import os
import queue
import shlex
import subprocess
import threading
import time
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
    _DEFAULT_TIMEOUT_SECONDS = 120.0
    _MAX_TIMEOUT_SECONDS = 1800.0
    _DEFAULT_IDLE_TIMEOUT_SECONDS = 30.0
    _MAX_IDLE_TIMEOUT_SECONDS = 600.0

    def classify_command(self, command: str) -> str:
        cmd = str(command or "").strip()
        if not cmd:
            return "reject"
        if _requires_shell(cmd):
            return "approval"
        try:
            parts = shlex.split(cmd)
        except ValueError:
            return "reject"
        if not parts:
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
        use_shell = _requires_shell(cmd)
        parts: list[str] = []
        if not use_shell:
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
            completed = _run_command(
                command=cmd,
                argv=parts,
                cwd=resolved_cwd,
                use_shell=use_shell,
                timeout_seconds=timeout_seconds,
            )
        except _TerminalInputRequiredError as exc:
            return TerminalExecutionResult(
                status="failed",
                stdout=exc.stdout,
                stderr=exc.message,
                exit_code=None,
            )
        except _TerminalWatchdogTimeout as exc:
            return TerminalExecutionResult(
                status="failed",
                stdout=exc.stdout,
                stderr=exc.message,
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
        timeout_seconds: float = 120.0,
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
        cmd = str(command or "").strip()
        use_shell = _requires_shell(cmd)
        argv: list[str] = []
        if not use_shell:
            try:
                argv = shlex.split(cmd)
            except ValueError as exc:
                return {
                    "status": "failed",
                    "result": None,
                    "error": {"code": "invalid_command", "message": str(exc)},
                    "metadata": metadata,
                }
        resolved_cwd = Path(decision.cwd)
        try:
            completed = _run_command(
                command=cmd,
                argv=argv,
                cwd=resolved_cwd,
                use_shell=use_shell,
                timeout_seconds=_effective_timeout(timeout_seconds),
            )
        except _TerminalInputRequiredError as exc:
            return {
                "status": "failed",
                "result": {
                    "exit_code": None,
                    "stdout": _truncate_text(exc.stdout),
                    "stderr": _truncate_text(exc.stderr),
                },
                "error": {"code": exc.code, "message": exc.message},
                "metadata": {
                    **metadata,
                    "elapsed_ms": exc.elapsed_ms,
                    "watchdog_reason": exc.code,
                },
            }
        except _TerminalWatchdogTimeout as exc:
            return {
                "status": "failed",
                "result": {
                    "exit_code": None,
                    "stdout": _truncate_text(exc.stdout),
                    "stderr": _truncate_text(exc.stderr),
                },
                "error": {"code": exc.code, "message": exc.message},
                "metadata": {
                    **metadata,
                    "elapsed_ms": exc.elapsed_ms,
                    "watchdog_reason": exc.code,
                },
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
                "metadata": {
                    **metadata,
                    "elapsed_ms": completed.elapsed_ms,
                },
            }
        return {
            "status": "ok",
            "result": {
                "exit_code": int(completed.returncode or 0),
                "stdout": _truncate_text(completed.stdout),
                "stderr": _truncate_text(completed.stderr),
            },
            "error": None,
            "metadata": {
                **metadata,
                "elapsed_ms": completed.elapsed_ms,
            },
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
        effective_mode = mode if mode in {"readonly", "dev", "ops"} else "readonly"
        uses_shell = _requires_shell(cmd)
        parts: list[str] = []
        if not uses_shell:
            try:
                parts = shlex.split(cmd)
            except ValueError:
                return _decision(False, "command_parse_failed", effective_mode, cmd, cwd, "reject", audit_id)
            if not parts:
                return _decision(False, "command_empty", effective_mode, cmd, cwd, "reject", audit_id)
        head = str(parts[0]).strip().lower() if parts else ""
        if head in self._BLOCK_ALWAYS:
            return _decision(False, "command_blocked", effective_mode, cmd, cwd, "reject", audit_id)

        resolved_roots = _resolve_allowed_roots(allowed_roots)
        resolved_cwd = _resolve_cwd_from_roots(cwd=cwd, allowed_roots=resolved_roots)
        if resolved_cwd is None:
            return _decision(False, "cwd_not_allowed", effective_mode, cmd, cwd, "reject", audit_id)

        if effective_mode != "ops" and uses_shell:
            return _decision(False, "shell_operators_disallowed", effective_mode, cmd, str(resolved_cwd), "reject", audit_id)

        if parts and not _paths_in_args_are_allowed(parts=parts, cwd=resolved_cwd, allowed_roots=resolved_roots):
            return _decision(False, "path_not_allowed", effective_mode, cmd, str(resolved_cwd), "reject", audit_id)

        if effective_mode == "readonly":
            if not parts:
                return _decision(False, "mode_readonly_command_disallowed", effective_mode, cmd, str(resolved_cwd), "reject", audit_id)
            if head not in self._READONLY_CMDS:
                return _decision(False, "mode_readonly_command_disallowed", effective_mode, cmd, str(resolved_cwd), "reject", audit_id)
            if _looks_like_write(parts, write_cmds=self._WRITE_HINT_CMDS):
                return _decision(False, "mode_readonly_write_disallowed", effective_mode, cmd, str(resolved_cwd), "reject", audit_id)
            if _looks_like_network(head, network_cmds=self._NETWORK_HINT_CMDS):
                return _decision(False, "mode_readonly_network_disallowed", effective_mode, cmd, str(resolved_cwd), "reject", audit_id)
            return _decision(True, "allowed_readonly", effective_mode, cmd, str(resolved_cwd), "auto", audit_id)
        if effective_mode == "dev":
            if not parts:
                return _decision(False, "mode_dev_command_disallowed", effective_mode, cmd, str(resolved_cwd), "reject", audit_id)
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


def _run_command(
    *,
    command: str,
    argv: list[str],
    cwd: Path,
    use_shell: bool,
    timeout_seconds: float | None,
) -> "_TerminalRunResult":
    env_overrides: dict[str, str] = {}
    argv_to_run = argv
    if not use_shell:
        env_overrides, argv_to_run = _split_leading_env_assignments(argv)
        if env_overrides and not argv_to_run:
            raise _TerminalInputRequiredError(
                code="invalid_command",
                message="Environment assignments provided without an executable command.",
                stdout="",
                stderr="",
                elapsed_ms=0,
            )

    if not use_shell and _requires_stdin_without_input(argv_to_run):
        raise _TerminalInputRequiredError(
            code="stdin_required_no_input",
            message="Command expects stdin input but none was provided.",
            stdout="",
            stderr="",
            elapsed_ms=0,
        )

    started = time.monotonic()
    wall_timeout = None if timeout_seconds is None else max(1.0, float(timeout_seconds))
    idle_timeout = _effective_idle_timeout(wall_timeout)
    env = _scrub_env(os.environ)
    if env_overrides:
        env.update(env_overrides)
    if use_shell:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            shell=True,
            executable="/bin/bash",
        )
    else:
        process = subprocess.Popen(
            argv_to_run,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

    out_chunks: list[str] = []
    err_chunks: list[str] = []
    output_queue: queue.Queue[tuple[str, str]] = queue.Queue()
    done_flags = {"stdout": False, "stderr": False}

    def _reader(stream_name: str, stream: object) -> None:
        if stream is None:
            done_flags[stream_name] = True
            return
        while True:
            chunk = stream.read(4096)  # type: ignore[attr-defined]
            if not chunk:
                break
            output_queue.put((stream_name, str(chunk)))
        done_flags[stream_name] = True

    stdout_thread = threading.Thread(target=_reader, args=("stdout", process.stdout), daemon=True)
    stderr_thread = threading.Thread(target=_reader, args=("stderr", process.stderr), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    last_output_at = started

    try:
        while True:
            now = time.monotonic()
            elapsed = now - started
            if wall_timeout is not None and elapsed >= wall_timeout:
                process.kill()
                process.wait(timeout=2)
                _drain_output_queue(output_queue=output_queue, out_chunks=out_chunks, err_chunks=err_chunks)
                raise _TerminalWatchdogTimeout(
                    code="timeout",
                    message="Command timed out",
                    stdout="".join(out_chunks),
                    stderr="".join(err_chunks),
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                )
            if idle_timeout > 0 and (now - last_output_at) >= idle_timeout and process.poll() is None:
                process.kill()
                process.wait(timeout=2)
                _drain_output_queue(output_queue=output_queue, out_chunks=out_chunks, err_chunks=err_chunks)
                raise _TerminalWatchdogTimeout(
                    code="idle_timeout",
                    message="Command produced no output for too long",
                    stdout="".join(out_chunks),
                    stderr="".join(err_chunks),
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                )
            wait_step = 0.2
            if wall_timeout is not None:
                remaining = max(0.01, wall_timeout - elapsed)
                wait_step = min(wait_step, remaining)
            try:
                stream_name, chunk = output_queue.get(timeout=wait_step)
                if stream_name == "stdout":
                    out_chunks.append(chunk)
                else:
                    err_chunks.append(chunk)
                last_output_at = time.monotonic()
                continue
            except queue.Empty:
                pass
            if process.poll() is not None and done_flags["stdout"] and done_flags["stderr"] and output_queue.empty():
                break

        return _TerminalRunResult(
            returncode=int(process.returncode or 0),
            stdout="".join(out_chunks),
            stderr="".join(err_chunks),
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
    finally:
        try:
            if process.stdout is not None:
                process.stdout.close()
        except Exception:
            pass
        try:
            if process.stderr is not None:
                process.stderr.close()
        except Exception:
            pass


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


def _effective_timeout(value: float | int | None) -> float:
    default_timeout = _read_positive_float(
        "ALPHONSE_TERMINAL_DEFAULT_TIMEOUT_SECONDS",
        TerminalTool._DEFAULT_TIMEOUT_SECONDS,
    )
    max_timeout = _read_positive_float(
        "ALPHONSE_TERMINAL_MAX_TIMEOUT_SECONDS",
        TerminalTool._MAX_TIMEOUT_SECONDS,
    )
    if max_timeout < 1.0:
        max_timeout = TerminalTool._MAX_TIMEOUT_SECONDS
    raw = default_timeout if value is None else value
    try:
        requested = float(raw)
    except (TypeError, ValueError):
        requested = default_timeout
    return max(1.0, min(requested, max_timeout))


def _effective_idle_timeout(timeout_seconds: float | None) -> float:
    configured = _read_positive_float(
        "ALPHONSE_TERMINAL_IDLE_TIMEOUT_SECONDS",
        TerminalTool._DEFAULT_IDLE_TIMEOUT_SECONDS,
    )
    max_idle = _read_positive_float(
        "ALPHONSE_TERMINAL_MAX_IDLE_TIMEOUT_SECONDS",
        TerminalTool._MAX_IDLE_TIMEOUT_SECONDS,
    )
    idle = max(0.0, min(configured, max_idle))
    if timeout_seconds is None:
        return idle
    return min(idle, max(0.0, timeout_seconds))


def _requires_shell(command: str) -> bool:
    text = str(command or "").strip()
    if not text:
        return False
    shell_markers = ("|", "||", "&&", ";", ">", ">>", "<", "$(", "`", "\n")
    return any(marker in text for marker in shell_markers)


def _requires_stdin_without_input(argv: list[str]) -> bool:
    if not argv:
        return False
    head = str(argv[0]).strip().lower()
    if head not in {"python", "python3", "bash", "sh", "zsh", "node"}:
        return False
    return any(str(token).strip() == "-" for token in argv[1:])


def _split_leading_env_assignments(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    env_overrides: dict[str, str] = {}
    idx = 0
    for token in argv:
        text = str(token or "").strip()
        if not text:
            break
        if "=" not in text:
            break
        name, value = text.split("=", 1)
        if not name or not _is_valid_env_name(name):
            break
        env_overrides[name] = value
        idx += 1
    return env_overrides, argv[idx:]


def _is_valid_env_name(name: str) -> bool:
    if not name:
        return False
    first = name[0]
    if not (first.isalpha() or first == "_"):
        return False
    return all(ch.isalnum() or ch == "_" for ch in name[1:])


def _drain_output_queue(
    *,
    output_queue: "queue.Queue[tuple[str, str]]",
    out_chunks: list[str],
    err_chunks: list[str],
) -> None:
    while True:
        try:
            stream_name, chunk = output_queue.get_nowait()
        except queue.Empty:
            return
        if stream_name == "stdout":
            out_chunks.append(chunk)
        else:
            err_chunks.append(chunk)


@dataclass(frozen=True)
class _TerminalRunResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_ms: int


class _TerminalWatchdogTimeout(Exception):
    def __init__(self, *, code: str, message: str, stdout: str, stderr: str, elapsed_ms: int) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.stdout = str(stdout or "")
        self.stderr = str(stderr or "")
        self.elapsed_ms = int(elapsed_ms)


class _TerminalInputRequiredError(Exception):
    def __init__(self, *, code: str, message: str, stdout: str, stderr: str, elapsed_ms: int) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.stdout = str(stdout or "")
        self.stderr = str(stderr or "")
        self.elapsed_ms = int(elapsed_ms)


def _read_positive_float(name: str, default: float) -> float:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    if value <= 0:
        return float(default)
    return float(value)
