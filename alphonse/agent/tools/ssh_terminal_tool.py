from __future__ import annotations

import os
import shlex
import time
from typing import Any


class SshTerminalTool:
    def execute(
        self,
        *,
        host: str,
        username: str,
        command: str,
        port: int | None = None,
        password: str | None = None,
        private_key_path: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        connect_timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        if not _ssh_terminal_enabled():
            return _failed("ssh_terminal_disabled", retryable=False)

        hostname = str(host or "").strip()
        user = str(username or "").strip()
        raw_command = str(command or "").strip()
        if not hostname:
            return _failed("host_required", retryable=False)
        if not user:
            return _failed("username_required", retryable=False)
        if not raw_command:
            return _failed("command_required", retryable=False)

        try:
            port_value = int(port if port is not None else 22)
        except (TypeError, ValueError):
            return _failed("invalid_port", retryable=False)
        if port_value <= 0 or port_value > 65535:
            return _failed("invalid_port", retryable=False)

        timeout_value = _normalize_command_timeout(timeout_seconds)
        connect_timeout_value = _normalize_connect_timeout(connect_timeout_seconds)
        remote_command = _with_cwd(raw_command, cwd)
        allow_agent = _env_bool("ALPHONSE_SSH_TERMINAL_ALLOW_AGENT", True)
        look_for_keys = _env_bool("ALPHONSE_SSH_TERMINAL_LOOK_FOR_KEYS", True)
        strict_host_key = _env_bool("ALPHONSE_SSH_TERMINAL_STRICT_HOST_KEY", False)
        known_hosts = str(os.getenv("ALPHONSE_SSH_TERMINAL_KNOWN_HOSTS_PATH") or "").strip()

        try:
            import paramiko  # type: ignore
        except Exception:
            return _failed("paramiko_not_installed", retryable=False)

        client = paramiko.SSHClient()
        if strict_host_key:
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
            if known_hosts:
                try:
                    client.load_host_keys(known_hosts)
                except Exception as exc:
                    return _failed(
                        "known_hosts_load_failed",
                        retryable=False,
                        detail=str(exc),
                        known_hosts_path=known_hosts,
                    )
        else:
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            client.connect(
                hostname=hostname,
                port=port_value,
                username=user,
                password=password or None,
                key_filename=(str(private_key_path).strip() or None) if private_key_path else None,
                timeout=connect_timeout_value,
                banner_timeout=connect_timeout_value,
                auth_timeout=connect_timeout_value,
                allow_agent=allow_agent,
                look_for_keys=look_for_keys,
            )
            transport = client.get_transport()
            if transport is None:
                return _failed("ssh_transport_unavailable", retryable=True)
            channel = transport.open_session(timeout=connect_timeout_value)
            channel.exec_command(remote_command)
            exit_code, stdout, stderr = _read_channel(channel=channel, timeout_seconds=timeout_value)
        except TimeoutError:
            return _failed("ssh_command_timeout", retryable=True)
        except Exception as exc:
            message = str(exc)
            lowered = message.lower()
            if "authentication failed" in lowered:
                return _failed("ssh_auth_failed", retryable=False, detail=message)
            return _failed("ssh_execution_failed", retryable=True, detail=message)
        finally:
            try:
                client.close()
            except Exception:
                pass

        metadata = {
            "tool": "ssh_terminal",
            "host": hostname,
            "port": port_value,
            "username": user,
            "timeout_seconds": timeout_value,
            "connect_timeout_seconds": connect_timeout_value,
        }
        if exit_code != 0:
            return {
                "status": "failed",
                "result": {
                    "exit_code": int(exit_code),
                    "stdout": _truncate_output(stdout),
                    "stderr": _truncate_output(stderr),
                },
                "error": {
                    "code": "ssh_non_zero_exit",
                    "message": "remote command returned non-zero exit",
                    "retryable": False,
                },
                "metadata": metadata,
            }

        return {
            "status": "ok",
            "result": {
                "exit_code": int(exit_code),
                "stdout": _truncate_output(stdout),
                "stderr": _truncate_output(stderr),
            },
            "error": None,
            "metadata": metadata,
        }


def _with_cwd(command: str, cwd: str | None) -> str:
    raw_cwd = str(cwd or "").strip()
    if not raw_cwd:
        return command
    return f"cd {shlex.quote(raw_cwd)} && {command}"


def _read_channel(*, channel: Any, timeout_seconds: float) -> tuple[int, str, str]:
    started = time.monotonic()
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    while True:
        if channel.recv_ready():
            stdout_chunks.append(channel.recv(4096).decode("utf-8", errors="replace"))
        if channel.recv_stderr_ready():
            stderr_chunks.append(channel.recv_stderr(4096).decode("utf-8", errors="replace"))
        if channel.exit_status_ready():
            break
        if (time.monotonic() - started) > timeout_seconds:
            try:
                channel.close()
            except Exception:
                pass
            raise TimeoutError("ssh command timed out")
        time.sleep(0.05)

    while channel.recv_ready():
        stdout_chunks.append(channel.recv(4096).decode("utf-8", errors="replace"))
    while channel.recv_stderr_ready():
        stderr_chunks.append(channel.recv_stderr(4096).decode("utf-8", errors="replace"))

    exit_code = int(channel.recv_exit_status())
    return exit_code, "".join(stdout_chunks), "".join(stderr_chunks)


def _normalize_command_timeout(value: float | None) -> float:
    default_timeout = _read_float("ALPHONSE_SSH_TERMINAL_DEFAULT_TIMEOUT_SECONDS", 30.0)
    max_timeout = _read_float("ALPHONSE_SSH_TERMINAL_MAX_TIMEOUT_SECONDS", 600.0)
    if max_timeout < 1.0:
        max_timeout = 1.0
    if value is None:
        timeout = default_timeout
    else:
        try:
            timeout = float(value)
        except (TypeError, ValueError):
            timeout = default_timeout
    return max(1.0, min(timeout, max_timeout))


def _normalize_connect_timeout(value: float | None) -> float:
    default_timeout = _read_float("ALPHONSE_SSH_TERMINAL_CONNECT_TIMEOUT_SECONDS", 10.0)
    if value is None:
        timeout = default_timeout
    else:
        try:
            timeout = float(value)
        except (TypeError, ValueError):
            timeout = default_timeout
    return max(1.0, min(timeout, 120.0))


def _ssh_terminal_enabled() -> bool:
    return _env_bool("ALPHONSE_ENABLE_SSH_TERMINAL", False)


def _read_float(name: str, fallback: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return fallback
    try:
        return float(raw)
    except (TypeError, ValueError):
        return fallback


def _env_bool(name: str, fallback: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return fallback
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _truncate_output(value: Any, *, limit: int = 8000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit]


def _failed(error_code: str, *, retryable: bool, **kwargs: Any) -> dict[str, Any]:
    return {
        "status": "failed",
        "result": None,
        "error": {
            "code": str(error_code or "ssh_terminal_failed"),
            "message": str(error_code or "ssh_terminal_failed"),
            "retryable": bool(retryable),
            "details": dict(kwargs),
        },
        "metadata": {"tool": "ssh_terminal"},
    }
