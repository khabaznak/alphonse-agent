from __future__ import annotations

import os
import queue
import signal
import shlex
import subprocess
import threading
import time
import uuid
from typing import Any


class BashTool:
    canonical_name: str = "execution.run_bash"
    capability: str = "execution"

    _DEFAULT_TIMEOUT_SECONDS = 120.0
    _OUTPUT_LIMIT_CHARS = 64 * 1024
    _STOP_TIMEOUT_SECONDS = 2.0

    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[str] = queue.Queue()
        self._stderr_queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.RLock()

    def execute(
        self,
        *,
        command: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = state
        started = time.monotonic()
        raw_command = str(command or "").strip()
        if not raw_command:
            return _failed(
                code="bash_command_empty",
                message="Command is empty.",
                output={"exit_code": None, "stdout": "", "stderr": ""},
                metadata=_metadata(started_at=started),
            )

        timeout = _resolve_timeout(timeout_seconds, self._DEFAULT_TIMEOUT_SECONDS)
        delimiter = f"__ALPHONSE_BASH_DONE_{uuid.uuid4().hex}__"

        with self._lock:
            process = self._ensure_process()
            if process.stdin is None:
                self._reset_process()
                return _failed(
                    code="bash_stdin_unavailable",
                    message="Bash process stdin is unavailable.",
                    output={"exit_code": None, "stdout": "", "stderr": ""},
                    metadata=_metadata(started_at=started, delimiter=delimiter),
                )

            self._drain_queues()
            wrapped = _wrap_command(command=raw_command, cwd=cwd, delimiter=delimiter)
            try:
                process.stdin.write(wrapped)
                process.stdin.flush()
            except Exception as exc:
                self._reset_process()
                return _failed(
                    code="bash_write_failed",
                    message=str(exc) or exc.__class__.__name__,
                    output={"exit_code": None, "stdout": "", "stderr": ""},
                    metadata=_metadata(started_at=started, delimiter=delimiter),
                )

            stdout, stderr, exit_code, timed_out = self._read_until_delimiter(
                delimiter=delimiter,
                timeout_seconds=timeout,
            )

            metadata = _metadata(started_at=started, delimiter=delimiter)
            if timed_out:
                self._reset_process()
                return _failed(
                    code="bash_command_timeout",
                    message="Bash command timed out.",
                    output={"exit_code": None, "stdout": stdout, "stderr": stderr},
                    metadata=metadata,
                )

            output = {
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
            }
            if exit_code != 0:
                return _failed(
                    code="bash_non_zero_exit",
                    message="Bash command returned non-zero exit status.",
                    output=output,
                    metadata=metadata,
                )
            return {
                "output": output,
                "exception": None,
                "metadata": metadata,
            }

    def close(self) -> None:
        with self._lock:
            self._reset_process()

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process

        self._process = subprocess.Popen(
            ["/bin/bash", "--noprofile", "--norc"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=_scrub_env(os.environ),
            start_new_session=True,
        )
        _start_reader(self._process.stdout, self._stdout_queue)
        _start_reader(self._process.stderr, self._stderr_queue)
        return self._process

    def _reset_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=self._STOP_TIMEOUT_SECONDS)
        except Exception:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=self._STOP_TIMEOUT_SECONDS)
            except Exception:
                pass

    def _read_until_delimiter(
        self,
        *,
        delimiter: str,
        timeout_seconds: float,
    ) -> tuple[str, str, int | None, bool]:
        deadline = time.monotonic() + timeout_seconds
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        exit_code: int | None = None

        while time.monotonic() < deadline:
            _drain_queue(self._stderr_queue, stderr_parts)
            try:
                chunk = self._stdout_queue.get(timeout=0.05)
            except queue.Empty:
                process = self._process
                if process is None or process.poll() is not None:
                    _drain_queue(self._stdout_queue, stdout_parts)
                    _drain_queue(self._stderr_queue, stderr_parts)
                    return (
                        _truncate("".join(stdout_parts)),
                        _truncate("".join(stderr_parts)),
                        exit_code,
                        False,
                    )
                continue

            stdout_parts.append(chunk)
            joined_stdout = "".join(stdout_parts)
            parsed = _split_delimiter(joined_stdout, delimiter)
            if parsed is None:
                continue
            stdout_text, exit_code = parsed
            _drain_queue(self._stderr_queue, stderr_parts)
            return (
                _truncate(stdout_text),
                _truncate("".join(stderr_parts)),
                exit_code,
                False,
            )

        _drain_queue(self._stdout_queue, stdout_parts)
        _drain_queue(self._stderr_queue, stderr_parts)
        return (
            _truncate("".join(stdout_parts)),
            _truncate("".join(stderr_parts)),
            None,
            True,
        )

    def _drain_queues(self) -> None:
        _drain_queue(self._stdout_queue, [])
        _drain_queue(self._stderr_queue, [])


def _wrap_command(*, command: str, cwd: str | None, delimiter: str) -> str:
    lines: list[str] = []
    if str(cwd or "").strip():
        lines.append(f"cd {shlex.quote(str(cwd).strip())}")
    lines.append(command)
    lines.append("__alphonse_bash_status=$?")
    lines.append(f"printf '\\n%s:%s\\n' {shlex.quote(delimiter)} \"$__alphonse_bash_status\"")
    return "\n".join(lines) + "\n"


def _split_delimiter(stdout: str, delimiter: str) -> tuple[str, int | None] | None:
    marker = f"{delimiter}:"
    index = stdout.find(marker)
    if index < 0:
        return None
    before = stdout[:index]
    after = stdout[index + len(marker) :]
    raw_code = after.splitlines()[0].strip() if after else ""
    try:
        exit_code = int(raw_code)
    except ValueError:
        exit_code = None
    return before.rstrip("\n"), exit_code


def _start_reader(stream: Any, output_queue: queue.Queue[str]) -> None:
    def _reader() -> None:
        if stream is None:
            return
        while True:
            try:
                chunk = stream.readline()
            except Exception:
                break
            if not chunk:
                break
            output_queue.put(str(chunk))

    threading.Thread(target=_reader, daemon=True).start()


def _drain_queue(source: queue.Queue[str], target: list[str]) -> None:
    while True:
        try:
            target.append(source.get_nowait())
        except queue.Empty:
            return


def _scrub_env(env: os._Environ[str]) -> dict[str, str]:
    safe = dict(env)
    for key in list(safe):
        upper = key.upper()
        if upper.endswith("_API_KEY") or upper.endswith("_TOKEN"):
            safe.pop(key, None)
    return safe


def _resolve_timeout(value: float | None, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(1.0, parsed)


def _truncate(value: str, *, limit: int = BashTool._OUTPUT_LIMIT_CHARS) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit]


def _metadata(*, started_at: float, delimiter: str | None = None) -> dict[str, Any]:
    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    metadata: dict[str, Any] = {
        "tool": BashTool.canonical_name,
        "elapsed_ms": elapsed_ms,
    }
    if delimiter:
        metadata["delimiter"] = delimiter
    return metadata


def _failed(
    *,
    code: str,
    message: str,
    output: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "output": output,
        "exception": {
            "code": code,
            "message": message,
        },
        "metadata": metadata,
    }
