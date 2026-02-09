from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

from alphonse.agent.nervous_system.terminal_tools import (
    get_terminal_command,
    get_terminal_sandbox,
    get_terminal_session,
    list_terminal_commands,
    record_terminal_command_output,
    update_terminal_command_status,
    update_terminal_session_status,
)
from alphonse.agent.nervous_system.tool_configs import get_active_tool_config
from alphonse.agent.nervous_system.senses.base import Sense, SignalSpec
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.agent.tools.terminal import TerminalTool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionPolicy:
    poll_seconds: float
    timeout_seconds: float
    max_batch: int


class TerminalExecutorSense(Sense):
    key = "terminal_executor"
    name = "Terminal Executor"
    description = "Executes approved terminal commands asynchronously."
    source_type = "system"
    signals = [
        SignalSpec(
            key="terminal.command_executed",
            name="Terminal Command Executed",
            description="Terminal command executed by background worker",
        )
    ]

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._bus: Bus | None = None
        config = _load_executor_config()
        self._enabled = config.get("enabled", False)
        poll = config.get("poll_seconds", 2.0)
        timeout = config.get("timeout_seconds", 30.0)
        max_batch = config.get("batch", 10)
        self._policy = ExecutionPolicy(
            poll_seconds=poll,
            timeout_seconds=timeout,
            max_batch=max_batch,
        )
        self._tool = TerminalTool()

    def start(self, bus: Bus) -> None:
        self._refresh_config()
        if not self._enabled:
            logger.info("TerminalExecutorSense disabled")
            return
        if self._thread and self._thread.is_alive():
            return
        self._bus = bus
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("TerminalExecutorSense started interval=%.2fs", self._policy.poll_seconds)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("TerminalExecutorSense stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._refresh_config()
            if not self._enabled:
                self._stop_event.wait(timeout=self._policy.poll_seconds)
                continue
            try:
                self._drain_once()
            except Exception:
                logger.exception("TerminalExecutorSense failed")
            self._stop_event.wait(timeout=self._policy.poll_seconds)

    def _refresh_config(self) -> None:
        config = _load_executor_config()
        self._enabled = bool(config.get("enabled", False))
        self._policy = ExecutionPolicy(
            poll_seconds=float(config.get("poll_seconds", self._policy.poll_seconds)),
            timeout_seconds=float(config.get("timeout_seconds", self._policy.timeout_seconds)),
            max_batch=int(config.get("batch", self._policy.max_batch)),
        )

    def _drain_once(self) -> None:
        commands = list_terminal_commands(status="approved", limit=self._policy.max_batch)
        for item in commands:
            command_id = item.get("command_id")
            if not command_id:
                continue
            command = get_terminal_command(command_id)
            if not command or command.get("status") != "approved":
                continue
            update_terminal_command_status(command_id, "running")
            session = get_terminal_session(command.get("session_id"))
            if not session:
                record_terminal_command_output(
                    command_id,
                    stdout="",
                    stderr="Missing terminal session",
                    exit_code=None,
                    status="failed",
                )
                continue
            sandbox = get_terminal_sandbox(session.get("sandbox_id"))
            if not sandbox or not sandbox.get("is_active"):
                record_terminal_command_output(
                    command_id,
                    stdout="",
                    stderr="Terminal sandbox missing or inactive",
                    exit_code=None,
                    status="failed",
                )
                update_terminal_session_status(session["session_id"], "failed")
                continue
            update_terminal_session_status(session["session_id"], "running")
            result = self._tool.execute(
                command=command.get("command") or "",
                cwd=command.get("cwd") or ".",
                sandbox_path=sandbox.get("path") or ".",
                timeout_seconds=_command_timeout(command, self._policy.timeout_seconds),
            )
            record_terminal_command_output(
                command_id,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                status=result.status,
            )
            update_terminal_session_status(session["session_id"], result.status)
            if self._bus:
                self._bus.emit(
                    Signal(
                        type="terminal.command_executed",
                        payload={
                            "command_id": command_id,
                            "session_id": session["session_id"],
                            "status": result.status,
                        },
                        source="terminal_executor",
                        correlation_id=command_id,
                    )
                )
            self._stop_event.wait(timeout=0.01)


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return max(1.0, float(raw))
    except ValueError:
        return default


def _parse_int(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _command_timeout(command: dict[str, object] | None, fallback: float) -> float:
    if not command:
        return fallback
    requested = command.get("timeout_seconds")
    if requested is None:
        return fallback
    try:
        return max(1.0, float(requested))
    except (TypeError, ValueError):
        return fallback


def _load_executor_config() -> dict[str, float | int | bool]:
    config = get_active_tool_config("terminal_executor")
    payload: dict[str, float | int | bool] = {
        "enabled": False,
        "poll_seconds": 2.0,
        "timeout_seconds": 30.0,
        "batch": 10,
    }
    if not config:
        return payload
    raw = config.get("config") or {}
    if isinstance(raw, dict):
        if "enabled" in raw:
            payload["enabled"] = bool(raw.get("enabled"))
        if "poll_seconds" in raw:
            payload["poll_seconds"] = _parse_float(str(raw.get("poll_seconds")), payload["poll_seconds"])
        if "timeout_seconds" in raw:
            payload["timeout_seconds"] = _parse_float(
                str(raw.get("timeout_seconds")), payload["timeout_seconds"]
            )
        if "batch" in raw:
            payload["batch"] = _parse_int(str(raw.get("batch")), payload["batch"])
    return payload
