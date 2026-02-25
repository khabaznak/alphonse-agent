from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from alphonse.agent.nervous_system.sandbox_dirs import list_sandbox_aliases
from alphonse.agent.nervous_system.terminal_tools import (
    create_terminal_command,
    ensure_terminal_session,
    get_terminal_command,
    record_terminal_command_output,
    update_terminal_command_status,
    update_terminal_session_status,
    upsert_terminal_sandbox,
)
from alphonse.agent.tools.terminal import TerminalTool
from alphonse.config import settings


class TerminalCommandSubmitTool:
    def __init__(self, terminal: TerminalTool | None = None) -> None:
        self._terminal = terminal or TerminalTool()

    def execute(
        self,
        *,
        command: str,
        cwd: str = ".",
        timeout_seconds: float | None = None,
        sandbox_alias: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        aliases = _enabled_aliases()
        if not aliases:
            return _failed("sandbox_roots_not_configured", "No enabled sandbox directories found.")
        selected = _select_alias(aliases=aliases, requested=sandbox_alias)
        if not selected:
            return _failed("sandbox_alias_not_found", "Requested sandbox alias is not enabled.")
        principal_id = _resolve_principal_id(state)
        sandbox_id = f"sandbox_alias:{selected['alias']}"
        try:
            upsert_terminal_sandbox(
                {
                    "sandbox_id": sandbox_id,
                    "owner_principal_id": principal_id,
                    "label": str(selected["alias"]),
                    "path": str(selected["base_path"]),
                    "is_active": True,
                }
            )
            session_id = ensure_terminal_session(principal_id=principal_id, sandbox_id=sandbox_id)
            command_id = create_terminal_command(
                {
                    "session_id": session_id,
                    "command": str(command or "").strip(),
                    "cwd": _normalize_cwd(cwd=cwd, roots=_allowed_roots()),
                    "status": "pending",
                    "requested_by": principal_id,
                    "approved_by": "system:auto",
                    "timeout_seconds": timeout_seconds,
                }
            )
        except Exception as exc:
            return _failed("terminal_submit_failed", str(exc))
        worker = threading.Thread(
            target=_run_terminal_command,
            kwargs={
                "terminal": self._terminal,
                "command_id": command_id,
                "session_id": session_id,
                "command": str(command or "").strip(),
                "cwd": _normalize_cwd(cwd=cwd, roots=_allowed_roots()),
                "timeout_seconds": timeout_seconds,
            },
            daemon=True,
        )
        worker.start()
        return _ok(
            {
                "command_id": command_id,
                "session_id": session_id,
                "status": "pending",
                "sandbox_alias": selected["alias"],
            }
        )


class TerminalCommandStatusTool:
    def execute(self, *, command_id: str) -> dict[str, Any]:
        item = get_terminal_command(str(command_id or "").strip())
        if not item:
            return _failed("terminal_command_not_found", "Terminal command not found.")
        status = str(item.get("status") or "")
        return _ok(
            {
                "command_id": str(item.get("command_id") or ""),
                "session_id": str(item.get("session_id") or ""),
                "status": status,
                "done": status in {"executed", "failed"},
                "exit_code": item.get("exit_code"),
                "stdout": str(item.get("stdout") or ""),
                "stderr": str(item.get("stderr") or ""),
                "updated_at": item.get("updated_at"),
            }
        )


def _run_terminal_command(
    *,
    terminal: TerminalTool,
    command_id: str,
    session_id: str,
    command: str,
    cwd: str,
    timeout_seconds: float | None,
) -> None:
    try:
        update_terminal_command_status(command_id, "running", approved_by="system:auto")
        update_terminal_session_status(session_id, "running")
        result = terminal.execute_with_policy(
            command=command,
            cwd=cwd,
            allowed_roots=_allowed_roots(),
            mode=settings.get_execution_mode(),
            timeout_seconds=timeout_seconds,
        )
        result_payload = result.get("result") if isinstance(result.get("result"), dict) else {}
        error_payload = result.get("error") if isinstance(result.get("error"), dict) else {}
        status = "executed" if str(result.get("status") or "") == "ok" else "failed"
        stdout = str(result_payload.get("stdout") or "")
        stderr = str(result_payload.get("stderr") or "")
        if not stderr and status == "failed":
            stderr = str(error_payload.get("message") or "terminal command failed")
        exit_code = result_payload.get("exit_code")
        record_terminal_command_output(
            command_id,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code if isinstance(exit_code, int) else None,
            status=status,
        )
        update_terminal_session_status(session_id, status)
    except Exception as exc:
        record_terminal_command_output(
            command_id,
            stdout="",
            stderr=str(exc),
            exit_code=None,
            status="failed",
        )
        update_terminal_session_status(session_id, "failed")


def _enabled_aliases() -> list[dict[str, str]]:
    rows = list_sandbox_aliases(enabled_only=True, limit=500)
    out: list[dict[str, str]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        alias = str(item.get("alias") or "").strip()
        base_path = str(item.get("base_path") or "").strip()
        if not alias or not base_path:
            continue
        out.append({"alias": alias, "base_path": base_path})
    return out


def _allowed_roots() -> list[str]:
    aliases = _enabled_aliases()
    prioritized = sorted(aliases, key=_root_priority)
    return [str(item["base_path"]) for item in prioritized]


def _select_alias(
    *,
    aliases: list[dict[str, str]],
    requested: str | None,
) -> dict[str, str] | None:
    req = str(requested or "").strip().lower()
    if req:
        for item in aliases:
            if str(item.get("alias") or "").strip().lower() == req:
                return item
        return None
    prioritized = sorted(aliases, key=_root_priority)
    return prioritized[0] if prioritized else None


def _root_priority(record: dict[str, str]) -> tuple[int, str]:
    alias = str(record.get("alias") or "").strip().lower()
    if alias == "main":
        return (0, alias)
    if alias == "dumpster":
        return (1, alias)
    return (2, alias)


def _resolve_principal_id(state: dict[str, Any] | None) -> str:
    payload = state if isinstance(state, dict) else {}
    for key in ("actor_person_id", "incoming_user_id", "channel_target", "chat_id"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return "default"


def _normalize_cwd(*, cwd: str | None, roots: list[str]) -> str:
    raw = str(cwd or "").strip()
    if raw and raw != ".":
        return raw
    if roots:
        return str(Path(roots[0]).resolve())
    return "."


def _ok(result: dict[str, Any]) -> dict[str, Any]:
    return {"status": "ok", "result": result, "error": None}


def _failed(code: str, message: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "result": None,
        "error": {"code": str(code), "message": str(message)},
    }
