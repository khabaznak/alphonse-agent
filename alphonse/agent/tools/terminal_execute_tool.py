from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from alphonse.agent.nervous_system.sandbox_dirs import list_sandbox_aliases
from alphonse.agent.tools.terminal import TerminalTool
from alphonse.config import settings


class TerminalExecuteTool:
    def __init__(self, terminal: TerminalTool | None = None) -> None:
        self._terminal = terminal or TerminalTool()

    def execute(
        self,
        *,
        command: str,
        cwd: str = ".",
        timeout_seconds: float | None = None,
        timeout_ms: float | int | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = state
        mode = settings.get_execution_mode()
        roots = _allowed_roots()
        if not roots:
            return {
                "status": "failed",
                "result": None,
                "error": {
                    "code": "sandbox_roots_not_configured",
                    "message": "No enabled sandbox directories found. Configure sandbox dirs in nerve-db.",
                },
                "metadata": {"mode": mode},
            }
        result = self._terminal.execute_with_policy(
            command=command,
            cwd=_normalize_cwd(cwd=cwd, roots=roots),
            allowed_roots=roots,
            mode=mode,
            timeout_seconds=_resolve_timeout(timeout_seconds, timeout_ms),
        )
        return result


def _allowed_roots() -> list[str]:
    try:
        rows = list_sandbox_aliases(enabled_only=True, limit=500)
    except Exception:
        return []
    prioritized = sorted(
        [item for item in rows if isinstance(item, dict)],
        key=_root_priority,
    )
    roots = [str(item.get("base_path") or "").strip() for item in prioritized]
    roots = [path for path in roots if path]
    return roots


def _root_priority(record: dict[str, Any]) -> tuple[int, str]:
    alias = str(record.get("alias") or "").strip().lower()
    if alias == "main":
        return (0, alias)
    if alias == "dumpster":
        return (1, alias)
    return (2, alias)


def _resolve_timeout(value_seconds: float | None, value_ms: float | int | None) -> float:
    default_timeout = _read_positive_float(
        "ALPHONSE_TERMINAL_DEFAULT_TIMEOUT_SECONDS",
        120.0,
    )
    if value_seconds is not None:
        try:
            requested = float(value_seconds)
        except (TypeError, ValueError):
            requested = default_timeout
    elif value_ms is not None:
        try:
            requested = float(value_ms) / 1000.0
        except (TypeError, ValueError):
            requested = default_timeout
    else:
        requested = default_timeout
    max_timeout = _read_positive_float(
        "ALPHONSE_TERMINAL_MAX_TIMEOUT_SECONDS",
        1800.0,
    )
    return max(1.0, min(requested, max_timeout))


def _normalize_cwd(*, cwd: str | None, roots: list[str]) -> str:
    raw = str(cwd or "").strip()
    if raw and raw != ".":
        return raw
    if roots:
        return str(Path(roots[0]).resolve())
    return "."


def _read_positive_float(name: str, default: float) -> float:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return float(default)
    try:
        parsed = float(raw)
    except ValueError:
        return float(default)
    if parsed <= 0:
        return float(default)
    return float(parsed)
