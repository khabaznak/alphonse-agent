from __future__ import annotations

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
        timeout_seconds: float = 30.0,
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
            cwd=cwd,
            allowed_roots=roots,
            mode=mode,
            timeout_seconds=timeout_seconds,
        )
        return result


def _allowed_roots() -> list[str]:
    try:
        rows = list_sandbox_aliases(enabled_only=True, limit=500)
    except Exception:
        return []
    roots = [
        str(item.get("base_path") or "").strip()
        for item in rows
        if isinstance(item, dict)
    ]
    roots = [path for path in roots if path]
    return roots
