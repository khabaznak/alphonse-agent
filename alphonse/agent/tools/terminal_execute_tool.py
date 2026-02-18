from __future__ import annotations

from typing import Any

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
        roots = settings.get_terminal_allowed_roots()
        result = self._terminal.execute_with_policy(
            command=command,
            cwd=cwd,
            allowed_roots=roots,
            mode=mode,
            timeout_seconds=timeout_seconds,
        )
        return result

