from __future__ import annotations

from alphonse.agent.tools.base import ToolExecutionState, tool_execution_mark


def test_tool_execution_mark_is_predictable() -> None:
    assert tool_execution_mark("getTime", ToolExecutionState.STARTED) == "tool::getTime::started"
    assert tool_execution_mark("terminal_sync", "FAILED") == "tool::terminal_sync::failed"

