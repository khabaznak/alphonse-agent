from __future__ import annotations

from alphonse.agent.tools.base import ToolExecutionState, tool_execution_mark


def test_tool_execution_mark_is_predictable() -> None:
    assert tool_execution_mark("get_time", ToolExecutionState.STARTED) == "tool::get_time::started"
    assert tool_execution_mark("execution.run_terminal", "FAILED") == "tool::execution.run_terminal::failed"

