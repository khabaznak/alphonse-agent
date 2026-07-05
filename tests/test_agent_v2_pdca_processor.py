from __future__ import annotations

from datetime import datetime
from importlib import import_module

from alphonse.agent_v2.core.core import AlphonseCore
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.core import LoopStepStatus
from alphonse.agent_v2.core.core import ProcessingStatus
from alphonse.agent_v2.core.core import StateSnapshot
from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.core import ToolKind
from alphonse.agent_v2.core.intelligence.pdca import PDCAIntelligenceProcessor
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.state import reset_state

plan_node_module = import_module("alphonse.agent_v2.core.intelligence.pdca.nodes.plan_node")


def test_pdca_processor_runs_graph_and_returns_completed_result() -> None:
    task = TaskState(goal="Write the file", user="alex")
    processor = PDCAIntelligenceProcessor()

    result = processor.process(task, CoreLoopContext(messages=InMemoryMessageQueue()))

    assert result.status == ProcessingStatus.COMPLETED
    assert result.snapshot.current_work == "Write the file"
    assert result.snapshot.task_owner == "alex"
    assert result.snapshot.metadata["check_verdict"] == "new"
    assert result.snapshot.metadata["task_state"]["check_verdict"] == "new"
    assert "acceptance_criteria_prompt" in result.snapshot.metadata["task_state"]["metadata"]
    assert result.snapshot.metadata["task_state"]["metadata"]["acceptance_criteria_llm_stubbed"] is True


def test_core_loop_can_use_pdca_processor_snapshot_metadata() -> None:
    reset_state()
    queue = InMemoryMessageQueue()
    queue.enqueue(CoreMessage(timestamp=datetime.now().astimezone(), prompt="Write the file", user="alex"))
    visible_state = _RecordingState()
    core = AlphonseCore(
        intelligence=PDCAIntelligenceProcessor(),
        messages=queue,
        tools=_NullTools(),
        prompts=_NullPrompts(),
        state=visible_state,
        memory=_NullMemory(),
    )

    result = core.step()

    assert result.status == LoopStepStatus.PROCESSED
    assert visible_state.snapshot().metadata["check_verdict"] == "new"
    assert visible_state.snapshot().metadata["task_state"]["goal"] == "Write the file"


def test_pdca_processor_snapshot_includes_planned_tool_call_when_present(monkeypatch) -> None:
    planned = {"tool_id": "tool-1", "tool_name": "write_file", "arguments": {"path": "a.txt"}}
    monkeypatch.setattr(plan_node_module, "_call_tool_planning_llm", lambda prompt: planned)
    task = TaskState(
        goal="Write the file",
        user="alex",
        check_verdict="new",
        acceptance_criteria_md="1.- [ ] File exists",
    )
    processor = PDCAIntelligenceProcessor()

    result = processor.process(
        task,
        CoreLoopContext(messages=InMemoryMessageQueue(), tools=_ToolRegistry()),
    )

    assert result.snapshot.metadata["planned_tool_call"] == planned
    assert result.snapshot.metadata["plan_md"].startswith("{")
    assert result.snapshot.metadata["task_state"]["metadata"]["planned_tool_call"] == planned


class _RecordingState:
    def __init__(self) -> None:
        self.value = StateSnapshot()

    def update(self, snapshot: StateSnapshot) -> None:
        self.value = snapshot

    def snapshot(self) -> StateSnapshot:
        return self.value


class _NullTools:
    def list(self) -> tuple[object, ...]:
        return ()


class _ToolRegistry:
    def list(self) -> tuple[ToolDescriptor, ...]:
        return (
            ToolDescriptor(
                tool_id="tool-1",
                name="write_file",
                kind=ToolKind.NATIVE,
                description="Writes a file",
            ),
        )


class _NullPrompts:
    pass


class _NullMemory:
    pass
