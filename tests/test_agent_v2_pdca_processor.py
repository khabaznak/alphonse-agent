from __future__ import annotations

from datetime import datetime

from alphonse.agent_v2.core.core import AlphonseCore
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.core import LoopStepStatus
from alphonse.agent_v2.core.core import ProcessingStatus
from alphonse.agent_v2.core.core import StateSnapshot
from alphonse.agent_v2.core.intelligence.pdca import PDCAIntelligenceProcessor
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.state import reset_state


def test_pdca_processor_runs_graph_and_returns_completed_result() -> None:
    task = TaskState(goal="Write the file", user="alex")
    processor = PDCAIntelligenceProcessor()

    result = processor.process(task, CoreLoopContext(messages=InMemoryMessageQueue()))

    assert result.status == ProcessingStatus.COMPLETED
    assert result.snapshot.current_work == "Write the file"
    assert result.snapshot.task_owner == "alex"
    assert result.snapshot.metadata["check_verdict"] == "new"
    assert result.snapshot.metadata["task_state"]["check_verdict"] == "new"


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


class _RecordingState:
    def __init__(self) -> None:
        self.value = StateSnapshot()

    def update(self, snapshot: StateSnapshot) -> None:
        self.value = snapshot

    def snapshot(self) -> StateSnapshot:
        return self.value


class _NullTools:
    pass


class _NullPrompts:
    pass


class _NullMemory:
    pass
