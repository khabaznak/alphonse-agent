from __future__ import annotations

from alphonse.agent_v2.core.core import LoopStepStatus
from alphonse.agent_v2.interfaces.tui import build_tui_runtime
from alphonse.agent_v2.interfaces.tui import submit_tui_input


def test_tui_runtime_factory_wires_core_services() -> None:
    runtime = build_tui_runtime(user="alex")

    assert runtime.user == "alex"
    assert runtime.queue.size() == 0
    assert runtime.core.messages is runtime.queue
    assert runtime.channel.messages is runtime.queue


def test_submitting_input_queues_steps_and_updates_visible_state() -> None:
    runtime = build_tui_runtime(user="alex")

    result = submit_tui_input(runtime, "hello")

    assert result.queued
    assert result.step_status == LoopStepStatus.PROCESSED
    assert runtime.queue.size() == 0
    assert runtime.visible_state.snapshot().current_work == "hello"
    assert result.response == "I received your message: hello"


def test_submit_stop_exits_without_queueing_message() -> None:
    runtime = build_tui_runtime(user="alex")

    result = submit_tui_input(runtime, "/stop")

    assert result.should_exit
    assert not result.queued
    assert runtime.queue.size() == 0
