from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import CompletionCondition
from alphonse.agent_v2.core.intelligence.v3 import FailurePolicy
from alphonse.agent_v2.core.intelligence.v3 import MutationScope
from alphonse.agent_v2.core.intelligence.v3 import PhaseExecutor
from alphonse.agent_v2.core.intelligence.v3 import PhaseLimits
from alphonse.agent_v2.core.intelligence.v3 import PhasePlan
from alphonse.agent_v2.core.intelligence.v3 import PhaseStatus
from alphonse.agent_v2.core.intelligence.v3 import PhaseSubgoal
from alphonse.agent_v2.core.intelligence.v3 import SideEffectClass
from alphonse.agent_v2.core.intelligence.v3 import TacticalState
from alphonse.agent_v2.core.intelligence.v3 import new_tactical_state
from alphonse.agent_v2.core.messages import CommunicationChannel, InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition


def _tool(tool_id: str, callback, *, read_only: bool) -> ToolDefinition:
    return ToolDefinition(
        descriptor=ToolDescriptor(
            tool_id=tool_id,
            name=tool_id.removeprefix("native."),
            kind=ToolKind.NATIVE,
            read_only=read_only,
        ),
        callable=callback,
    )


def _registry(calls: list[str]) -> InMemoryToolRegistry:
    registry = InMemoryToolRegistry()

    def search(arguments: dict[str, Any]) -> dict[str, Any]:
        calls.append("search")
        return {"path": "mejoras_hogar/backlog.md", "anchor": "Solar | Idea"}

    def edit(arguments: dict[str, Any]) -> dict[str, Any]:
        calls.append("edit")
        return {
            "affected_paths": [arguments["path"]],
            "verification": {"status": "verified"},
        }

    def read(arguments: dict[str, Any]) -> dict[str, Any]:
        calls.append("read")
        return {"path": arguments["path"], "status": "Complete"}

    registry.register(_tool("native.search", search, read_only=True))
    registry.register(_tool("native.exact_text_edit", edit, read_only=False))
    registry.register(_tool("native.read", read, read_only=True))
    return registry


def _phase() -> PhasePlan:
    return PhasePlan(
        phase_id="solar-phase",
        objective="Complete and verify solar project",
        authorized_capabilities=("native.search", "native.exact_text_edit", "native.read"),
        mutation_scope=MutationScope(("mejoras_hogar/backlog.md",)),
        limits=PhaseLimits(max_tool_calls=5, max_duration_seconds=30),
        subgoals=(
            PhaseSubgoal(
                "locate", "Locate record", "record_reference",
                allowed_capabilities=("native.search",),
                completion=CompletionCondition("output_present", output_type="record_reference"),
            ),
            PhaseSubgoal(
                "update", "Update record", "verified_mutation", depends_on=("locate",),
                allowed_capabilities=("native.exact_text_edit",),
                allowed_side_effects=(SideEffectClass.PROJECT_MUTATION,),
                completion=CompletionCondition("field_equals", field="verification.status", expected="verified"),
            ),
            PhaseSubgoal(
                "verify", "Verify record", "record_observation", depends_on=("update",),
                allowed_capabilities=("native.read",),
                completion=CompletionCondition("output_present", output_type="record_observation"),
            ),
        ),
    )


def _selector(actions: list[dict[str, Any]]):
    remaining = deque(actions)

    def select(state, subgoal, tools):
        _ = state, subgoal, tools
        return remaining.popleft()

    return select


def test_phase_executor_runs_search_edit_verify_in_one_phase() -> None:
    calls: list[str] = []
    task = TaskState(task_id="task", user="alex", project_id="home")
    state = new_tactical_state(_phase())
    state.revealed_tool_ids = ["native.search", "native.exact_text_edit", "native.read"]
    executor = PhaseExecutor(
        action_selector=_selector(
            [
                {"tool_id": "native.search", "arguments": {"query": "solar"}},
                {"tool_id": "native.exact_text_edit", "arguments": {"path": "mejoras_hogar/backlog.md"}},
                {"tool_id": "native.read", "arguments": {"path": "mejoras_hogar/backlog.md"}},
            ]
        )
    )

    outcome = executor.run(
        task, state, CoreLoopContext(messages=InMemoryMessageQueue(), tools=_registry(calls))
    )

    assert outcome.status == PhaseStatus.PHASE_COMPLETE
    assert calls == ["search", "edit", "read"]
    assert state.completed_subgoal_ids == ["locate", "update", "verify"]
    assert state.bindings["record_reference"]["value"]["path"] == "mejoras_hogar/backlog.md"
    assert len(state.evidence.entries) == 3
    assert task.hierarchical_state["status"] == "phase_complete"


def test_phase_executor_rejects_unrevealed_tool_before_execution() -> None:
    calls: list[str] = []
    state = new_tactical_state(_phase())
    state.revealed_tool_ids = ["native.search"]
    executor = PhaseExecutor(action_selector=_selector([{"tool_id": "native.read", "arguments": {"path": "x"}}]))

    outcome = executor.run(TaskState(user="alex", project_id="home"), state, CoreLoopContext(messages=InMemoryMessageQueue(), tools=_registry(calls)))

    assert outcome.status == PhaseStatus.BLOCKED
    assert "tool_not_revealed" in outcome.reason
    assert calls == []


def test_phase_executor_rejects_mutation_outside_phase_scope() -> None:
    calls: list[str] = []
    state = new_tactical_state(_phase())
    state.status = PhaseStatus.RUNNING
    state.active_subgoal_id = "update"
    state.completed_subgoal_ids = ["locate"]
    state.revealed_tool_ids = ["native.exact_text_edit"]
    executor = PhaseExecutor(action_selector=_selector([{"tool_id": "native.exact_text_edit", "arguments": {"path": "other.md"}}]))

    outcome = executor.run(TaskState(user="alex", project_id="home"), state, CoreLoopContext(messages=InMemoryMessageQueue(), tools=_registry(calls)))

    assert outcome.status == PhaseStatus.BLOCKED
    assert "mutation_path_not_authorized" in outcome.reason
    assert calls == []


def test_phase_executor_uses_bounded_local_fallback() -> None:
    calls: list[str] = []
    registry = InMemoryToolRegistry()

    def broken(arguments):
        calls.append("broken")
        raise RuntimeError("index unavailable")

    def fallback(arguments):
        calls.append("fallback")
        return {"path": "backlog.md"}

    registry.register(_tool("native.broken", broken, read_only=True))
    registry.register(_tool("native.fallback", fallback, read_only=True))
    phase = PhasePlan(
        "recover", "Locate record",
        (PhaseSubgoal(
            "locate", "Locate", "record_reference",
            allowed_capabilities=("native.broken", "native.fallback"),
            failure_policy=FailurePolicy.LOCAL_FALLBACK,
            limits=PhaseLimits(max_tool_calls=2, max_duration_seconds=10),
            completion=CompletionCondition("output_present", output_type="record_reference"),
        ),),
        authorized_capabilities=("native.broken", "native.fallback"),
        limits=PhaseLimits(max_tool_calls=2, max_duration_seconds=10),
    )
    state = new_tactical_state(phase)
    state.revealed_tool_ids = ["native.broken", "native.fallback"]
    executor = PhaseExecutor(action_selector=_selector([
        {"tool_id": "native.broken", "arguments": {}},
        {"tool_id": "native.fallback", "arguments": {}},
    ]))

    outcome = executor.run(TaskState(user="alex"), state, CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry))

    assert outcome.status == PhaseStatus.PHASE_COMPLETE
    assert calls == ["broken", "fallback"]
    assert [item.status for item in state.actions] == ["failed", "success"]


def test_phase_executor_stops_on_failure_and_preserves_evidence() -> None:
    registry = InMemoryToolRegistry()
    registry.register(_tool("native.fail", lambda arguments: (_ for _ in ()).throw(RuntimeError("boom")), read_only=True))
    phase = PhasePlan(
        "fail", "Fail safely",
        (PhaseSubgoal("one", "One", "result", allowed_capabilities=("native.fail",)),),
        authorized_capabilities=("native.fail",),
    )
    state = new_tactical_state(phase)
    state.revealed_tool_ids = ["native.fail"]

    outcome = PhaseExecutor(action_selector=_selector([{"tool_id": "native.fail", "arguments": {}}])).run(
        TaskState(user="alex"), state, CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry)
    )

    assert outcome.status == PhaseStatus.BLOCKED
    assert "boom" in outcome.reason
    assert state.evidence.entries[0]["status"] == "failed"


def test_phase_executor_checks_steering_before_each_action() -> None:
    queue = InMemoryMessageQueue()
    CommunicationChannel(queue).queue_message(
        prompt="Stop", user="alex", project_id="home", metadata={"routing_disposition": "steering"}
    )
    state = new_tactical_state(_phase())

    outcome = PhaseExecutor(action_selector=lambda *_: pytest.fail("selector should not run")).run(
        TaskState(user="alex", project_id="home"), state, CoreLoopContext(messages=queue, tools=_registry([]))
    )

    assert outcome.status == PhaseStatus.BLOCKED
    assert outcome.blockers == ("steering_pending",)
    assert state.steering_pending is True


def test_phase_executor_checks_cancellation_before_action() -> None:
    state = new_tactical_state(_phase())
    outcome = PhaseExecutor(action_selector=lambda *_: pytest.fail("selector should not run")).run(
        TaskState(user="alex"), state,
        CoreLoopContext(messages=InMemoryMessageQueue(), tools=_registry([]), cancellation_checker=lambda: True),
    )

    assert outcome.status == PhaseStatus.CANCELLED
    assert state.cancellation_pending is True


def test_phase_executor_resumes_checkpoint_without_repeating_completed_action() -> None:
    calls: list[str] = []
    task = TaskState(user="alex", project_id="home")
    state = new_tactical_state(_phase())
    state.status = PhaseStatus.RUNNING
    state.completed_subgoal_ids = ["locate"]
    state.active_subgoal_id = "update"
    state.bind_subgoal_output("locate", "record_reference", {"path": "mejoras_hogar/backlog.md"})
    state.revealed_tool_ids = ["native.exact_text_edit", "native.read"]
    restored = TacticalState.from_dict(state.to_dict())

    outcome = PhaseExecutor(action_selector=_selector([
        {"tool_id": "native.exact_text_edit", "arguments": {"path": "mejoras_hogar/backlog.md"}},
        {"tool_id": "native.read", "arguments": {"path": "mejoras_hogar/backlog.md"}},
    ])).run(task, restored, CoreLoopContext(messages=InMemoryMessageQueue(), tools=_registry(calls)))

    assert outcome.status == PhaseStatus.PHASE_COMPLETE
    assert calls == ["edit", "read"]


def test_phase_executor_enforces_subgoal_budget_when_completion_is_unmet() -> None:
    calls: list[str] = []
    registry = InMemoryToolRegistry()
    registry.register(_tool("native.empty", lambda arguments: calls.append("empty") or {}, read_only=True))
    phase = PhasePlan(
        "budget", "Find value",
        (PhaseSubgoal(
            "find", "Find", "value", allowed_capabilities=("native.empty",),
            limits=PhaseLimits(max_tool_calls=1, max_duration_seconds=10),
            completion=CompletionCondition("field_equals", field="found", expected=True),
        ),),
        authorized_capabilities=("native.empty",),
        limits=PhaseLimits(max_tool_calls=2, max_duration_seconds=10),
    )
    state = new_tactical_state(phase)
    state.revealed_tool_ids = ["native.empty"]

    outcome = PhaseExecutor(action_selector=_selector([{"tool_id": "native.empty", "arguments": {}}])).run(
        TaskState(user="alex"), state, CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry)
    )

    assert outcome.status == PhaseStatus.BUDGET_EXHAUSTED
    assert calls == ["empty"]
