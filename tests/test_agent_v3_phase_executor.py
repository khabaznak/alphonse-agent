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
from alphonse.agent_v2.system_one import SystemOneToolRegistrySelection
from alphonse.agent_v2.system_one import SystemOneTacticalReview


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


def _accept_successful_actions():
    class SystemOne:
        def evaluate_tactical_progress(self, **values):
            complete = (
                values["action"].get("status") == "success"
                and values["action"].get("result") not in (None, {})
            )
            return SystemOneTacticalReview(
                complete=complete,
                confidence=0.99 if complete else 0.01,
                confident=True,
            )
    return SystemOne()


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
        task, state, CoreLoopContext(messages=InMemoryMessageQueue(), tools=_registry(calls), system_one=_accept_successful_actions())
    )

    assert outcome.status == PhaseStatus.PHASE_COMPLETE
    assert calls == ["search", "edit", "read"]
    assert state.completed_subgoal_ids == ["locate", "update", "verify"]
    assert state.bindings["record_reference"]["value"]["path"] == "mejoras_hogar/backlog.md"
    assert len(state.evidence.entries) == 3
    assert task.hierarchical_state["status"] == "phase_complete"


def test_phase_executor_supplies_revealed_tools_to_tactical_inference() -> None:
    calls: list[str] = []
    task = TaskState(task_id="task", user="alex", project_id="home")
    state = new_tactical_state(_phase())
    selected_tool_sets = []

    class SystemOne:
        def select_plan_tools(self, **values):
            assert {item.tool_id for item in values["tools"]} == {
                "native.search", "native.exact_text_edit", "native.read",
            }
            return SystemOneToolRegistrySelection(
                selected_tool_ids=("native.search",),
                ambiguous_tool_ids=("native.read",),
                rejected_tool_ids=("native.exact_text_edit",),
                probabilities={"native.search": 0.95, "native.read": 0.63, "native.exact_text_edit": 0.1},
            )

        def evaluate_tactical_progress(self, **values):
            return SystemOneTacticalReview(complete=True, confidence=0.99, confident=True)

    def selector(current_state, subgoal, tools):
        selected_tool_sets.append([item.tool_id for item in tools])
        return {"tool_id": tools[0].tool_id, "arguments": {"query": "solar"}}

    executor = PhaseExecutor(action_selector=selector)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=_registry(calls), system_one=SystemOne())

    # Injected action selectors intentionally bypass model/System One selection;
    # verify the provider path separately through the real selector below.
    executor._action_selector = None
    class Inference:
        def generate_json(self, request):
            selected_tool_sets.append([item.tool_id for item in request.tools])
            return type("Result", (), {"json_value": {
                "tool_id": request.tools[0].tool_id,
                "arguments": {"query": "solar"},
                "acceptance_questions": [{
                    "question_id": "found_record",
                    "type": "noul",
                    "instructions": "Was a record found?",
                    "criteria": {"true": "A record is present.", "false": "No record is present."},
                }],
            }})()
    context.inference = Inference()
    state.phase = PhasePlan(
        "search", "Search", (PhaseSubgoal("locate", "Locate", "record", allowed_capabilities=("native.search", "native.read")),),
        authorized_capabilities=("native.search", "native.read"), limits=PhaseLimits(1, 20),
    )
    state.active_subgoal_id = "locate"
    state.remaining_tool_calls = 1
    state.revealed_tool_ids = []

    outcome = executor.run(task, state, context)

    assert outcome.status == PhaseStatus.PHASE_COMPLETE
    assert selected_tool_sets == [["native.search", "native.read"]]


def test_phase_executor_uses_system_one_to_check_semantic_subgoal_completion() -> None:
    calls: list[str] = []
    phase = PhasePlan(
        "search", "Find authoritative record",
        (PhaseSubgoal(
            "locate", "Locate exact solar record", "record",
            allowed_capabilities=("native.search",),
            limits=PhaseLimits(1, 20),
        ),),
        authorized_capabilities=("native.search",),
        limits=PhaseLimits(1, 20),
    )
    state = new_tactical_state(phase)
    state.revealed_tool_ids = ["native.search"]

    class SystemOne:
        def evaluate_tactical_progress(self, **values):
            assert values["action"]["status"] == "success"
            return SystemOneTacticalReview(complete=False, confidence=0.05, confident=True)

    executor = PhaseExecutor(
        action_selector=_selector([{"tool_id": "native.search", "arguments": {"query": "solar"}}])
    )
    outcome = executor.run(
        TaskState(task_id="task", user="alex", project_id="home"),
        state,
        CoreLoopContext(messages=InMemoryMessageQueue(), tools=_registry(calls), system_one=SystemOne()),
    )

    assert outcome.status == PhaseStatus.BUDGET_EXHAUSTED
    assert state.completed_subgoal_ids == []


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

    outcome = executor.run(TaskState(user="alex"), state, CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, system_one=_accept_successful_actions()))

    assert outcome.status == PhaseStatus.BLOCKED
    assert calls == ["broken"]
    assert [item.status for item in state.actions] == ["failed"]


def test_phase_executor_retries_same_read_only_action_when_jev_approves(monkeypatch) -> None:
    import alphonse.agent_v2.core.intelligence.v3.executor as executor_module

    monkeypatch.setattr(executor_module, "TACTICAL_RETRY_BASE_SECONDS", 0)
    calls: list[dict[str, Any]] = []
    registry = InMemoryToolRegistry()

    def flaky(arguments):
        calls.append(dict(arguments))
        if len(calls) == 1:
            return {
                "output": None,
                "exception": {
                    "code": "server_timeout",
                    "message": "temporary timeout",
                    "retryable": True,
                },
            }
        return {"path": "backlog.md"}

    registry.register(_tool("native.flaky_read", flaky, read_only=True))
    phase = PhasePlan(
        "retry", "Read record",
        (PhaseSubgoal("read", "Read record", "record", allowed_capabilities=("native.flaky_read",)),),
        authorized_capabilities=("native.flaky_read",), limits=PhaseLimits(3, 20),
    )
    state = new_tactical_state(phase)
    state.revealed_tool_ids = ["native.flaky_read"]

    class Jev:
        def evaluate_tactical_progress(self, **values):
            assert len(values["questions"]) == 1
            return SystemOneTacticalReview(
                complete=values["action"]["status"] == "success",
                confidence=0.99,
                confident=True,
                retry_approved=True,
            )

    outcome = PhaseExecutor(action_selector=_selector([
        {"tool_id": "native.flaky_read", "arguments": {"record_id": "r-1"}},
    ])).run(
        TaskState(user="alex"), state,
        CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, system_one=Jev()),
    )

    assert outcome.status == PhaseStatus.PHASE_COMPLETE
    assert calls == [{"record_id": "r-1"}, {"record_id": "r-1"}]
    assert [item.status for item in state.actions] == ["failed", "success"]
    assert [item["retry_attempt"] for item in state.evidence.entries] == [0, 1]


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
        TaskState(user="alex"), state, CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, system_one=_accept_successful_actions())
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
    ])).run(task, restored, CoreLoopContext(messages=InMemoryMessageQueue(), tools=_registry(calls), system_one=_accept_successful_actions()))

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
        TaskState(user="alex"), state, CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, system_one=_accept_successful_actions())
    )

    assert outcome.status == PhaseStatus.BUDGET_EXHAUSTED
    assert calls == ["empty"]
