from __future__ import annotations

from typing import Any

from alphonse.agent_v2.core.intelligence.pdca import ACT_NODE
from alphonse.agent_v2.core.intelligence.pdca import CHECK_NODE
from alphonse.agent_v2.core.intelligence.pdca import DO_NODE
from alphonse.agent_v2.core.intelligence.pdca import PLAN_NODE
from alphonse.agent_v2.core.intelligence.pdca import build_pdca_graph
from alphonse.agent_v2.core.intelligence.pdca import run_pdca_once
from alphonse.agent_v2.core.intelligence.pdca.graph import _route_after_act
from alphonse.agent_v2.core.intelligence.pdca.nodes import act_node
from alphonse.agent_v2.core.intelligence.pdca.nodes import check_node
from alphonse.agent_v2.core.intelligence.pdca.nodes import do_node
from alphonse.agent_v2.core.intelligence.pdca.nodes import plan_node
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.core import ToolKind
from alphonse.agent_v2.core.messages import InMemoryMessageQueue


def test_pdca_do_node_stub_returns_task_state_unchanged() -> None:
    state = TaskState(goal="Review this request")

    assert do_node(state) is state


def test_pdca_plan_node_returns_task_state() -> None:
    state = TaskState(goal="Review this request", acceptance_criteria_md="1.- [ ] Done")

    assert plan_node(state) is state
    assert "tool_call_plan_prompt" in state.metadata


def test_pdca_act_node_returns_task_state() -> None:
    state = TaskState(goal="Review this request")

    assert act_node(state) is state


def test_pdca_check_node_sets_initial_verdict() -> None:
    state = TaskState(goal="Review this request")

    assert check_node(state) is state
    assert state.check_verdict == "new"


def test_pdca_graph_uses_check_as_entry_point() -> None:
    graph = build_pdca_graph().get_graph()

    start_edges = [edge for edge in graph.edges if edge.source == "__start__"]

    assert len(start_edges) == 1
    assert start_edges[0].target == CHECK_NODE


def test_pdca_graph_contains_all_cycle_nodes() -> None:
    graph = build_pdca_graph().get_graph()

    assert CHECK_NODE in graph.nodes
    assert PLAN_NODE in graph.nodes
    assert DO_NODE in graph.nodes
    assert ACT_NODE in graph.nodes


def test_pdca_graph_routes_check_directly_to_act() -> None:
    graph = build_pdca_graph().get_graph()

    assert any(edge.source == CHECK_NODE and edge.target == ACT_NODE for edge in graph.edges)
    assert not any(edge.source == CHECK_NODE and edge.target == PLAN_NODE for edge in graph.edges)


def test_pdca_graph_routes_after_act_to_plan_when_acceptance_criteria_exist() -> None:
    assert _route_after_act(TaskState(metadata={"act_route": "plan"})) == PLAN_NODE
    assert _route_after_act(TaskState(metadata={"act_route": "end"})) == "__end__"


def test_pdca_graph_routes_plan_to_do_and_do_to_check() -> None:
    graph = build_pdca_graph().get_graph()

    assert any(edge.source == PLAN_NODE and edge.target == DO_NODE for edge in graph.edges)
    assert any(edge.source == DO_NODE and edge.target == CHECK_NODE for edge in graph.edges)
    assert not any(edge.source == PLAN_NODE and edge.target == "__end__" for edge in graph.edges)


def test_run_pdca_once_preserves_task_state_container() -> None:
    state = TaskState(task_id="task-1", goal="Review this request", user="alex")

    result = run_pdca_once(state)

    assert isinstance(result, TaskState)
    assert result.task_id == "task-1"
    assert result.goal == "Review this request"
    assert result.user == "alex"


def test_run_pdca_once_passes_context_tools_to_plan() -> None:
    state = TaskState(task_id="task-1", goal="Review this request", user="alex", acceptance_criteria_md="1.- [ ] Done")

    result = run_pdca_once(
        state,
        context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=_ToolRegistry()),
    )

    assert "tool_call_plan_prompt" in result.metadata
    assert "write_file" in result.metadata["tool_call_plan_prompt"]
    assert result.metadata["act_route"] == "end"


def test_run_pdca_once_executes_planned_tool_and_returns_to_check(monkeypatch) -> None:
    from importlib import import_module

    plan_node_module = import_module("alphonse.agent_v2.core.intelligence.pdca.nodes.plan_node")
    monkeypatch.setattr(
        plan_node_module,
        "_call_tool_planning_llm",
        lambda prompt: {
            "id": "plan-call-1",
            "tool_id": "tool-1",
            "tool_name": "write_file",
            "arguments": {"path": "a.txt"},
            "internal_state": "Writing the requested file.",
        },
    )
    state = TaskState(task_id="task-1", goal="Review this request", user="alex", acceptance_criteria_md="1.- [ ] Done")
    tools = _ToolRegistry()

    result = run_pdca_once(
        state,
        context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=tools),
    )

    assert tools.calls == [("tool-1", {"path": "a.txt"})]
    assert result.check_verdict == "wip"
    assert result.pdca_cycle_count == 1
    assert result.metadata["act_route"] == "end"
    assert result.metadata["act_stop_reason"] == "temporary_cycle_limit"
    assert '"status": "success"' in result.plan_json


class _ToolRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list(self) -> tuple[ToolDescriptor, ...]:
        return (
            ToolDescriptor(
                tool_id="tool-1",
                name="write_file",
                kind=ToolKind.NATIVE,
                description="Writes a file",
            ),
        )

    def execute(self, tool_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool_id, dict(arguments)))
        return {"ok": True}
