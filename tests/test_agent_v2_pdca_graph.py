from __future__ import annotations

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
    assert _route_after_act(TaskState(acceptance_criteria_md="1.- [ ] Done")) == PLAN_NODE
    assert _route_after_act(TaskState(acceptance_criteria_md="- (none)")) == "__end__"


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
