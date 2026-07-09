"""LangGraph scaffold for the v2 PDCA intelligence cycle."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END
from langgraph.graph import StateGraph

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.pdca.nodes.act_node import act_node
from alphonse.agent_v2.core.intelligence.pdca.nodes.check_node import check_node
from alphonse.agent_v2.core.intelligence.pdca.nodes.do_node import do_node
from alphonse.agent_v2.core.intelligence.pdca.nodes.plan_node import plan_node
from alphonse.agent_v2.core.intelligence.task_state import TaskState


CHECK_NODE = "check"
PLAN_NODE = "plan"
DO_NODE = "do"
ACT_NODE = "act"


def build_pdca_graph(context: CoreLoopContext | None = None) -> Any:
    """Build the first LangGraph scaffold for the PDCA cycle.

    The cycle enters at Check because v2 first reviews the incoming TaskState
    before planning work.
    """
    graph = StateGraph(TaskState)
    graph.add_node(CHECK_NODE, lambda task: check_node(task, context=context))
    graph.add_node(PLAN_NODE, lambda task: plan_node(task, context=context))
    graph.add_node(DO_NODE, lambda task: do_node(task, context=context))
    graph.add_node(ACT_NODE, lambda task: act_node(task, context=context))

    graph.set_entry_point(CHECK_NODE)
    graph.add_edge(CHECK_NODE, ACT_NODE)
    graph.add_conditional_edges(
        ACT_NODE,
        _route_after_act,
        {
            PLAN_NODE: PLAN_NODE,
            END: END,
        },
    )
    graph.add_edge(PLAN_NODE, DO_NODE)
    graph.add_conditional_edges(
        DO_NODE,
        _route_after_do,
        {
            CHECK_NODE: CHECK_NODE,
            END: END,
        },
    )
    return graph.compile()


def run_pdca_once(task: TaskState, context: CoreLoopContext | None = None) -> TaskState:
    """Run one stubbed PDCA pass and return a TaskState container."""
    result = build_pdca_graph(context=context).invoke(task, config={"recursion_limit": 64})
    return TaskState.from_dict(dict(result))


def _route_after_act(task: TaskState) -> str:
    if task.metadata.get("act_route") == PLAN_NODE:
        return PLAN_NODE
    return END


def _route_after_do(task: TaskState) -> str:
    if str(task.status or "").strip().lower() == "waiting_user" or task.metadata.get("task_parked") is True:
        return END
    return CHECK_NODE
