"""LangGraph scaffold for the v2 PDCA intelligence cycle."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END
from langgraph.graph import StateGraph

from alphonse.agent_v2.core.intelligence.pdca.nodes.act_node import act_node
from alphonse.agent_v2.core.intelligence.pdca.nodes.check_node import check_node
from alphonse.agent_v2.core.intelligence.pdca.nodes.do_node import do_node
from alphonse.agent_v2.core.intelligence.pdca.nodes.plan_node import plan_node
from alphonse.agent_v2.core.intelligence.task_state import TaskState


CHECK_NODE = "check"
PLAN_NODE = "plan"
DO_NODE = "do"
ACT_NODE = "act"


def build_pdca_graph() -> Any:
    """Build the first LangGraph scaffold for the PDCA cycle.

    The cycle enters at Check because v2 first reviews the incoming TaskState
    before planning work.
    """
    graph = StateGraph(TaskState)
    graph.add_node(CHECK_NODE, check_node)
    graph.add_node(PLAN_NODE, plan_node)
    graph.add_node(DO_NODE, do_node)
    graph.add_node(ACT_NODE, act_node)

    graph.set_entry_point(CHECK_NODE)
    graph.add_edge(CHECK_NODE, PLAN_NODE)
    graph.add_edge(PLAN_NODE, DO_NODE)
    graph.add_edge(DO_NODE, ACT_NODE)
    graph.add_edge(ACT_NODE, END)
    return graph.compile()


def run_pdca_once(task: TaskState) -> TaskState:
    """Run one stubbed PDCA pass and return a TaskState container."""
    result = build_pdca_graph().invoke(task)
    return TaskState.from_dict(dict(result))
