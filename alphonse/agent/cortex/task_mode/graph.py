from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph

from alphonse.agent.cortex.task_mode.pdca import act_node
from alphonse.agent.cortex.task_mode.pdca import build_next_step_node
from alphonse.agent.cortex.task_mode.pdca import execute_step_node
from alphonse.agent.cortex.task_mode.pdca import route_after_act
from alphonse.agent.cortex.task_mode.pdca import route_after_next_step
from alphonse.agent.cortex.task_mode.pdca import route_after_progress_critic
from alphonse.agent.cortex.task_mode.pdca import progress_critic_node
from alphonse.agent.cortex.task_mode.pdca import update_state_node


def wire_task_mode_pdca(graph: StateGraph, *, tool_registry: Any) -> None:
    """PDCA loop: Plan-Do-Check-Act; one executable step per iteration."""
    graph.add_node(
        "next_step_node",
        build_next_step_node(tool_registry=tool_registry),
    )
    graph.add_node(
        "execute_step_node",
        lambda state: execute_step_node(state, tool_registry=tool_registry),
    )
    graph.add_node("update_state_node", update_state_node)
    graph.add_node("progress_critic_node", progress_critic_node)
    graph.add_node("act_node", act_node)

    graph.add_edge("task_mode_entry_node", "next_step_node")
    graph.add_conditional_edges(
        "next_step_node",
        route_after_next_step,
        {
            "execute_step_node": "execute_step_node",
            "respond_node": "respond_node",
        },
    )
    graph.add_edge("execute_step_node", "update_state_node")
    graph.add_edge("update_state_node", "progress_critic_node")
    graph.add_conditional_edges(
        "progress_critic_node",
        route_after_progress_critic,
        {
            "act_node": "act_node",
            "respond_node": "respond_node",
        },
    )
    graph.add_conditional_edges(
        "act_node",
        route_after_act,
        {
            "next_step_node": "next_step_node",
            "respond_node": "respond_node",
        },
    )
