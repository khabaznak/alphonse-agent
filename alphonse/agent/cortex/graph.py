from __future__ import annotations

import os
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, StateGraph

from alphonse.agent.cognition.plans import CortexPlan, CortexResult
from alphonse.agent.cortex.nodes import build_apology_node
from alphonse.agent.cortex.nodes import build_first_decision_node
from alphonse.agent.cortex.nodes import task_mode_entry_node
from alphonse.agent.cortex.nodes import respond_finalize_node
from alphonse.agent.cortex.nodes import route_after_first_decision
from alphonse.agent.cortex.task_mode.state import TaskState
from alphonse.agent.cortex.task_mode.graph import wire_task_mode_pdca
from alphonse.agent.cortex.transitions import emit_transition_event
from alphonse.agent.cortex.utils import build_cognition_state, build_meta
from alphonse.agent.tools.registry import ToolRegistry, build_default_tool_registry

class LLMClient(Protocol):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        ...


class CortexState(TypedDict, total=False):
    chat_id: str
    channel_type: str
    channel_target: str
    conversation_key: str
    actor_person_id: str | None
    incoming_raw_message: dict[str, Any] | None
    last_user_message: str
    intent: str | None
    intent_category: str | None
    intent_confidence: float
    tone: str | None
    address_style: str | None
    slots: dict[str, Any]
    messages: list[dict[str, str]]
    response_text: str | None
    timezone: str
    intent_evidence: dict[str, Any]
    correlation_id: str | None
    plans: list[dict[str, Any]]
    last_intent: str | None
    locale: str | None
    autonomy_level: float | None
    planning_mode: str | None
    events: list[dict[str, Any]]
    pending_interaction: dict[str, Any] | None
    ability_state: dict[str, Any] | None
    planning_context: dict[str, Any] | None
    task_state: TaskState | None
    plan_retry: bool
    plan_repair_attempts: int
    selected_step_index: int | None
    route_decision: str | None
    session_id: str | None
    session_state: dict[str, Any] | None
    recent_conversation_block: str | None
    _llm_client: Any
    _transition_sink: Any
    _bus: Any


class CortexGraph:
    def __init__(self, *, tool_registry: ToolRegistry | None = None) -> None:
        self._tool_registry = tool_registry or build_default_tool_registry()

    def build(self) -> StateGraph:
        graph = StateGraph(CortexState)
        graph.add_node(
            "first_decision_node",
            build_first_decision_node(
                llm_client_from_state=self._llm_client_from_state,
            ),
        )
        graph.add_node(
            "apology_node",
            build_apology_node(
                llm_client_from_state=self._llm_client_from_state,
            ),
        )
        graph.add_node(
            "respond_node",
            lambda state: respond_finalize_node(
                state,
                emit_transition_event=emit_transition_event,
            ),
        )
        graph.add_node(
            "task_mode_entry_node",
            task_mode_entry_node,
        )
        wire_task_mode_pdca(graph, tool_registry=self._tool_registry)

        graph.set_entry_point("first_decision_node")
        graph.add_conditional_edges(
            "first_decision_node",
            route_after_first_decision,
            {
                "task_mode_entry_node": "task_mode_entry_node",
                "respond_node": "respond_node",
            },
        )
        graph.add_edge("apology_node", "respond_node")
        graph.add_edge("respond_node", END)
        return graph

    def invoke(
        self,
        state: dict[str, Any],
        text: str,
        *,
        llm_client: LLMClient | None = None,
    ) -> CortexResult:
        runner = self.build().compile()
        recursion_limit = _resolve_recursion_limit()
        result_state = runner.invoke(
            {**state, "last_user_message": text, "_llm_client": llm_client},
            config={"recursion_limit": recursion_limit},
        )
        plans = [
            CortexPlan.model_validate(plan) for plan in result_state.get("plans") or []
        ]
        response_text = result_state.get("response_text")
        return CortexResult(
            reply_text=response_text,
            plans=plans,
            cognition_state=build_cognition_state(result_state),
            meta=build_meta(result_state),
        )

    @staticmethod
    def _llm_client_from_state(state: dict[str, Any]) -> LLMClient | None:
        client = state.get("_llm_client")
        return client if client is not None else None


def _resolve_recursion_limit() -> int:
    raw = str(os.getenv("ALPHONSE_GRAPH_RECURSION_LIMIT") or "").strip()
    if not raw:
        return 1000
    try:
        parsed = int(raw)
    except ValueError:
        return 1000
    return max(100, min(parsed, 1000))
