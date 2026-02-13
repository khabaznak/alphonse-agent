from __future__ import annotations

from functools import partial
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, StateGraph

from alphonse.agent.cognition.plans import CortexPlan, CortexResult
from alphonse.agent.cortex.nodes import build_apology_node
from alphonse.agent.cortex.nodes import build_ask_question_node
from alphonse.agent.cortex.nodes import build_critic_node
from alphonse.agent.cortex.nodes import build_planning_cycle_runner
from alphonse.agent.cortex.nodes import compose_response_from_state
from alphonse.agent.cortex.nodes import build_execute_stage_node
from alphonse.agent.cortex.nodes import ingest_node
from alphonse.agent.cortex.nodes import plan_node_stateful
from alphonse.agent.cortex.nodes import respond_finalize_node
from alphonse.agent.cortex.nodes import route_after_plan
from alphonse.agent.cortex.nodes import route_after_critic
from alphonse.agent.cortex.nodes import route_after_step_selection
from alphonse.agent.cortex.nodes import select_next_step_node_stateful
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
    slots: dict[str, Any]
    messages: list[dict[str, str]]
    response_text: str | None
    response_key: str | None
    response_vars: dict[str, Any] | None
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
    selected_step_index: int | None
    route_decision: str | None
    _llm_client: Any


class CortexGraph:
    def __init__(self, *, tool_registry: ToolRegistry | None = None) -> None:
        self._tool_registry = tool_registry or build_default_tool_registry()

    def build(self) -> StateGraph:
        graph = StateGraph(CortexState)
        planning_cycle_runner = build_planning_cycle_runner(
            tool_registry=self._tool_registry,
        )
        graph.add_node("ingest_node", ingest_node)
        graph.add_node(
            "plan_node",
            partial(
                plan_node_stateful,
                llm_client_from_state=self._llm_client_from_state,
                run_planning_cycle_with_llm=planning_cycle_runner,
            ),
        )
        graph.add_node("select_next_step_node", select_next_step_node_stateful)
        graph.add_node("critic_node", build_critic_node())
        graph.add_node(
            "execute_tool_node",
            build_execute_stage_node(
                tool_registry=self._tool_registry,
                llm_client_from_state=self._llm_client_from_state,
            ),
        )
        graph.add_node(
            "ask_question_node",
            build_ask_question_node(),
        )
        graph.add_node(
            "apology_node",
            build_apology_node(
                llm_client_from_state=self._llm_client_from_state,
            ),
        )
        graph.add_node(
            "respond_node",
            partial(
                respond_finalize_node,
                emit_transition_event=emit_transition_event,
            ),
        )

        graph.set_entry_point("ingest_node")
        graph.add_edge("ingest_node", "plan_node")
        graph.add_conditional_edges(
            "plan_node",
            route_after_plan,
            {
                "select_next_step_node": "select_next_step_node",
                "apology_node": "apology_node",
                "respond_node": "respond_node",
            },
        )
        graph.add_conditional_edges(
            "select_next_step_node",
            route_after_step_selection,
            {
                "critic_node": "critic_node",
                "ask_question_node": "ask_question_node",
                "apology_node": "apology_node",
                "respond_node": "respond_node",
            },
        )
        graph.add_conditional_edges(
            "critic_node",
            route_after_critic,
            {
                "execute_tool_node": "execute_tool_node",
                "select_next_step_node": "select_next_step_node",
                "apology_node": "apology_node",
                "respond_node": "respond_node",
            },
        )
        graph.add_edge("execute_tool_node", "select_next_step_node")
        graph.add_edge("ask_question_node", "respond_node")
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
        result_state = runner.invoke(
            {**state, "last_user_message": text, "_llm_client": llm_client}
        )
        plans = [
            CortexPlan.model_validate(plan) for plan in result_state.get("plans") or []
        ]
        response_text = result_state.get("response_text")
        if not response_text and result_state.get("response_key"):
            response_text = compose_response_from_state(result_state)
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
