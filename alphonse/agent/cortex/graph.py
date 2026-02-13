from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import partial
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, StateGraph

from alphonse.agent.cognition.planning_catalog import (
    discover_plan,
    format_available_ability_catalog,
    format_available_abilities,
)
from alphonse.agent.cognition.intent_types import IntentCategory
from alphonse.agent.cognition.pending_interaction import (
    PendingInteractionType,
    build_pending_interaction,
    serialize_pending_interaction,
)
from alphonse.agent.cognition.plans import CortexPlan, CortexResult
from alphonse.agent.cognition.preferences.store import get_or_create_principal_for_channel
from alphonse.agent.cognition.prompt_templates_runtime import (
    CAPABILITY_GAP_APOLOGY_SYSTEM_PROMPT,
    CAPABILITY_GAP_APOLOGY_USER_TEMPLATE,
    GRAPH_PLAN_CRITIC_SYSTEM_PROMPT,
    GRAPH_PLAN_CRITIC_USER_TEMPLATE,
    render_prompt_template,
)
from alphonse.agent.cognition.step_validation import (
    StepValidationResult,
    is_internal_tool_question,
    validate_step,
)
from alphonse.agent.cortex.nodes import AbilityStateCycleDeps
from alphonse.agent.cortex.nodes import PlanningLoopDeps
from alphonse.agent.cortex.nodes import DispatchCycleDeps
from alphonse.agent.cortex.nodes import EmptyCycleResultDeps
from alphonse.agent.cortex.nodes import FreshPlanningDeps
from alphonse.agent.cortex.nodes import PendingCycleDeps
from alphonse.agent.cortex.nodes import PlanningCycleDeps
from alphonse.agent.cortex.nodes import apology_node_stateful
from alphonse.agent.cortex.nodes import ask_question_node_stateful
from alphonse.agent.cortex.nodes import bind_answer_to_steps
from alphonse.agent.cortex.nodes import build_planning_loop_state
from alphonse.agent.cortex.nodes import build_gap_plan
from alphonse.agent.cortex.nodes import compose_response_from_state
from alphonse.agent.cortex.nodes import dispatch_cycle_result
from alphonse.agent.cortex.nodes import execute_tool_node
from alphonse.agent.cortex.nodes import execute_tool_node_stateful
from alphonse.agent.cortex.nodes import execution_helpers
from alphonse.agent.cortex.nodes import handle_ability_state_for_cycle
from alphonse.agent.cortex.nodes import handle_pending_interaction_for_cycle
from alphonse.agent.cortex.nodes import ingest_node
from alphonse.agent.cortex.nodes import is_effectively_empty_cycle_result
from alphonse.agent.cortex.nodes import plan_node_stateful
from alphonse.agent.cortex.nodes import respond_finalize_node
from alphonse.agent.cortex.nodes import run_capability_gap_tool
from alphonse.agent.cortex.nodes import run_planning_loop_step
from alphonse.agent.cortex.nodes import run_fresh_planning_pass
from alphonse.agent.cortex.nodes import run_planning_cycle
from alphonse.agent.cortex.nodes import select_next_step_node_stateful
from alphonse.agent.cortex.nodes.plan import next_step_index
from alphonse.agent.cortex.providers import get_ability_registry
from alphonse.agent.cortex.transitions import emit_transition_event
from alphonse.agent.cortex.utils import build_cognition_state, build_meta, safe_json
from alphonse.agent.tools.registry import ToolRegistry, build_default_tool_registry

logger = logging.getLogger(__name__)


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
        graph.add_node("ingest_node", ingest_node)
        graph.add_node(
            "plan_node",
            partial(
                plan_node_stateful,
                llm_client_from_state=self._llm_client_from_state,
                run_planning_cycle_with_llm=self._run_planning_cycle,
            ),
        )
        graph.add_node("select_next_step_node", select_next_step_node_stateful)
        graph.add_node(
            "execute_tool_node",
            partial(
                execute_tool_node_stateful,
                llm_client_from_state=self._llm_client_from_state,
                execute_tool_node_impl=execute_tool_node,
                run_planning_loop_step_with_llm=lambda s, loop_state, llm_client: run_planning_loop_step(
                    s,
                    loop_state,
                    llm_client,
                    deps=PlanningLoopDeps(
                        next_step_index=next_step_index,
                        safe_json=lambda value, limit: safe_json(value, limit=limit),
                        available_tool_catalog_data=self._available_tool_catalog_data,
                        validate_loop_step=self._validate_loop_step,
                        critic_repair_invalid_step=lambda *, state, step, llm_client, validation: execution_helpers.critic_repair_invalid_step(
                            state=state,
                            step=step,
                            llm_client=llm_client,
                            validation=validation,
                            render_prompt_template=render_prompt_template,
                            plan_critic_user_template=GRAPH_PLAN_CRITIC_USER_TEMPLATE,
                            plan_critic_system_prompt=GRAPH_PLAN_CRITIC_SYSTEM_PROMPT,
                            safe_json=lambda value, limit: safe_json(value, limit=limit),
                            format_available_abilities=format_available_abilities,
                            format_available_ability_catalog=format_available_ability_catalog,
                            ability_exists=lambda tool_name: get_ability_registry().get(tool_name)
                            is not None,
                            is_internal_tool_question=is_internal_tool_question,
                        ),
                        build_gap_plan=lambda *, state, reason, missing_slots: build_gap_plan(
                            state=state,
                            reason=reason,
                            missing_slots=missing_slots,
                            get_or_create_principal_for_channel=get_or_create_principal_for_channel,
                        ),
                        execute_tool_node=lambda s: execution_helpers.run_ask_question_step(
                            s,
                            (
                                s.get("ability_state", {}).get("steps", [])[
                                    int(s.get("selected_step_index") or 0)
                                ]
                                if isinstance(s.get("ability_state"), dict)
                                and isinstance(s.get("ability_state", {}).get("steps"), list)
                                and isinstance(s.get("selected_step_index"), int)
                                and 0
                                <= s["selected_step_index"]
                                < len(s.get("ability_state", {}).get("steps", []))
                                else {}
                            ),
                            s.get("ability_state")
                            if isinstance(s.get("ability_state"), dict)
                            else {},
                            s.get("selected_step_index")
                            if isinstance(s.get("selected_step_index"), int)
                            else None,
                            build_pending_interaction=build_pending_interaction,
                            pending_interaction_type_slot_fill=PendingInteractionType.SLOT_FILL,
                            serialize_pending_interaction=serialize_pending_interaction,
                            emit_transition_event=emit_transition_event,
                        ),
                        ability_registry_getter=get_ability_registry,
                        tool_registry=self._tool_registry,
                        emit_transition_event=emit_transition_event,
                        has_missing_params=execution_helpers.has_missing_params,
                        is_planning_loop_state=_is_planning_loop_state,
                        task_plane_category=IntentCategory.TASK_PLANE.value,
                    ),
                ),
            ),
        )
        graph.add_node(
            "ask_question_node",
            partial(
                ask_question_node_stateful,
                run_ask_question_step=lambda s, step, loop_state, step_index: execution_helpers.run_ask_question_step(
                    s,
                    step,
                    loop_state,
                    step_index,
                    build_pending_interaction=build_pending_interaction,
                    pending_interaction_type_slot_fill=PendingInteractionType.SLOT_FILL,
                    serialize_pending_interaction=serialize_pending_interaction,
                    emit_transition_event=emit_transition_event,
                ),
                next_step_index=next_step_index,
            ),
        )
        graph.add_node(
            "apology_node",
            partial(
                apology_node_stateful,
                build_capability_gap_apology=lambda **kwargs: execution_helpers.build_capability_gap_apology(
                    **kwargs,
                    render_prompt_template=render_prompt_template,
                    apology_user_template=CAPABILITY_GAP_APOLOGY_USER_TEMPLATE,
                    apology_system_prompt=CAPABILITY_GAP_APOLOGY_SYSTEM_PROMPT,
                    locale_for_state=lambda s: (
                        s.get("locale")
                        if isinstance(s.get("locale"), str) and s.get("locale")
                        else "en-US"
                    ),
                    logger_exception=lambda msg, chat_id, correlation_id, rsn: logger.exception(
                        msg,
                        chat_id,
                        correlation_id,
                        rsn,
                    ),
                ),
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
            self._route_after_plan,
            {
                "select_next_step_node": "select_next_step_node",
                "apology_node": "apology_node",
                "respond_node": "respond_node",
            },
        )
        graph.add_conditional_edges(
            "select_next_step_node",
            self._route_after_step_selection,
            {
                "execute_tool_node": "execute_tool_node",
                "ask_question_node": "ask_question_node",
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

    def _run_planning_cycle(
        self,
        state: CortexState,
        llm_client: LLMClient | None,
    ) -> dict[str, Any] | None:
        return run_planning_cycle(
            state,
            llm_client,
            deps=PlanningCycleDeps(
                require_last_user_message=_require_last_user_message,
                handle_pending_interaction_for_cycle=lambda **kwargs: self._handle_pending_interaction_for_cycle(
                    **kwargs
                ),
                handle_ability_state_for_cycle=lambda **kwargs: self._handle_ability_state_for_cycle(
                    **kwargs
                ),
                run_fresh_planning_pass=lambda **kwargs: self._run_fresh_planning_pass(
                    **kwargs
                ),
            ),
        )

    def _run_fresh_planning_pass(
        self,
        *,
        state: CortexState,
        llm_client: LLMClient | None,
        text: str,
    ) -> dict[str, Any] | None:
        return run_fresh_planning_pass(
            state=state,
            llm_client=llm_client,
            text=text,
            deps=FreshPlanningDeps(
                run_capability_gap_tool=run_capability_gap_tool,
                format_available_abilities=format_available_abilities,
                planning_context_for_cycle=lambda s, _text: (
                    s.get("planning_context")
                    if isinstance(s.get("planning_context"), dict)
                    else None
                ),
                discover_plan=discover_plan,
                locale_for_state=lambda s: (
                    s.get("locale")
                    if isinstance(s.get("locale"), str) and s.get("locale")
                    else "en-US"
                ),
                dispatch_cycle_result=lambda **kwargs: self._dispatch_cycle_result(
                    **kwargs
                ),
            ),
        )

    def _dispatch_cycle_result(
        self,
        *,
        state: CortexState,
        llm_client: LLMClient | None,
        source_text: str,
        discovery: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return dispatch_cycle_result(
            state=state,
            llm_client=llm_client,
            source_text=source_text,
            discovery=discovery,
            deps=DispatchCycleDeps(
                run_capability_gap_tool=run_capability_gap_tool,
                build_planning_loop_state=build_planning_loop_state,
            ),
        )

    def _handle_pending_interaction_for_cycle(
        self,
        *,
        state: CortexState,
        llm_client: LLMClient | None,
        text: str,
    ) -> dict[str, Any] | None:
        return handle_pending_interaction_for_cycle(
            state=state,
            llm_client=llm_client,
            text=text,
            deps=PendingCycleDeps(
                now_iso_utc=lambda: datetime.now(timezone.utc).isoformat(),
                is_effectively_empty_cycle_result=lambda result: is_effectively_empty_cycle_result(
                    result,
                    deps=EmptyCycleResultDeps(
                        is_planning_loop_state=_is_planning_loop_state,
                        next_step_index=next_step_index,
                    ),
                ),
                is_planning_loop_state=_is_planning_loop_state,
                next_step_index=next_step_index,
                bind_answer_to_steps=bind_answer_to_steps,
                ability_registry_getter=get_ability_registry,
                tool_registry=self._tool_registry,
                logger_info=lambda msg, chat_id, correlation_id: logger.info(
                    msg,
                    chat_id,
                    correlation_id,
                ),
            ),
        )

    def _handle_ability_state_for_cycle(
        self,
        *,
        state: CortexState,
        llm_client: LLMClient | None,
        text: str,
    ) -> dict[str, Any] | None:
        return handle_ability_state_for_cycle(
            state=state,
            llm_client=llm_client,
            text=text,
            deps=AbilityStateCycleDeps(
                is_planning_loop_state=_is_planning_loop_state,
                ability_registry_getter=get_ability_registry,
                tool_registry=self._tool_registry,
                logger_info=lambda msg, chat_id, correlation_id, prev_message, new_message: logger.info(
                    msg,
                    chat_id,
                    correlation_id,
                    prev_message,
                    new_message,
                ),
            ),
        )

    def _route_after_plan(self, state: CortexState) -> str:
        if _has_capability_gap_plan(state):
            return "apology_node"
        ability_state = state.get("ability_state")
        if isinstance(ability_state, dict) and _is_planning_loop_state(ability_state):
            steps = ability_state.get("steps")
            if isinstance(steps, list) and steps:
                return "select_next_step_node"
        return "respond_node"

    def _route_after_step_selection(self, state: CortexState) -> str:
        if _has_capability_gap_plan(state):
            return "apology_node"
        decision = str(state.get("route_decision") or "").strip()
        if decision in {"execute_tool_node", "ask_question_node", "respond_node"}:
            return decision
        if decision == "execute_tool":
            return "execute_tool_node"
        if decision == "ask_question":
            return "ask_question_node"
        return "respond_node"

    def _available_tool_catalog_data(self) -> dict[str, Any]:
        return execution_helpers.available_tool_catalog_data(
            format_available_ability_catalog=format_available_ability_catalog,
            list_registered_intents=get_ability_registry().list_intents,
        )

    def _validate_loop_step(
        self,
        step: dict[str, Any],
        catalog: dict[str, Any],
    ) -> StepValidationResult:
        return execution_helpers.validate_loop_step(
            step,
            catalog,
            validate_step=validate_step,
        )

    @staticmethod
    def _llm_client_from_state(state: dict[str, Any]) -> LLMClient | None:
        client = state.get("_llm_client")
        return client if client is not None else None


def _require_last_user_message(state: CortexState) -> str:
    raw = state.get("last_user_message")
    if not isinstance(raw, str):
        raise TypeError("last_user_message must be a string in CortexState")
    text = raw.strip()
    if not text:
        raise ValueError("last_user_message must be non-empty")
    return text


def _is_planning_loop_state(ability_state: dict[str, Any]) -> bool:
    return str(ability_state.get("kind") or "") == "discovery_loop"


def _has_capability_gap_plan(state: CortexState) -> bool:
    plans = state.get("plans")
    if not isinstance(plans, list):
        return False
    return any(
        isinstance(item, dict)
        and str(item.get("plan_type") or "") == "CAPABILITY_GAP"
        for item in plans
    )
