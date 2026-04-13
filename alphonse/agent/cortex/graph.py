from __future__ import annotations

import os
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from alphonse.agent.cognition.providers.contracts import TextCompletionProvider
from alphonse.agent.cognition.providers.contracts import require_text_completion_provider
from alphonse.agent.cognition.plans import CortexPlan, CortexResult
from alphonse.agent.cortex.task_mode.pdca import act_node
from alphonse.agent.cortex.task_mode.pdca import build_next_step_node
from alphonse.agent.cortex.task_mode.pdca import check_node
from alphonse.agent.cortex.task_mode.pdca import execute_step_node
from alphonse.agent.cortex.task_mode.pdca import route_after_act
from alphonse.agent.cortex.task_mode.pdca import route_after_next_step
from alphonse.agent.cortex.task_mode.plan import PlannerOutput
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.cortex.utils import build_cognition_state, build_meta
from alphonse.agent.nervous_system.pdca_queue_store import append_pdca_event
from alphonse.agent.tools.registry import ToolRegistry, build_default_tool_registry


class CortexState(TypedDict, total=False):
    chat_id: str
    channel_type: str
    channel_target: str
    conversation_key: str
    actor_person_id: str | None
    incoming_raw_message: dict[str, Any] | None
    last_user_message: str
    response_text: str | None
    timezone: str
    correlation_id: str | None
    locale: str | None
    events: list[dict[str, Any]]
    pending_interaction: dict[str, Any] | None
    task_record: TaskRecord | None
    check_result: dict[str, Any] | None
    planner_output: PlannerOutput | None
    act_result: dict[str, Any] | None
    check_provenance: str | None
    recent_conversation_block: str | None
    _llm_client: Any
    _transition_sink: Any
    _bus: Any


class CortexGraph:
    def __init__(self, *, tool_registry: ToolRegistry | None = None) -> None:
        self._tool_registry = tool_registry or build_default_tool_registry()

    def build(self) -> StateGraph:
        graph = StateGraph(CortexState)
        graph.add_node("task_record_entry_node", task_record_entry_node)
        graph.add_node("check_node", check_node_state_adapter)
        graph.add_node("act_node", act_node_state_adapter)
        graph.add_node("next_step_node", next_step_state_adapter(tool_registry=self._tool_registry))
        graph.add_node("execute_step_node", execute_step_state_adapter)

        graph.set_entry_point("task_record_entry_node")
        graph.add_edge("task_record_entry_node", "check_node")
        graph.add_edge("check_node", "act_node")
        graph.add_conditional_edges(
            "act_node",
            _route_after_act_state,
            {"next_step_node": "next_step_node", "end": END},
        )
        graph.add_conditional_edges(
            "next_step_node",
            _route_after_next_step_state,
            {"execute_step_node": "execute_step_node"},
        )
        graph.add_edge("execute_step_node", "check_node")
        return graph

    def invoke(
        self,
        state: dict[str, Any],
        text: str,
        *,
        llm_client: TextCompletionProvider | None = None,
    ) -> CortexResult:
        validated_llm_client: TextCompletionProvider | None = None
        if llm_client is not None:
            validated_llm_client = require_text_completion_provider(
                llm_client,
                source="cortex.graph.invoke",
            )
        runner = self.build().compile()
        recursion_limit = _resolve_recursion_limit()
        result_state = runner.invoke(
            {**state, "last_user_message": text, "_llm_client": validated_llm_client},
            config={"recursion_limit": recursion_limit},
        )
        plans = [
            CortexPlan.model_validate(plan) for plan in result_state.get("plans") or []
        ]
        return CortexResult(
            reply_text=result_state.get("response_text"),
            plans=plans,
            cognition_state=build_cognition_state(result_state),
            meta=build_meta(result_state),
        )


def task_record_entry_node(state: dict[str, Any]) -> dict[str, Any]:
    task_record = _hydrate_task_record_from_state(state)
    provenance = _select_check_provenance(state)
    return {
        "task_record": task_record,
        "check_provenance": provenance,
        "recent_conversation_block": task_record.recent_conversation_md,
    }


def check_node_state_adapter(state: dict[str, Any]) -> dict[str, Any]:
    task_record = _hydrate_task_record_from_state(state)
    provenance = _select_check_provenance(state)
    result = check_node(task_record, provenance=provenance)
    updated_task_record = result["task_record"]
    consumed_inputs = result.get("consumed_inputs") if isinstance(result.get("consumed_inputs"), list) else []
    if consumed_inputs:
        latest_text = str(consumed_inputs[-1].get("text") or "").strip()
        if latest_text:
            state["last_user_message"] = latest_text
    _append_check_criteria_snapshot_event(updated_task_record, result.get("judge_result"))
    return {
        "task_record": updated_task_record,
        "check_result": result,
        "check_provenance": "slice_resume" if consumed_inputs else provenance,
        "recent_conversation_block": updated_task_record.recent_conversation_md,
    }


def act_node_state_adapter(state: dict[str, Any]) -> dict[str, Any]:
    task_record = state.get("task_record")
    check_result = state.get("check_result")
    if not isinstance(task_record, TaskRecord):
        raise ValueError("act_node_state_adapter.missing_task_record")
    if not isinstance(check_result, dict):
        raise ValueError("act_node_state_adapter.missing_check_result")
    _ = check_result
    result = act_node(task_record)
    out = {
        "task_record": task_record,
        "act_result": result,
    }
    response_text = str(result.get("response_text") or "").strip()
    if response_text:
        out["response_text"] = response_text
    return out


def next_step_state_adapter(*, tool_registry: ToolRegistry | None = None):
    semantic_node = build_next_step_node(tool_registry=tool_registry or build_default_tool_registry())

    def _adapter(state: dict[str, Any]) -> dict[str, Any]:
        task_record = state.get("task_record")
        if not isinstance(task_record, TaskRecord):
            raise ValueError("next_step_state_adapter.missing_task_record")
        planner_output = semantic_node(task_record)
        return {"task_record": task_record, "planner_output": planner_output}

    return _adapter


def execute_step_state_adapter(state: dict[str, Any]) -> dict[str, Any]:
    task_record = state.get("task_record")
    planner_output = state.get("planner_output")
    if not isinstance(task_record, TaskRecord):
        raise ValueError("execute_step_state_adapter.missing_task_record")
    if not isinstance(planner_output, dict):
        raise ValueError("execute_step_state_adapter.missing_planner_output")
    result = execute_step_node(task_record, planner_output)
    updated_task_record = result["task_record"]
    return {
        "task_record": updated_task_record,
        "check_provenance": result["provenance"],
        "recent_conversation_block": updated_task_record.recent_conversation_md,
    }


def _route_after_act_state(state: dict[str, Any]) -> str:
    return route_after_act(state.get("act_result"))


def _route_after_next_step_state(state: dict[str, Any]) -> str:
    return route_after_next_step(state.get("planner_output"))


def _select_check_provenance(state: dict[str, Any]) -> str:
    provenance = str(state.get("check_provenance") or "").strip().lower()
    if provenance in {"entry", "do", "slice_resume"}:
        return provenance
    return "entry"


def _hydrate_task_record_from_state(state: dict[str, Any]) -> TaskRecord:
    existing = state.get("task_record")
    if isinstance(existing, TaskRecord):
        return existing
    elif isinstance(existing, dict):
        return TaskRecord.from_dict(existing)
    raise ValueError("pdca_graph.missing_task_record")


def _append_check_criteria_snapshot_event(task_record: TaskRecord, judge_result: Any) -> None:
    if not task_record.task_id or not isinstance(judge_result, dict):
        return
    criteria = []
    for index, line in enumerate(task_record.get_acceptance_criteria_md().splitlines(), start=1):
        rendered = line.strip().removeprefix("- ").strip()
        if not rendered or rendered == "(none)":
            continue
        status = "satisfied" if rendered.startswith("[x] ") else "pending"
        text = rendered[4:].strip() if rendered.startswith(("[x] ", "[ ] ")) else rendered
        criteria.append({"id": f"ac_{index}", "text": text[:180], "status": status})
    payload = {
        "case_type": str(judge_result.get("case_type") or "").strip()[:40],
        "acceptance_criteria": criteria[:12],
        "verdict": {
            "kind": str(judge_result.get("kind") or "").strip()[:32],
            "confidence": float(judge_result.get("confidence") or 0.0),
        },
    }
    reason = str(judge_result.get("reason") or "").strip()
    if reason:
        payload["verdict"]["reason"] = reason[:220]
    try:
        append_pdca_event(
            task_id=task_record.task_id,
            event_type="check.criteria_snapshot",
            payload=payload,
            correlation_id=task_record.correlation_id or None,
        )
    except Exception:
        return



def _resolve_recursion_limit() -> int:
    raw = str(os.getenv("ALPHONSE_GRAPH_RECURSION_LIMIT") or "").strip()
    if not raw:
        return 1000
    try:
        parsed = int(raw)
    except ValueError:
        return 1000
    return max(100, min(parsed, 1000))
