from __future__ import annotations

import json
import os
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from alphonse.agent.cognition.providers.contracts import TextCompletionProvider
from alphonse.agent.cognition.providers.contracts import require_text_completion_provider
from alphonse.agent.cognition.plans import CortexPlan, CortexResult
from alphonse.agent.cortex.nodes.respond import respond_finalize_node
from alphonse.agent.cortex.task_mode.pdca import act_node
from alphonse.agent.cortex.task_mode.pdca import build_next_step_node
from alphonse.agent.cortex.task_mode.pdca import check_node
from alphonse.agent.cortex.task_mode.pdca import execute_step_node
from alphonse.agent.cortex.task_mode.pdca import route_after_act
from alphonse.agent.cortex.task_mode.pdca import route_after_next_step
from alphonse.agent.cortex.task_mode.plan import PlannerOutput
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.cortex.transitions import emit_transition_event
from alphonse.agent.cortex.utils import build_cognition_state, build_meta
from alphonse.agent.nervous_system.pdca_queue_store import append_pdca_event
from alphonse.agent.session.day_state import render_recent_conversation_block
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
        graph.add_node(
            "respond_node",
            lambda state: respond_finalize_node(
                state,
                emit_transition_event=emit_transition_event,
            ),
        )

        graph.set_entry_point("task_record_entry_node")
        graph.add_edge("task_record_entry_node", "check_node")
        graph.add_edge("check_node", "act_node")
        graph.add_conditional_edges(
            "act_node",
            _route_after_act_state,
            {"next_step_node": "next_step_node", "respond_node": "respond_node"},
        )
        graph.add_conditional_edges(
            "next_step_node",
            _route_after_next_step_state,
            {"execute_step_node": "execute_step_node"},
        )
        graph.add_edge("execute_step_node", "check_node")
        graph.add_edge("respond_node", END)
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
    provenance = "slice_resume" if bool(task_record.task_id) else "entry"
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
    verdict = str(check_result.get("verdict") or "").strip().lower()
    return {
        "task_record": task_record,
        "act_result": act_node(verdict, task_record),
    }


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
    task_record = state.get("task_record")
    if isinstance(task_record, TaskRecord) and task_record.task_id:
        return "slice_resume"
    return "entry"


def _hydrate_task_record_from_state(state: dict[str, Any]) -> TaskRecord:
    existing = state.get("task_record")
    record = existing if isinstance(existing, TaskRecord) else TaskRecord()
    record.task_id = _first_non_empty(record.task_id, state.get("task_id"))
    record.user_id = _first_non_empty(record.user_id, state.get("actor_person_id"))
    record.set_correlation_id(_first_non_empty(record.correlation_id, state.get("correlation_id")) or "")
    if not str(record.goal or "").strip():
        record.goal = _resolve_goal_text(state)
    if not str(record.status or "").strip():
        record.status = "running"
    record.set_recent_conversation_md(_resolve_recent_conversation_md(state=state, task_record=record))
    _append_runtime_facts(record, state)
    return record


def _resolve_recent_conversation_md(*, state: dict[str, Any], task_record: TaskRecord) -> str:
    existing = str(task_record.recent_conversation_md or "").strip()
    if existing and existing != "- (none)":
        return existing
    recent = str(state.get("recent_conversation_block") or "").strip()
    if recent:
        return recent
    session_state = state.get("session_state") if isinstance(state.get("session_state"), dict) else None
    if session_state:
        rendered = render_recent_conversation_block(session_state)
        if rendered:
            return rendered
    last_user_message = str(state.get("last_user_message") or "").strip()
    if last_user_message:
        return f"- User: {last_user_message}"
    return "- (none)"


def _resolve_goal_text(state: dict[str, Any]) -> str:
    incoming = state.get("incoming_raw_message")
    if isinstance(incoming, dict):
        extracted = _extract_goal_from_payload(incoming)
        if extracted:
            return extracted
    return _extract_goal_from_packed_message(str(state.get("last_user_message") or ""))


def _extract_goal_from_payload(payload: dict[str, Any]) -> str:
    direct = str(payload.get("text") or "").strip()
    if direct:
        return direct
    message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
    nested = str(message.get("text") or "").strip()
    if nested:
        return nested
    provider_event = payload.get("provider_event") if isinstance(payload.get("provider_event"), dict) else {}
    provider_message = provider_event.get("message") if isinstance(provider_event.get("message"), dict) else {}
    provider_text = str(provider_message.get("text") or "").strip()
    if provider_text:
        return provider_text
    return ""


def _extract_goal_from_packed_message(last_user_message: str) -> str:
    rendered = str(last_user_message or "").strip()
    if not rendered:
        return ""
    marker = "```json"
    if marker in rendered:
        after = rendered.split(marker, 1)[1]
        json_payload = after.split("```", 1)[0].strip()
        try:
            parsed = json.loads(json_payload)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            extracted = _extract_goal_from_payload(parsed)
            if extracted:
                return extracted
    for line in rendered.splitlines():
        if line.lower().startswith("- text:"):
            candidate = line.split(":", 1)[1].strip()
            if candidate:
                return candidate
    return rendered[:240]


def _append_runtime_facts(record: TaskRecord, state: dict[str, Any]) -> None:
    for key in (
        "channel_type",
        "channel_target",
        "locale",
        "timezone",
        "message_id",
        "conversation_key",
        "actor_person_id",
    ):
        value = _first_non_empty(state.get(key))
        if value and f"- {key}: " not in record.get_facts_md():
            record.append_fact(f"{key}: {value}")


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


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        rendered = str(value or "").strip()
        if rendered:
            return rendered
    return None


def _resolve_recursion_limit() -> int:
    raw = str(os.getenv("ALPHONSE_GRAPH_RECURSION_LIMIT") or "").strip()
    if not raw:
        return 1000
    try:
        parsed = int(raw)
    except ValueError:
        return 1000
    return max(100, min(parsed, 1000))
