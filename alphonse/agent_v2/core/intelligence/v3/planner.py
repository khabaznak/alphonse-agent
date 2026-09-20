"""Strategic phase planning for hierarchical CAPD."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest
from alphonse.agent_v2.core.intelligence.v3.contracts import PhasePlan
from alphonse.agent_v2.core.intelligence.v3.revealing import tool_capabilities

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.task_state import TaskState


def plan_phase(task: "TaskState", context: "CoreLoopContext") -> PhasePlan:
    if context.inference is None:
        raise RuntimeError("v3_phase_planning_inference_unavailable")
    tools = tuple(context.tools.list()) if context.tools is not None else ()
    catalog = sorted({capability for tool in tools for capability in tool_capabilities(tool)})
    prompt = (
        "Plan one bounded strategic execution phase. Return one JSON object matching the PhasePlan contract. "
        "Use meaningful subgoals, not one outer CAPD cycle per tool. Do not include a final user response subgoal. "
        "Authorize only capabilities and project-relative mutation paths needed in this phase. "
        "Never target .alphonse, memory ledgers, prompts, plans, acceptance criteria, or other agent-internal state. "
        "A mutation path must already be established by the user or prior verified evidence; otherwise plan a read-only discovery phase first. "
        "Use native.project_search and native.read_project_file for bounded file discovery; native.bash is unavailable in V3.\n\n"
        f"Goal: {task.goal}\n"
        f"Immutable acceptance contract: {json.dumps(task.ensure_acceptance_contract(), ensure_ascii=False)}\n"
        f"Prior V3 phase history: {json.dumps(task.metadata.get('v3_phase_history') or [], ensure_ascii=False)}\n"
        f"Available capability identifiers: {json.dumps(catalog)}\n"
        "Required top-level fields: phase_id, objective, subgoals, criterion_ids, limits, "
        "authorized_capabilities, mutation_scope, originating_decision, schema_version."
    )
    result = context.inference.generate_json(
        InferenceRequest(
            prompt=prompt,
            purpose=InferencePurpose.PHASE_PLANNING,
            project_id=task.project_id,
            user=task.user,
            task_id=task.task_id,
            tools=(),
            cancel_checker=context.is_cancelled if context.cancellation_checker is not None else None,
        )
    )
    if not isinstance(result.json_value, dict):
        raise ValueError("v3_phase_plan_missing")
    return PhasePlan.from_dict(result.json_value)
