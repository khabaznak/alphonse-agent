"""Deterministic offline adapters that execute the real V2 and V3 processors."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest, InferenceResult
from alphonse.agent_v2.core.inference import InferenceRouter, ModelProfile
from alphonse.agent_v2.core.intelligence import PDCAIntelligenceProcessor
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import HierarchicalCAPDProcessor
from alphonse.agent_v2.core.intelligence.v3.contracts import PhaseStatus, TacticalAction, new_tactical_state
from alphonse.agent_v2.core.intelligence.v3.contracts import PhasePlan
from alphonse.agent_v2.system_one import SystemOneActRecommendation, SystemOneReviewResult
from alphonse.agent_v2.core.messages import CommunicationChannel, InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
from alphonse.agent_v2.evaluation.replay import EngineTrace, EvaluationCase


@dataclass(frozen=True)
class _Action:
    tool_id: str
    capability: str
    arguments: dict[str, Any] = field(default_factory=dict)
    role: str = "work"
    read_only: bool = True
    behavior: str = "static"
    result: dict[str, Any] = field(default_factory=lambda: {"found": True})
    side_effect: str = "read_only"
    failure_policy: str = "stop"


@dataclass(frozen=True)
class _Scenario:
    actions: tuple[_Action, ...]
    acceptance: str
    final_response: str
    mutation_paths: tuple[str, ...] = ()
    initial_acceptance: bool = False
    steering: str = ""
    restart_after_role: str = ""


@dataclass
class _RunState:
    scenario: _Scenario
    root: Path
    engine: str
    tool_calls: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    v2_index: int = 0
    v3_index: int = 0


class _DeterministicProvider:
    def __init__(self, state: _RunState) -> None:
        self.state = state

    def generate_markdown(self, request: InferenceRequest) -> InferenceResult:
        if request.purpose == InferencePurpose.ACCEPTANCE_CRITERIA:
            content = f"- [ ] {self.state.scenario.acceptance}"
        elif request.purpose == InferencePurpose.FINAL_RESPONSE:
            content = self.state.scenario.final_response
        else:
            content = ""
        return self._result(content=content, request=request)

    def generate_json(self, request: InferenceRequest) -> InferenceResult:
        if request.purpose == InferencePurpose.PHASE_PLANNING:
            phase = _phase_plan(self.state.scenario).to_dict()
            if '"response_required": true' in request.prompt.lower() or '"response_required":true' in request.prompt.lower():
                phase.update({
                    "phase_id": "final-response", "objective": "Respond to the requester",
                    "authorized_capabilities": ["user_response"],
                    "mutation_scope": {"allowed_paths": [], "allow_external_effects": True},
                    "limits": {"max_tool_calls": 1, "max_duration_seconds": 10},
                    "subgoals": [{
                        "subgoal_id": "respond", "objective": "Provide the final response",
                        "required_output_type": "user_response", "depends_on": [],
                        "allowed_capabilities": ["user_response"], "allowed_side_effects": ["user_response"],
                        "limits": {"max_tool_calls": 1, "max_duration_seconds": 10},
                        "completion": {"kind": "output_present", "output_type": "user_response"},
                        "failure_policy": "stop",
                    }],
                })
            value = phase
        elif request.purpose == InferencePurpose.TACTICAL_ACTION:
            value = self._next_v3_action()
        elif request.purpose == InferencePurpose.PHASE_REVIEW:
            value = _satisfied_patch(request.prompt)
        elif request.purpose == InferencePurpose.CRITERIA_REVIEW:
            completed = sum(
                tool_id not in {"native.respond", "native.ask_question"}
                for tool_id in self.state.tool_calls
            )
            if self.state.scenario.restart_after_role:
                completed += 1
            required = sum(
                action.tool_id not in {"native.respond", "native.ask_question"}
                for action in self.state.scenario.actions
            )
            value = _satisfied_patch(request.prompt) if completed >= required else {"updates": []}
        elif request.purpose == InferencePurpose.ACCEPTANCE_CRITERIA:
            value = {"operations": []}
        else:
            value = {}
        return self._result(json_value=value, request=request)

    def plan_tool_call(self, request: InferenceRequest) -> InferenceResult:
        actions = list(self.state.scenario.actions)
        if '"response_required": true' in request.prompt.lower() or '"response_required":true' in request.prompt.lower():
            action = _respond_action(self.state.scenario.final_response)
        elif self.state.v2_index >= len(actions):
            action = _respond_action(self.state.scenario.final_response)
        else:
            action = actions[self.state.v2_index]
            self.state.v2_index += 1
        value = {
            "id": f"v2-{self.state.v2_index}-{action.role}",
            "execution_mode": "direct",
            "tool_id": action.tool_id,
            "tool_name": action.tool_id,
            "arguments": dict(action.arguments),
            "internal_state": f"{action.role}: {action.capability}",
        }
        return self._result(json_value=value, tool_call=value, request=request)

    def _next_v3_action(self) -> dict[str, Any]:
        actions = list(self.state.scenario.actions)
        if self.state.v3_index >= len(actions):
            return {}
        action = actions[self.state.v3_index]
        self.state.v3_index += 1
        return {
            "action_id": f"v3-{self.state.v3_index}-{action.role}",
            "tool_id": action.tool_id,
            "arguments": dict(action.arguments),
        }

    @staticmethod
    def _result(*, request: InferenceRequest, content: str = "", json_value: Any = None, tool_call: Any = None) -> InferenceResult:
        return InferenceResult(
            content=content,
            json_value=json_value,
            tool_call=tool_call,
            usage={
                "input_tokens": max(1, (len(request.prompt) + 3) // 4),
                "output_tokens": max(1, (len(content or json.dumps(json_value, default=str)) + 3) // 4),
            },
        )


def v2_runner(
    case: EvaluationCase,
    fixture_root: Path,
    telemetry_sink: Callable[[dict[str, Any]], None],
) -> EngineTrace:
    return _run(case, fixture_root, telemetry_sink, engine="tactical_v2")


def v3_runner(
    case: EvaluationCase,
    fixture_root: Path,
    telemetry_sink: Callable[[dict[str, Any]], None],
) -> EngineTrace:
    return _run(case, fixture_root, telemetry_sink, engine="hierarchical_v3")


def _run(
    case: EvaluationCase,
    fixture_root: Path,
    telemetry_sink: Callable[[dict[str, Any]], None],
    *,
    engine: str,
) -> EngineTrace:
    scenario = _scenario(case.case_id)
    state = _RunState(scenario=scenario, root=fixture_root / "project", engine=engine)
    registry = _registry(state)
    provider = _DeterministicProvider(state)
    router = InferenceRouter(
        provider=provider,
        default_profile=ModelProfile(provider="deterministic", model="replay-v1", profile_id="offline-replay"),
        telemetry_sink=telemetry_sink,
    )
    queue = InMemoryMessageQueue()
    ui_events: list[dict[str, Any]] = []
    context = CoreLoopContext(
        messages=queue,
        tools=registry,
        inference=router,
        system_one=_ReplayJev(),
        telemetry_sink=telemetry_sink,
        ui_event_sink=lambda event: ui_events.append({"event_type": event.event_type, "payload": dict(event.payload)}),
    )
    task = TaskState(
        task_id=f"replay-{case.case_id}-{engine}",
        message_id=f"message-{case.case_id}",
        user="evaluation-user",
        project_id="fixture-project",
        goal=case.goal,
        intelligence_engine=engine,
        intelligence_schema_version=3 if engine == "hierarchical_v3" else 2,
        metadata={"attachments": list(case.fixture.get("attachments") or [])},
    )
    if scenario.initial_acceptance or scenario.restart_after_role:
        task.set_acceptance_contract_from_markdown(f"- [ ] {scenario.acceptance}")
    if scenario.steering:
        CommunicationChannel(queue).queue_message(
            prompt=scenario.steering,
            user=task.user or "evaluation-user",
            project_id=task.project_id,
            metadata={"routing_disposition": "steering"},
        )
    if scenario.restart_after_role:
        _restore_checkpoint(task, scenario, engine)
        if engine == "tactical_v2":
            state.v2_index = 1
        else:
            state.v3_index = 1

    processor = HierarchicalCAPDProcessor() if engine == "hierarchical_v3" else PDCAIntelligenceProcessor()
    try:
        result = processor.process(task, context)
    except Exception as exc:
        return EngineTrace(
            engine=engine,
            status="failed",
            outcome=f"processor exception: {type(exc).__name__}",
            error=f"{type(exc).__name__}: {exc}",
            capabilities=_used_capabilities(state, scenario),
            questions=list(state.questions),
            events=_evaluation_events(ui_events),
        )
    final_task = TaskState.from_dict(result.snapshot.metadata.get("task_state") or task.to_dict())
    metadata = _trace_metadata(case, scenario, state, final_task, ui_events)
    return EngineTrace(
        engine=engine,
        status=final_task.status,
        outcome=str((final_task.outcome or {}).get("reason") or final_task.status),
        capabilities=_used_capabilities(state, scenario),
        effects=_affected_paths_from_events(ui_events),
        questions=list(state.questions),
        events=_evaluation_events(ui_events),
        metadata=metadata,
        error="" if final_task.status not in {"failed", "error"} else str((final_task.outcome or {}).get("reason") or "task failed"),
    )


def _registry(state: _RunState) -> InMemoryToolRegistry:
    registry = InMemoryToolRegistry()
    by_id: dict[str, _Action] = {}
    for action in (*state.scenario.actions, _respond_action(state.scenario.final_response)):
        by_id.setdefault(action.tool_id, action)
    for tool_id, action in by_id.items():
        kind = ToolKind.ARTIFACT if tool_id.startswith("artifact.") else ToolKind.NATIVE
        descriptor = ToolDescriptor(
            tool_id=tool_id,
            name=tool_id,
            kind=kind,
            description=f"Deterministic replay tool for {action.capability}.",
            argument_schema={"type": "object", "additionalProperties": True},
            metadata={"v3_capabilities": [action.capability]},
            read_only=action.read_only,
        )

        def invoke(arguments: dict[str, Any], *, selected: _Action = action) -> dict[str, Any]:
            state.tool_calls.append(selected.tool_id)
            if selected.behavior == "fail":
                state.failures.append(selected.tool_id)
                raise RuntimeError("fixture index unavailable")
            if selected.behavior == "edit":
                return _edit_fixture(state.root, arguments)
            if selected.behavior == "read":
                return _read_fixture(state.root, arguments)
            if selected.behavior == "ask":
                question = str(arguments.get("question") or selected.result.get("question") or "Which source should be authoritative?")
                state.questions.append(question)
                return {"waiting_for_answer": True, "question": question}
            if selected.behavior == "respond":
                return {"message": str(arguments.get("message") or state.scenario.final_response)}
            return json.loads(json.dumps(selected.result))

        registry.register(ToolDefinition(descriptor=descriptor, callable=invoke))
    return registry


def _phase_plan(scenario: _Scenario) -> PhasePlan:
    subgoals = []
    prior: list[str] = []
    for index, action in enumerate(scenario.actions, start=1):
        allowed_effects = [action.side_effect]
        completion = {"kind": "output_present", "output_type": f"result_{index}"}
        if action.behavior == "edit":
            completion = {"kind": "field_equals", "field": "verification.status", "expected": "verified"}
        subgoals.append({
            "subgoal_id": action.role,
            "objective": f"Execute {action.role}",
            "required_output_type": f"result_{index}",
            "depends_on": list(prior[-1:]),
            "allowed_capabilities": [action.capability],
            "allowed_side_effects": allowed_effects,
            "limits": {"max_tool_calls": 2 if action.failure_policy == "local_fallback" else 1, "max_duration_seconds": 30},
            "completion": completion,
            "failure_policy": action.failure_policy,
        })
        if not (action.failure_policy == "local_fallback" and index < len(scenario.actions)):
            prior.append(action.role)
    # Consecutive fallback actions are one tactical subgoal with two eligible tools.
    if any(action.failure_policy == "local_fallback" for action in scenario.actions):
        first, second = scenario.actions[:2]
        subgoals = [{
            "subgoal_id": "locate",
            "objective": "Locate using bounded fallback",
            "required_output_type": "result_1",
            "depends_on": [],
            "allowed_capabilities": [first.capability, second.capability],
            "allowed_side_effects": ["read_only"],
            "limits": {"max_tool_calls": 2, "max_duration_seconds": 30},
            "completion": {"kind": "output_present", "output_type": "result_1"},
            "failure_policy": "local_fallback",
        }]
    return PhasePlan.from_dict({
        "acceptance_criteria": [scenario.acceptance],
        "phase_id": "fixture-phase",
        "objective": scenario.acceptance,
        "subgoals": subgoals,
        "criterion_ids": ["ac-1"],
        "limits": {"max_tool_calls": max(2, len(scenario.actions) + 1), "max_duration_seconds": 60},
        "authorized_capabilities": list(dict.fromkeys(action.capability for action in scenario.actions)),
        "mutation_scope": {
            "allowed_paths": list(scenario.mutation_paths),
            "allow_external_effects": any(action.side_effect.startswith("external_") for action in scenario.actions),
        },
        "originating_decision": "offline_replay",
        "schema_version": 3,
    })


class _ReplayJev:
    """Deterministic Jev stand-in for offline V2/V3 evaluation runs."""

    def evaluate(self, *, contract, phase, evidence):
        _ = phase
        entries = evidence.get("entries") if isinstance(evidence, dict) else []
        successful = [
            str(item.get("evidence_ref")) for item in entries
            if isinstance(item, dict) and item.get("status") == "success" and item.get("evidence_ref")
        ]
        updates = tuple({
            "criterion_id": str(item.get("id") or ""),
            "status": "satisfied",
            "evidence_refs": successful[-1:],
            "reason": "Deterministic replay evidence",
        } for item in contract.get("criteria") or [] if isinstance(item, dict) and successful)
        return SystemOneReviewResult(updates, (), model="replay-jev")

    def select_plan_tools(self, *, tools, **_values):
        from alphonse.agent_v2.system_one import SystemOneToolRegistrySelection
        tool_ids = tuple(str(getattr(tool, "tool_id", "")) for tool in tools)
        return SystemOneToolRegistrySelection(selected_tool_ids=tool_ids)

    def triage_plan_messages(self, *, candidates, **_values):
        return tuple(str(item.get("message_id")) for item in candidates if isinstance(item, dict))

    def recommend_act(self, *, state):
        verdict = str(state.get("check_verdict") or "wip")
        action = "complete" if verdict in {"success", "mission_success"} else "continue" if verdict == "wip" else "fail_explain"
        return SystemOneActRecommendation(
            action=action, confidence=0.99, confident=True, rationale=f"Replay Jev recommends {action}.",
            answers={
                "continuation_is_worthwhile": 0.99,
                "user_input_can_unblock": 0.1,
                "closure_explanation_is_warranted": 0.99,
            }, model="replay-jev",
        )

    def evaluate_tactical_progress(self, *, questions=None, **_values):
        from alphonse.agent_v2.system_one import SystemOneTacticalReview
        return SystemOneTacticalReview(
            complete=True, confidence=0.99, confident=True, model="replay-jev",
            answers={str(item.get("question_id")): 0.99 for item in questions or []},
            retry_approved=False,
        )


def _restore_checkpoint(task: TaskState, scenario: _Scenario, engine: str) -> None:
    first = scenario.actions[0]
    if engine == "hierarchical_v3":
        tactical = new_tactical_state(_phase_plan(scenario))
        tactical.status = PhaseStatus.RUNNING
        tactical.active_subgoal_id = scenario.actions[1].role
        tactical.completed_subgoal_ids = [first.role]
        tactical.remaining_tool_calls = max(1, int(tactical.remaining_tool_calls or 1) - 1)
        tactical.actions.append(TacticalAction("checkpoint-locate", first.role, first.tool_id, dict(first.arguments), "success", first.result))
        tactical.evidence.append({
            "evidence_ref": "tactical-action:checkpoint-locate", "phase_id": "fixture-phase",
            "subgoal_id": first.role, "tool_id": first.tool_id, "status": "success", "result": first.result,
        })
        task.hierarchical_state = tactical.to_dict()
    else:
        task.plan_json = json.dumps([{
            "id": "checkpoint-locate", "tool_id": first.tool_id, "tool_name": first.tool_id,
            "arguments": first.arguments, "internal_state": first.role,
            "execution": {"status": "success", "result": first.result},
        }])
        task.evidence_journal = [{
            "evidence_ref": "tool-call:checkpoint-locate", "call_id": "checkpoint-locate",
            "tool_id": first.tool_id, "status": "success", "result": first.result,
        }]


def _edit_fixture(root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    relative = str(arguments.get("path") or "").replace("\\", "/")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise PermissionError("fixture_edit_outside_project")
    old = str(arguments.get("old") or "")
    new = str(arguments.get("new") or "")
    content = path.read_text(encoding="utf-8")
    if old not in content:
        raise ValueError("fixture_edit_anchor_missing")
    path.write_text(content.replace(old, new, 1), encoding="utf-8")
    return {"affected_paths": [relative], "verification": {"status": "verified"}}


def _read_fixture(root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    relative = str(arguments.get("path") or "")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise PermissionError("fixture_read_outside_project")
    return {"path": relative, "content": path.read_text(encoding="utf-8")}


def _satisfied_patch(prompt: str) -> dict[str, Any]:
    ids = list(dict.fromkeys(re.findall(r'"id"\s*:\s*"(ac-[^"]+)"', prompt))) or ["ac-1"]
    refs = re.findall(r'(?:tool-call|tactical-action):[A-Za-z0-9._:-]+', prompt)
    evidence = [refs[-1]] if refs else []
    return {"updates": [
        {"criterion_id": criterion_id, "status": "satisfied", "evidence_refs": evidence, "reason": "fixture evidence"}
        for criterion_id in ids
    ]}


def _used_capabilities(state: _RunState, scenario: _Scenario) -> list[str]:
    capabilities = [
        action.capability for tool_id in state.tool_calls for action in scenario.actions
        if action.tool_id == tool_id
    ]
    return list(dict.fromkeys(capabilities))


def _affected_paths_from_events(events: list[dict[str, Any]]) -> list[str]:
    affected: list[str] = []
    for event in events:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        for path in result.get("affected_paths") or []:
            affected.append(str(path))
    return list(dict.fromkeys(affected))


def _evaluation_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for event in events:
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if event_type == "tactical_tools_revealed":
            projected.append({
                "event_type": event_type,
                "phase_id": payload.get("phase_id"),
                "subgoal_id": payload.get("subgoal_id"),
                "capabilities": list(payload.get("capabilities") or []),
                "tool_ids": list(payload.get("tool_ids") or []),
            })
        elif event_type == "tool_call_result":
            result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
            projected.append({
                "event_type": event_type,
                "tool_id": payload.get("tool_id"),
                "status": payload.get("status"),
                "affected_paths": [str(path) for path in result.get("affected_paths") or []],
            })
    return projected


def _trace_metadata(
    case: EvaluationCase,
    scenario: _Scenario,
    state: _RunState,
    task: TaskState,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "phase_shape": [action.role for action in scenario.actions if action.tool_id not in {"native.respond", "native.ask_question"}],
        "tool_ids": list(state.tool_calls),
        "task_status": task.status,
    }
    if case.case_id == "tool-failure-local-fallback":
        metadata["recovery"] = "bounded project-file search" if state.failures and "fixture.search" in state.tool_calls else ""
    if case.case_id == "conflicting-authoritative-records":
        route = str(task.metadata.get("v3_route") or task.metadata.get("act_route") or "")
        metadata["outer_route"] = "strategic_replan_or_ask_user" if state.questions and task.status == "waiting_user" else route
    if case.case_id == "steering-during-phase":
        reviewed = int(task.check_new_message_count or 0) > 0 or bool(task.metadata.get("acceptance_criteria_amendment_prompt"))
        metadata["outer_route"] = "explicit_contract_review" if reviewed else str(task.metadata.get("v3_route") or "")
    if case.case_id == "restart-mid-phase":
        locate_repeated = scenario.actions[0].tool_id in state.tool_calls
        metadata["property"] = "checkpointed action idempotency" if not locate_repeated else "locate repeated"
    return metadata


def _respond_action(message: str) -> _Action:
    return _Action(
        "native.respond", "user_response", {"message": message}, role="respond",
        behavior="respond", result={"message": message},
    )


def _scenario(case_id: str) -> _Scenario:
    scenarios = {
        "solar-project-completion": _Scenario(
            actions=(
                _Action("fixture.search", "project_record_search", {"query": "solar"}, role="locate", result={"path": "mejoras_hogar/backlog.md"}),
                _Action("native.exact_text_edit", "exact_text_mutation", {"path": "mejoras_hogar/backlog.md", "old": "Solar: open", "new": "Solar: complete"}, role="update", read_only=False, behavior="edit", side_effect="project_mutation"),
                _Action("fixture.read", "project_record_search", {"path": "mejoras_hogar/backlog.md"}, role="verify", behavior="read"),
            ),
            acceptance="The solar project is marked complete and verified.",
            final_response="Listo, el proyecto de celdas solares quedó marcado como completo.",
            mutation_paths=("mejoras_hogar/backlog.md",),
        ),
        "lg-temperature-native-client": _Scenario(
            actions=(_Action("artifact.lg", "device_control", {}, role="read_temperature", result={"temperature_c": 23}),),
            acceptance="The indoor temperature is read through the LG client.", final_response="En casa tenemos 23 °C.",
        ),
        "medical-treatment-recall": _Scenario(
            actions=(
                _Action("native.search_memory", "memory_recall", {"query": "treatment"}, role="recall", result={"medicine": "amoxicillin"}),
                _Action("artifact.medical", "project_artifact_query", {"medicine": "amoxicillin"}, role="record", read_only=False, result={"affected_paths": ["resolved medical artifact"], "recorded": True}, side_effect="external_reversible"),
            ),
            acceptance="The known treatment is recalled and the health event is recorded.", final_response="Listo, registré el evento con el tratamiento conocido.",
        ),
        "prescription-image-ocr": _Scenario(
            actions=(_Action("native.analyze_image", "attachment_analysis", {"asset_id": "prescription-image"}, role="extract", result={"medicines": ["Medicine A"]}),),
            acceptance="Medicine names are extracted from the attached prescription.", final_response="Extraje el nombre del medicamento de la receta.",
        ),
        "markdown-prescription-no-ocr": _Scenario(
            actions=(_Action("fixture.read", "project_record_search", {"path": "Receta.md"}, role="read_record", behavior="read"),),
            acceptance="Medicine names are read from the existing markdown record.", final_response="Encontré el medicamento en Receta.md.",
        ),
        "tool-failure-local-fallback": _Scenario(
            actions=(
                _Action("artifact.index", "project_artifact_query", {}, role="index", behavior="fail", failure_policy="local_fallback"),
                _Action("fixture.search", "project_record_search", {"query": "record"}, role="fallback", result={"path": "records/item.md"}),
            ),
            acceptance="The record is located with a bounded local fallback.", final_response="Encontré el registro usando la búsqueda local de respaldo.",
        ),
        "conflicting-authoritative-records": _Scenario(
            actions=(
                _Action("artifact.records", "project_artifact_query", {}, role="compare", result={"conflict": True, "sources": ["A", "B"]}),
                _Action("native.ask_question", "user_interaction", {"question": "Which source should be authoritative?"}, role="clarify", behavior="ask"),
            ),
            acceptance="Conflicting authoritative sources are resolved by the user.", final_response="Necesito confirmar cuál fuente es la autoritativa.",
        ),
        "steering-during-phase": _Scenario(
            actions=(_Action("artifact.lg", "device_control", {}, role="use_lg", result={"temperature_c": 23}),),
            acceptance="The LG client is used instead of Home Assistant.", final_response="Usé el cliente LG; la temperatura es 23 °C.",
            initial_acceptance=True, steering="Do not use Home Assistant; use the LG client.",
        ),
        "restart-mid-phase": _Scenario(
            actions=(
                _Action("fixture.search", "project_record_search", {"query": "record"}, role="locate", result={"path": "records/item.md"}),
                _Action("native.exact_text_edit", "exact_text_mutation", {"path": "records/item.md", "old": "Status: open", "new": "Status: complete"}, role="update", read_only=False, behavior="edit", side_effect="project_mutation"),
                _Action("fixture.read", "project_record_search", {"path": "records/item.md"}, role="verify", behavior="read"),
            ),
            acceptance="The resumed task updates and verifies the located record without repeating locate.",
            final_response="Reanudé la tarea y verifiqué la actualización sin repetir la búsqueda.",
            mutation_paths=("records/item.md",), restart_after_role="locate",
        ),
    }
    try:
        return scenarios[case_id]
    except KeyError as exc:
        raise ValueError(f"deterministic_scenario_missing:{case_id}") from exc
