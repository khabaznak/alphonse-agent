"""Disposable, scripted comparison of browser fill phase boundaries.

The conservative control forces the old stop after a fill and after its
reconciliation. It is a control-flow reconstruction, not a provider benchmark
or a live-browser qualification. Both arms use the real executor and service.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace
from dataclasses import replace

from alphonse.agent_v2.browser_workflow_service import BrowserWorkflowService
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import MutationScope, PhaseExecutor, PhasePlan, PhaseSubgoal, PhaseStatus, SideEffectClass, new_tactical_state
from alphonse.agent_v2.core.intelligence.v3 import browser_workflow
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.browser_workflow import build_browser_workflow_tools
from alphonse.agent_v2.system_one import SystemOneTacticalReview


class _Form:
    def __init__(self, scenario="normal"):
        self.values = {}
        self.calls = []
        self.changed = False
        self.scenario = scenario

    def session_identity(self, actor, task):
        if self.scenario == "stale" and self.calls:
            return "changed-session"
        return "fixture-session" if (actor, task) == ("fixture-owner", "fixture-task") else ""

    def snapshot(self):
        return {"targets": ["1_1", "1_2"], "values": dict(self.values), "changed": self.changed}

    def observe(self):
        return self.snapshot()

    def fill_if_unchanged(self, target, value, observation):
        if observation != self.snapshot():
            return False
        self.values[target] = value
        self.calls.append(target)
        return True

    def verify_fill(self, target, value, observation):
        return self.scenario != "retry" and observation == self.snapshot() and observation["values"].get(target) == value


def run_case(*, control: bool, changed: bool = False, sequence: bool = False, inference=None,
             scenario: str = "normal") -> dict:
    """Return counts and a content-free independent oracle for one frozen case."""
    if changed:
        if scenario != "normal":
            raise ValueError("conflicting_scenario")
        scenario = "changed"
    if scenario not in {"normal", "changed", "stale", "denied", "cancelled", "retry", "malformed_once", "malformed_always", "malformed_cancelled"}:
        raise ValueError("unknown_scenario")
    with TemporaryDirectory(prefix="alphonse-browser-comparison-") as directory:
        form = _Form(scenario)
        store = SQLiteQuestionStore(Path(directory) / "fixture.sqlite")
        service = BrowserWorkflowService(checkpoint_store=store, transport=form,
            authorized=lambda actor, task: "fixture-grant", cancelled=lambda: False, enabled=True)
        registry = InMemoryToolRegistry()
        for tool in build_browser_workflow_tools(enabled=True):
            registry.register(tool)
        def make_state(number):
            phase = PhasePlan(f"phase-{number}", "Complete the synthetic fields", (PhaseSubgoal(
                "fill", "Fill and verify fields", "form", allowed_capabilities=("browser",),
                allowed_side_effects=(SideEffectClass.READ_ONLY, SideEffectClass.EXTERNAL_IRREVERSIBLE)),),
                authorized_capabilities=("browser",), mutation_scope=MutationScope(allow_external_effects=scenario != "denied"))
            state = new_tactical_state(phase)
            state.revealed_tool_ids = ["native.browser_workflow_" + name for name in ("fill", "observe", "reconcile")]
            return state
        state = make_state(0)
        state.transition(PhaseStatus.RUNNING)
        task = TaskState(task_id="fixture-task", user="fixture-owner", intelligence_engine="hierarchical_v3",
                         goal="Fill the synthetic name field at target 1_1 and number field at target 1_2. "
                              "Both private inputs are already supplied. Independently reconcile each fill.",
                         hierarchical_state=state.to_dict())
        service.begin(task)
        service.supply(task, actor_id=task.user, field="name", value="SYNTHETIC_NAME")
        service.supply(task, actor_id=task.user, field="number", value="SYNTHETIC_NUMBER")
        observations = []
        observe = service.observe
        def counted_observe(*args, **kwargs):
            ref = observe(*args, **kwargs)
            observations.append(json.dumps(form.snapshot(), sort_keys=True))
            return ref
        service.observe = counted_observe
        service.observe(task)
        if scenario == "changed":
            form.changed = True
        fills = [("fill", {"field": "name", "target": "1_1"}), ("observe", {}), ("reconcile", {}),
                 ("fill", {"field": "number", "target": "1_2"}), ("observe", {}), ("reconcile", {})]
        segments = [fills] if not control else [fills[0:1], fills[1:3], fills[3:4], fills[4:6]]
        phases, actions, outcomes, compositions, recovery_guidance_calls, sequence_proposals = 0, 0, [], 0, 0, 0
        tool_ids = []
        provider_choices = []
        provider_shapes = []
        failure_type = failure_code = None
        start = perf_counter()
        try:
            for index, segment in enumerate(segments):
                if index:
                    state = make_state(index)
                selected = iter(segment)
                class ScriptedProvider:
                    def generate_json(self, request):
                        nonlocal compositions, recovery_guidance_calls, sequence_proposals
                        compositions += 1
                        recovery_guidance_calls += int("A previous browser fill has an unresolved effect." in request.prompt)
                        if scenario in {"malformed_always", "malformed_cancelled"} or (scenario == "malformed_once" and compositions == 1):
                            raise ValueError("openai_codex_invalid_json")
                        item = next(selected, None)
                        question = {"question_id": "result", "type": "noul", "instructions": "Was the step completed?",
                                    "criteria": {"true": "The step was completed.", "false": "The step was not completed."}}
                        def action(step):
                            return {"tool_id": "native.browser_workflow_" + step[0], "arguments": step[1],
                                    "acceptance_questions": [question]}
                        if item and sequence:
                            sequence_proposals += 1
                            response = {"browser_fill_sequence": [action(step) for step in (item, next(selected), next(selected))]}
                        else:
                            response = action(item) if item else None
                        return SimpleNamespace(json_value=response)
                class MeasuredProvider:
                    def generate_json(self, request):
                        nonlocal compositions, recovery_guidance_calls, sequence_proposals
                        compositions += 1
                        recovery_guidance_calls += int("A previous browser fill has an unresolved effect." in request.prompt)
                        if not sequence:
                            prompt = request.prompt.replace(
                                "For a browser fill whose field, target and private input are already known, you may instead return",
                                "Do not return browser_fill_sequence. For this comparison you must not return")
                            prompt = prompt.replace(
                                "For a read-only subgoal only, you may instead return {actions:[...]}",
                                "Do not return an actions array. For this comparison you may not return {actions:[...]}")
                            request = replace(request, prompt=("For this control, return exactly one action object with "
                                "top-level tool_id, arguments, and acceptance_questions; do not wrap it.\n" + prompt))
                        result = inference.generate_json(request)
                        if isinstance(result.json_value, dict):
                            known_keys = {"tool_id", "tool_name", "name", "arguments", "action", "actions", "tool_call",
                                          "browser_fill_sequence", "acceptance_questions", "uncertainty_assessment"}
                            provider_shapes.append(sorted(known_keys.intersection(result.json_value)))
                        else:
                            provider_shapes.append([])
                        if isinstance(result.json_value, dict) and "browser_fill_sequence" in result.json_value:
                            sequence_proposals += 1
                            provider_choices.append("browser_fill_sequence")
                        elif isinstance(result.json_value, dict):
                            tool_id = result.json_value.get("tool_id")
                            allowed_ids = {"native.browser_workflow_" + operation for operation in
                                           ("fill", "observe", "reconcile", "submit", "observe_result", "verify_submission")}
                            provider_choices.append(tool_id if isinstance(tool_id, str) and tool_id in allowed_ids else "other_or_invalid")
                        else:
                            provider_choices.append("other_or_invalid")
                        if not sequence and isinstance(result.json_value, dict) and "browser_fill_sequence" in result.json_value:
                            return SimpleNamespace(json_value=None)
                        return result
                def review(**_):
                    return SystemOneTacticalReview(False, .99, True, answers={
                        "runtime_action_added_progress": .99, "runtime_declared_subgoal_complete": .01})
                try:
                    outcome = PhaseExecutor().run(task, state,
                        CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, question_store=store,
                                        browser_workflow_service=service,
                                        inference=MeasuredProvider() if inference is not None else ScriptedProvider(),
                                        system_one=SimpleNamespace(evaluate_tactical_progress=review),
                                        cancellation_checker=lambda: compositions >= 12 or
                                            (scenario == "cancelled" and bool(form.calls)) or
                                            (scenario == "malformed_cancelled" and compositions >= 1)))
                    outcome_status = outcome.status.value
                except Exception as exc:
                    failure_type = type(exc).__name__
                    message = str(exc)
                    allowed = {"openai_codex_invalid_json", "openai_codex_timeout", "openai_codex_cli_missing",
                               "openai_codex_auth_required", "openai_codex_cli_upgrade_required",
                               "openai_codex_empty_response"}
                    if message in allowed:
                        failure_code = message
                    elif message.startswith("openai_codex_exec_failed: exit_code="):
                        failure_code = "openai_codex_exec_failed"
                    elif message.startswith("openai_codex_model_access_rejected:"):
                        failure_code = "openai_codex_model_access_rejected"
                    outcome_status = "error"
                phases += 1
                actions += len(state.actions)
                tool_ids.extend(action.tool_id for action in state.actions
                                if action.tool_id.startswith('native.browser_workflow_'))
                outcomes.append(outcome_status)
                # A failed action, unknown effect or changed form is not a
                # completed control segment and must never be replayed.
                if failure_type or any(action.status == "failed" for action in state.actions) or (
                        scenario != "normal" and outcome_status in {"blocked", "cancelled"}):
                    break
            workflow = browser_workflow.load(task)
            verified = [e["field"] for e in workflow["events"] if e["kind"] == "verified"]
            oracle = form.values == {"1_1": "SYNTHETIC_NAME", "1_2": "SYNTHETIC_NUMBER"}
            if scenario == "changed":
                expected = not form.calls and browser_workflow.unresolved(workflow) and not verified
            elif scenario == "denied":
                expected = not form.calls and not browser_workflow.unresolved(workflow) and not verified
            elif scenario in {"stale", "cancelled", "retry"}:
                expected = len(form.calls) == 1 and browser_workflow.unresolved(workflow) and not verified
            elif scenario in {"malformed_always", "malformed_cancelled"}:
                expected = not form.calls and not browser_workflow.unresolved(workflow) and not verified
            else:
                expected = oracle and verified == ["name", "number"] and not browser_workflow.unresolved(workflow)
            expected = expected and failure_type is None and bool(outcomes) and outcomes[-1] == (
                "cancelled" if scenario in {"cancelled", "malformed_cancelled"} else "blocked")
            return {"variant": "fill_sequence" if sequence else "forced_phase_breaks" if control else "continued_phase", "changed": scenario == "changed", "scenario": scenario,
                    "composition_calls": compositions,
                    "recovery_guidance_calls": recovery_guidance_calls,
                    "sequence_proposals": sequence_proposals,
                    "provider_choices": provider_choices,
                    "provider_shapes": provider_shapes,
                    "tool_ids": tool_ids,
                    "provider_mode": "live" if inference is not None else "scripted",
                    "failure_type": failure_type,
                    "failure_code": failure_code,
                    "scripted_phase_replans": max(0, phases - 1), "phase_count": phases, "tool_calls": actions,
                    "observations": len(observations), "repeated_observations": len(observations) - len(set(observations)),
                    "verified_fields": len(verified), "effect_oracle_pass": oracle and verified == ["name", "number"],
                    "scenario_expectation_met": expected,
                    "unresolved_effect": browser_workflow.unresolved(workflow), "phase_outcomes": outcomes,
                    "elapsed_seconds": perf_counter() - start, "task_completion": "not_qualified"}
        finally:
            service.private_data.clear()


def report(repeats: int = 3) -> dict:
    if type(repeats) is not int or not 1 <= repeats <= 100:
        raise ValueError("invalid_repeats")
    return {"scope": "scripted disposable backend; verified field effects, no submission or live browser",
            "runs": [run_case(control=control, changed=changed) for changed in (False, True)
                     for control in (True, False) for _ in range(repeats)]}


def sequence_report(repeats: int = 3) -> dict:
    """Compare the current loop and sequence on identical disposable cases."""
    if type(repeats) is not int or not 1 <= repeats <= 100:
        raise ValueError("invalid_repeats")
    runs = [run_case(control=False, sequence=sequence, scenario=scenario)
            for scenario in ("normal", "changed", "stale", "denied", "cancelled", "retry")
            for sequence in (False, True) for _ in range(repeats)]
    return {"scope": "scripted inference through real executor and disposable browser service; no submission",
            "limitations": ["Provider responses and tactical reviews are scripted.",
                "Retry scenario tests that failed verification does not replay a possibly effective fill.",
                "Independent field oracle is not task-completion or real-browser qualification."],
            "safety_oracle_failures": sum(not row["scenario_expectation_met"] for row in runs),
            "runs": runs}


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, sort_keys=True))
