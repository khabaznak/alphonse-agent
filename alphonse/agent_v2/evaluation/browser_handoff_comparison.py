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

from alphonse.agent_v2.browser_workflow_service import BrowserWorkflowService
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import MutationScope, PhaseExecutor, PhasePlan, PhaseSubgoal, PhaseStatus, SideEffectClass, new_tactical_state
from alphonse.agent_v2.core.intelligence.v3 import browser_workflow
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.browser_workflow import build_browser_workflow_tools


class _Form:
    def __init__(self):
        self.values = {}
        self.calls = []
        self.changed = False

    def session_identity(self, actor, task):
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
        return observation == self.snapshot() and observation["values"].get(target) == value


def run_case(*, control: bool, changed: bool = False) -> dict:
    """Return counts and a content-free independent oracle for one frozen case."""
    with TemporaryDirectory(prefix="alphonse-browser-comparison-") as directory:
        form = _Form()
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
                authorized_capabilities=("browser",), mutation_scope=MutationScope(allow_external_effects=True))
            state = new_tactical_state(phase)
            state.revealed_tool_ids = ["native.browser_workflow_" + name for name in ("fill", "observe", "reconcile")]
            return state
        state = make_state(0)
        state.transition(PhaseStatus.RUNNING)
        task = TaskState(task_id="fixture-task", user="fixture-owner", intelligence_engine="hierarchical_v3",
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
        if changed:
            form.changed = True
        fills = [("fill", {"field": "name", "target": "1_1"}), ("observe", {}), ("reconcile", {}),
                 ("fill", {"field": "number", "target": "1_2"}), ("observe", {}), ("reconcile", {})]
        segments = [fills] if not control else [fills[0:1], fills[1:3], fills[3:4], fills[4:6]]
        phases, actions, outcomes = 0, 0, []
        start = perf_counter()
        try:
            for index, segment in enumerate(segments):
                if index:
                    state = make_state(index)
                selected = iter(segment)
                def choose(*_):
                    item = next(selected, None)
                    return {"tool_id": "native.browser_workflow_" + item[0], "arguments": item[1]} if item else None
                outcome = PhaseExecutor(action_selector=choose).run(task, state,
                    CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, question_store=store,
                                    browser_workflow_service=service))
                phases += 1
                actions += len(state.actions)
                outcomes.append(outcome.status.value)
                # A failed action, unknown effect or changed form is not a
                # completed control segment and must never be replayed.
                if any(action.status == "failed" for action in state.actions):
                    break
            workflow = browser_workflow.load(task)
            verified = [e["field"] for e in workflow["events"] if e["kind"] == "verified"]
            oracle = form.values == {"1_1": "SYNTHETIC_NAME", "1_2": "SYNTHETIC_NUMBER"}
            expected = (not form.calls and browser_workflow.unresolved(workflow) and not verified
                        if changed else oracle and verified == ["name", "number"] and not browser_workflow.unresolved(workflow))
            return {"variant": "forced_phase_breaks" if control else "continued_phase", "changed": changed,
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


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, sort_keys=True))
