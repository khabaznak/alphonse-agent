"""Synthetic browser submission through V3, preserving the real completion guard."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

from alphonse.agent_v2.browser_workflow_service import BrowserWorkflowService
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import MutationScope, PhaseExecutor, PhasePlan, PhaseSubgoal, PhaseStatus, SideEffectClass, new_tactical_state
from alphonse.agent_v2.core.intelligence.v3 import browser_workflow
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.browser_workflow import build_browser_workflow_tools
from alphonse.agent_v2.evaluation.browser_metrics import summarize
from alphonse.agent_v2.system_one import SystemOneTacticalReview


class Fixture:
    """Isolated synthetic form and result oracle; no actual browser or network."""
    def __init__(self, *, changed=False, bad_result=False):
        self.values = {}
        self.submissions = []
        self.observations = []
        self.changed = changed
        self.bad_result = bad_result

    def session_identity(self, actor, task):
        return "fixture-session" if (actor, task) == ("fixture-owner", "fixture-task") else ""

    def snapshot(self):
        return {"targets": ["1_1", "1_2", "1_3"], "values": dict(self.values),
                "result": "accepted" if self.submissions and not self.bad_result else "missing"}

    def observe(self):
        snapshot = self.snapshot()
        self.observations.append(snapshot)
        return snapshot

    def fill_if_unchanged(self, target, value, observation):
        if self.changed or observation != self.snapshot():
            return False
        self.values[target] = value
        return True

    def verify_fill(self, target, value, observation):
        return observation == self.snapshot() and observation["values"].get(target) == value

    def submit_if_unchanged(self, target, observation):
        if target != "1_3" or observation != self.snapshot() or self.submissions:
            return False
        self.submissions.append(dict(self.values))
        return True

    def verify_submission(self, target, before, after):
        return (target == "1_3" and before["result"] == "missing"
                and after == self.snapshot() and after["result"] == "accepted")

    def oracle(self):
        return (self.submissions == [{"1_1": "SYNTHETIC_NAME", "1_2": "SYNTHETIC_NUMBER"}]
                and self.snapshot()["result"] == "accepted")


def run_case(*, changed=False, bad_result=False, cancelled=False, fixture=None):
    with TemporaryDirectory(prefix="alphonse-submission-fixture-") as directory:
        fixture = fixture or Fixture(changed=changed, bad_result=bad_result)
        store = SQLiteQuestionStore(Path(directory) / "checkpoint.sqlite")
        cancelled_flag = [False]
        service = BrowserWorkflowService(checkpoint_store=store, transport=fixture,
            authorized=lambda actor, task: "fixture-grant", cancelled=lambda: cancelled_flag[0], enabled=True)
        registry = InMemoryToolRegistry()
        for tool in build_browser_workflow_tools(enabled=True):
            registry.register(tool)
        def state_for(number):
            plan = PhasePlan(f"phase-{number}", "Complete synthetic form", (PhaseSubgoal(
                "work", "Complete form", "result", allowed_capabilities=("browser",),
                allowed_side_effects=(SideEffectClass.READ_ONLY, SideEffectClass.EXTERNAL_IRREVERSIBLE)),),
                authorized_capabilities=("browser",), mutation_scope=MutationScope(allow_external_effects=True))
            state = new_tactical_state(plan)
            state.revealed_tool_ids = ["native.browser_workflow_" + op for op in
                ("fill", "observe", "reconcile", "submit", "observe_result", "verify_submission")]
            state.transition(PhaseStatus.RUNNING)
            return state
        state = state_for(0)
        task = TaskState(task_id="fixture-task", user="fixture-owner", intelligence_engine="hierarchical_v3",
                         hierarchical_state=state.to_dict())
        events = []
        def review(**_):
            return SystemOneTacticalReview(False, .99, True, answers={
                "runtime_action_added_progress": .99, "runtime_declared_subgoal_complete": .01})
        context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, question_store=store,
            browser_workflow_service=service, telemetry_sink=events.append,
            system_one=SimpleNamespace(evaluate_tactical_progress=review), cancellation_checker=lambda: cancelled_flag[0])
        start = perf_counter()
        service.begin(task)
        service.supply(task, actor_id=task.user, field="name", value="SYNTHETIC_NAME")
        service.supply(task, actor_id=task.user, field="number", value="SYNTHETIC_NUMBER")
        service.observe(task)
        phases = [
            [("fill", {"field": "name", "target": "1_1"}), ("observe", {}), ("reconcile", {}),
             ("fill", {"field": "number", "target": "1_2"}), ("observe", {}), ("reconcile", {}),
             ("submit", {"target": "1_3"})],
            [("observe_result", {}), ("verify_submission", {})],
        ]
        statuses, counts = [], 0
        try:
            for i, steps in enumerate(phases):
                if i:
                    state = state_for(i)
                selected = iter(steps)
                def select(*_):
                    nonlocal counts
                    counts += 1
                    item = next(selected, None)
                    return {"tool_id": "native.browser_workflow_" + item[0], "arguments": item[1]} if item else None
                outcome = PhaseExecutor(action_selector=select).run(task, state, context)
                statuses.append(outcome.status.value)
                if cancelled and i == 0:
                    cancelled_flag[0] = True
                if not fixture.submissions or cancelled_flag[0]:
                    break
            state_record = browser_workflow.load(task)
            verified = [event["field"] for event in state_record["events"] if event["kind"] == "verified"]
            oracle_result = fixture.oracle()
            completion_blocked = browser_workflow.completion_blocked(task, service)
            elapsed_ms = round((perf_counter() - start) * 1000, 3)
            return {"elapsed_ms": elapsed_ms, "phase_statuses": statuses,
                "task_outcome": "cancelled" if cancelled_flag[0] else "blocked" if completion_blocked else "unqualified",
                "oracle_result": "verified" if oracle_result else "unverified",
                "v3_completion_guard": "blocked" if completion_blocked else "not_blocked",
                "submission_verified": "submit" in verified,
                "verified_fields": sum(field in {"name", "number"} for field in verified),
                "unresolved_effect": browser_workflow.unresolved(state_record),
                "compositions": counts, "observations": len(fixture.observations),
                "repeated_observations": len(fixture.observations) - len({json.dumps(x, sort_keys=True) for x in fixture.observations}),
                "spans": summarize(events, task_id=task.task_id)["spans"]}
        finally:
            service.private_data.clear()


if __name__ == "__main__":
    print(json.dumps({"scope": "synthetic; scripted V3 phase execution; task guard retained",
                      "runs": [run_case() for _ in range(5)]}, indent=2))
