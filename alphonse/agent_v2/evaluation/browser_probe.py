"""Repeat a disposable, blocked V3 browser-foundation probe (not a benchmark)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from time import perf_counter

from alphonse.agent_v2.browser_workflow_service import BrowserWorkflowService
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import MutationScope, PhaseExecutor, PhasePlan, PhaseSubgoal, SideEffectClass, new_tactical_state
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.browser_workflow import build_browser_workflow_tools
from alphonse.agent_v2.evaluation.browser_metrics import summarize


class SyntheticPage:
    """In-memory two-field form; no browser, network or household data."""
    def __init__(self):
        self.fills = 0
        self.observations = 0

    def session_identity(self, actor, task):
        return "synthetic-session"

    def observe(self):
        self.observations += 1
        return {"targets": ["1_1", "1_2"]}

    def fill_if_unchanged(self, target, value, observed):
        self.fills += 1
        return target in observed["targets"]


def run_once(path: Path) -> dict:
    start = perf_counter()
    task = TaskState(task_id="synthetic-task", user="synthetic-actor", intelligence_engine="hierarchical_v3",
        hierarchical_state={"status": "running", "active_subgoal_id": "fill", "phase": {
            "phase_id": "phase-1", "mutation_scope": {"allow_external_effects": True},
            "subgoals": [{"subgoal_id": "fill", "allowed_capabilities": ["browser"],
                          "allowed_side_effects": ["external_irreversible"]}]}})
    page = SyntheticPage()
    store = SQLiteQuestionStore(path / "questions.sqlite")
    service = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task_id: "disposable-grant", cancelled=lambda: False, enabled=True)
    service.begin(task)
    service.supply(task, actor_id=task.user, field="name", value="synthetic name")
    service.supply(task, actor_id=task.user, field="number", value="synthetic number")
    service.observe(task)
    plan = PhasePlan("phase-1", "Complete synthetic form", (PhaseSubgoal("fill", "Fill", "receipt",
        allowed_capabilities=("browser",), allowed_side_effects=(SideEffectClass.EXTERNAL_IRREVERSIBLE,)),),
        authorized_capabilities=("browser",), mutation_scope=MutationScope(allow_external_effects=True))
    state = new_tactical_state(plan)
    state.revealed_tool_ids = ["native.browser_workflow_fill"]
    registry = InMemoryToolRegistry()
    for tool in build_browser_workflow_tools(enabled=True):
        registry.register(tool)
    events = []
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, question_store=store,
        browser_workflow_service=service, telemetry_sink=events.append)
    executor = PhaseExecutor(action_selector=lambda *_: {"tool_id": "native.browser_workflow_fill",
        "arguments": {"field": "name", "target": "1_1"}})
    try:
        outcome = executor.run(task, state, context)
        # Only a known blocker is exported; do not serialize arbitrary model or backend text.
        reason = "reconciliation_required" if "browser_workflow_reconciliation_required" in outcome.blockers else "other"
        return {"elapsed_ms": round((perf_counter() - start) * 1000, 3),
                "outcome": "blocked" if reason == "reconciliation_required" else "unexpected",
                "reason": reason, "observations": page.observations, "fills": page.fills,
                "spans": summarize(events, task_id=task.task_id)["spans"]}
    finally:
        service.private_data.clear()


def main():
    runs = []
    for _ in range(5):
        with tempfile.TemporaryDirectory(prefix="alphonse-browser-probe-") as folder:
            runs.append(run_once(Path(folder)))
    print(json.dumps({"kind": "incomplete_synthetic_probe", "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
