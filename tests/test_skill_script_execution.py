"""Disposable standard packages executed through the ordinary native Bash path."""
from pathlib import Path
import shlex
import sys
import time
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.invocation import ToolInvocationService
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.core.intelligence.v3.context import selected_skill_guidance
from alphonse.agent_v2.core.intelligence.v3 import (
    CompletionCondition, MutationScope, PhaseExecutor, PhasePlan, PhaseStatus,
    PhaseSubgoal, SideEffectClass, TacticalAction, TacticalState, new_tactical_state,
)
from alphonse.agent_v2.system_one import SystemOneTacticalReview


@pytest.fixture
def package(tmp_path):
    store = SkillStore(tmp_path / "skills with spaces")
    root = store.skills_dir / "sample"
    (root / "scripts").mkdir(parents=True)
    (root / "SKILL.md").write_text("---\nname: sample\ndescription: Disposable script fixture\n---\nRun scripts with the documented interpreter; read back writes.\n")
    (root / "scripts" / "read write.py").write_text(
        "from pathlib import Path\nimport sys\np = Path(sys.argv[2])\n"
        "if sys.argv[1] == 'write': p.write_text(p.read_text() + 'once\\n' if p.exists() else 'once\\n')\n"
        "print(p.read_text(), end='')\n"
    )
    executable = root / "scripts" / "hello.sh"
    executable.write_text("#!/bin/sh\nprintf ordinary\n")
    executable.chmod(0o755)
    task = TaskState(user="alex", project_id="unrelated")
    task.metadata["v3_skill_selection"] = {"selected_skill_ids": ["skill:sample"]}
    project = tmp_path / "unrelated project"
    project.mkdir()
    context = CoreLoopContext(
        messages=InMemoryMessageQueue(), skill_store=store, tools=build_native_tool_registry(),
        project_store=SimpleNamespace(get_project=lambda *a, **k: SimpleNamespace(root_path=str(project))),
    )
    return root, task, context


def invoke(task, context, **arguments):
    return ToolInvocationService(context=context, task=task).invoke("native.bash", {"skill_id": "skill:sample", **arguments})


def test_standard_script_runtime_spaces_project_and_daemon_cwd_independence(package, tmp_path, monkeypatch):
    root, task, context = package
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "household data.txt"
    command = '"$ALPHONSE_SKILL_RUNTIME" "scripts/read write.py" write ' + shlex.quote(str(target))
    outcome = invoke(task, context, command=command, runtime=sys.executable)
    assert outcome["status"] == "success"
    assert outcome["result"]["cwd"] == str(root)
    assert outcome["result"]["stdout"] == "once\n"
    assert target.read_text() == "once\n"
    readback = invoke(task, context, command='"$ALPHONSE_SKILL_RUNTIME" "scripts/read write.py" read ' + shlex.quote(str(target)), runtime=sys.executable)
    assert readback["result"]["stdout"] == "once\n"
    guidance, ids = selected_skill_guidance(task, context)
    assert str(root) in guidance and "no registration" in guidance
    assert ids == ["skill:sample"]


def test_ordinary_executable_and_relative_cwd(package):
    root, task, context = package
    outcome = invoke(task, context, cwd="scripts", command='./hello.sh; printf "|%s" "$ALPHONSE_SKILL_ROOT"')
    assert outcome["status"] == "success"
    assert outcome["result"]["stdout"] == f"ordinary|{root}"


@pytest.mark.parametrize("arguments,diagnostic", [
    ({"runtime": "alphonse-nonexistent-runtime"}, "bash_runtime_unavailable"),
    ({"runtime": ".venv/bin/python"}, "bash_runtime_unavailable"),
    ({"required_files": ["missing config.json"]}, "bash_required_file_unavailable"),
    ({"cwd": "../"}, "bash_skill_cwd_outside_package"),
    ({"required_files": ["../secret"]}, "bash_required_file_outside_package"),
    ({"runtime": "../python"}, "bash_runtime_outside_package"),
])
def test_preflight_failure_never_dispatches(package, arguments, diagnostic):
    root, task, context = package
    outcome = invoke(task, context, command="touch should-not-exist", **arguments)
    assert outcome["status"] == "failed"
    assert diagnostic in outcome["error"]["message"]
    assert not (root / "should-not-exist").exists()


def test_available_config_and_contained_interpreter_link(package, tmp_path):
    root, task, context = package
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").symlink_to(sys.executable)
    config = tmp_path / "user config.json"
    config.write_text("{}")
    result = invoke(task, context, runtime=".venv/bin/python", required_files=[str(config)], command='"$ALPHONSE_SKILL_RUNTIME" -c "print(42)"')
    assert result["status"] == "success" and result["result"]["stdout"] == "42\n"
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    assert invoke(task, context, cwd="escape", command="true")["status"] == "failed"
    assert invoke(task, context, required_files=["escape/user config.json"], command="true")["status"] == "failed"


def test_documented_dependency_check_retains_error(package):
    _, task, context = package
    outcome = invoke(task, context, runtime=sys.executable, command='"$ALPHONSE_SKILL_RUNTIME" -c "import alphonse_missing_fixture_dependency"')
    assert outcome["status"] == "failed"
    assert outcome["result"]["exit_code"] != 0
    assert "ModuleNotFoundError" in outcome["result"]["stderr"]


def test_unselected_and_removed_packages_are_rechecked(package):
    root, task, context = package
    task.metadata["v3_skill_selection"]["selected_skill_ids"] = []
    assert "bash_skill_not_selected" in invoke(task, context, command="true")["error"]["message"]
    task.metadata["v3_skill_selection"]["selected_skill_ids"] = ["skill:sample"]
    (root / "SKILL.md").unlink()
    assert invoke(task, context, command="true")["status"] == "failed"


def test_cancellation_before_dispatch_and_during_process_preserves_output(package):
    root, task, context = package
    context.cancellation_checker = lambda: True
    result = invoke(task, context, command="touch forbidden")
    assert result["status"] == "failed" and result["result"]["cancelled"]
    assert not (root / "forbidden").exists()
    started = time.monotonic()
    context.cancellation_checker = lambda: time.monotonic() - started > .15
    result = invoke(task, context, command="printf started; sleep 30; touch forbidden")
    assert time.monotonic() - started < 2
    assert result["status"] == "failed" and result["result"]["cancelled"]
    assert result["result"]["stdout"] == "started"
    assert not result["result"]["timed_out"]
    assert not (root / "forbidden").exists()


def question():
    return [{"question_id": "verified", "type": "noul", "instructions": "Does output prove this subgoal?", "criteria": {"true": "Direct evidence", "false": "Missing or failed"}}]


def phase():
    return PhasePlan("scripts", "Write once then verify", (
        PhaseSubgoal("write", "Write", "command_result", allowed_capabilities=("native.bash",), allowed_side_effects=(SideEffectClass.PROJECT_MUTATION,), completion=CompletionCondition("field_equals", field="exit_code", expected=0)),
        PhaseSubgoal("read", "Read back", "readback", depends_on=("write",), allowed_capabilities=("native.bash",), completion=CompletionCondition("field_equals", field="exit_code", expected=0)),
    ), authorized_capabilities=("native.bash",), mutation_scope=MutationScope(("output.txt",)))


def test_phase_write_resume_readback_does_not_duplicate_mutation(package):
    root, task, context = package
    # Household output is live task data, outside the pinned reusable package.
    target = Path(context.project_store.get_project(task.project_id).root_path) / "output.txt"
    state = new_tactical_state(phase())
    state.revealed_tool_ids = ["native.bash"]
    def review(**values):
        # Simulate restart after the write's completed subgoal was checkpointed.
        if values["subgoal"]["subgoal_id"] == "write":
            context.cancellation_checker = lambda: True
        return SystemOneTacticalReview(complete=True, confident=True, confidence=.99, answers={}, retry_approved=False)
    context.system_one = SimpleNamespace(evaluate_tactical_progress=review)
    write = {"tool_id": "native.bash", "arguments": {"skill_id": "skill:sample", "command": "printf once >> " + shlex.quote(str(target))}, "acceptance_questions": question()}
    PhaseExecutor(action_selector=lambda *a: write).run(task, state, context)
    assert target.read_text() == "once"
    restored = TacticalState.from_dict(state.to_dict())
    restored.status = PhaseStatus.RUNNING
    restored.cancellation_pending = False
    context.cancellation_checker = None
    read = {"tool_id": "native.bash", "arguments": {"skill_id": "skill:sample", "command": "cat " + shlex.quote(str(target))}, "acceptance_questions": question()}
    outcome = PhaseExecutor(action_selector=lambda *a: read).run(task, restored, context)
    assert outcome.status == PhaseStatus.PHASE_COMPLETE
    assert restored.actions[-1].result["stdout"] == "once"
    assert target.read_text() == "once"


@pytest.mark.parametrize("status", ["planned", "running", "success", "failed"])
def test_uncertain_or_prior_mutation_cannot_be_blindly_replayed(package, status):
    root, task, context = package
    state = new_tactical_state(phase())
    arguments = {"skill_id": "skill:sample", "command": "printf once >> output.txt"}
    state.actions = [TacticalAction("prior-write", "write", "native.bash", arguments, status=status)]
    state.revealed_tool_ids = ["native.bash"]
    selected = {"tool_id": "native.bash", "arguments": arguments, "acceptance_questions": question()}
    outcome = PhaseExecutor(action_selector=lambda *a: selected).run(task, state, context)
    assert outcome.status == PhaseStatus.BLOCKED
    assert "reconciliation" in outcome.blockers[0]
    assert not (root / "output.txt").exists()


def test_failed_command_cannot_complete_or_retry_even_if_jev_approves(package):
    root, task, context = package
    state = new_tactical_state(phase())
    state.revealed_tool_ids = ["native.bash"]
    context.system_one = SimpleNamespace(evaluate_tactical_progress=lambda **k: SystemOneTacticalReview(complete=True, confident=True, confidence=.99, answers={}, retry_approved=True))
    selected = {"tool_id": "native.bash", "arguments": {"skill_id": "skill:sample", "command": "printf partial; printf failure >&2; exit 7"}, "acceptance_questions": question()}
    outcome = PhaseExecutor(action_selector=lambda *a: selected).run(task, state, context)
    assert outcome.status == PhaseStatus.BLOCKED and not state.completed_subgoal_ids
    assert len(state.actions) == 1
    evidence = state.evidence.entries[-1]
    assert evidence["status"] == "failed"
    assert evidence["result"]["exit_code"] == 7
    assert evidence["result"]["stdout"] == "partial" and evidence["result"]["stderr"] == "failure"
    assert evidence["progress_review"]["complete"] is False


def test_package_execution_without_project_and_authenticated_actor(package):
    _, task, context = package
    task.project_id = None
    assert invoke(task, context, command="scripts/hello.sh")["status"] == "success"
    task.user = ""
    result = invoke(task, context, command="true")
    assert "bash_skill_authenticated_actor_required" in result["error"]["message"]


def test_skill_timeout_retains_partial_evidence(package):
    _, task, context = package
    result = invoke(task, context, command="printf partial; sleep 30", timeout_seconds=.1)
    assert result["status"] == "failed"
    assert result["result"]["timed_out"] and not result["result"]["cancelled"]
    assert result["result"]["stdout"] == "partial"


def test_skill_cancellation_terminates_child_group(package):
    root, task, context = package
    started = time.monotonic()
    context.cancellation_checker = lambda: time.monotonic() - started > .2
    result = invoke(task, context, command="sh -c 'trap \"printf stopped > child-stopped; exit 0\" TERM; while :; do sleep 1; done' & wait")
    assert result["result"]["cancelled"]
    assert (root / "child-stopped").read_text() == "stopped"
