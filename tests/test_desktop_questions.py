"""Synthetic Desktop -> IPC -> persisted question -> continuation regressions."""
import pytest

from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def desktop(tmp_path):
    users = V2UserStore(":memory:")
    admin = users.onboard(display_name="Fixture owner", users_root=tmp_path / "users")
    member = users.create_user(display_name="Fixture member")
    store = SQLiteQuestionStore(str(tmp_path / "questions.db"))
    runtime = build_runtime_host(user=admin.user_id, user_store=users,
                                 question_store=store, schedule_store=ScheduledTaskStore(":memory:"))
    daemon = V2Daemon(runtime)
    def request(method, **params):
        return daemon.ipc._dispatch({"method": method, "params": params})
    return daemon, store, admin.user_id, member.user_id, request


def test_desktop_rejects_wrong_respondent_without_impersonating_owner(desktop):
    daemon, store, owner, member, request = desktop
    question = store.create_question(task=TaskState(task_id="fixture", user=owner, goal="Fixture"), question="Which fixture?")
    with pytest.raises(PermissionError):
        request("answer_question", user=member, question_id=question.question_id, payload={"text": "Wrong actor"})
    assert store.get_question(question.question_id).status == "pending"
    assert daemon.runtime.queue.size() == 0


def test_desktop_cancel_requires_respondent(desktop):
    _, store, owner, member, request = desktop
    question = store.create_question(task=TaskState(task_id="fixture", user=owner, goal="Fixture"), question="Which fixture?")
    with pytest.raises(PermissionError):
        request("cancel_question", user=member, question_id=question.question_id)
    assert store.get_question(question.question_id).status == "pending"


def poll(request, owner, client="fixture"):
    return request("desktop_poll", user=owner, client_id=client,
                   client_capabilities={"supportedCatalogIds": ["alphonse.desktop.catalog.v2"]})


@pytest.mark.parametrize("kind,payload,choices", [
    ("open_text", {"text": "A fixture answer"}, []),
    ("yes_no", {"answer": False}, []),
    ("single_choice", {"choice_id": "b"}, [{"id": "a", "label": "Alpha"}, {"id": "b", "label": "Beta"}]),
    ("multi_choice", {"choice_ids": ["a", "b"]}, [{"id": "a", "label": "Alpha"}, {"id": "b", "label": "Beta"}]),
    ("datetime", {"datetime": "2026-11-02T14:30"}, []),
])
def test_question_answer_queues_exact_checkpoint_once(desktop, kind, payload, choices):
    daemon, store, owner, _, request = desktop
    task = TaskState(task_id="fixture", user=owner, goal="Continue fixture", correlation_id="fixture-correlation")
    q = store.create_question(task=task, question="Fixture question", kind=kind, choices=choices)
    other = store.create_question(task=TaskState(task_id="other", user=owner, goal="Other fixture"), question="Other question")
    assert len(poll(request, owner)["questions"]) == 2
    invalid = request("answer_question", user=owner, question_id=q.question_id, payload={"choice_ids": ["bad", "bad"]})
    assert invalid["invalid"]
    assert store.get_question(q.question_id).status == "pending"
    result = request("answer_question", user=owner, question_id=q.question_id, payload=payload)
    assert result["handled"] and not result["invalid"]
    assert result["question_id"] == q.question_id and result["task_id"] == "fixture"
    assert store.get_question(q.question_id).status == "answered"
    queued = daemon.runtime.queue.peek()
    assert queued.message.metadata["answered_question_id"] == q.question_id
    resumed = queued.message.metadata["task_state"]
    assert resumed["task_id"] == "fixture"
    assert resumed["metadata"]["question_interrupt"]["answer"] == result["answer"]
    assert not request("answer_question", user=owner, question_id=q.question_id, payload=payload)["handled"]
    assert daemon.runtime.queue.size() == 1
    after = poll(request, owner)
    assert [item["question_id"] for item in after["questions"]] == [other.question_id]
    assert f"question:{q.question_id}" in str(after["ui_events"])
    assert "deleteSurface" in str(after["ui_events"])
    with pytest.raises(ValueError, match="a2ui_question_not_available"):
        request("a2ui_action", user=owner, client_id="fixture", surface_id=f"question:{q.question_id}",
                source_component_id="submit", action_name="answer_question", context={"question_id": q.question_id}, data_model={"answer": payload})


@pytest.mark.parametrize("terminal", ["expired", "cancelled", "answered"])
def test_poll_reconciles_terminal_questions_and_reconnect(desktop, terminal):
    daemon, store, owner, _, request = desktop
    q = store.create_question(task=TaskState(task_id="fixture", user=owner, goal="Fixture"), question="Fixture?")
    poll(request, owner)
    if terminal == "expired":
        store.expire_questions(now="9999-01-01T00:00:00+00:00")
    elif terminal == "cancelled":
        assert request("cancel_question", user=owner, question_id=q.question_id)["cancelled"]
    else:
        # The store is shared with the other channels; no Desktop event needed.
        assert store.route_answer(respondent_user_id=owner, question_id=q.question_id, text="Elsewhere").handled
    after = poll(request, owner)
    assert after["questions"] == []
    assert "deleteSurface" in str(after["ui_events"])
    assert poll(request, owner, client="reconnected")["questions"] == []
    assert not request("answer_question", user=owner, question_id=q.question_id, payload={"text": "Stale"})["handled"]
    assert store.get_question(q.question_id).status == terminal


def test_desktop_missing_actor_and_stale_context_do_not_authorize(desktop):
    daemon, store, owner, member, request = desktop
    q = store.create_question(task=TaskState(task_id="fixture", user=owner, goal="Fixture"), question="Fixture?",
                              delivery_metadata={"question_context": {"text": "Review this fixture", "artifact_ids": ["artifact.missing"]}})
    for method in ["answer_question", "cancel_question", "pending_questions"]:
        with pytest.raises(PermissionError):
            request(method, question_id=q.question_id, payload={"text": "Missing actor"})
    with pytest.raises(PermissionError):
        poll(request, member)
    current = poll(request, owner)["questions"][0]
    assert current["context"] == {"text": "Review this fixture", "artifacts": [{"status": "unavailable"}]}
    assert current["metadata"] == {}
    # A yes/no question supplies input, never an approval or a tool permission.
    assert "authorization" not in current


def test_question_context_checks_current_artifact_access_and_availability(desktop, tmp_path):
    daemon, store, owner, member, request = desktop
    artifact = daemon.runtime.artifact_store.register(
        artifact_id="artifact.fixture", name="Disposable resource", description="Fixture metadata only",
        project_id="", owner_user_id=owner, entrypoint_path="fixture.py", argument_schema={"type": "object"},
        storage_kind="shared", skill_id="skill.missing", shared_with_user_ids=(member,),
    )
    q = store.create_question(task=TaskState(task_id="delegated", user=owner, goal="Private task"),
        respondent_user_id=member, question="Review the resource?", delivery_metadata={"question_context": {"artifact_ids": [artifact.artifact_id]}})
    context = daemon._question_context(store.get_question(q.question_id), member)
    assert context["artifacts"][0]["name"] == "Disposable resource"
    assert context["artifacts"][0]["status"] == "unavailable"  # Missing installed skill, no fabricated preview.
    assert "entrypoint_path" not in str(context)
    from dataclasses import replace
    daemon.runtime.artifact_store.update_location(replace(artifact, shared_with_user_ids=()))
    assert daemon._question_context(store.get_question(q.question_id), member)["artifacts"] == [{"status": "unavailable"}]
    # Even the local administrator cannot answer a delegated member's question.
    with pytest.raises(PermissionError):
        request("answer_question", user=owner, question_id=q.question_id, payload={"text": "Impersonate"})
    assert store.get_question(q.question_id).status == "pending"


def test_missing_checkpoint_does_not_report_answer_success(desktop):
    _, store, owner, _, request = desktop
    q = store.create_question(task=TaskState(task_id="fixture", user=owner, goal="Fixture"), question="Fixture?")
    with store._connect() as conn:
        conn.execute("DELETE FROM v2_task_checkpoints WHERE task_id=?", (q.task_id,))
    result = request("answer_question", user=owner, question_id=q.question_id, payload={"text": "Answer"})
    assert result["invalid"] and not result["answer"]
    assert store.get_question(q.question_id).status == "pending"


def test_ipc_answer_runs_resumed_continuation_once(desktop, tmp_path):
    """The real daemon consumes the resumed checkpoint, with a deterministic fixture processor."""
    from alphonse.agent_v2.core.core import ProcessingResult, ProcessingStatus, StateSnapshot, LoopStepStatus
    from alphonse.agent_v2.core.state import reset_state
    reset_state()
    daemon, store, owner, _, request = desktop
    output = tmp_path / "verified-continuation.txt"
    class FixtureProcessor:
        calls = 0
        def process(self, task, context):
            self.calls += 1
            answer = task.metadata["question_interrupt"]
            assert task.task_id == "resume-fixture" and answer["status"] == "answered"
            assert answer["respondent_user_id"] == owner
            output.write_text(answer["answer"]["text"])
            task.status = "completed"
            task.outcome = {"status": "success", "verified_output": str(output)}
            return ProcessingResult(status=ProcessingStatus.COMPLETED, snapshot=StateSnapshot(metadata={"task_state": task.to_dict()}))
    processor = FixtureProcessor()
    daemon.runtime.core.intelligence = processor
    q = store.create_question(task=TaskState(task_id="resume-fixture", user=owner, goal="Write disposable answer"), question="What should the fixture write?")
    request("answer_question", user=owner, question_id=q.question_id, payload={"text": "Verified answer"})
    assert daemon.run_once().status == LoopStepStatus.PROCESSED
    assert output.read_text() == "Verified answer"
    assert processor.calls == 1 and daemon.runtime.queue.size() == 0
    request("answer_question", user=owner, question_id=q.question_id, payload={"text": "Duplicate"})
    daemon.run_once()
    assert processor.calls == 1 and output.read_text() == "Verified answer"


def test_native_question_context_schema_persists_explicit_context(desktop):
    from alphonse.agent_v2.core.core import ToolExecutionContext
    from alphonse.agent_v2.core.tools.registry.native.ask_question import execute_ask_question
    from jsonschema import ValidationError
    daemon, store, owner, _, request = desktop
    context = ToolExecutionContext(task=TaskState(task_id="native-context", user=owner, goal="Fixture"), messages=daemon.runtime.queue, question_store=store)
    execute_ask_question({"question":"Review?", "question_kind":"open_text", "question_context":{"text":"Fixture context", "artifact_ids":["artifact.missing"]}}, context=context)
    assert poll(request, owner)["questions"][0]["context"]["text"] == "Fixture context"
    with pytest.raises(ValidationError):
        execute_ask_question({"question":"Review?", "question_kind":"open_text", "question_context":{"html":"<script>bad</script>"}}, context=context)


def test_question_cannot_resurrect_terminal_checkpoint(desktop):
    daemon, store, owner, _, request = desktop
    q = store.create_question(task=TaskState(task_id="terminal", user=owner, goal="Fixture"), question="Continue?")
    store.mark_task_checkpoint_terminal(q.task_id, status="cancelled")
    result = request("answer_question", user=owner, question_id=q.question_id, payload={"text": "Stale"})
    assert not result["handled"] and daemon.runtime.queue.size() == 0
    with store._connect() as conn:
        assert conn.execute("SELECT status FROM v2_task_checkpoints WHERE task_id=?", (q.task_id,)).fetchone()[0] == "cancelled"
