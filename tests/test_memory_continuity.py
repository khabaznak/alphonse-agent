"""Invented conversations exercise bounded continuity without provider calls."""
import json
from types import SimpleNamespace

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.memory.continuity import active_task_continuity
from alphonse.agent_v2.core.memory.formatting import join_context_sections
from alphonse.agent_v2.core.memory.transcript import conversation_transcript
from alphonse.agent_v2.core.questions import SQLiteQuestionStore


def record(store, role, content, index, **kwargs):
    return store.record(owner_user_id="alex", role=role, content=content,
                        source="desktop", source_message_id=str(index), **kwargs)


def test_complete_exchange_omission_and_exact_manifest():
    store = SQLiteConversationStore()
    old_user = record(store, "user", "Long old request " * 300, 1)
    old_reply = record(store, "assistant", "Long old reply " * 300, 2)
    task = TaskState(user="alex")
    task.append_conversation_message("alex", "try again")
    rendered = conversation_transcript(task, store, max_chars=300)
    assert "try again" in rendered and "Long old" not in rendered
    assert len(rendered) <= 300
    assert task.metadata["conversation_sources"] == []
    assert {x["source_id"] for x in task.metadata["conversation_projection"]["omitted"]} == {
        f"conversation:{old_user.event_id}", f"conversation:{old_reply.event_id}"}
    assert store.read(event_id=old_reply.event_id, requester_user_id="alex").content == old_reply.content


def test_recent_exchanges_preserve_retry_pronouns_and_channel_changes():
    store = SQLiteConversationStore()
    record(store, "user", "Show the two blue fixtures", 1, project_id="old", created_at="2026-10-16T23:59:00+00:00")
    reply = record(store, "assistant", "Fixture A and fixture B", 2, created_at="2026-10-17T00:01:00+00:00")
    for followup in ("try again", "yes, that one"):
        task = TaskState(user="alex", project_id="new", metadata={"channel": {"provider_key": "telegram"}})
        task.append_conversation_message("alex", followup)
        rendered = conversation_transcript(task, store)
        assert "Show the two blue fixtures" in rendered and reply.content in rendered and followup in rendered


def test_completed_history_does_not_accumulate_and_repeated_yes_remains_distinct():
    store = SQLiteConversationStore()
    for i in range(8):
        record(store, "user", "yes", 2 * i)
        record(store, "assistant", f"Reply {i}", 2 * i + 1)
    task = TaskState(user="alex")
    task.append_conversation_message("alex", "yes")
    rendered = conversation_transcript(task, store)
    assert "Reply 4" not in rendered and "Reply 5" in rendered and "Reply 7" in rendered
    assert rendered.count("yes") == 4
    assert len(task.metadata["conversation_sources"]) == 6
    assert len(task.metadata["conversation_projection"]["omitted"]) == 10


def test_multiline_steering_is_one_checkpoint_record():
    task = TaskState(user="alex")
    task.append_conversation_message("alex", "Keep both constraints:\n1. blue\n2. under 20")
    restored = TaskState.from_dict(json.loads(json.dumps(task.to_checkpoint_dict())))
    assert len(restored.conversation_records) == 1
    rendered = conversation_transcript(restored, None, max_chars=100)
    assert "1. blue" not in rendered and "2. under 20" not in rendered
    assert restored.metadata["conversation_projection"]["current_records_omitted"] == 1


def test_legacy_markdown_is_omitted_as_one_unit():
    task = TaskState(user="alex", recent_conversation_md="- alex: Long legacy request\n- Alphonse: " + "reply " * 200)
    task.append_conversation_message("alex", "New steering")
    text = conversation_transcript(task, None, max_chars=200)
    assert "New steering" in text and "Long legacy" not in text and "reply" not in text


def test_tiny_budgets_never_slice_any_record():
    task = TaskState(user="alex")
    task.append_conversation_message("alex", "A whole message")
    for budget in range(100):
        text = conversation_transcript(task, None, max_chars=budget)
        assert len(text) <= budget
        if "A whole" in text:
            assert "A whole message" in text


def test_question_answer_and_constraints_survive_restart_and_compaction(tmp_path):
    path = tmp_path / "questions.sqlite"
    store = SQLiteQuestionStore(path)
    task = TaskState(task_id="task-one", user="alex", goal="Choose a fixture", acceptance_criteria_md="- under 20")
    question = store.create_question(task=task, question="Fixture A or B?", kind="open_text")
    resumed = SQLiteQuestionStore(path).route_answer(respondent_user_id="alex", text="B", question_id=question.question_id).resumed_task
    resumed.plan_json = "- (none)"
    resumed.memory_session_id = "tomorrow"
    store.save_task_checkpoint(resumed)
    restored = SQLiteQuestionStore(path).load_task_checkpoint("task-one")
    context = active_task_continuity(restored)
    assert "Choose a fixture" in context and "under 20" in context
    assert "Fixture A or B?" in context and '"text": "B"' in context
    assert '"status": "answered"' in context and "authorization" in context


def test_multiple_tasks_require_explicit_answer_target_and_owner():
    store = SQLiteQuestionStore()
    questions = [store.create_question(task=TaskState(task_id=f"task-{i}", user="alex", goal=f"Goal {i}"),
                                       question=f"Question {i}?") for i in range(2)]
    assert store.route_answer(respondent_user_id="alex", text="yes").ambiguous
    assert not store.route_answer(respondent_user_id="sam", text="yes", question_id=questions[0].question_id).handled
    resumed = store.route_answer(respondent_user_id="alex", text="yes", question_id=questions[1].question_id).resumed_task
    assert resumed.task_id == "task-1"
    assert "Goal 0" not in active_task_continuity(resumed)
    assert store.get_question(questions[0].question_id).status == "pending"


def test_pending_question_and_failed_evidence_do_not_become_success():
    store = SQLiteQuestionStore()
    task = TaskState(task_id="task-one", user="alex", goal="Read the fixture")
    task.evidence_journal = [{"evidence_ref": "tool-call:read", "status": "failed", "result": "not found"}]
    store.create_question(task=task, question="Which fixture?")
    rendered = active_task_continuity(task)
    assert '"status": "pending"' in rendered and '"status": "failed"' in rendered
    assert "task-evidence:task-one:0" in rendered and '"status": "success"' not in rendered
    fresh = TaskState(task_id="new-task", user="alex", goal="Read a new fixture")
    assert "Which fixture?" not in active_task_continuity(fresh)


def test_overflow_and_unavailable_history_are_explicit():
    task = TaskState(user="alex", goal="x" * 10000, acceptance_criteria_md="- preserve this")
    assert "preserve this" in active_task_continuity(task, max_chars=500)
    assert task.metadata["active_continuity_projection"]["reserved_overflow"]
    def unavailable(**kwargs):
        raise OSError("offline")
    conversation_transcript(task, SimpleNamespace(list_recent=unavailable))
    assert task.metadata["conversation_projection"]["history_unavailable"]


def test_downstream_section_budget_never_cuts_a_question():
    assert join_context_sections(["Goal", "Question?\nAnswer."], 12) == "Goal"
