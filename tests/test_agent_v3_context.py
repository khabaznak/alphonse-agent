from types import SimpleNamespace

from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.context import _recent_conversation
from alphonse.agent_v2.core.tools.registry.native.memory import execute_search_memory


from alphonse.agent_v2.conversations import ConversationEvent


def event(**kwargs):
    values = dict(sequence=1, event_id="test-event", owner_user_id="alex", project_id="",
                  memory_session_id="", role="user", content="", source="test", source_message_id="",
                  created_at="2026-10-11", recorded_at="2026-10-11", observed_at="2026-10-11")
    values.update(kwargs)
    return ConversationEvent(**values)


class _ConversationStore:
    def list_recent(self, *, owner_user_id: str, limit: int):
        assert owner_user_id == "alex"
        assert limit == 30
        return [
            event(event_id="earlier", role="user", content="Earlier unassociated question", project_id=""),
            event(event_id="reply", role="assistant", content="Reply from archived project task", project_id="old-project"),
            event(role="user", content="Current project request", project_id="current-project"),
        ]


def test_recent_conversation_keeps_mixed_and_historical_project_events() -> None:
    context = SimpleNamespace(conversation_store=_ConversationStore())

    rendered = _recent_conversation(TaskState(user="alex"), context)

    assert "Earlier unassociated question" in rendered
    assert "Reply from archived project task" in rendered
    assert "Current project request" in rendered


def test_memory_recall_searches_user_history_without_project_scope() -> None:
    class _SearchStore:
        def search(self, *, owner_user_id: str, query: str, limit: int):
            assert owner_user_id == "alex"
            assert query == "Adrian"
            assert limit == 8
            return [event(created_at="2026-10-11", project_id="health", role="assistant", content="Adrian's appointment is Friday.")]

    class _Memory:
        def search(self, **_kwargs):
            self.searched = True
            return ""

    context = SimpleNamespace(
        task=TaskState(user="alex"),
        conversation_store=_SearchStore(),
        memory=_Memory(),
    )

    result = execute_search_memory({"query": "Adrian"}, context=context)

    assert context.memory.searched
    assert result["project_id"] == ""
    assert "appointment is Friday" in result["matches_markdown"]


def test_transcript_preserves_current_message_and_labels_legacy_imports() -> None:
    from alphonse.agent_v2.core.memory.transcript import conversation_transcript

    class Store:
        def list_recent(self, **_kwargs):
            return [event(role="user", content="Old imported question", source="ledger", provenance="legacy_import")]

    task = TaskState(user="alex")
    task.append_conversation_message("alex", 'Check the "study" temperature')
    rendered = conversation_transcript(task, Store())
    assert "legacy ledger import; unverified provenance" in rendered
    assert 'Check the \\"study\\" temperature' in rendered


def test_transcript_keeps_newest_messages_when_history_exceeds_budget() -> None:
    from alphonse.agent_v2.core.memory.transcript import conversation_transcript

    class Store:
        def list_recent(self, **_kwargs):
            return [event(role="assistant", content="old " * 5000)]

    task = TaskState(user="alex")
    task.append_conversation_message("alex", "Current temperature request")
    rendered = conversation_transcript(task, Store(), max_chars=500)
    assert len(rendered) <= 500
    assert "Current temperature request" in rendered
    assert "earlier conversation omitted" in rendered


def test_transcript_does_not_duplicate_current_recorded_user_message() -> None:
    from alphonse.agent_v2.core.memory.transcript import conversation_transcript

    class Store:
        def list_recent(self, **_kwargs):
            return [event(role="user", content='Check the "study" temperature')]

    task = TaskState(user="alex")
    task.append_conversation_message("alex", 'Check the "study" temperature')
    assert conversation_transcript(task, Store()).count("temperature") == 1


def test_decision_context_keeps_memory_and_conversation_separate() -> None:
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.v3.context import task_context_for_decision
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue

    task = TaskState(user="alex", goal="Current request", memory_context_md="Known project file: journal.md")
    task.append_conversation_message("alex", "Current request")
    rendered = task_context_for_decision(task, CoreLoopContext(messages=InMemoryMessageQueue()))
    assert "Current request" in rendered
    assert "Known project file: journal.md" in rendered
    assert "Durable household memory (person/audience scoped; project association is optional):" in rendered
