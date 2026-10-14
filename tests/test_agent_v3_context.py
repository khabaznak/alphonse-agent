from types import SimpleNamespace

from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.context import _recent_conversation
from alphonse.agent_v2.core.tools.registry.native.memory import execute_search_memory


class _ConversationStore:
    def list_recent(self, *, owner_user_id: str, limit: int):
        assert owner_user_id == "alex"
        assert limit == 30
        return [
            SimpleNamespace(role="user", content="Earlier unassociated question", project_id=""),
            SimpleNamespace(role="assistant", content="Reply from archived project task", project_id="old-project"),
            SimpleNamespace(role="user", content="Current project request", project_id="current-project"),
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
            return [SimpleNamespace(created_at="2026-10-11", project_id="health", role="assistant", content="Adrian's appointment is Friday.")]

    class _Memory:
        def search(self, **_kwargs):
            raise AssertionError("conversation history should be searched first")

    context = SimpleNamespace(
        task=TaskState(user="alex"),
        conversation_store=_SearchStore(),
        memory=_Memory(),
    )

    result = execute_search_memory({"query": "Adrian"}, context=context)

    assert result["project_id"] == ""
    assert "appointment is Friday" in result["matches_markdown"]
