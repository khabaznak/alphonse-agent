from __future__ import annotations

from types import SimpleNamespace

from alphonse.agent_v2.core.io import SQLiteOutboundStore
from alphonse.agent_v2.core.memory.ledger import LedgerMemory, estimate_tokens
from alphonse.agent_v2.core.messages import CommunicationChannel, InMemoryMessageQueue
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.memory_sessions import MemorySessionBindingKey, SQLiteMemorySessionStore
from alphonse.agent_v2.memory_settings import MemorySettings, SQLiteMemorySettingsStore
from alphonse.agent_v2.services.project_sessions import ProjectInboundRouter, ProjectSessionKey, SQLiteProjectSessionStore
from alphonse.agent_v2.core.intelligence.task_state import TaskState


def _memory(tmp_path, project_root, *, budget=4096, summarizer=None):
    settings = SQLiteMemorySettingsStore(tmp_path / "settings.sqlite3")
    settings.save(MemorySettings(memory_context_token_budget=budget))
    return LedgerMemory(
        users_root=lambda: tmp_path / "users",
        settings_store=settings,
        summarizer=summarizer,
        project_root_provider=lambda _project_id: project_root,
    )


def _task(session_id: str, *, goal: str = "hello"):
    return SimpleNamespace(user="alex", project_id="project", memory_session_id=session_id, task_id="task", message_id="message", goal=goal, metadata={}, outcome=None, status="completed")


def test_bounded_context_never_loads_full_session_ledger(tmp_path) -> None:
    root = tmp_path / "project"
    memory = _memory(tmp_path, root, budget=512)
    task = _task("session-one", goal="x" * 20_000)

    context = memory.start_task(task)

    assert estimate_tokens(context) <= 512
    assert task.metadata["memory_context_token_budget"] == 512
    assert task.metadata["memory_context_truncated"] is True
    assert (root / ".alphonse" / "memory" / "sessions" / "session-one" / "ledger-0001.md").stat().st_size > len(context)


def test_session_close_promotes_project_memory_atomically(tmp_path) -> None:
    root = tmp_path / "project"
    memory = _memory(tmp_path, root, summarizer=lambda source: "stable preference and unresolved commitment")
    task = _task("session-one")
    memory.start_task(task)
    memory.event(task, "Outcome", {"status": "success"})

    summary = memory.close_session(user_id="alex", project_id="project", session_id="session-one")

    assert "stable preference" in summary
    assert "stable preference" in (root / ".alphonse" / "memory" / "project-summary.md").read_text()
    assert (root / ".alphonse" / "memory" / "sessions" / "session-one" / "summary.md").exists()


def test_router_stamps_selected_session_before_queueing_and_supports_commands(tmp_path) -> None:
    queue = InMemoryMessageQueue()
    projects = ProjectStore(":memory:")
    project = projects.create_project(name="Alpha", root_path=str(tmp_path / "alpha"), owner_user_id="alex")
    sessions = SQLiteMemorySessionStore(":memory:")
    memory = _memory(tmp_path, tmp_path / "alpha", summarizer=lambda source: "durable")
    router = ProjectInboundRouter(
        channel=CommunicationChannel(queue), outbox=SQLiteOutboundStore(), projects=projects,
        sessions=SQLiteProjectSessionStore(":memory:"), memory_sessions=sessions, memory=memory,
    )
    key = ProjectSessionKey("alex", "tui", "alex")
    router.select_project(key, project.project_id)

    router.ingest(prompt="/session new Research", user="alex", integration_id="tui", provider_key="tui", channel_target="alex")
    selected = sessions.get_binding(MemorySessionBindingKey("alex", "tui", "alex", "", project.project_id))
    assert selected is not None and selected.name == "Research"

    routed = router.ingest(prompt="Investigate", user="alex", integration_id="tui", provider_key="tui", channel_target="alex")
    router.ingest(prompt="/session new Other", user="alex", integration_id="tui", provider_key="tui", channel_target="alex")

    assert routed.queued is not None
    assert routed.queued.message.memory_session_id == selected.session_id
    assert routed.queued.message.metadata["memory_session_id"] == selected.session_id


def test_legacy_ledgers_are_archived_and_summarized_once(tmp_path) -> None:
    root = tmp_path / "project"
    legacy = root / ".alphonse" / "memory" / "ledger-0001.md"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("important old decision " * 500)
    memory = _memory(tmp_path, root, budget=512, summarizer=lambda source: "old decision")

    assert memory.migrate_project_legacy(user_id="alex", project_id="project") == 1
    assert memory.migrate_project_legacy(user_id="alex", project_id="project") == 0
    assert (legacy.parent / "legacy" / "ledger-0001.md").exists()
    assert "old decision" in (legacy.parent / "project-summary.md").read_text()


def test_memory_session_store_keeps_bindings_per_project_and_rejects_closed_sessions() -> None:
    store = SQLiteMemorySessionStore(":memory:")
    first = store.create(project_id="one", name="Research", created_by_user_id="alex")
    second = store.create(project_id="two", name="Research", created_by_user_id="alex")
    key_one = MemorySessionBindingKey("alex", "tui", "alex", "", "one")
    key_two = MemorySessionBindingKey("alex", "tui", "alex", "", "two")
    store.bind(key_one, first); store.bind(key_two, second)

    assert store.get_binding(key_one) == first
    assert store.get_binding(key_two) == second
    store.close(first.session_id)
    assert store.get_binding(key_one) is None
    assert store.resolve_open("two", "Research") == second


def test_operational_task_prompt_omits_memory_context() -> None:
    task = TaskState(goal="continue", conversation_history_md="expensive historical secret")

    assert "expensive historical secret" in task.to_markdown_prompt()
    assert "expensive historical secret" not in task.to_markdown_prompt(include_memory=False)


def test_memory_setting_persists_configurable_context_budget(tmp_path) -> None:
    path = tmp_path / "settings.sqlite3"
    store = SQLiteMemorySettingsStore(path)
    store.save(MemorySettings(memory_context_token_budget=8192))

    assert SQLiteMemorySettingsStore(path).get().memory_context_token_budget == 8192
