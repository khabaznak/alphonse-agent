from __future__ import annotations

from alphonse.agent_v2.core.io import SQLiteOutboundStore
from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.services.project_sessions import ProjectInboundRouter
from alphonse.agent_v2.services.project_sessions import ProjectSessionKey
from alphonse.agent_v2.services.project_sessions import SQLiteProjectSessionStore


def _router() -> tuple[ProjectInboundRouter, InMemoryMessageQueue, SQLiteOutboundStore, ProjectStore]:
    queue = InMemoryMessageQueue()
    outbox = SQLiteOutboundStore()
    projects = ProjectStore(":memory:")
    return (
        ProjectInboundRouter(
            channel=CommunicationChannel(queue),
            outbox=outbox,
            projects=projects,
            sessions=SQLiteProjectSessionStore(":memory:"),
        ),
        queue,
        outbox,
        projects,
    )


def test_project_session_isolated_by_user_channel_and_thread(tmp_path) -> None:
    router, queue, _, projects = _router()
    project = projects.create_project(name="Exercise", root_path=str(tmp_path / "exercise"), owner_user_id="alex")
    telegram = ProjectSessionKey("alex", "telegram-home", "chat-1")
    tui = ProjectSessionKey("alex", "tui", "alex")
    thread = ProjectSessionKey("alex", "telegram-home", "chat-1", "topic-2")

    router.select_project(telegram, project.project_id)
    router.ingest(prompt="Routine?", user="alex", integration_id="telegram-home", provider_key="telegram", channel_target="chat-1")
    router.ingest(prompt="Routine?", user="alex", integration_id="tui", provider_key="tui", channel_target="alex")
    router.ingest(prompt="Routine?", user="alex", integration_id="telegram-home", provider_key="telegram", channel_target="chat-1", thread_id="topic-2")

    assert queue.dequeue().message.project_id == project.project_id
    assert queue.dequeue().message.project_id == ""
    assert queue.dequeue().message.project_id == ""
    assert router.active_project(tui) is None
    assert router.active_project(thread) is None


def test_project_commands_are_deterministic_and_do_not_queue_capd_work(tmp_path, monkeypatch) -> None:
    router, queue, outbox, projects = _router()
    project = projects.create_project(name="Exercise", root_path=str(tmp_path / "exercise"), owner_user_id="alex")
    monkeypatch.setenv("ALPHONSE_V2_MANAGED_PROJECTS_DIR", str(tmp_path / "managed"))

    selected = router.ingest(prompt="/project Exercise", user="alex", integration_id="telegram-home", provider_key="telegram", channel_target="chat")
    listed = router.ingest(prompt="/projects", user="alex", integration_id="telegram-home", provider_key="telegram", channel_target="chat")
    created = router.ingest(prompt="/project create Medicines", user="alex", integration_id="telegram-home", provider_key="telegram", channel_target="chat")

    assert selected.handled_command and listed.handled_command and created.handled_command
    assert queue.size() == 0
    active = router.active_project(ProjectSessionKey("alex", "telegram-home", "chat"))
    assert active is not None and active.name == "Medicines" and active.visibility == "private"
    assert active.root_path.startswith(str(tmp_path / "managed"))
    messages = [message.message for message in outbox.list()]
    assert any("Active project: Exercise." in message for message in messages)
    assert any(project.project_id in message for message in messages)


def test_project_context_mutation_requires_owner_and_unknown_slash_reaches_capd(tmp_path) -> None:
    router, queue, outbox, projects = _router()
    shared = projects.create_project(name="Shared", root_path=str(tmp_path / "shared"), owner_user_id="alex", visibility="shared")
    gaby = ProjectSessionKey("gaby", "telegram-home", "chat")
    router.select_project(gaby, shared.project_id)

    denied = router.ingest(prompt="/project context set private note", user="gaby", integration_id="telegram-home", provider_key="telegram", channel_target="chat")
    normal = router.ingest(prompt="/agent-config", user="gaby", integration_id="telegram-home", provider_key="telegram", channel_target="chat")

    assert denied.handled_command
    assert "Only the project owner" in outbox.list()[-1].message
    assert normal.queued is not None
    assert queue.dequeue().message.project_id == shared.project_id
