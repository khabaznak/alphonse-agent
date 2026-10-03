from __future__ import annotations

import pytest

from alphonse.agent_v2.core.io import SQLiteOutboundStore
from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
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
    home = projects.home_project("alex")
    assert home is not None and home.is_system_home
    assert queue.dequeue().message.project_id == home.project_id
    assert queue.dequeue().message.project_id == home.project_id
    assert router.active_project(tui) == home
    assert router.active_project(thread) == home


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


def test_project_context_mutation_requires_owner_and_unknown_slash_bypasses_capd(tmp_path) -> None:
    router, queue, outbox, projects = _router()
    shared = projects.create_project(name="Shared", root_path=str(tmp_path / "shared"), owner_user_id="alex", visibility="shared")
    gaby = ProjectSessionKey("gaby", "telegram-home", "chat")
    router.select_project(gaby, shared.project_id)

    denied = router.ingest(prompt="/project context set private note", user="gaby", integration_id="telegram-home", provider_key="telegram", channel_target="chat")
    normal = router.ingest(prompt="/agent-config", user="gaby", integration_id="telegram-home", provider_key="telegram", channel_target="chat")

    assert denied.handled_command
    assert normal.handled_command
    assert "Only the project owner" in outbox.list()[-2].message
    assert "Unsupported command: /agent-config." in outbox.list()[-1].message
    assert queue.size() == 0


def test_project_commands_do_not_grant_admins_access_to_other_users_private_projects(tmp_path) -> None:
    queue = InMemoryMessageQueue()
    outbox = SQLiteOutboundStore()
    projects = ProjectStore(":memory:")
    router = ProjectInboundRouter(
        channel=CommunicationChannel(queue),
        outbox=outbox,
        projects=projects,
        sessions=SQLiteProjectSessionStore(":memory:"),
        is_admin=lambda user: user == "alex",
    )
    owned = projects.create_project(name="Alex private", root_path=str(tmp_path / "alex"), owner_user_id="alex")
    private = projects.create_project(name="Gaby private", root_path=str(tmp_path / "gaby"), owner_user_id="gaby")
    shared = projects.create_project(name="Shared", root_path=str(tmp_path / "shared"), owner_user_id="gaby", visibility="shared")

    router.ingest(prompt="/projects", user="alex", integration_id="telegram-home", provider_key="telegram", channel_target="chat")
    listed = outbox.list()[-1].message
    router.ingest(prompt=f"/project {private.project_id}", user="alex", integration_id="telegram-home", provider_key="telegram", channel_target="chat")
    denied = outbox.list()[-1].message
    router.ingest(prompt=f"/project {shared.project_id}", user="alex", integration_id="telegram-home", provider_key="telegram", channel_target="chat")

    assert owned.project_id in listed
    assert shared.project_id in listed
    assert private.project_id not in listed
    assert "Project not found or not visible" in denied
    assert f"Active project: {shared.name}." in outbox.list()[-1].message


@pytest.mark.parametrize(
    ("answer_integration", "answer_provider", "answer_target"),
    [("telegram-home", "telegram", "chat-1"), ("desktop", "tui", "alex")],
)
def test_cross_channel_reply_resumes_pending_question_with_original_task_context(
    tmp_path, answer_integration, answer_provider, answer_target,
) -> None:
    queue = InMemoryMessageQueue()
    projects = ProjectStore(":memory:")
    project = projects.create_project(name="Home", root_path=str(tmp_path / "home"), owner_user_id="alex")
    questions = SQLiteQuestionStore()
    router = ProjectInboundRouter(
        channel=CommunicationChannel(queue),
        outbox=SQLiteOutboundStore(),
        projects=projects,
        sessions=SQLiteProjectSessionStore(":memory:"),
        question_store=questions,
    )
    task = TaskState(
        task_id="reminder-task", user="alex", project_id=project.project_id,
        memory_session_id="home-session", goal="Remind me at 9 to bring yogurt.",
        metadata={"channel": {
            "integration_id": "telegram-home", "provider_key": "telegram",
            "channel_target": "chat-1", "provider_user_id": "alex-telegram",
            "alphonse_user_id": "alex",
        }},
    )
    task.append_conversation_message("Alphonse", "¿Quieres que te recuerde hoy o mañana?")
    question = questions.create_question(task=task, question="¿Quieres que te recuerde hoy o mañana?")

    routed = router.ingest(
        prompt="Hoy", user="alex", integration_id=answer_integration, provider_key=answer_provider,
        provider_user_id="alex-telegram", channel_target=answer_target, provider_message_id="answer-2",
    )

    assert routed.queued is not None
    assert routed.disposition == "correlated_response"
    assert routed.queued.message.project_id == project.project_id
    assert routed.queued.message.memory_session_id == "home-session"
    resumed = TaskState.from_queued_message(routed.queued)
    assert resumed.metadata["channel"]["integration_id"] == "telegram-home"
    assert 'Alphonse: "¿Quieres que te recuerde hoy o mañana?"' in resumed.recent_conversation_md
    assert 'alex: "Hoy"' in resumed.recent_conversation_md
    assert questions.get_question(question.question_id).status == "answered"
