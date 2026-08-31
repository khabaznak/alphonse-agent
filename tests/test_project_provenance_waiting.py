from __future__ import annotations

import pytest

from alphonse.agent_v2.core.io import SQLiteOutboundStore
from alphonse.agent_v2.core.messages import CommunicationChannel, SQLiteMessageQueue
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.services.project_sessions import ProjectInboundRouter, SQLiteProjectSessionStore


def _router(tmp_path):
    queue = SQLiteMessageQueue(tmp_path / "queue.sqlite3")
    projects = ProjectStore(tmp_path / "queue.sqlite3")
    outbox = SQLiteOutboundStore(tmp_path / "queue.sqlite3")
    active: dict[str, str] = {}
    router = ProjectInboundRouter(
        channel=CommunicationChannel(queue),
        outbox=outbox,
        projects=projects,
        sessions=SQLiteProjectSessionStore(tmp_path / "queue.sqlite3"),
        managed_root=lambda user: tmp_path / "projects" / user,
        active_task_lookup=lambda: active,
    )
    return router, queue, projects, outbox, active


def test_new_conversation_creates_protected_home_project(tmp_path) -> None:
    router, _, projects, _, _ = _router(tmp_path)

    routed = router.ingest(prompt="Hello", user="alex", integration_id="desktop", provider_key="tui", channel_target="alex")

    home = projects.home_project("alex")
    assert home is not None and home.is_system_home
    assert routed.project_id == home.project_id
    with pytest.raises(ValueError, match="system_home_project_protected"):
        projects.archive_project(home.project_id, requester_user_id="alex")


def test_independent_message_receives_exact_turns_ahead_notice(tmp_path) -> None:
    router, queue, projects, outbox, active = _router(tmp_path)
    first = router.ingest(prompt="First", user="alex", integration_id="desktop", provider_key="tui", channel_target="alex")
    claimed = queue.claim_next()
    assert claimed is not None
    active.update({"user": "alex", "project_id": first.project_id, "routing_disposition": "pdca_task"})

    routed = router.ingest(prompt="Second", user="gaby", integration_id="desktop", provider_key="tui", channel_target="gaby")

    assert routed.disposition == "pdca_task"
    assert routed.turns_ahead == 1
    assert projects.home_project("gaby") is not None
    assert "after 1 task ahead" in outbox.list()[-1].message


def test_steering_and_event_prompts_do_not_become_waiting_turns(tmp_path) -> None:
    router, _, _, outbox, active = _router(tmp_path)
    first = router.ingest(prompt="First", user="alex", integration_id="desktop", provider_key="tui", channel_target="alex")
    active.update({"user": "alex", "project_id": first.project_id, "routing_disposition": "pdca_task"})

    steering = router.ingest(prompt="Add tests", user="alex", integration_id="desktop", provider_key="tui", channel_target="alex")
    event = router.ingest(prompt="Check sensors", user="alex", integration_id="system", provider_key="system", channel_target="alex", metadata={"source": "event_automation"})

    assert steering.disposition == "steering"
    assert event.disposition == "pdca_task"
    assert outbox.list() == []
