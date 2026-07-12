from __future__ import annotations

from alphonse.agent_v2.core.inference import InferencePurpose
from alphonse.agent_v2.core.inference import InferenceRouter
from alphonse.agent_v2.core.inference import ModelProfile
from alphonse.agent_v2.core.inference import StubInferenceProvider
from alphonse.agent_v2.core.core import CoreActivityEvent
from alphonse.agent_v2.core.core import CoreUiEvent
from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.io import ChannelAddress
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.agent_config import AgentConfigStore
from alphonse.agent_v2.agent_config import PHILOSOPHY_FILE
from alphonse.agent_v2.services.project_sessions import SQLiteProjectSessionStore
from alphonse.agent_v2.runtime import build_runtime_host


def _router() -> InferenceRouter:
    return InferenceRouter(
        provider=StubInferenceProvider(
            markdown_by_purpose={
                InferencePurpose.ACCEPTANCE_CRITERIA: "1.- [ ] response delivered",
                InferencePurpose.CRITERIA_REVIEW: "1.- [x] response delivered",
            },
            tool_call={
                "tool_id": "native.respond",
                "tool_name": "respond",
                "arguments": {"message": "Hello from daemon."},
            },
        ),
        default_profile=ModelProfile(provider="stub", model="stub", profile_id="stub"),
    )


def test_daemon_ipc_dispatches_ping_status_and_queue_message() -> None:
    runtime = build_runtime_host(
        inference=_router(),
        schedule_store=ScheduledTaskStore(":memory:"),
    )
    daemon = V2Daemon(runtime)
    assert daemon.ipc._dispatch({"method": "ping"})["status"] == "ready"
    queued = daemon.ipc._dispatch({"method": "queue_message", "params": {"prompt": "hello", "user": "alex"}})
    assert queued["message_id"]
    assert daemon.ipc._dispatch({"method": "status"})["queue_size"] == 1
    daemon.run_once()
    assert daemon.ipc._dispatch({"method": "status"})["queue_size"] == 0


def test_daemon_ipc_exposes_inference_configuration() -> None:
    runtime = build_runtime_host(schedule_store=ScheduledTaskStore(":memory:"))
    daemon = V2Daemon(runtime)

    settings = daemon.ipc._dispatch({"method": "inference_settings"})["settings"]
    providers = daemon.ipc._dispatch({"method": "inference_providers"})["providers"]

    assert settings["provider_key"] == "openai_codex"
    assert providers[0]["provider_key"] == "openai_codex"


def test_model_settings_request_uses_validation_timeout(monkeypatch) -> None:
    client = __import__("alphonse.agent_v2.ipc", fromlist=["V2DaemonClient"]).V2DaemonClient("/tmp/test.sock", timeout_sec=2)
    captured = {}

    def fake_request(self, method, **params):
        captured["timeout"] = self.timeout_sec
        captured["method"] = method
        return {"settings": {}}

    monkeypatch.setattr("alphonse.agent_v2.ipc.V2DaemonClient.request", fake_request)

    client.set_inference_settings(provider_key="openai_codex", model_id="gpt-5.5")

    assert captured == {"timeout": 35.0, "method": "set_inference_settings"}


def test_daemon_ipc_reads_and_saves_agent_configuration(tmp_path) -> None:
    runtime = build_runtime_host(
        schedule_store=ScheduledTaskStore(":memory:"),
        agent_config_store=AgentConfigStore(tmp_path / "agent-config"),
    )
    daemon = V2Daemon(runtime)

    documents = daemon.ipc._dispatch({"method": "agent_config_documents"})["documents"]
    saved = daemon.ipc._dispatch(
        {"method": "save_agent_config", "params": {"file_name": PHILOSOPHY_FILE, "content": "Updated philosophy"}}
    )["document"]

    assert {document["file_name"] for document in documents} == {"CoreContext.md", "Philosophy.md"}
    assert saved["content"] == "Updated philosophy"


def test_daemon_ipc_project_session_enriches_queued_message(tmp_path) -> None:
    runtime = build_runtime_host(
        schedule_store=ScheduledTaskStore(":memory:"),
        project_session_store=SQLiteProjectSessionStore(":memory:"),
    )
    daemon = V2Daemon(runtime)
    project = runtime.project_store.create_project(name="Exercise", root_path=str(tmp_path / "exercise"), owner_user_id="alex")

    daemon.ipc._dispatch(
        {
            "method": "select_project_session",
            "params": {"user": "alex", "integration_id": "tui", "channel_target": "alex", "project_id": project.project_id},
        }
    )
    queued = daemon.ipc._dispatch(
        {
            "method": "queue_message",
            "params": {"prompt": "Routine?", "user": "alex", "integration_id": "tui", "provider_key": "tui", "channel_target": "alex"},
        }
    )

    assert queued["project_id"] == project.project_id
    assert runtime.queue.peek().message.project_id == project.project_id


def test_desktop_poll_is_cursor_based_and_acknowledges_only_its_delivery() -> None:
    runtime = build_runtime_host(inference=_router(), schedule_store=ScheduledTaskStore(":memory:"))
    daemon = V2Daemon(runtime)

    runtime.outbox.enqueue(
        address=ChannelAddress("desktop", "tui", "alex", alphonse_user_id="alex"),
        message="Hello from Desktop.",
    )
    runtime.activity_events.append(
        CoreActivityEvent(
            phase=ImprovementPhase.PLAN,
            label="thinking",
            message="Preparing response.",
            user="alex",
            integration_id="desktop",
            channel_target="alex",
        )
    )

    poll = daemon.ipc._dispatch(
        {"method": "desktop_poll", "params": {"client_id": "desktop-a", "user": "alex", "after_sequence": 0}}
    )

    assert poll["events"][0]["sequence"] == 1
    assert poll["deliveries"][0]["integration_id"] == "desktop"
    delivery_id = poll["deliveries"][0]["outbox_message_id"]
    assert daemon.ipc._dispatch(
        {"method": "desktop_ack_delivery", "params": {"client_id": "other-client", "outbox_message_id": delivery_id}}
    )["acknowledged"] is False
    assert daemon.ipc._dispatch(
        {"method": "desktop_ack_delivery", "params": {"client_id": "desktop-a", "outbox_message_id": delivery_id}}
    )["acknowledged"] is True
    repeat = daemon.ipc._dispatch(
        {"method": "desktop_poll", "params": {"client_id": "desktop-a", "user": "alex", "after_sequence": poll["next_sequence"]}}
    )
    assert repeat["events"] == []
    assert repeat["deliveries"] == []


def test_desktop_conversation_history_is_project_scoped() -> None:
    runtime = build_runtime_host(inference=_router(), schedule_store=ScheduledTaskStore(":memory:"))
    daemon = V2Daemon(runtime)
    first = runtime.outbox.enqueue(
        address=ChannelAddress("desktop", "tui", "alex", alphonse_user_id="alex"),
        message="Project one response.",
        metadata={"project_id": "project-one"},
    )
    runtime.outbox.mark_delivered(first.outbox_message_id)
    runtime.outbox.enqueue(
        address=ChannelAddress("desktop", "tui", "alex", alphonse_user_id="alex"),
        message="Project two response.",
        metadata={"project_id": "project-two"},
    )

    history = daemon.ipc._dispatch(
        {"method": "desktop_conversation_history", "params": {"user": "alex", "project_id": "project-one"}}
    )["messages"]

    assert history == [
        {
            "id": first.outbox_message_id,
            "role": "assistant",
            "content": "Project one response.",
            "created_at": first.created_at,
        }
    ]


def test_desktop_a2ui_question_surface_is_negotiated_and_actions_resume_only_the_question() -> None:
    store = SQLiteQuestionStore(":memory:")
    runtime = build_runtime_host(inference=_router(), schedule_store=ScheduledTaskStore(":memory:"), question_store=store)
    daemon = V2Daemon(runtime)
    question = store.create_question(
        task=TaskState(task_id="task-a2ui", goal="Need confirmation", user="alex"),
        question="Continue?",
        kind="yes_no",
    )
    runtime.ui_events.append(CoreUiEvent("question_interrupt_opened", {"question": question.to_dict()}))

    no_catalog = daemon.ipc._dispatch(
        {"method": "desktop_poll", "params": {"client_id": "plain", "user": "alex"}}
    )
    assert not any(event["event"].get("name") == "a2ui.envelope" for event in no_catalog["ui_events"])

    poll = daemon.ipc._dispatch(
        {
            "method": "desktop_poll",
            "params": {
                "client_id": "a2ui", "user": "alex",
                "client_capabilities": {"supportedCatalogIds": ["alphonse.desktop.catalog.v1"]},
            },
        }
    )
    envelopes = [item["event"]["value"] for item in poll["ui_events"] if item["event"].get("name") == "a2ui.envelope"]
    assert envelopes[0]["createSurface"]["catalogId"] == "alphonse.desktop.catalog.v1"

    result = daemon.ipc._dispatch(
        {
            "method": "a2ui_action",
            "params": {
                "client_id": "a2ui", "user": "alex", "surface_id": f"question:{question.question_id}",
                "source_component_id": "answer_yes", "action_name": "answer_question",
                "context": {"question_id": question.question_id, "answer": True}, "data_model": {},
            },
        }
    )
    assert result["handled"] is True
    assert store.get_question(question.question_id).status == "answered"

    after = daemon.ipc._dispatch(
        {"method": "desktop_poll", "params": {"client_id": "a2ui", "user": "alex", "client_capabilities": {"supportedCatalogIds": ["alphonse.desktop.catalog.v1"]}}}
    )
    assert any(event["event"].get("name") == "a2ui.envelope" and "deleteSurface" in event["event"]["value"] for event in after["ui_events"])


def test_desktop_project_ipc_uses_the_daemon_owned_store(tmp_path) -> None:
    runtime = build_runtime_host(inference=_router(), schedule_store=ScheduledTaskStore(":memory:"))
    daemon = V2Daemon(runtime)

    created = daemon.ipc._dispatch(
        {
            "method": "create_project",
            "params": {
                "user": "alex",
                "name": "Desktop",
                "description": "Desktop project",
                "root_path": str(tmp_path / "desktop-project"),
                "visibility": "private",
            },
        }
    )["project"]
    daemon.ipc._dispatch(
        {"method": "save_project_context", "params": {"user": "alex", "project_id": created["project_id"], "content": "Focus here."}}
    )

    assert daemon.ipc._dispatch({"method": "projects", "params": {"user": "alex"}})["projects"][0]["project_id"] == created["project_id"]
    assert daemon.ipc._dispatch(
        {"method": "project_context", "params": {"user": "alex", "project_id": created["project_id"]}}
    )["content"] == "Focus here."
