from __future__ import annotations

import os

import pytest

from alphonse.agent_v2.core.inference import InferencePurpose
from alphonse.agent_v2.core.inference import InferenceRouter
from alphonse.agent_v2.core.inference import ModelProfile
from alphonse.agent_v2.core.inference import StubInferenceProvider
from alphonse.agent_v2.core.core import CoreActivityEvent
from alphonse.agent_v2.core.core import CoreUiEvent
from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.core import _task_progress_snapshot
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.io import ChannelAddress
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.agent_config import AgentConfigStore
from alphonse.agent_v2.agent_config import GLOBAL_CONTEXT_FILE
from alphonse.agent_v2.agent_config import PHILOSOPHY_FILE
from alphonse.agent_v2.interfaces.a2ui import A2UiAdapter
from alphonse.agent_v2.interfaces.a2ui import ALPHONSE_DESKTOP_CATALOG_ID
from alphonse.agent_v2.services.project_sessions import SQLiteProjectSessionStore
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore
from alphonse.agent_v2.web_tools_settings import SQLiteWebToolsSettingsStore


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


def test_daemon_settings_validate_and_persist_timezone(tmp_path) -> None:
    users = V2UserStore(":memory:")
    users.onboard(display_name="Admin", users_root=tmp_path / "users")
    daemon = V2Daemon(build_runtime_host(user_store=users, schedule_store=ScheduledTaskStore(":memory:")))

    saved = daemon.ipc._dispatch({"method": "save_settings", "params": {"users_root": str(tmp_path / "users"), "timezone": "America/Mexico_City"}})
    assert saved["timezone"] == "America/Mexico_City"
    assert daemon.ipc._dispatch({"method": "settings"})["timezone"] == "America/Mexico_City"
    with pytest.raises(ValueError, match="invalid_timezone"):
        daemon.ipc._dispatch({"method": "save_settings", "params": {"users_root": str(tmp_path / "users"), "timezone": "Not/A_Timezone"}})


def test_daemon_ipc_web_tools_require_admin_and_refresh_registry(tmp_path) -> None:
    users = V2UserStore(":memory:")
    admin = users.onboard(display_name="Admin", users_root=tmp_path / "users")
    runtime = build_runtime_host(user_store=users, web_tools_settings_store=SQLiteWebToolsSettingsStore(":memory:"), schedule_store=ScheduledTaskStore(":memory:"))
    daemon = V2Daemon(runtime)
    with pytest.raises(PermissionError, match="admin_required"):
        daemon.ipc._dispatch({"method": "web_tools_settings", "params": {"actor_user_id": "not-admin"}})
    saved = daemon.ipc._dispatch({"method": "save_web_tools_settings", "params": {"actor_user_id": admin.user_id, "values": {"enabled": True, "searxng_base_url": "http://127.0.0.1:8080", "search_timeout_seconds": 10, "fetch_timeout_seconds": 10, "fetch_max_chars": 12000}}})["settings"]
    assert saved["available"] is True
    assert runtime.core.tools.get("web_search") is not None


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

    assert {document["file_name"] for document in documents} == {GLOBAL_CONTEXT_FILE, "Philosophy.md"}
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


def test_scheduled_task_confirmation_card_uses_persisted_schedule_fields() -> None:
    adapter = A2UiAdapter()
    task = {
        "scheduled_task_id": "schedule-1",
        "name": "Charge Tesla",
        "description": "Before the trip",
        "schedule_summary": "Once at 2026-07-26T03:00:00+00:00",
        "next_run_at": "2026-07-26T03:00:00+00:00",
        "timezone": "America/Mexico_City",
    }

    envelopes = adapter.scheduled_task_created(task, project_name="Road trip")

    assert envelopes[0]["createSurface"]["surfaceId"] == "scheduled-task:schedule-1"
    components = {item["id"]: item for item in envelopes[1]["updateComponents"]["components"]}
    assert components["name"]["text"] == "Charge Tesla"
    assert "America/Mexico_City" in components["details"]["text"]
    assert components["view"]["action"] == {"name": "view_scheduled_task", "context": {"scheduled_task_id": "schedule-1"}}


def test_scheduled_task_a2ui_action_requires_capability_and_owner(tmp_path) -> None:
    users = V2UserStore(":memory:")
    admin = users.onboard(display_name="Admin", users_root=tmp_path / "users")
    schedules = ScheduledTaskStore(":memory:")
    task = schedules.create_task(owner_user_id=admin.user_id, project_id="", name="Reminder", description="", prompt="Remember", schedule_kind="once", run_at="2026-07-26T03:00:00+00:00", timezone_name="UTC")
    daemon = V2Daemon(build_runtime_host(user_store=users, schedule_store=schedules, inference=_router()))
    daemon._desktop_capabilities["desktop-a"] = {ALPHONSE_DESKTOP_CATALOG_ID}

    result = daemon.a2ui_action(client_id="desktop-a", user=admin.user_id, surface_id=f"scheduled-task:{task.scheduled_task_id}", source_component_id="view", action_name="view_scheduled_task", context={"scheduled_task_id": task.scheduled_task_id})
    assert result == {"action": "view_scheduled_task", "scheduled_task_id": task.scheduled_task_id}
    with pytest.raises(ValueError, match="a2ui_surface_or_context_invalid"):
        daemon.a2ui_action(client_id="desktop-a", user=admin.user_id, surface_id=f"scheduled-task:{task.scheduled_task_id}", source_component_id="view", action_name="view_scheduled_task", context={"scheduled_task_id": "other"})
    with pytest.raises(ValueError, match="a2ui_catalog_not_negotiated"):
        daemon.a2ui_action(client_id="desktop-b", user=admin.user_id, surface_id=f"scheduled-task:{task.scheduled_task_id}", source_component_id="view", action_name="view_scheduled_task", context={"scheduled_task_id": task.scheduled_task_id})


def test_desktop_task_progress_a2ui_is_admin_desktop_only_and_sanitized(tmp_path) -> None:
    users = V2UserStore(":memory:")
    admin = users.onboard(display_name="Admin", users_root=tmp_path / "users")
    runtime = build_runtime_host(user_store=users, schedule_store=ScheduledTaskStore(":memory:"), inference=_router())
    daemon = V2Daemon(runtime)
    task = TaskState(task_id="task-progress", user=admin.user_id, goal="Check status", acceptance_criteria_md="1. [ ] Return a safe result")
    task.metadata["planned_tool_call"] = {"tool_name": "Search", "arguments": {"query": "weather", "api_key": "secret-value"}}
    task.append_plan_call({"id": "call-1", "tool_name": "Search", "arguments": {"query": "weather"}})
    task.record_plan_call_success("call-1", {"summary": "Sunny", "access_token": "hidden-token"})
    progress = _task_progress_snapshot(task)
    assert progress["tool_arguments"]["api_key"] == "[redacted]"
    assert progress["tool_result"]["access_token"] == "[redacted]"
    runtime.activity_events.append(CoreActivityEvent(
        phase=ImprovementPhase.PLAN,
        label="thinking",
        message="Selecting a tool.",
        task_id="task-progress",
        user=admin.user_id,
        integration_id="desktop",
        channel_target=admin.user_id,
        progress=progress,
    ))

    plain = daemon.ipc._dispatch({"method": "desktop_poll", "params": {"client_id": "plain", "user": admin.user_id}})
    assert not any("task-progress:" in str(item) for item in plain["ui_events"])
    rich = daemon.ipc._dispatch({"method": "desktop_poll", "params": {"client_id": "rich", "user": admin.user_id, "client_capabilities": {"supportedCatalogIds": [ALPHONSE_DESKTOP_CATALOG_ID]}}})
    envelopes = [item["event"]["value"] for item in rich["ui_events"] if item["event"].get("name") == "a2ui.envelope"]
    assert any(item.get("createSurface", {}).get("surfaceId") == "task-progress:task-progress" for item in envelopes)
    assert "secret-value" not in str(envelopes)

    runtime.activity_events.append(CoreActivityEvent(
        phase=ImprovementPhase.PLAN, label="thinking", message="Hidden", task_id="telegram-task", user=admin.user_id,
        integration_id="telegram", channel_target=admin.user_id, progress=progress,
    ))
    excluded = daemon.ipc._dispatch({"method": "desktop_poll", "params": {"client_id": "rich", "user": admin.user_id, "client_capabilities": {"supportedCatalogIds": [ALPHONSE_DESKTOP_CATALOG_ID]}}})
    assert "telegram-task" not in str(excluded["ui_events"])


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


def test_project_recent_files_ipc_lists_newest_visible_direct_children(tmp_path) -> None:
    runtime = build_runtime_host(inference=_router(), schedule_store=ScheduledTaskStore(":memory:"))
    daemon = V2Daemon(runtime)
    project = runtime.project_store.create_project(name="Recent files", root_path=str(tmp_path / "recent-files"), owner_user_id="alex")
    root = tmp_path / "recent-files"
    children = [root / "old.txt", root / "notes.md", root / "assets", root / "latest.py", root / "fifth.txt"]
    children[2].mkdir()
    for child in children:
        if child.is_dir():
            continue
        child.write_text(child.name, encoding="utf-8")
    (root / ".hidden.txt").write_text("hidden", encoding="utf-8")
    nested = root / "assets" / "nested.txt"
    nested.write_text("nested", encoding="utf-8")
    for index, child in enumerate(children, start=1):
        modified_at = 2_000_000_000 + index
        os.utime(child, (modified_at, modified_at))
    os.utime(nested, (2_000_000_999, 2_000_000_999))

    result = daemon.ipc._dispatch(
        {"method": "project_recent_files", "params": {"user": "alex", "project_id": project.project_id, "limit": 99}}
    )["files"]

    assert [item["name"] for item in result] == ["fifth.txt", "latest.py", "assets", "notes.md"]
    assert [item["kind"] for item in result] == ["file", "file", "directory", "file"]
    assert all(not item["name"].startswith(".") for item in result)


def test_project_recent_files_ipc_rejects_unknown_project(tmp_path) -> None:
    runtime = build_runtime_host(inference=_router(), schedule_store=ScheduledTaskStore(":memory:"))
    daemon = V2Daemon(runtime)

    with pytest.raises(ValueError, match="project_not_found"):
        daemon.ipc._dispatch(
            {"method": "project_recent_files", "params": {"user": "alex", "project_id": "missing"}}
        )


def test_scheduled_task_ipc_scopes_owners_and_manages_task_lifecycle(tmp_path) -> None:
    runtime = build_runtime_host(inference=_router(), schedule_store=ScheduledTaskStore(":memory:"))
    admin = runtime.user_store.onboard(display_name="Alex", users_root=tmp_path / "users")
    member = runtime.user_store.create_user(display_name="Gaby")
    daemon = V2Daemon(runtime)
    task = runtime.schedule_store.create_task(
        owner_user_id=member.user_id,
        name="Morning check-in",
        prompt="Ask how the morning is going.",
        schedule_kind="rrule",
        rrule="FREQ=DAILY",
        dtstart="2026-07-10T09:00:00+00:00",
    )
    runtime.schedule_store.record_execution(scheduled_task_id=task.scheduled_task_id, run_id="run-1", status="delivered")

    listed = daemon.ipc._dispatch({"method": "scheduled_tasks", "params": {"actor_user_id": admin.user_id, "owner_user_id": member.user_id}})["tasks"]
    assert listed[0]["latest_execution"]["status"] == "delivered"
    with pytest.raises(PermissionError, match="scheduled_task_owner_forbidden"):
        daemon.ipc._dispatch({"method": "scheduled_tasks", "params": {"actor_user_id": member.user_id, "owner_user_id": admin.user_id}})

    updated = daemon.ipc._dispatch({"method": "update_scheduled_task", "params": {"actor_user_id": admin.user_id, "scheduled_task_id": task.scheduled_task_id, "name": "Daily check-in", "prompt": "Ask about the day."}})["task"]
    assert updated["name"] == "Daily check-in"
    assert daemon.ipc._dispatch({"method": "pause_schedule", "params": {"actor_user_id": admin.user_id, "scheduled_task_id": task.scheduled_task_id}})["task"]["status"] == "paused"
    assert daemon.ipc._dispatch({"method": "resume_schedule", "params": {"actor_user_id": admin.user_id, "scheduled_task_id": task.scheduled_task_id}})["task"]["status"] == "active"
    assert daemon.ipc._dispatch({"method": "cancel_schedule", "params": {"actor_user_id": admin.user_id, "scheduled_task_id": task.scheduled_task_id}})["task"]["status"] == "cancelled"
    assert daemon.ipc._dispatch({"method": "delete_scheduled_task", "params": {"actor_user_id": admin.user_id, "scheduled_task_id": task.scheduled_task_id}})["deleted"] is True


def test_project_management_ipc_archives_imports_and_safely_removes(tmp_path) -> None:
    users = V2UserStore(tmp_path / "users.sqlite3")
    admin = users.onboard(display_name="Alex", users_root=tmp_path / "users")
    runtime = build_runtime_host(inference=_router(), user=admin.user_id, user_store=users, schedule_store=ScheduledTaskStore(":memory:"))
    daemon = V2Daemon(runtime)
    external = tmp_path / "external"; external.mkdir()
    imported = daemon.ipc._dispatch({"method": "import_project", "params": {"user": admin.user_id, "name": "Imported", "description": "", "root_path": str(external), "visibility": "private"}})["project"]

    assert daemon.ipc._dispatch({"method": "manageable_projects", "params": {"user": admin.user_id}})["projects"][0]["owner"]["display_name"] == "Alex"
    archived = daemon.ipc._dispatch({"method": "archive_project", "params": {"user": admin.user_id, "project_id": imported["project_id"]}})["project"]
    assert archived["status"] == "archived"
    assert daemon.ipc._dispatch({"method": "restore_project", "params": {"user": admin.user_id, "project_id": imported["project_id"]}})["project"]["status"] == "active"
    with pytest.raises(ValueError, match="delete_confirmation_must_match_project_id"):
        daemon.ipc._dispatch({"method": "delete_project", "params": {"user": admin.user_id, "project_id": imported["project_id"], "confirmation": "wrong"}})
    removed = daemon.ipc._dispatch({"method": "delete_project", "params": {"user": admin.user_id, "project_id": imported["project_id"], "confirmation": imported["project_id"]}})
    assert removed["removed_managed_files"] is False
    assert external.exists()
