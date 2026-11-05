"""Only disposable skill declarations and historical records cross the bridge."""
import json
from pathlib import Path

from alphonse.agent_v2.artifacts import SQLiteArtifactStore
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.users import V2UserStore
from alphonse.agent_v2.widget_context import definitions


def fixture(tmp_path):
    users = V2UserStore(":memory:")
    admin = users.onboard(display_name="Fixture", users_root=tmp_path / "workspace" / "users")
    users.set_workspace_root(tmp_path / "workspace")
    skills = SkillStore(tmp_path / "skills")
    source = tmp_path / "fixture"
    source.mkdir()
    (source / "SKILL.md").write_text("---\nname: fixture\ndescription: Disposable planning and exercise fixture\n---\nRead authorized context.")
    runtime = build_runtime_host(user=admin.user_id, user_store=users, skill_store=skills, artifact_store=SQLiteArtifactStore(":memory:"), schedule_store=ScheduledTaskStore(":memory:"))
    event = runtime.conversation_store.record(owner_user_id=admin.user_id, project_id="", role="assistant", content="Past session: completed three exercise sets.", source="desktop", source_message_id="fixture")
    spec = {"version": 1, "widgets": [
        {"key": "calendar", "title": "Calendar", "binding": {"type": "agenda"}},
        {"key": "exercise", "title": "Exercise progress", "binding": {"type": "memory", "source_id": f"conversation:{event.event_id}"}},
        {"key": "missing", "title": "Unavailable", "binding": {"type": "memory", "source_id": "conversation:missing"}},
    ]}
    (source / "widgets.json").write_text(json.dumps(spec))
    skill = skills.install_directory(source)
    daemon = V2Daemon(runtime)
    task = TaskState(task_id="planning", message_id="fixture", user=admin.user_id, goal="Plan dinner and a trip", metadata={"v3_skill_selection": {"selected_skill_ids": [skill.skill_id]}})
    runtime.question_store.save_task_checkpoint(task)
    daemon.kill_switch.activate(message_id=task.message_id, task_id=task.task_id, user_id=admin.user_id, project_id="", metadata={})
    return daemon, task, skill, spec


def test_poll_projects_existing_skill_decision_and_rechecks_reads(tmp_path):
    daemon, task, skill, spec = fixture(tmp_path)
    first = daemon.poll_desktop(client_id="fixture", user=task.user)["widget_context"]
    assert first["member"] == task.user
    calendar, exercise, missing = first["widgets"]
    assert calendar["relevant"] and calendar["binding"] == "agenda"
    assert exercise["text"] == "Past session: completed three exercise sets."
    assert exercise["historical"] and missing["status"] == "unavailable"
    # A past-session question reuses the existing Jev-selected skill, no classifier.
    task.goal = "How did my past exercise session go?"
    daemon.runtime.question_store.save_task_checkpoint(task)
    second = daemon.poll_desktop(client_id="reconnect", user=task.user)["widget_context"]
    assert second == first
    # The source belongs to the first actor; relevance cannot grant another actor a read.
    other = daemon.desktop_widget_context("other")["widgets"]
    assert not any(row["relevant"] for row in other)
    assert all(row["text"] == "" for row in other if row["binding"] == "memory")
    assert not set(row["id"] for row in other) & set(row["id"] for row in first["widgets"])
    # Owner erasure revokes the actual source, without changing the declaration.
    source_id = spec["widgets"][1]["binding"]["source_id"].removeprefix("conversation:")
    assert daemon.runtime.conversation_store.delete(event_id=source_id, owner_user_id=task.user)
    erased = daemon.desktop_widget_context(task.user)["widgets"][1]
    assert erased["status"] == "unavailable" and erased["text"] == ""
    # Revoking/changing a binding removes its old payload on the very next snapshot.
    spec["widgets"][1]["binding"]["source_id"] = "conversation:revoked"
    (Path(skill.directory) / "widgets.json").write_text(json.dumps(spec))
    revoked = daemon.desktop_widget_context(task.user)["widgets"][1]
    assert revoked["id"] == exercise["id"] and revoked["status"] == "unavailable" and revoked["text"] == ""
    daemon.kill_switch.clear()
    assert daemon.desktop_widget_context(task.user)["widgets"] == []
    retained = daemon.desktop_widget_context(task.user, [calendar["id"], exercise["id"]])["widgets"]
    assert len(retained) == 2 and not any(row["relevant"] for row in retained)
    assert retained[0]["status"] == "ready" and retained[1]["text"] == ""
    assert daemon.desktop_widget_context("other", [calendar["id"], exercise["id"]])["widgets"] == []
    daemon.runtime.skill_store.uninstall(skill.skill_id)
    assert daemon.desktop_widget_context(task.user)["widgets"] == []


def test_resource_validation_rejects_code_bindings_and_escape(tmp_path):
    _, _, skill, spec = fixture(tmp_path)
    path = Path(skill.directory) / "widgets.json"
    spec["widgets"] += [
        {"key": "bad-suggestion", "title": "Bad", "binding": {"type": "agenda"}, "suggestion": []},
        {"key": "evil", "title": "Evil", "binding": {"type": "html", "code": "<script>alert(1)</script>"}},
        {"key": "url", "title": "URL", "binding": {"type": "memory", "source_id": "conversation:x", "url": "file:///private"}},
    ]
    path.write_text(json.dumps(spec))
    assert len(definitions(skill)) == 3
    external = tmp_path / "outside.json"
    external.write_text(json.dumps(spec))
    path.unlink()
    path.symlink_to(external)
    assert definitions(skill) == []
