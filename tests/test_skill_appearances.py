"""Disposable appearance packages; no household assets or programs are imported."""
import base64
import struct
import zlib
from pathlib import Path

import pytest
import yaml

from alphonse.agent_v2.skill_appearances import STATES, contained_image, leading_skill, resolve_appearance, validate_png
from alphonse.agent_v2.skills import SkillStore


def png():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(b"\0\0\0\0\0")) + chunk(b"IEND", b"")


def package(tmp_path, appearance=None):
    source = tmp_path / "source" / "fixture"
    source.mkdir(parents=True, exist_ok=True)
    metadata = {"name": "fixture", "description": "Disposable appearance fixture"}
    if appearance is not None:
        metadata["appearance"] = appearance
    (source / "SKILL.md").write_text("---\n" + yaml.safe_dump(metadata) + "---\n\nRead fixture instructions.\n")
    (source / "idle.png").write_bytes(png())
    return source


def test_install_update_remove_and_missing_resource_fallback(tmp_path):
    source = package(tmp_path, {"version": 1, "states": {"idle": "idle.png"}})
    store = SkillStore(tmp_path / "installed")
    record = store.install_directory(source)
    first = resolve_appearance(record, include_resources=True)
    assert first["states"] == {"idle": {"resource": "idle.png"}}
    assert base64.b64decode(first["resources"]["idle.png"].split(",")[1]) == png()
    assert "resources" not in resolve_appearance(record)
    assert "checking" not in first["states"]  # UI uses the matching base state.
    package(tmp_path, {"version": 1, "preset": "doctor"})
    updated = store.replace_directory(record.skill_id, source)
    assert resolve_appearance(updated)["revision"] != first["revision"]
    assert set(resolve_appearance(updated)["states"]) == set(STATES)
    store.set_appearance(record.skill_id, "exercise-coach")
    assert resolve_appearance(store.get(record.skill_id))["label"] == "Exercise coach"
    store.update_description(record.skill_id, "Updated fixture")
    assert store.get(record.skill_id).appearance["preset"] == "exercise-coach"
    store.set_appearance(record.skill_id, "")
    assert resolve_appearance(store.get(record.skill_id)) is None
    package(tmp_path, {"version": 1, "states": {"idle": "missing.png"}})
    record = store.replace_directory(record.skill_id, source)
    assert resolve_appearance(record) is None
    assert "Read fixture" in record.instructions  # Optional failure never disables the skill.
    store.uninstall(record.skill_id)
    assert store.get(record.skill_id) is None


@pytest.mark.parametrize("spec", [
    {"version": 2, "preset": "doctor"}, {"version": True, "preset": "doctor"},
    {"version": 1, "preset": "unknown"}, {"version": 1, "states": {"unknown": "idle.png"}},
    {"version": 1, "states": {"idle": "../private.png"}},
    {"version": 1, "states": {"idle": "/tmp/private.png"}},
    {"version": 1, "states": {"idle": "https://example.com/a.png"}},
    {"version": 1, "states": {"idle": "..\\private.png"}},
    {"version": 1, "states": {"idle": "active.svg"}},
    {"version": 1, "states": {"idle": 42}}, "not-a-contract",
])
def test_incompatible_appearance_does_not_break_installation(tmp_path, spec):
    store = SkillStore(tmp_path / "installed")
    record = store.install_directory(package(tmp_path, spec))
    assert resolve_appearance(record) is None
    assert store.get(record.skill_id).instructions


def test_symlink_escape_corrupt_and_animated_png_are_rejected(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    outside = tmp_path / "private.png"
    outside.write_bytes(png())
    (root / "escape.png").symlink_to(outside)
    (root / "escape-dir").symlink_to(tmp_path, target_is_directory=True)
    for path in ["escape.png", "escape-dir/private.png"]:
        with pytest.raises(ValueError, match="outside_package"):
            contained_image(root, path)
    with pytest.raises(ValueError):
        validate_png(png()[:-1])
    with pytest.raises(ValueError):
        validate_png(b"<svg onload='alert(1)'/>")
    data = png()
    actl = b"acTL" + struct.pack(">II", 1, 0)
    animated = data[:33] + struct.pack(">I", 8) + actl + struct.pack(">I", zlib.crc32(actl)) + data[33:]
    with pytest.raises(ValueError, match="static_png"):
        validate_png(animated)


def test_install_and_update_do_not_launder_external_resource_links(tmp_path):
    source = package(tmp_path, {"version": 1, "states": {"idle": "linked.png"}})
    outside = tmp_path / "private.png"
    outside.write_bytes(png())
    (source / "linked.png").symlink_to(outside)
    store = SkillStore(tmp_path / "installed")
    record = store.install_directory(source)
    assert resolve_appearance(record) is None
    assert (Path(record.directory) / "linked.png").is_symlink()
    updated = store.replace_directory(record.skill_id, source)
    assert resolve_appearance(updated) is None
    assert outside.read_bytes() == png()


@pytest.mark.parametrize("preset", ["doctor", "exercise-coach"])
def test_shipped_sets_cover_every_existing_state(tmp_path, preset):
    store = SkillStore(tmp_path / "installed")
    record = store.install_directory(package(tmp_path, {"version": 1, "preset": preset}))
    appearance = resolve_appearance(record, include_resources=True)
    assert set(appearance["states"]) == set(STATES)
    assert len(appearance["resources"]) == 1
    assert len(set(tuple(state["frame"]) for state in appearance["states"].values())) == 7


def test_leading_skill_uses_only_existing_explicit_selection():
    assert leading_skill({"selected_skill_ids": ["doctor"], "primary_skill_ids": []}) == "doctor"
    assert leading_skill({"selected_skill_ids": ["doctor", "coach"], "primary_skill_ids": ["doctor"]}) == "doctor"
    assert leading_skill({"selected_skill_ids": ["doctor", "coach"], "primary_skill_ids": ["doctor", "coach"]}) is None
    assert leading_skill({"selected_skill_ids": ["doctor", "coach"]}) is None
    assert leading_skill({"selected_skill_ids": [], "primary_skill_ids": ["doctor"]}) is None
    assert leading_skill({"selected_skill_ids": "doctor"}) is None
    assert leading_skill(None) is None


def test_ipc_appearance_authorization_and_active_task_projection(tmp_path):
    from alphonse.agent_v2.artifacts import SQLiteArtifactStore
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.runtime import build_runtime_host
    from alphonse.agent_v2.users import V2UserStore

    users = V2UserStore(":memory:")
    admin = users.onboard(display_name="Fixture administrator", users_root=tmp_path / "workspace" / "users")
    users.set_workspace_root(tmp_path / "workspace")
    member = users.create_user(display_name="Fixture member")
    skills = SkillStore(tmp_path / "installed")
    record = skills.install_directory(package(tmp_path))
    runtime = build_runtime_host(user=admin.user_id, user_store=users, skill_store=skills, artifact_store=SQLiteArtifactStore(":memory:"), schedule_store=ScheduledTaskStore(":memory:"))
    daemon = V2Daemon(runtime)

    def request(method, actor=admin.user_id, **params):
        return daemon.ipc._dispatch({"method": method, "params": {"actor_user_id": actor, "skill_id": record.skill_id, **params}})

    for method in ["skill_appearance", "set_skill_appearance"]:
        with pytest.raises(PermissionError, match="admin_required"):
            request(method, actor=member.user_id, preset="doctor")
    request("set_skill_appearance", preset="doctor")
    assert request("skill_appearance")["appearance"]["label"] == "Family health assistant"
    catalog = request("skills")["skills"]
    assert next(s for s in catalog if s["id"] == record.skill_id)["appearance_configured"]
    task = TaskState(task_id="fixture-task", message_id="fixture-message", user=admin.user_id, metadata={"v3_skill_selection": {"selected_skill_ids": [record.skill_id], "primary_skill_ids": [record.skill_id]}})
    runtime.question_store.save_task_checkpoint(task)
    daemon.kill_switch.activate(message_id=task.message_id, task_id=task.task_id, user_id=admin.user_id, project_id="", metadata={})
    context = daemon.desktop_appearance_context(admin.user_id)
    assert context["appearance"]["skill_id"] == record.skill_id
    assert "resources" not in context["appearance"]
    assert daemon.desktop_appearance_context(member.user_id)["appearance"] is None
    request("set_skill_appearance", preset="exercise-coach")
    assert daemon.desktop_appearance_context(admin.user_id)["appearance"]["revision"] != context["appearance"]["revision"]
    task.metadata["v3_skill_selection"]["selected_skill_ids"].append("skill:other")
    task.metadata["v3_skill_selection"]["primary_skill_ids"].append("skill:other")
    runtime.question_store.save_task_checkpoint(task)
    assert daemon.desktop_appearance_context(admin.user_id)["appearance"] is None
    task.metadata["v3_skill_selection"]["primary_skill_ids"] = [record.skill_id]
    runtime.question_store.save_task_checkpoint(task)
    daemon.kill_switch.request_cancel()
    assert daemon.desktop_appearance_context(admin.user_id)["appearance"] is None
    daemon.kill_switch.clear()
    assert daemon.desktop_appearance_context(admin.user_id)["task_id"] == ""
    daemon.kill_switch.activate(message_id=task.message_id, task_id=task.task_id, user_id=admin.user_id, project_id="", metadata={})
    for status in ["completed", "failed", "cancelled", "waiting_user"]:
        task.status = status
        runtime.question_store.save_task_checkpoint(task, status="done" if status == "completed" else status)
        assert daemon.desktop_appearance_context(admin.user_id)["appearance"] is None
    skills.uninstall(record.skill_id)
    assert request("skill_appearance")["appearance"] is None


def test_direct_conversation_publishes_skill_selection_before_composing_reply(tmp_path):
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.inference import InferenceResult
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.questions import SQLiteQuestionStore

    skills = SkillStore(tmp_path / "installed")
    record = skills.install_directory(package(tmp_path, {"version": 1, "preset": "doctor"}))
    questions = SQLiteQuestionStore(":memory:")
    task = TaskState(task_id="conversation", message_id="message", user="fixture", metadata={"v3_skill_selection": {"selected_skill_ids": [record.skill_id], "primary_skill_ids": [record.skill_id]}})

    class Inference:
        def generate_markdown(self, request):
            checkpoint = questions.load_task_checkpoint(task.task_id)
            assert checkpoint is not None and checkpoint.status == "running"
            assert leading_skill(checkpoint.metadata["v3_skill_selection"]) == record.skill_id
            return InferenceResult(content="Fixture reply.", model_profile=request.model_profile)

    context = CoreLoopContext(messages=InMemoryMessageQueue(), question_store=questions, skill_store=skills, inference=Inference())
    HierarchicalCAPDProcessor._complete_direct_response(task, context)
    assert task.status == "completed"
