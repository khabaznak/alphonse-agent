"""Exercise the supported IPC operations used by Settings with disposable storage."""
import pytest

from alphonse.agent_v2.artifacts import SQLiteArtifactStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def workspace(tmp_path):
    users = V2UserStore(":memory:")
    admin = users.onboard(display_name="Fixture administrator", users_root=tmp_path / "workspace" / "users")
    users.set_workspace_root(tmp_path / "workspace")
    member = users.create_user(display_name="Fixture member")
    runtime = build_runtime_host(
        user=admin.user_id, user_store=users,
        skill_store=SkillStore(tmp_path / "workspace" / "skills"),
        artifact_store=SQLiteArtifactStore(":memory:"),
        schedule_store=ScheduledTaskStore(":memory:"),
    )
    daemon = V2Daemon(runtime)

    def request(method, **params):
        return daemon.ipc._dispatch({"method": method, "params": {"actor_user_id": admin.user_id, **params}})

    return request, runtime, member.user_id


def test_settings_skill_package_lifecycle_and_member_denial(workspace, tmp_path):
    request, runtime, member = workspace
    source = tmp_path / "disposable"
    source.mkdir()
    definition = "---\nname: disposable\ndescription: Disposable fixture\n---\n\n# Workflow\nRead a fixture.\n"
    (source / "SKILL.md").write_text(definition)
    skill_id = request("install_skill", source_directory=str(source))["skill"]["id"]
    assert any(s["id"] == skill_id for s in request("skills")["skills"])
    request("update_skill", skill_id=skill_id, description="Updated fixture")
    request("write_skill_file", skill_id=skill_id, path="notes.txt", content="A local note")
    assert request("read_skill_file", skill_id=skill_id, path="notes.txt")["content"] == "A local note"
    imported = tmp_path / "reference.txt"
    imported.write_text("Disposable reference")
    added = request("add_skill_file", skill_id=skill_id, source_path=str(imported))["path"]
    assert request("read_skill_file", skill_id=skill_id, path=added)["content"] == "Disposable reference"
    request("delete_skill_file", skill_id=skill_id, path="notes.txt")
    with pytest.raises(PermissionError, match="admin_required"):
        request("skills", actor_user_id=member)
    for method, params in [
        ("install_skill", {"source_directory": str(source)}),
        ("replace_skill", {"skill_id": skill_id, "source_directory": str(source)}),
        ("write_skill_file", {"skill_id": skill_id, "path": "notes.txt", "content": "denied"}),
        ("delete_skill", {"skill_id": skill_id}),
    ]:
        with pytest.raises(PermissionError, match="admin_required"):
            request(method, actor_user_id=member, **params)
    (source / "SKILL.md").write_text(definition.replace("Read a fixture.", "Read the updated fixture."))
    request("replace_skill", skill_id=skill_id, source_directory=str(source))
    assert "updated fixture" in request("read_skill_file", skill_id=skill_id, path="SKILL.md")["content"]
    request("delete_skill", skill_id=skill_id)
    assert all(s["id"] != skill_id for s in request("skills")["skills"])


def test_settings_artifact_attachment_and_unregister_preserve_files(workspace, tmp_path):
    request, runtime, member = workspace
    skill_id = request("create_skill", name="fixture", description="Fixture", instructions="Read a fixture.")["skill"]["id"]
    root = runtime.core.artifact_resources.shared_directory(skill_id, "artifact.fixture")
    root.mkdir(parents=True, exist_ok=True)
    program = root / "fixture.py"
    program.write_text("print('fixture')\n")
    record = runtime.artifact_store.register(
        artifact_id="artifact.fixture", name="Fixture", description="Disposable fixture",
        owner_user_id=runtime.user_store.admin_user().user_id, project_id="", storage_kind="shared", skill_id=skill_id,
        entrypoint_path="fixture.py", argument_schema={"type": "object"},
    )
    request("attach_artifact_to_skill", artifact_id=record.artifact_id, skill_id=skill_id, instructions="Read fixture.py using Bash.")
    catalog = request("skills")
    skill = next(s for s in catalog["skills"] if s["id"] == skill_id)
    assert skill["artifacts"][0]["artifact_id"] == record.artifact_id
    assert runtime.core.tools.get(record.artifact_id) is None
    assert record.artifact_id in request("read_skill_file", skill_id=skill_id, path="SKILL.md")["content"]
    with pytest.raises(ValueError, match="skill_has_registered_artifacts"):
        request("replace_skill", skill_id=skill_id, source_directory=str(root))
    with pytest.raises(PermissionError):
        request("delete_artifact", actor_user_id=member, artifact_id=record.artifact_id)
    request("delete_artifact", artifact_id=record.artifact_id)
    assert program.read_text() == "print('fixture')\n"
    assert runtime.artifact_store.get(record.artifact_id) is None
