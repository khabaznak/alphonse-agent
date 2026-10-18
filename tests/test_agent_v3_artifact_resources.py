from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import sys

import pytest

from alphonse.agent_v2.artifacts import SQLiteArtifactStore, build_artifact_tool_definitions
from alphonse.agent_v2.artifact_resources import ArtifactResources
from alphonse.agent_v2.core.core import CoreLoopContext, ToolExecutionContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.context import selected_skill_guidance
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.core.tools.registry.native.artifact_registration import execute_artifact_registration
from alphonse.agent_v2.core.tools.registry.native.bash import execute_bash
from alphonse.agent_v2.skills import SkillStore


@pytest.fixture
def setup(tmp_path):
    skills = SkillStore(tmp_path / "skills")
    skill = skills.create_skill("sensor", "Read a disposable sensor", "# Sensor\nRead the registered fixture.")
    store = SQLiteArtifactStore(tmp_path / "catalog.sqlite3")
    projects = ProjectStore(":memory:")
    resources = ArtifactResources(store, skills, projects, is_admin=lambda actor: actor == "admin")
    resources.ensure_repository()
    return skills, skill, store, projects, resources


def register(setup, *, kind="shared", project=None, execution=None, instructions="Read the fixture; never change it."):
    skills, skill, store, projects, resources = setup
    task = TaskState(user="alex", project_id=project.project_id if project else "")
    if kind == "shared":
        root = resources.shared_directory(skill.skill_id, "artifact.fixture")
    elif kind == "skill":
        root = Path(skill.directory)
    else:
        root = Path(project.root_path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "reader.py").write_text("import json\nfrom pathlib import Path\nprint(json.dumps({'value': Path('reading.txt').read_text()}))\n")
    (root / "reading.txt").write_text("42")
    args = {"name": "Fixture", "artifact_id": "artifact.fixture", "description": "Disposable sensor reader",
            "skill_id": skill.skill_id, "entrypoint_path": "reader.py", "storage_kind": kind,
            "instructions": instructions, "argument_schema": {"type": "object"},
            "execution": execution or {"runtime_path": sys.executable, "required_files": ["reading.txt"]}}
    context = ToolExecutionContext(task=task, messages=InMemoryMessageQueue(), project_store=projects, artifact_resources=resources)
    execute_artifact_registration(args, context=context, store=store, skill_store=skills)
    return store.get("artifact.fixture"), root


@pytest.mark.parametrize("project_mode", ["none", "unrelated"])
def test_selected_skill_executes_shared_artifact_independent_of_task_project_and_daemon_cwd(setup, tmp_path, monkeypatch, project_mode):
    skills, skill, store, projects, resources = setup
    record, root = register(setup)
    unrelated = projects.create_project(name="Other", owner_user_id="alex", root_path=str(tmp_path / "other"))
    monkeypatch.chdir(tmp_path / "other")
    task = TaskState(user="alex", project_id=unrelated.project_id if project_mode == "unrelated" else "",
                     metadata={"v3_skill_selection": {"selected_skill_ids": [skill.skill_id]}})
    context = CoreLoopContext(messages=InMemoryMessageQueue(), skill_store=skills, artifact_resources=resources)
    text, loaded = selected_skill_guidance(task, context)
    assert str(root / "reader.py") in text
    assert "Read the fixture; never change it." in text
    assert loaded == [skill.skill_id]
    result = execute_bash({"artifact_id": record.artifact_id,
                          "command": '"$ALPHONSE_ARTIFACT_RUNTIME" "$ALPHONSE_ARTIFACT_ENTRYPOINT"'},
                         context=ToolExecutionContext(task=task, messages=InMemoryMessageQueue(), project_store=projects, artifact_resources=resources))
    assert result["exit_code"] == 0
    assert json.loads(result["stdout"]) == {"value": "42"}
    assert result["cwd"] == str(root)
    assert build_artifact_tool_definitions(store, projects) == []


def test_legacy_artifact_is_available_without_membership_in_storage_project(setup, tmp_path):
    skills, skill, store, projects, resources = setup
    project = projects.create_project(name="Legacy", owner_user_id="alex", root_path=str(tmp_path / "legacy"))
    record, root = register(setup, kind="project", project=project)
    assert resources.require_ready(record.artifact_id, "alex", [skill.skill_id])["storage_root"] == str(root)
    store.update_location(replace(record, shared_with_user_ids=("gaby",)))
    # Sharing the resource does not add the user to the legacy project.
    assert projects.get_project(project.project_id, requester_user_id="gaby") is None
    assert resources.require_ready(record.artifact_id, "gaby", [skill.skill_id])["storage_root"] == str(root)


def test_permissions_selection_and_disabled_checks_happen_before_shell_execution(setup):
    _, skill, store, _, resources = setup
    record, root = register(setup)
    for actor, selected, error in [("gaby", [skill.skill_id], "access_denied"), ("alex", [], "skill_not_selected")]:
        task = TaskState(user=actor, metadata={"v3_skill_selection": {"selected_skill_ids": selected}})
        with pytest.raises(PermissionError, match=error):
            execute_bash({"artifact_id": record.artifact_id, "command": "touch unauthorized"}, context=ToolExecutionContext(task=task, messages=InMemoryMessageQueue(), artifact_resources=resources))
        assert not (root / "unauthorized").exists()
    store.set_enabled(record.artifact_id, False)
    with pytest.raises(ValueError, match="disabled"):
        resources.require_ready(record.artifact_id, "alex", [skill.skill_id])


def test_private_artifact_sections_do_not_leak_into_another_users_skill_prompt(setup):
    skills, skill, _, _, resources = setup
    record, root = register(setup, instructions="PRIVATE_ARTIFACT_INSTRUCTION")
    task = TaskState(user="gaby", metadata={"v3_skill_selection": {"selected_skill_ids": [skill.skill_id]}})
    text, _ = selected_skill_guidance(task, CoreLoopContext(messages=InMemoryMessageQueue(), skill_store=skills, artifact_resources=resources))
    assert "PRIVATE_ARTIFACT_INSTRUCTION" not in text
    assert str(root) not in text
    assert record.artifact_id not in text


@pytest.mark.parametrize("field", ["entrypoint_path", "working_directory", "configuration_paths", "required_files"])
def test_registration_rejects_traversal_and_keeps_programs_unchanged(setup, field):
    _, _, store, _, resources = setup
    record, root = register(setup)
    execution = dict(record.execution)
    if field == "entrypoint_path":
        candidate = replace(record, entrypoint_path="../escape.py")
    else:
        execution[field] = ["../escape"] if field.endswith("paths") or field == "required_files" else "../escape"
        candidate = replace(record, execution=execution)
    with pytest.raises(ValueError, match="outside_storage"):
        resources.validate(candidate)
    assert store.get(record.artifact_id).entrypoint_path == "reader.py"
    assert (root / "reading.txt").read_text() == "42"


def test_symlink_entrypoint_and_shared_directory_escape_are_rejected(setup, tmp_path):
    _, skill, _, _, resources = setup
    record, root = register(setup)
    outside = tmp_path / "outside.py"
    outside.write_text("print('outside')")
    (root / "escape.py").symlink_to(outside)
    with pytest.raises(ValueError, match="outside_storage"):
        resources.validate(replace(record, entrypoint_path="escape.py"))
    alias = resources.repository_root / "sensor" / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        resources.shared_directory(skill.skill_id, "artifact.alias")


def test_explicit_venv_runtime_symlink_is_supported_without_relaxing_program_containment(setup):
    _, _, _, _, resources = setup
    record, root = register(setup)
    runtime = root / ".venv" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.symlink_to(sys.executable)
    resources.validate(replace(record, execution={"runtime_path": ".venv/bin/python"}))


def test_missing_resources_have_actionable_diagnostics_and_no_configuration_values(setup):
    _, _, _, _, resources = setup
    record, root = register(setup)
    (root / "secret.env").write_text("TOKEN=DO_NOT_DISCLOSE")
    record = replace(record, execution={"runtime_path": "missing-python", "configuration_paths": ["secret.env", "missing.env"], "required_files": ["missing.db"]})
    detail = resources.describe(record, "alex")
    assert detail["status"] == "unavailable"
    assert set(detail["issues"]) == {"runtime_missing_or_not_executable", "configuration_paths_missing", "required_files_missing"}
    assert "DO_NOT_DISCLOSE" not in json.dumps(detail)


def test_catalog_upgrade_keeps_legacy_ids_and_programs_and_is_idempotent(tmp_path):
    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE v2_artifacts(artifact_id TEXT PRIMARY KEY, name TEXT, description TEXT, project_id TEXT, owner_user_id TEXT, entrypoint_path TEXT, argument_schema_json TEXT, timeout_seconds REAL, enabled INTEGER, created_at TEXT, updated_at TEXT)")
        connection.execute("INSERT INTO v2_artifacts VALUES('artifact.old','Old','Reader','old-project','alex','reader.py','{}',30,1,'then','then')")
    for _ in range(2):
        record = SQLiteArtifactStore(path).get("artifact.old")
        assert record.storage_kind == "project"
        assert record.project_id == "old-project"
        assert record.execution == {}
        assert record.shared_with_user_ids == ()


def test_reconciliation_reports_ambiguity_without_selecting_a_copy(setup):
    _, skill, store, _, resources = setup
    record, root = register(setup)
    duplicate = Path(skill.directory) / "reader.py"
    duplicate.write_text("print('different copy')")
    report = resources.reconcile(record, "alex")
    assert report["reconciliation"] == "ambiguous"
    assert len(report["location_candidates"]) == 2
    assert store.get(record.artifact_id).storage_kind == "shared"
    assert duplicate.read_text() == "print('different copy')"
    assert (root / "reading.txt").read_text() == "42"


def test_relocation_updates_definition_and_preserves_operating_instructions_and_files(setup):
    skills, skill, store, _, resources = setup
    record, root = register(setup)
    target = Path(skill.directory) / "replacement.py"
    target.write_text("print('replacement')")
    with pytest.raises(PermissionError, match="manager_required"):
        resources.relocate(record.artifact_id, "gaby", storage_kind="skill", entrypoint_path="replacement.py", execution={"runtime_path": sys.executable})
    saved = resources.relocate(record.artifact_id, "alex", storage_kind="skill", entrypoint_path="replacement.py", execution={"runtime_path": sys.executable})
    assert saved.artifact_id == record.artifact_id
    instructions = skills.get(skill.skill_id).instructions
    assert '"entrypoint_path": "replacement.py"' in instructions
    assert '"argument_schema": {' in instructions
    assert "Read the fixture; never change it." in instructions
    assert (root / "reader.py").is_file() and target.is_file()
    assert SQLiteArtifactStore(store.db_path).get(record.artifact_id).execution["runtime_path"] == sys.executable


def test_archiving_or_deleting_a_goal_project_does_not_hide_registered_resource(setup, tmp_path):
    _, skill, store, projects, resources = setup
    project = projects.create_project(name="Finite goal", owner_user_id="alex", root_path=str(tmp_path / "finite"))
    record, root = register(setup, kind="project", project=project)
    projects.set_status(project.project_id, "archived", requester_user_id="alex")
    assert resources.require_ready(record.artifact_id, "alex", [skill.skill_id])["storage_root"] == str(root)
    projects.delete_project(project.project_id, requester_user_id="alex")
    assert resources.require_ready(record.artifact_id, "alex", [skill.skill_id])["storage_root"] == str(root)
    assert (root / "reader.py").is_file()


def test_capture_legacy_roots_is_metadata_only_and_survives_project_deletion(setup, tmp_path):
    _, skill, store, projects, resources = setup
    project = projects.create_project(name="Old", owner_user_id="alex", root_path=str(tmp_path / "old"))
    record, root = register(setup, kind="project", project=project)
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("UPDATE v2_artifacts SET legacy_root_path='' WHERE artifact_id=?", (record.artifact_id,))
    before = (root / "reader.py").read_bytes()
    store.capture_legacy_roots(projects)
    store.capture_legacy_roots(projects)
    assert store.get(record.artifact_id).legacy_root_path == str(root)
    projects.set_status(project.project_id, "archived", requester_user_id="alex")
    projects.delete_project(project.project_id, requester_user_id="alex")
    assert resources.require_ready(record.artifact_id, "alex", [skill.skill_id])["working_directory"] == str(root)
    assert (root / "reader.py").read_bytes() == before


def test_registration_rolls_back_catalog_if_skill_definition_cannot_be_saved(setup, monkeypatch):
    skills, _, store, _, _ = setup
    monkeypatch.setattr(skills, "add_artifact_instructions", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        register(setup)
    assert store.get("artifact.fixture") is None


def test_unregistration_does_not_delete_shared_program_data_or_configuration(setup):
    _, _, store, _, _ = setup
    record, root = register(setup)
    (root / "config.env").write_text("configuration")
    before = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
    store.delete(record.artifact_id)
    assert {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()} == before


def test_ipc_exposes_diagnostics_and_authorized_location_updates_without_touching_files(setup, tmp_path):
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.runtime import build_runtime_host
    from alphonse.agent_v2.users import V2UserStore

    skills, skill, store, projects, _ = setup
    record, root = register(setup)
    users = V2UserStore(":memory:")
    users.set_workspace_root(tmp_path)
    users.onboard(display_name="Alex", user_id="alex")
    runtime = build_runtime_host(user_store=users, skill_store=skills, artifact_store=store, project_store=projects)
    daemon = V2Daemon(runtime)
    workspace = daemon.ipc._dispatch({"method": "skills", "params": {"actor_user_id": "alex"}})
    owned = next(item for item in workspace["skills"] if item["id"] == skill.skill_id)
    assert owned["artifacts"][0]["resource"]["status"] == "ready"
    assert workspace["artifact_repository_root"] == str(tmp_path / "artifacts")
    report = daemon.ipc._dispatch({"method": "reconcile_artifacts", "params": {"actor_user_id": "alex"}})
    assert report["artifacts"][0]["reconciliation"] == "registered"
    replacement = Path(skill.directory) / "replacement.py"
    replacement.write_text("print('replacement')")
    request = {"method": "relocate_artifact", "params": {"actor_user_id": "gaby", "artifact_id": record.artifact_id,
               "storage_kind": "skill", "entrypoint_path": "replacement.py", "execution": {"runtime_path": sys.executable}}}
    with pytest.raises(PermissionError, match="manager_required"):
        daemon.ipc._dispatch(request)
    request["params"]["actor_user_id"] = "alex"
    updated = daemon.ipc._dispatch(request)["artifact"]
    assert updated["resource"]["resolved_entrypoint_path"] == str(replacement)
    daemon.update_artifact(actor_user_id="alex", artifact_id=record.artifact_id, name="New name", description="New description")
    assert "New description" in skills.get(skill.skill_id).instructions
    with pytest.raises(ValueError, match="skill_has_registered_artifacts"):
        daemon.delete_skill(actor_user_id="alex", skill_id=skill.skill_id)
    daemon.delete_artifact(actor_user_id="alex", artifact_id=record.artifact_id)
    assert replacement.is_file() and (root / "reader.py").is_file()


def test_fresh_workspace_initializes_shared_repository_without_any_project(tmp_path):
    from alphonse.agent_v2.runtime import build_runtime_host
    from alphonse.agent_v2.users import V2UserStore

    users = V2UserStore(":memory:")
    users.set_workspace_root(tmp_path)
    runtime = build_runtime_host(user_store=users, skill_store=SkillStore(tmp_path / "skills"))
    assert (tmp_path / "artifacts").is_dir()
    assert runtime.project_store.list_visible_projects("alex") == []
    assert runtime.core.artifact_resources.repository_root == tmp_path / "artifacts"
    assert not any(item.tool_id.startswith("artifact.") for item in runtime.core.tools.list())


def test_python_registration_requires_explicit_runtime_instead_of_path_guessing(setup):
    _, _, store, _, _ = setup
    with pytest.raises(ValueError, match="runtime_not_registered"):
        register(setup, execution={"working_directory": "."})
    assert store.get("artifact.fixture") is None


def test_native_catalog_location_repair_preserves_schema_and_is_not_an_artifact_tool(setup):
    from alphonse.agent_v2.core.tools.registry.native.artifact_metadata_update import execute_artifact_metadata_update
    from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry

    skills, skill, store, _, resources = setup
    record, root = register(setup)
    (Path(skill.directory) / "repaired.py").write_text("print('fixture')")
    task = TaskState(user="alex", project_id="")
    context = ToolExecutionContext(task=task, messages=InMemoryMessageQueue(), artifact_resources=resources)
    result = execute_artifact_metadata_update({"artifact_id": record.artifact_id, "name": "Verified fixture", "description": "Updated reader",
        "storage_kind": "skill", "entrypoint_path": "repaired.py", "execution": {"runtime_path": sys.executable}}, context=context, store=store)
    assert result["artifact"]["storage_kind"] == "skill"
    assert result["artifact"]["argument_schema"] == record.argument_schema
    assert (root / "reader.py").is_file()
    tools = build_native_tool_registry(artifact_store=store, skill_store=skills).list()
    assert any(item.tool_id == "native.artifact_metadata_update" for item in tools)
    assert not any(item.tool_id.startswith("artifact.") for item in tools)


def test_onboarding_wires_skill_library_and_artifact_repository_to_new_workspace(tmp_path):
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.runtime import build_runtime_host
    from alphonse.agent_v2.users import V2UserStore

    skills = SkillStore(tmp_path / "old-skills")
    skills.create_skill("sensor", "Disposable reader", "# Sensor\nRead its registered resources.")
    users = V2UserStore(":memory:")
    # Keep pre-workspace user/profile migration inside the disposable fixture.
    users.set_users_root(tmp_path / "old-users")
    runtime = build_runtime_host(user_store=users, skill_store=skills)
    workspace = tmp_path / "workspace"
    V2Daemon(runtime).onboard(display_name="Alex", workspace_root=str(workspace))
    assert (workspace / "artifacts").is_dir()
    assert runtime.skill_store.skills_dir == workspace / "skills"
    assert runtime.skill_store.get("skill:sensor").directory == str(workspace / "skills" / "sensor")


def test_candidate_projection_keeps_complete_authorized_artifact_instructions(setup, monkeypatch):
    from alphonse.agent_v2.core.intelligence.v3.projections import project
    skills, skill, _, _, resources = setup
    instructions = 'Read this disposable fixture carefully.\n' * 500 + 'ARTIFACT_INSTRUCTION_TAIL'
    record, root = register(setup, instructions=instructions)
    task = TaskState(user='alex', goal='Read fixture', metadata={'v3_skill_selection': {'selected_skill_ids': [skill.skill_id]}})
    context = CoreLoopContext(messages=InMemoryMessageQueue(), skill_store=skills, artifact_resources=resources)
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_TOKEN_BUDGET', '60000')
    text = project(task, context, purpose='execution')
    assert 'ARTIFACT_INSTRUCTION_TAIL' in text
    assert str(root / 'reader.py') in text
    assert 'Registered argument schema' in text
    assert build_artifact_tool_definitions(resources.store, resources.projects) == []
