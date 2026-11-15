from __future__ import annotations

from pathlib import Path
import json

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.core import ToolExecutionContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.core.tools.registry.native import BASH_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native import BASH_TOOL_NAME
from alphonse.agent_v2.core.tools.registry.native import build_bash_tool_definition
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.core.tools.registry.native import execute_bash


def test_project_store_creates_directory_and_config_file(tmp_path: Path) -> None:
    store = ProjectStore(":memory:")
    root = tmp_path / "alpha"

    project = store.create_project(
        name="Alpha",
        description="Important work",
        root_path=str(root),
        visibility="private",
        owner_user_id="alex",
    )
    updated = store.write_project_config(project.project_id, {"scope": "# Context\nBuild carefully."}, requester_user_id="alex")

    assert root.is_dir()
    assert project.config_path == str(root / "project_config.json")
    assert not Path(project.context_path).exists()
    assert store.read_project_config(project.project_id, requester_user_id="alex")["scope"] == "# Context\nBuild carefully."
    assert updated.updated_at >= project.updated_at


def test_project_store_filters_private_and_shared_projects(tmp_path: Path) -> None:
    store = ProjectStore(":memory:")
    private = store.create_project(
        name="Private",
        root_path=str(tmp_path / "private"),
        visibility="private",
        owner_user_id="alex",
    )
    shared = store.create_project(
        name="Shared",
        root_path=str(tmp_path / "shared"),
        visibility="shared",
        owner_user_id="alex",
    )

    assert store.list_visible_projects("gaby") == []
    store.add_member(shared.project_id, "gaby")
    alex_projects = {project.project_id for project in store.list_visible_projects("alex")}
    gaby_projects = {project.project_id for project in store.list_visible_projects("gaby")}

    assert alex_projects == {private.project_id, shared.project_id}
    assert gaby_projects == {shared.project_id}
    assert store.get_project(private.project_id, requester_user_id="gaby") is None


def test_project_store_archives_restores_and_updates_metadata(tmp_path: Path) -> None:
    store = ProjectStore(":memory:")
    project = store.create_project(name="Alpha", root_path=str(tmp_path / "alpha"), owner_user_id="alex")

    archived = store.archive_project(project.project_id, requester_user_id="alex")

    assert archived.status == "archived"
    assert archived.archived_at
    assert store.get_project(project.project_id, requester_user_id="alex") is None
    assert store.list_manageable_projects("alex", status="archived") == [archived]
    restored = store.restore_project(project.project_id, requester_user_id="alex")
    updated = store.update_project(project.project_id, name="Renamed", description="Details", visibility="shared", requester_user_id="alex")
    assert restored.status == "active"
    assert updated.name == "Renamed"
    assert updated.description == "Details"
    assert updated.visibility == "shared"
    store.add_member(project.project_id, "gaby")
    with pytest.raises(PermissionError, match="project_owner_required"):
        store.update_project(project.project_id, name="No", description="", visibility="private", requester_user_id="gaby")










def test_project_context_includes_configured_scope(tmp_path: Path) -> None:
    store = ProjectStore(":memory:")
    project = store.create_project(name="Alpha", description="Build docs", root_path=str(tmp_path / "alpha"), owner_user_id="alex")
    store.write_project_config(project.project_id, {"scope": "Prefer concise markdown."}, requester_user_id="alex")
    assert "Prefer concise markdown." in store.render_project_context(project.project_id, requester_user_id="alex")


def test_project_contexts_are_isolated_between_users(tmp_path: Path) -> None:
    store = ProjectStore(":memory:")
    alex_project = store.create_project(name="Alex project", root_path=str(tmp_path / "alex"), owner_user_id="alex")
    gaby_project = store.create_project(name="Gaby project", root_path=str(tmp_path / "gaby"), owner_user_id="gaby")
    store.write_project_config(alex_project.project_id, {"scope": "ALEX_PROJECT_SENTINEL"}, requester_user_id="alex")
    store.write_project_config(gaby_project.project_id, {"scope": "GABY_PROJECT_SENTINEL"}, requester_user_id="gaby")
    alex_prompt = store.render_project_context(alex_project.project_id, requester_user_id='alex')
    gaby_prompt = store.render_project_context(gaby_project.project_id, requester_user_id='gaby')
    assert "ALEX_PROJECT_SENTINEL" in alex_prompt
    assert "GABY_PROJECT_SENTINEL" not in alex_prompt
    assert store.render_project_context(gaby_project.project_id, requester_user_id="alex") == ""
    assert "GABY_PROJECT_SENTINEL" in gaby_prompt
    assert "ALEX_PROJECT_SENTINEL" not in gaby_prompt


def test_project_context_uses_requested_project(tmp_path: Path) -> None:
    store = ProjectStore(":memory:")
    first = store.create_project(name="First", root_path=str(tmp_path / "first"), owner_user_id="alex")
    second = store.create_project(name="Second", root_path=str(tmp_path / "second"), owner_user_id="alex")
    store.write_project_config(first.project_id, {"scope": "FIRST_PROJECT_SENTINEL"}, requester_user_id="alex")
    store.write_project_config(second.project_id, {"scope": "SECOND_PROJECT_SENTINEL"}, requester_user_id="alex")
    first_prompt = store.render_project_context(first.project_id, requester_user_id='alex')
    second_prompt = store.render_project_context(second.project_id, requester_user_id='alex')
    assert "FIRST_PROJECT_SENTINEL" in first_prompt
    assert "SECOND_PROJECT_SENTINEL" not in first_prompt
    assert "SECOND_PROJECT_SENTINEL" in second_prompt
    assert "FIRST_PROJECT_SENTINEL" not in second_prompt


def test_bash_defaults_to_project_root_when_cwd_omitted(tmp_path: Path) -> None:
    store = ProjectStore(":memory:")
    project = store.create_project(name="Alpha", root_path=str(tmp_path / "alpha"), owner_user_id="alex")
    task = TaskState(goal="pwd", user="alex", project_id=project.project_id)
    context = ToolExecutionContext(task=task, messages=InMemoryMessageQueue(), project_store=store)

    result = execute_bash({"command": "pwd"}, context=context)

    assert result["exit_code"] == 0
    assert result["cwd"] == str(tmp_path / "alpha")
    assert result["stdout"].strip() == str(tmp_path / "alpha")


def test_bash_tool_definition_accepts_context() -> None:
    definition = build_bash_tool_definition()

    assert definition.descriptor.tool_id == BASH_TOOL_ID
    assert definition.descriptor.name == BASH_TOOL_NAME
    assert definition.accepts_context is True
