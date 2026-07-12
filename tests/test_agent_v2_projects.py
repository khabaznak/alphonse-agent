from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.core import ToolExecutionContext
from alphonse.agent_v2.core.intelligence.pdca.nodes.act_node import _render_acceptance_criteria_prompt
from alphonse.agent_v2.core.intelligence.pdca.nodes.check_node import _render_criteria_review_prompt
from alphonse.agent_v2.core.intelligence.pdca.nodes.plan_node import _render_tool_call_plan_prompt
from alphonse.agent_v2.core.intelligence.pdca.nodes.plan_node import plan_node
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.core.tools.registry.native import BASH_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native import BASH_TOOL_NAME
from alphonse.agent_v2.core.tools.registry.native import build_bash_tool_definition
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.core.tools.registry.native import execute_bash
from alphonse.agent_v2.interfaces.tui import build_tui_runtime
from alphonse.agent_v2.interfaces.tui import create_tui_project
from alphonse.agent_v2.interfaces.tui import detect_tui_slash_command
from alphonse.agent_v2.interfaces.tui import queue_tui_input
from alphonse.agent_v2.interfaces.tui import save_tui_project_context
from alphonse.agent_v2.interfaces.tui import select_tui_project


def test_project_store_creates_directory_and_context_file(tmp_path: Path) -> None:
    store = ProjectStore(":memory:")
    root = tmp_path / "alpha"

    project = store.create_project(
        name="Alpha",
        description="Important work",
        root_path=str(root),
        visibility="private",
        owner_user_id="alex",
    )
    updated = store.write_project_context(project.project_id, "# Context\nBuild carefully.", requester_user_id="alex")

    assert root.is_dir()
    assert project.context_path == str(root / "project_context.md")
    assert store.read_project_context(project.project_id, requester_user_id="alex") == "# Context\nBuild carefully."
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
    with pytest.raises(PermissionError, match="project_owner_required"):
        store.update_project(project.project_id, name="No", description="", visibility="private", requester_user_id="gaby")


def test_tui_project_commands_are_intercepted_and_not_queued(tmp_path: Path) -> None:
    runtime = build_tui_runtime(user="alex", project_store=ProjectStore(":memory:"))

    result = queue_tui_input(runtime, "/project")
    indented = queue_tui_input(runtime, " /project")

    assert result.command == "project"
    assert result.queued is False
    assert runtime.queue.size() == 1
    assert indented.queued is True
    assert indented.command == ""


def test_tui_project_selection_routes_future_messages(tmp_path: Path) -> None:
    runtime = build_tui_runtime(user="alex", project_store=ProjectStore(":memory:"))
    project = create_tui_project(
        runtime,
        name="Alpha",
        description="",
        root_path=str(tmp_path / "alpha"),
        visibility="private",
    )
    save_tui_project_context(runtime, "Use the alpha rules.")

    result = queue_tui_input(runtime, "List files")

    assert runtime.active_project_id == project.project_id
    assert result.queued is True
    queued = runtime.queue.peek()
    assert queued is not None
    assert queued.message.project_id == project.project_id
    assert runtime.project_store.read_project_context(project.project_id, requester_user_id="alex") == "Use the alpha rules."


def test_select_tui_project_sets_active_project(tmp_path: Path) -> None:
    runtime = build_tui_runtime(user="alex", project_store=ProjectStore(":memory:"))
    project = runtime.project_store.create_project(
        name="Alpha",
        root_path=str(tmp_path / "alpha"),
        owner_user_id="alex",
    )

    selected = select_tui_project(runtime, project.project_id)

    assert selected.project_id == project.project_id
    assert runtime.active_project_id == project.project_id


def test_detect_tui_slash_command_requires_first_character() -> None:
    assert detect_tui_slash_command("/project") == "project"
    assert detect_tui_slash_command("/project-context") == "project-context"
    assert detect_tui_slash_command(" /project") == ""
    assert detect_tui_slash_command("please /project") == ""


def test_prompt_render_paths_include_project_context(tmp_path: Path) -> None:
    store = ProjectStore(":memory:")
    project = store.create_project(name="Alpha", description="Build docs", root_path=str(tmp_path / "alpha"), owner_user_id="alex")
    store.write_project_context(project.project_id, "Prefer concise markdown.", requester_user_id="alex")
    task = TaskState(
        goal="Write docs",
        user="alex",
        project_id=project.project_id,
        acceptance_criteria_md="1.- [ ] Docs exist",
    )
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=build_native_tool_registry(), project_store=store)

    plan_node(task, context=context)

    assert "Prefer concise markdown." in task.metadata["tool_call_plan_prompt"]
    assert "Alpha" in task.metadata["tool_call_plan_prompt"]
    assert "Prefer concise markdown." in _render_tool_call_plan_prompt(
        task,
        tuple(),
        project_context_md=store.render_project_context(project.project_id, requester_user_id="alex"),
    )
    assert "Prefer concise markdown." in _render_acceptance_criteria_prompt(
        task,
        project_context_md=store.render_project_context(project.project_id, requester_user_id="alex"),
    )
    assert "Prefer concise markdown." in _render_criteria_review_prompt(
        task,
        {},
        project_context_md=store.render_project_context(project.project_id, requester_user_id="alex"),
    )


def test_capd_prompts_keep_user_and_project_contexts_isolated(tmp_path: Path) -> None:
    store = ProjectStore(":memory:")
    alex_project = store.create_project(name="Alex project", root_path=str(tmp_path / "alex"), owner_user_id="alex")
    gaby_project = store.create_project(name="Gaby project", root_path=str(tmp_path / "gaby"), owner_user_id="gaby")
    store.write_project_context(alex_project.project_id, "ALEX_PROJECT_SENTINEL", requester_user_id="alex")
    store.write_project_context(gaby_project.project_id, "GABY_PROJECT_SENTINEL", requester_user_id="gaby")
    context = CoreLoopContext(
        messages=InMemoryMessageQueue(),
        tools=build_native_tool_registry(),
        project_store=store,
        user_context_provider=lambda user_id: f"{user_id.upper()}_USER_SENTINEL",
    )

    alex_task = TaskState(goal="Respond", user="alex", project_id=alex_project.project_id, acceptance_criteria_md="1.- [ ] Reply")
    gaby_task = TaskState(goal="Respond", user="gaby", project_id=gaby_project.project_id, acceptance_criteria_md="1.- [ ] Reply")
    plan_node(alex_task, context=context)
    plan_node(gaby_task, context=context)

    alex_prompt = alex_task.metadata["tool_call_plan_prompt"]
    gaby_prompt = gaby_task.metadata["tool_call_plan_prompt"]
    assert "ALEX_USER_SENTINEL" in alex_prompt
    assert "ALEX_PROJECT_SENTINEL" in alex_prompt
    assert "GABY_USER_SENTINEL" not in alex_prompt
    assert "GABY_PROJECT_SENTINEL" not in alex_prompt
    assert "GABY_USER_SENTINEL" in gaby_prompt
    assert "GABY_PROJECT_SENTINEL" in gaby_prompt
    assert "ALEX_USER_SENTINEL" not in gaby_prompt
    assert "ALEX_PROJECT_SENTINEL" not in gaby_prompt


def test_capd_prompt_uses_the_project_attached_to_each_new_task(tmp_path: Path) -> None:
    store = ProjectStore(":memory:")
    first = store.create_project(name="First", root_path=str(tmp_path / "first"), owner_user_id="alex")
    second = store.create_project(name="Second", root_path=str(tmp_path / "second"), owner_user_id="alex")
    store.write_project_context(first.project_id, "FIRST_PROJECT_SENTINEL", requester_user_id="alex")
    store.write_project_context(second.project_id, "SECOND_PROJECT_SENTINEL", requester_user_id="alex")
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=build_native_tool_registry(), project_store=store)

    first_task = TaskState(goal="Respond", user="alex", project_id=first.project_id, acceptance_criteria_md="1.- [ ] Reply")
    second_task = TaskState(goal="Respond", user="alex", project_id=second.project_id, acceptance_criteria_md="1.- [ ] Reply")
    plan_node(first_task, context=context)
    plan_node(second_task, context=context)

    assert "FIRST_PROJECT_SENTINEL" in first_task.metadata["tool_call_plan_prompt"]
    assert "SECOND_PROJECT_SENTINEL" not in first_task.metadata["tool_call_plan_prompt"]
    assert "SECOND_PROJECT_SENTINEL" in second_task.metadata["tool_call_plan_prompt"]
    assert "FIRST_PROJECT_SENTINEL" not in second_task.metadata["tool_call_plan_prompt"]


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
