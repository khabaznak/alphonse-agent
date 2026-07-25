from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent_v2.artifacts import SQLiteArtifactStore
from alphonse.agent_v2.artifacts import build_artifact_tool_definitions
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.core.tools.registry.native.artifact_registration import execute_artifact_registration


def _executable(root: Path, name: str = "artifact.py") -> Path:
    path = root / name
    path.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps({'value': json.load(sys.stdin)['value']}))\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)
    return path


def _project(tmp_path: Path):
    projects = ProjectStore(str(tmp_path / "projects.sqlite3"))
    project = projects.create_project(name="Home", root_path=str(tmp_path / "home"), owner_user_id="alex")
    return projects, project


def test_registration_requires_owner_and_project_local_executable(tmp_path: Path) -> None:
    projects, project = _project(tmp_path); _executable(Path(project.root_path))
    store = SQLiteArtifactStore(tmp_path / "artifacts.sqlite3")
    task = TaskState(task_id="task", user="alex", project_id=project.project_id, goal="register")
    # ToolExecutionContext is intentionally constructed directly for the native executor.
    from alphonse.agent_v2.core.core import ToolExecutionContext
    result = execute_artifact_registration({"name": "Sensor check", "description": "Reads sensor", "entrypoint_path": "artifact.py", "argument_schema": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}}, context=ToolExecutionContext(task=task, messages=InMemoryMessageQueue(), project_store=projects), store=store)
    assert result["artifact"]["artifact_id"] == "artifact.sensor-check"
    assert store.get("artifact.sensor-check") is not None
    with pytest.raises(ValueError, match="outside_project"):
        execute_artifact_registration({"name": "Bad", "description": "Bad", "entrypoint_path": "../outside", "argument_schema": {"type": "object"}}, context=ToolExecutionContext(task=task, messages=InMemoryMessageQueue(), project_store=projects), store=store)


def test_enabled_artifact_executes_json_and_disabled_is_not_materialized(tmp_path: Path) -> None:
    projects, project = _project(tmp_path); _executable(Path(project.root_path))
    store = SQLiteArtifactStore(tmp_path / "artifacts.sqlite3")
    record = store.register(artifact_id="artifact.sensor", name="Sensor", description="Read", project_id=project.project_id, owner_user_id="alex", entrypoint_path="artifact.py", argument_schema={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]})
    tools = build_artifact_tool_definitions(store, projects)
    assert tools[0].callable({"value": "42"}) == {"value": "42"}
    store.set_enabled(record.artifact_id, False)
    assert build_artifact_tool_definitions(store, projects) == []


def test_unregister_does_not_delete_program_or_data(tmp_path: Path) -> None:
    projects, project = _project(tmp_path); script = _executable(Path(project.root_path)); data = Path(project.root_path) / "data.sqlite"; data.write_text("data")
    store = SQLiteArtifactStore(tmp_path / "artifacts.sqlite3")
    store.register(artifact_id="artifact.sensor", name="Sensor", description="Read", project_id=project.project_id, owner_user_id="alex", entrypoint_path="artifact.py", argument_schema={"type": "object"})
    store.delete("artifact.sensor")
    assert script.exists() and data.exists()
