from __future__ import annotations

import pytest

from alphonse.agent_v2.core.core import ToolExecutionContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import MutationScope
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.core.tools.registry.native.exact_text_edit import execute_exact_text_edit
from alphonse.agent_v2.core.tools.registry.native.project_files import execute_project_read
from alphonse.agent_v2.core.tools.registry.native.project_files import execute_project_search


def _context(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    projects = ProjectStore(":memory:")
    project = projects.create_project(name="Test", root_path=str(root), owner_user_id="alex")
    task = TaskState(user="alex", project_id=project.project_id)
    return root, ToolExecutionContext(task=task, messages=InMemoryMessageQueue(), project_store=projects)


def test_project_search_excludes_internal_alphonse_memory(tmp_path) -> None:
    root, context = _context(tmp_path)
    (root / "backlog.md").write_text("- [ ] real target\n", encoding="utf-8")
    internal = root / ".alphonse" / "memory"
    internal.mkdir(parents=True)
    (internal / "ledger.md").write_text("- [ ] real target\n", encoding="utf-8")

    result = execute_project_search({"query": "real target"}, context=context)

    assert result["match_count"] == 1
    assert result["matches"][0]["path"] == "backlog.md"


def test_project_search_fails_closed_when_no_project_record_matches(tmp_path) -> None:
    root, context = _context(tmp_path)
    (root / "backlog.md").write_text("- [ ] unrelated item\n", encoding="utf-8")

    with pytest.raises(LookupError, match="project_search_no_matches"):
        execute_project_search({"query": "solar record"}, context=context)


def test_project_read_and_exact_edit_reject_internal_alphonse_paths(tmp_path) -> None:
    root, context = _context(tmp_path)
    internal = root / ".alphonse" / "memory"
    internal.mkdir(parents=True)
    (internal / "ledger.md").write_text("secret", encoding="utf-8")

    with pytest.raises(PermissionError, match="project_tool_path_protected"):
        execute_project_read({"path": ".alphonse/memory/ledger.md"}, context=context)
    with pytest.raises(PermissionError, match="exact_text_edit_path_protected"):
        execute_exact_text_edit({
            "path": ".alphonse/memory/ledger.md", "expected_text": "secret", "replacement_text": "changed",
        }, context=context)


def test_v3_mutation_scope_rejects_internal_alphonse_paths() -> None:
    with pytest.raises(ValueError, match="mutation_scope_path_protected"):
        MutationScope((".alphonse/memory/sessions/current/ledger-0001.md",))
