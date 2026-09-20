from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from alphonse.agent_v2.core.core import ToolExecutionContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.core.tools.registry.native.exact_text_edit import EXACT_TEXT_EDIT_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native.exact_text_edit import execute_exact_text_edit


@dataclass
class _Project:
    root_path: str


class _ProjectStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def get_project(self, project_id: str, requester_user_id: str | None = None) -> _Project | None:
        return _Project(str(self.root)) if project_id == "home" and requester_user_id == "alex" else None


def _context(tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(
        task=TaskState(project_id="home", user="alex"),
        messages=InMemoryMessageQueue(),
        project_store=_ProjectStore(tmp_path),
    )


def test_exact_text_edit_atomically_replaces_and_verifies_read_back(tmp_path: Path) -> None:
    target = tmp_path / "backlog.md"
    target.write_text("| Solar | Idea |\n| Paint | Pending |\n", encoding="utf-8")

    result = execute_exact_text_edit(
        {"path": "backlog.md", "expected_text": "| Solar | Idea |", "replacement_text": "| Solar | Complete |"},
        context=_context(tmp_path),
    )

    assert target.read_text(encoding="utf-8") == "| Solar | Complete |\n| Paint | Pending |\n"
    assert result["affected_paths"] == ["backlog.md"]
    assert result["before_sha256"] != result["after_sha256"]
    assert result["verification"]["status"] == "verified"
    assert "| Solar | Complete |" in result["verification"]["read_back_excerpt"]
    assert "+| Solar | Complete |" in result["diff"]
    assert "-| Solar | Idea |" in result["diff"]
    assert "+| Paint | Pending |" not in result["diff"]


def test_exact_text_edit_rejects_ambiguous_match_without_writing(tmp_path: Path) -> None:
    target = tmp_path / "notes.md"
    target.write_text("pending\npending\n", encoding="utf-8")

    with pytest.raises(ValueError, match="match_count_mismatch"):
        execute_exact_text_edit(
            {"path": "notes.md", "expected_text": "pending", "replacement_text": "done"},
            context=_context(tmp_path),
        )

    assert target.read_text(encoding="utf-8") == "pending\npending\n"


def test_exact_text_edit_rejects_path_outside_authorized_project(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("old", encoding="utf-8")
    try:
        with pytest.raises(PermissionError, match="outside_project"):
            execute_exact_text_edit(
                {"path": str(outside), "expected_text": "old", "replacement_text": "new"},
                context=_context(tmp_path),
            )
    finally:
        outside.unlink(missing_ok=True)


def test_exact_text_edit_is_registered_as_native_tool() -> None:
    descriptor = build_native_tool_registry().get(EXACT_TEXT_EDIT_TOOL_ID)

    assert descriptor is not None
    assert "verified post-write read-back" in descriptor.description
