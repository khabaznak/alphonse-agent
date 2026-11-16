from __future__ import annotations

from pathlib import Path

from jinja2 import Environment
from jinja2 import FileSystemLoader

from alphonse.agent_v2.core.intelligence.task_state import TaskState


TEMPLATE_DIR = Path("alphonse/agent_v2/core/intelligence/templates")




def test_acceptance_criteria_prompt_template_renders_expected_sections() -> None:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

    rendered = env.get_template("acceptance_criteria_prompt.j2").render(
        check_verdict="new",
        check_reason="No acceptance criteria were present.",
        existing_acceptance_criteria_md="- (none)",
        task_state_md="Task state markdown",
    )

    assert "# System Prompt" in rendered
    assert "1.- [ ] The required outcome is true" in rendered
    assert "2.- [x] An already completed requirement is true" in rendered
    assert "## Philosophy.md" in rendered
    assert "## GlobalContext.md" in rendered
    assert "## User Context" in rendered
    assert "## Project Context" in rendered
    assert "# Check Verdict" in rendered
    assert "new" in rendered
    assert "# Existing Acceptance Criteria" in rendered
    assert "# Task State" in rendered
    assert "Task state markdown" in rendered




def test_task_state_exposes_pending_silent_bash_confirmation_to_planner() -> None:
    task = TaskState(
        goal="Add sunglasses",
        metadata={
            "pending_silent_bash_confirmation": {
                "tool_call_id": "bash-call",
                "internal_state": "Added sunglasses to the TODO list.",
            }
        },
    )

    rendered = task.to_markdown_prompt()

    assert "# Required User Confirmation" in rendered
    assert "Call `native.respond` next" in rendered
    assert "Added sunglasses to the TODO list." in rendered


def test_task_state_describes_desktop_project_file_attachments() -> None:
    task = TaskState(
        goal="Analyze the attached files",
        metadata={
            "attachments": [
                {
                    "filename": "notes.txt",
                    "mime_type": "text/plain",
                    "kind": "desktop_project_file",
                    "ingestion_status": "copied",
                    "project_path": "/tmp/project/notes.txt",
                }
            ]
        },
    )

    rendered = task.to_markdown_prompt()

    assert "Name: notes.txt" in rendered
    assert "Project path: /tmp/project/notes.txt" in rendered
    assert "copied into the selected project directory" in rendered
    assert "Inspect them with normal local tools" in rendered
