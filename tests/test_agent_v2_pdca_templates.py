from __future__ import annotations

from alphonse.agent_v2.core.intelligence.task_state import TaskState


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
