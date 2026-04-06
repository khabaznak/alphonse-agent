from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TaskRecord:
    task_id: str | None = None
    user_id: str | None = None
    goal: str = ""
    facts_md: str = "- (none)"
    recent_conversation_md: str = "- (none)"
    plan_md: str = "- (none)"
    acceptance_criteria_md: str = "- (none)"
    memory_facts_md: str = "- (none)"
    tool_call_history_md: str = "- (none)"
    status: str = "running"
    outcome: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "goal": self.goal,
            "facts_md": self.facts_md,
            "recent_conversation_md": self.recent_conversation_md,
            "plan_md": self.plan_md,
            "acceptance_criteria_md": self.acceptance_criteria_md,
            "memory_facts_md": self.memory_facts_md,
            "tool_call_history_md": self.tool_call_history_md,
            "status": self.status,
            "outcome": dict(self.outcome) if isinstance(self.outcome, dict) else None,
        }

    def append_fact(self, fact: str) -> None:
        self.facts_md = _append_markdown_line(self.facts_md, fact)

    def get_facts_md(self) -> str:
        return self.facts_md if str(self.facts_md or "").strip() else "- (none)"

    def append_plan_line(self, line: str) -> None:
        self.plan_md = _append_markdown_line(self.plan_md, line)

    def get_plan_md(self) -> str:
        return self.plan_md if str(self.plan_md or "").strip() else "- (none)"

    def append_acceptance_criterion(self, criterion: str) -> None:
        self.acceptance_criteria_md = _append_markdown_line(self.acceptance_criteria_md, criterion)

    def get_acceptance_criteria_md(self) -> str:
        return self.acceptance_criteria_md if str(self.acceptance_criteria_md or "").strip() else "- (none)"

    def append_memory_fact(self, fact: str) -> None:
        self.memory_facts_md = _append_markdown_line(self.memory_facts_md, fact)

    def get_memory_facts_md(self) -> str:
        return self.memory_facts_md if str(self.memory_facts_md or "").strip() else "- (none)"

    def append_tool_call_history_entry(self, entry: str) -> None:
        self.tool_call_history_md = _append_markdown_line(self.tool_call_history_md, entry)

    def get_tool_call_history_md(self) -> str:
        return self.tool_call_history_md if str(self.tool_call_history_md or "").strip() else "- (none)"

    def set_recent_conversation_md(self, content: str) -> None:
        self.recent_conversation_md = str(content or "").strip() or "- (none)"

    def append_recent_conversation_line(self, line: str) -> None:
        self.recent_conversation_md = _append_markdown_line(self.recent_conversation_md, line)

    def clear_acceptance_criteria(self) -> None:
        self.acceptance_criteria_md = "- (none)"

    def replan(self) -> None:
        self.goal = ""
        self.clear_acceptance_criteria()
        self.status = "running"
        self.outcome = None


def _append_markdown_line(current: str, value: str) -> str:
    rendered = str(value or "").strip()
    if not rendered:
        return current
    line = rendered if rendered.startswith("- ") else f"- {rendered}"
    existing = str(current or "").strip()
    if not existing or existing == "- (none)":
        return line
    return f"{existing}\n{line}"
