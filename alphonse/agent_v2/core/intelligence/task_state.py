"""Markdown-first task state for the future v2 PDCA intelligence graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import select_autoescape

from alphonse.agent_v2.core.core import CoreMessage

if TYPE_CHECKING:
    from alphonse.agent_v2.core.messages.queue import QueuedMessage


CHECK_VERDICTS = {"new", "steer", "wip", "mission_success", "mission_failed"}
EMPTY_MARKDOWN = "- (none)"
_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


@dataclass
class TaskState:
    """Markdown container carried through the future PDCA processor."""

    task_id: str | None = None
    message_id: str | None = None
    user: str | None = None
    project_id: str = ""
    tag: str = ""
    correlation_id: str = ""
    goal: str = ""
    facts_md: str = EMPTY_MARKDOWN
    recent_conversation_md: str = EMPTY_MARKDOWN
    plan_md: str = EMPTY_MARKDOWN
    acceptance_criteria_md: str = EMPTY_MARKDOWN
    memory_facts_md: str = EMPTY_MARKDOWN
    tool_call_history_md: str = EMPTY_MARKDOWN
    updates_md: str = EMPTY_MARKDOWN
    status: str = "running"
    outcome: dict[str, Any] | None = None
    check_verdict: str | None = None
    check_reason: str = ""
    check_confidence: float = 0.0
    check_evidence_refs: list[str] | None = None
    check_new_message_count: int = 0
    pdca_cycle_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_message(cls, message: CoreMessage, message_id: str | None = None) -> "TaskState":
        """Create task state from a canonical queued message."""
        prompt = str(message.prompt or "").strip()
        state = cls(
            message_id=str(message_id or "").strip() or None,
            user=str(message.user or "").strip() or None,
            project_id=str(message.project_id or "").strip(),
            tag=str(message.tag or "").strip(),
            correlation_id=str(message.correlation_id or "").strip(),
            goal=prompt,
            metadata=dict(message.metadata),
        )
        state.append_conversation_message(message.user, prompt)
        return state

    @classmethod
    def from_queued_message(cls, queued: QueuedMessage) -> "TaskState":
        """Create task state from a queue-owned message envelope."""
        return cls.from_message(queued.message, message_id=queued.message_id)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskState":
        """Restore task state from a plain dictionary."""
        outcome = value.get("outcome")
        refs = value.get("check_evidence_refs")
        return cls(
            task_id=_optional_string(value.get("task_id")),
            message_id=_optional_string(value.get("message_id")),
            user=_optional_string(value.get("user")),
            project_id=str(value.get("project_id") or "").strip(),
            tag=str(value.get("tag") or "").strip(),
            correlation_id=str(value.get("correlation_id") or "").strip(),
            goal=str(value.get("goal") or "").strip(),
            facts_md=_markdown_or_default(value.get("facts_md")),
            recent_conversation_md=_markdown_or_default(value.get("recent_conversation_md")),
            plan_md=_markdown_or_default(value.get("plan_md")),
            acceptance_criteria_md=_markdown_or_default(value.get("acceptance_criteria_md")),
            memory_facts_md=_markdown_or_default(value.get("memory_facts_md")),
            tool_call_history_md=_markdown_or_default(value.get("tool_call_history_md")),
            updates_md=_markdown_or_default(value.get("updates_md")),
            status=str(value.get("status") or "").strip() or "running",
            outcome=dict(outcome) if isinstance(outcome, dict) else None,
            check_verdict=_normalize_check_verdict(value.get("check_verdict")),
            check_reason=str(value.get("check_reason") or "").strip(),
            check_confidence=_coerce_float(value.get("check_confidence")),
            check_evidence_refs=[str(item).strip() for item in refs if str(item).strip()]
            if isinstance(refs, list)
            else [],
            check_new_message_count=max(0, _coerce_int(value.get("check_new_message_count"))),
            pdca_cycle_count=max(0, _coerce_int(value.get("pdca_cycle_count"))),
            metadata=dict(value.get("metadata")) if isinstance(value.get("metadata"), dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize task state to plain Python values."""
        return {
            "task_id": self.task_id,
            "message_id": self.message_id,
            "user": self.user,
            "project_id": self.project_id,
            "tag": self.tag,
            "correlation_id": self.correlation_id,
            "goal": self.goal,
            "facts_md": self.facts_md,
            "recent_conversation_md": self.recent_conversation_md,
            "plan_md": self.plan_md,
            "acceptance_criteria_md": self.acceptance_criteria_md,
            "memory_facts_md": self.memory_facts_md,
            "tool_call_history_md": self.tool_call_history_md,
            "updates_md": self.updates_md,
            "status": self.status,
            "outcome": dict(self.outcome) if isinstance(self.outcome, dict) else None,
            "check_verdict": self.check_verdict,
            "check_reason": self.check_reason,
            "check_confidence": self.check_confidence,
            "check_evidence_refs": list(self.check_evidence_refs or []),
            "check_new_message_count": self.check_new_message_count,
            "pdca_cycle_count": self.pdca_cycle_count,
            "metadata": dict(self.metadata or {}),
        }

    def to_markdown_prompt(self) -> str:
        """Render this task state into the markdown prompt container."""
        env = Environment(
            loader=FileSystemLoader(_TEMPLATE_DIR),
            autoescape=select_autoescape(default_for_string=False),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = env.get_template("task_state_prompt.md.j2")
        return template.render(task=self).strip()

    def append_fact(self, fact: str) -> None:
        self.facts_md = _append_markdown_line(self.facts_md, fact)

    def append_plan_line(self, line: str) -> None:
        self.plan_md = _append_markdown_line(self.plan_md, line)

    def append_acceptance_criterion(self, criterion: str) -> None:
        self.acceptance_criteria_md = _append_markdown_line(self.acceptance_criteria_md, criterion)

    def append_memory_fact(self, fact: str) -> None:
        self.memory_facts_md = _append_markdown_line(self.memory_facts_md, fact)

    def append_tool_call_history_entry(self, entry: str) -> None:
        self.tool_call_history_md = _append_markdown_line(self.tool_call_history_md, entry)

    def append_recent_conversation_line(self, line: str) -> None:
        self.recent_conversation_md = _append_markdown_line(self.recent_conversation_md, line)

    def append_conversation_message(self, user: str, prompt: str) -> None:
        speaker = str(user or "").strip() or "unknown"
        rendered_prompt = str(prompt or "").strip()
        if not rendered_prompt:
            return
        escaped_prompt = rendered_prompt.replace('"', '\\"')
        self.append_recent_conversation_line(f'{speaker}: "{escaped_prompt}"')

    def append_update(self, update: str) -> None:
        self.updates_md = _append_markdown_line(self.updates_md, update)

    def clear_acceptance_criteria(self) -> None:
        self.acceptance_criteria_md = EMPTY_MARKDOWN

    def set_correlation_id(self, correlation_id: str) -> None:
        self.correlation_id = str(correlation_id or "").strip()

    def replan(self) -> None:
        self.goal = ""
        self.clear_acceptance_criteria()
        self.status = "running"
        self.outcome = None

    def set_check_result(
        self,
        *,
        verdict: str,
        reason: str = "",
        confidence: float = 0.0,
        evidence_refs: list[str] | None = None,
        new_message_count: int = 0,
    ) -> None:
        normalized = _normalize_check_verdict(verdict)
        if normalized is None:
            raise ValueError(f"invalid_check_verdict: {verdict}")
        self.check_verdict = normalized
        self.check_reason = str(reason or "").strip()
        self.check_confidence = max(0.0, min(1.0, _coerce_float(confidence)))
        self.check_evidence_refs = [str(item).strip() for item in evidence_refs or [] if str(item).strip()]
        self.check_new_message_count = max(0, _coerce_int(new_message_count))


def _append_markdown_line(current: str, value: str) -> str:
    rendered = str(value or "").strip()
    if not rendered:
        return current
    line = rendered if rendered.startswith("- ") else f"- {rendered}"
    existing = str(current or "").strip()
    if not existing or existing == EMPTY_MARKDOWN:
        return line
    return f"{existing}\n{line}"


def _normalize_check_verdict(value: Any) -> str | None:
    rendered = str(value or "").strip().lower()
    return rendered if rendered in CHECK_VERDICTS else None


def _markdown_or_default(value: Any) -> str:
    return str(value or "").strip() or EMPTY_MARKDOWN


def _optional_string(value: Any) -> str | None:
    rendered = str(value or "").strip()
    return rendered or None


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
