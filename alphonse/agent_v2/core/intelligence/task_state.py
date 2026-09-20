"""Markdown-first task state for the future v2 PDCA intelligence graph."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import select_autoescape

from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.intelligence.acceptance_contract import contract_from_markdown
from alphonse.agent_v2.core.intelligence.acceptance_contract import normalize_contract
from alphonse.agent_v2.core.intelligence.acceptance_contract import render_contract
from alphonse.agent_v2.core.intelligence.acceptance_contract import required_criteria_complete

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
    memory_session_id: str = ""
    tag: str = ""
    correlation_id: str = ""
    goal: str = ""
    facts_md: str = EMPTY_MARKDOWN
    recent_conversation_md: str = EMPTY_MARKDOWN
    conversation_history_md: str = EMPTY_MARKDOWN
    plan_json: str = EMPTY_MARKDOWN
    acceptance_criteria_md: str = EMPTY_MARKDOWN
    acceptance_contract: dict[str, Any] = field(default_factory=dict)
    memory_facts_md: str = EMPTY_MARKDOWN
    updates_md: str = EMPTY_MARKDOWN
    status: str = "running"
    outcome: dict[str, Any] | None = None
    check_verdict: str | None = None
    check_reason: str = ""
    check_confidence: float = 0.0
    check_evidence_refs: list[str] | None = None
    evidence_journal: list[dict[str, Any]] = field(default_factory=list)
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
            memory_session_id=str(message.memory_session_id or message.metadata.get("memory_session_id") or "").strip(),
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
        raw_task_state = queued.message.metadata.get("task_state")
        if isinstance(raw_task_state, dict):
            restored = cls.from_dict(raw_task_state)
            restored.message_id = queued.message_id
            if queued.message.prompt:
                restored.append_conversation_message(queued.message.user, queued.message.prompt)
            if queued.message.user == restored.user:
                restored.merge_attachments(queued.message.metadata)
            return restored
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
            memory_session_id=str(value.get("memory_session_id") or "").strip(),
            tag=str(value.get("tag") or "").strip(),
            correlation_id=str(value.get("correlation_id") or "").strip(),
            goal=str(value.get("goal") or "").strip(),
            facts_md=_markdown_or_default(value.get("facts_md")),
            recent_conversation_md=_markdown_or_default(value.get("recent_conversation_md")),
            conversation_history_md=_markdown_or_default(value.get("conversation_history_md")),
            plan_json=_markdown_or_default(
                value.get("plan_json") if value.get("plan_json") is not None else value.get("plan_md")
            ),
            acceptance_criteria_md=_markdown_or_default(value.get("acceptance_criteria_md")),
            acceptance_contract=normalize_contract(
                value.get("acceptance_contract"),
                fallback_markdown=_markdown_or_default(value.get("acceptance_criteria_md")),
                source_message_id=str(value.get("message_id") or ""),
            ),
            memory_facts_md=_markdown_or_default(value.get("memory_facts_md")),
            updates_md=_markdown_or_default(value.get("updates_md")),
            status=str(value.get("status") or "").strip() or "running",
            outcome=dict(outcome) if isinstance(outcome, dict) else None,
            check_verdict=_normalize_check_verdict(value.get("check_verdict")),
            check_reason=str(value.get("check_reason") or "").strip(),
            check_confidence=_coerce_float(value.get("check_confidence")),
            check_evidence_refs=[str(item).strip() for item in refs if str(item).strip()]
            if isinstance(refs, list)
            else [],
            evidence_journal=[dict(item) for item in value.get("evidence_journal") or [] if isinstance(item, dict)],
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
            "memory_session_id": self.memory_session_id,
            "tag": self.tag,
            "correlation_id": self.correlation_id,
            "goal": self.goal,
            "facts_md": self.facts_md,
            "recent_conversation_md": self.recent_conversation_md,
            "conversation_history_md": self.conversation_history_md,
            "plan_json": self.plan_json,
            "acceptance_criteria_md": self.acceptance_criteria_md,
            "acceptance_contract": dict(self.acceptance_contract or {}),
            "memory_facts_md": self.memory_facts_md,
            "updates_md": self.updates_md,
            "status": self.status,
            "outcome": dict(self.outcome) if isinstance(self.outcome, dict) else None,
            "check_verdict": self.check_verdict,
            "check_reason": self.check_reason,
            "check_confidence": self.check_confidence,
            "check_evidence_refs": list(self.check_evidence_refs or []),
            "evidence_journal": [dict(item) for item in self.evidence_journal],
            "check_new_message_count": self.check_new_message_count,
            "pdca_cycle_count": self.pdca_cycle_count,
            "metadata": dict(self.metadata or {}),
        }

    def to_checkpoint_dict(self) -> dict[str, Any]:
        """Serialize resumable state without duplicating the project memory ledger."""
        checkpoint = self.to_dict()
        checkpoint["conversation_history_md"] = EMPTY_MARKDOWN
        checkpoint["metadata"] = {
            key: value
            for key, value in dict(self.metadata or {}).items()
            if not str(key).endswith("_prompt")
        }
        return checkpoint

    def to_markdown_prompt(self, *, include_memory: bool = True, include_plan: bool = True) -> str:
        """Render this task state into the markdown prompt container."""
        env = Environment(
            loader=FileSystemLoader(_TEMPLATE_DIR),
            autoescape=select_autoescape(default_for_string=False),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = env.get_template("task_state_prompt.md.j2")
        return template.render(
            task=self,
            conversation_history_md=self.conversation_history_md if include_memory else EMPTY_MARKDOWN,
            operational_plan_json=self.operational_plan_json() if include_plan else EMPTY_MARKDOWN,
        ).strip()

    def ensure_acceptance_contract(self) -> dict[str, Any]:
        """Create the structured contract once for legacy Markdown-only tasks."""
        self.acceptance_contract = normalize_contract(
            self.acceptance_contract,
            fallback_markdown=self.acceptance_criteria_md,
            source_message_id=str(self.message_id or ""),
        )
        self.acceptance_criteria_md = render_contract(self.acceptance_contract)
        return self.acceptance_contract

    def set_acceptance_contract_from_markdown(self, markdown: str) -> None:
        self.acceptance_contract = contract_from_markdown(markdown, source_message_id=str(self.message_id or ""))
        self.acceptance_criteria_md = render_contract(self.acceptance_contract)

    def sync_acceptance_criteria_view(self) -> None:
        self.acceptance_criteria_md = render_contract(self.ensure_acceptance_contract())

    def operational_plan_json(self, *, max_calls: int = 4, max_chars: int = 12_000) -> str:
        """Return bounded execution evidence for prompts without mutating the audit plan."""
        calls = _json_list_or_empty(self.plan_json)[-max(1, int(max_calls)):]
        compact: list[dict[str, Any]] = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            item = {key: call.get(key) for key in ("id", "tool_id", "tool_name", "internal_state", "execution_mode") if call.get(key) is not None}
            execution = call.get("execution")
            if isinstance(execution, dict):
                item["execution"] = {
                    "status": execution.get("status"),
                    "exception": execution.get("exception"),
                    "result": _bounded_json_value(execution.get("result"), max_chars=4_000),
                }
            compact.append(item)
        rendered = json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True)
        return rendered if len(rendered) <= max_chars else rendered[: max_chars - 18].rstrip() + '\n"... truncated"'

    def merge_attachments(self, metadata: dict[str, Any]) -> None:
        """Preserve attachment references when same-owner steering or answers arrive."""
        attachments = list(self.metadata.get("attachments") or [])
        for item in metadata.get("attachments") or []:
            if isinstance(item, dict) and item not in attachments:
                attachments.append(dict(item))
        ids = list(self.metadata.get("asset_ids") or [])
        for asset_id in list(metadata.get("asset_ids") or []) + [item.get("asset_id") for item in attachments if isinstance(item, dict)]:
            if asset_id and asset_id not in ids:
                ids.append(asset_id)
        self.metadata["attachments"] = attachments
        self.metadata["asset_ids"] = ids

    def append_fact(self, fact: str) -> None:
        self.facts_md = _append_markdown_line(self.facts_md, fact)

    def append_plan_call(self, planned_call: dict[str, Any]) -> None:
        calls = _json_list_or_empty(self.plan_json)
        calls.append(dict(planned_call))
        self.plan_json = json.dumps(calls, indent=2, sort_keys=True)

    def get_next_planned_call(self) -> dict[str, Any] | None:
        calls = _json_list_or_empty(self.plan_json)
        for call in reversed(calls):
            if isinstance(call, dict) and not isinstance(call.get("execution"), dict):
                return dict(call)
        return None

    def get_latest_executed_plan_call(self) -> dict[str, Any] | None:
        calls = _json_list_or_empty(self.plan_json)
        for call in reversed(calls):
            if isinstance(call, dict) and isinstance(call.get("execution"), dict):
                return dict(call)
        return None

    def acceptance_criteria_all_complete(self) -> bool:
        contract = self.ensure_acceptance_contract()
        return required_criteria_complete(contract)

    def has_prepared_user_response(self) -> bool:
        response = self.metadata.get("prepared_user_response")
        return isinstance(response, dict) and bool(str(response.get("message") or "").strip())

    def count_plan_call_exceptions(self) -> int:
        count = 0
        for call in _json_list_or_empty(self.plan_json):
            if not isinstance(call, dict):
                continue
            execution = call.get("execution")
            if isinstance(execution, dict) and str(execution.get("status") or "").strip().lower() in {"exception", "failed"}:
                count += 1
        return count

    def record_plan_call_success(self, call_id: str, result: Any) -> None:
        self._record_plan_call_execution(
            call_id,
            {
                "status": "success",
                "result": _json_safe(result),
                "exception": "",
            },
        )

    def record_plan_call_failure(self, call_id: str, result: Any, exception: Any) -> None:
        self._record_plan_call_execution(
            call_id,
            {
                "status": "failed",
                "result": _json_safe(result),
                "exception": _exception_payload(exception),
            },
        )

    def record_plan_call_exception(self, call_id: str, exception: Any) -> None:
        self._record_plan_call_execution(
            call_id,
            {
                "status": "exception",
                "result": None,
                "exception": _exception_payload(exception),
            },
        )

    def record_plan_call_waiting(self, call_id: str, result: Any) -> None:
        self._record_plan_call_execution(
            call_id,
            {
                "status": "waiting",
                "result": _json_safe(result),
                "exception": "",
            },
        )

    def _record_plan_call_execution(self, call_id: str, execution: dict[str, Any]) -> None:
        planned_id = str(call_id or "").strip()
        if not planned_id:
            return
        calls = _json_list_or_empty(self.plan_json)
        now = datetime.now().astimezone().isoformat()
        execution_payload = {
            **execution,
            "started_at": now,
            "finished_at": now,
        }
        for index, call in enumerate(calls):
            if isinstance(call, dict) and str(call.get("id") or "").strip() == planned_id:
                updated = dict(call)
                updated["execution"] = execution_payload
                calls[index] = updated
                self.plan_json = json.dumps(calls, indent=2, sort_keys=True)
                self._append_evidence_journal(updated, execution_payload)
                return

    def _append_evidence_journal(self, call: dict[str, Any], execution: dict[str, Any]) -> None:
        call_id = str(call.get("id") or "").strip()
        if not call_id:
            return
        entry = {
            "evidence_ref": f"tool-call:{call_id}",
            "call_id": call_id,
            "tool_id": str(call.get("tool_id") or "program").strip(),
            "tool_name": str(call.get("tool_name") or call.get("tool_id") or "program").strip(),
            "execution_mode": str(call.get("execution_mode") or "direct").strip(),
            "status": str(execution.get("status") or "").strip(),
            "result": _bounded_json_value(execution.get("result"), max_chars=6_000),
            "exception": str(execution.get("exception") or "").strip(),
            "recorded_at": str(execution.get("finished_at") or ""),
        }
        self.evidence_journal.append(entry)

    def append_acceptance_criterion(self, criterion: str) -> None:
        self.acceptance_criteria_md = _append_markdown_line(self.acceptance_criteria_md, criterion)
        self.acceptance_contract = contract_from_markdown(
            self.acceptance_criteria_md,
            source_message_id=str(self.message_id or ""),
        )

    def append_memory_fact(self, fact: str) -> None:
        self.memory_facts_md = _append_markdown_line(self.memory_facts_md, fact)

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
        self.acceptance_contract = {}

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


def _json_list_or_empty(value: str) -> list[Any]:
    rendered = str(value or "").strip()
    if not rendered or rendered == EMPTY_MARKDOWN:
        return []
    try:
        parsed = json.loads(rendered)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _acceptance_criteria_checkbox_states(value: str) -> list[bool]:
    states: list[bool] = []
    for line in str(value or "").splitlines():
        match = re.search(r"\[(?P<state>x|\s)\]", line, flags=re.IGNORECASE)
        if match is None:
            continue
        states.append(match.group("state").lower() == "x")
    return states


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return str(value)


def _bounded_json_value(value: Any, *, max_chars: int) -> Any:
    safe = _json_safe(value)
    rendered = json.dumps(safe, ensure_ascii=False, sort_keys=True)
    if len(rendered) <= max_chars:
        return safe
    return rendered[: max(1, max_chars - 18)].rstrip() + "... [truncated]"


def _exception_payload(exception: Any) -> str:
    if isinstance(exception, BaseException):
        return f"{type(exception).__name__}: {exception}"
    return str(exception or "").strip()


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
