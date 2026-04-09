from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentMetricsBudget:
    repair_attempts: int = 0
    planner_invalid_streak: int = 0
    planner_invalid_last_history_index: int | None = None
    repeated_failure_signature_last: str | None = None
    repeated_failure_signature_streak: int = 0
    zero_progress_last_signature: str | None = None
    zero_progress_streak: int = 0
    judge_invalid_streak: int = 0
    check_provenance: str = "entry"
    steering_consumed_in_check: bool = False

    @classmethod
    def from_value(cls, value: Any) -> "AgentMetricsBudget":
        if not isinstance(value, dict):
            return cls()
        return cls(
            repair_attempts=_as_int(value.get("repair_attempts"), default=0),
            planner_invalid_streak=_as_int(value.get("planner_invalid_streak"), default=0),
            planner_invalid_last_history_index=_as_optional_int(value.get("planner_invalid_last_history_index")),
            repeated_failure_signature_last=_as_optional_str(value.get("repeated_failure_signature_last")),
            repeated_failure_signature_streak=_as_int(value.get("repeated_failure_signature_streak"), default=0),
            zero_progress_last_signature=_as_optional_str(value.get("zero_progress_last_signature")),
            zero_progress_streak=_as_int(value.get("zero_progress_streak"), default=0),
            judge_invalid_streak=_as_int(value.get("judge_invalid_streak"), default=0),
            check_provenance=_as_optional_str(value.get("check_provenance")) or "entry",
            steering_consumed_in_check=bool(value.get("steering_consumed_in_check")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "repair_attempts": int(self.repair_attempts or 0),
            "planner_invalid_streak": int(self.planner_invalid_streak or 0),
            "planner_invalid_last_history_index": self.planner_invalid_last_history_index,
            "repeated_failure_signature_last": _as_optional_str(self.repeated_failure_signature_last),
            "repeated_failure_signature_streak": int(self.repeated_failure_signature_streak or 0),
            "zero_progress_last_signature": _as_optional_str(self.zero_progress_last_signature),
            "zero_progress_streak": int(self.zero_progress_streak or 0),
            "judge_invalid_streak": int(self.judge_invalid_streak or 0),
            "check_provenance": _as_optional_str(self.check_provenance) or "entry",
            "steering_consumed_in_check": bool(self.steering_consumed_in_check),
        }


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_str(value: Any) -> str | None:
    rendered = str(value or "").strip()
    return rendered or None
