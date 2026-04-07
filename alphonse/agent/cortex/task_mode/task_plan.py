from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskPlan:
    version: int = 1
    steps: list[dict[str, Any]] = field(default_factory=list)
    current_step_id: str | None = None

    @classmethod
    def from_value(cls, value: Any) -> "TaskPlan":
        if not isinstance(value, dict):
            return cls()
        steps_raw = value.get("steps")
        steps = [dict(item) for item in steps_raw if isinstance(item, dict)] if isinstance(steps_raw, list) else []
        current_step_id = str(value.get("current_step_id") or "").strip() or None
        try:
            version = int(value.get("version") or 1)
        except (TypeError, ValueError):
            version = 1
        return cls(version=version, steps=steps, current_step_id=current_step_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version or 1),
            "steps": [dict(step) for step in self.steps if isinstance(step, dict)],
            "current_step_id": str(self.current_step_id or "").strip() or None,
        }

    def append_step(self, step: dict[str, Any]) -> None:
        if isinstance(step, dict):
            self.steps.append(dict(step))

    def current_step(self) -> dict[str, Any] | None:
        current_id = str(self.current_step_id or "").strip()
        if not current_id:
            return None
        for step in self.steps:
            if not isinstance(step, dict):
                continue
            if str(step.get("step_id") or "").strip() == current_id:
                return step
        return None
