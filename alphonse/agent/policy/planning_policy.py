from __future__ import annotations

from dataclasses import dataclass

from alphonse.agent.cognition.planning import PlanningMode


@dataclass(frozen=True)
class PlanningPolicy:
    def adjust_autonomy_level(self, autonomy_level: float, *, intent: str | None = None) -> float:
        _ = intent
        return autonomy_level

    def adjust_planning_mode(
        self,
        mode: PlanningMode,
        *,
        intent: str | None = None,
        autonomy_level: float | None = None,
    ) -> PlanningMode:
        _ = intent
        _ = autonomy_level
        return mode
