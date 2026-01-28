from __future__ import annotations

from typing import List

from rex.actions.models import ActionResult
from rex.extremities.base import Extremity


class ExtremityRegistry:
    def __init__(self) -> None:
        self._extremities: List[Extremity] = []

    def register(self, extremity: Extremity) -> None:
        self._extremities.append(extremity)

    def dispatch(self, result: ActionResult, narration: str | None = None) -> None:
        for extremity in self._extremities:
            if extremity.can_handle(result):
                extremity.execute(result, narration)
