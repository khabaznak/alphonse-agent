from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from alphonse.actions.models import ActionResult


class Extremity(ABC):
    @abstractmethod
    def can_handle(self, result: ActionResult) -> bool:
        raise NotImplementedError

    @abstractmethod
    def execute(self, result: ActionResult, narration: str | None = None) -> None:
        raise NotImplementedError
