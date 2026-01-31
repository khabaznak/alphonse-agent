from __future__ import annotations

from abc import ABC, abstractmethod

from alphonse.agent.actions.models import ActionResult


class Action(ABC):
    key: str

    @abstractmethod
    def execute(self, context: dict) -> ActionResult:
        raise NotImplementedError
