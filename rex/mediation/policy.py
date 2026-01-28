from __future__ import annotations

from rex.actions.models import ActionResult


class NarrationPolicy:
    def should_narrate(self, result: ActionResult, context: dict | None = None) -> bool:
        if result.requires_narration:
            return True
        if result.urgency in {"high", "critical"}:
            return True
        return False
