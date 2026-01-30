from __future__ import annotations

from dataclasses import dataclass

from alphonse.actions.registry import ActionRegistry
from alphonse.actions.models import ActionResult
from alphonse.extremities.notification import NotificationExtremity
from alphonse.actions.system_reminder import SystemReminderAction
from alphonse.extremities.registry import ExtremityRegistry
from alphonse.mediation.narrator import Narrator
from alphonse.mediation.policy import NarrationPolicy


@dataclass
class IntentPipeline:
    actions: ActionRegistry
    extremities: ExtremityRegistry
    narration_policy: NarrationPolicy
    narrator: Narrator

    def handle(self, action_key: str | None, context: dict) -> None:
        if not action_key:
            return
        factory = self.actions.get(action_key)
        if not factory:
            return
        action = factory(context)
        result = action.execute(context)
        narration = self._maybe_narrate(result, context)
        self.extremities.dispatch(result, narration)

    def _maybe_narrate(self, result: ActionResult, context: dict) -> str | None:
        if self.narration_policy.should_narrate(result, context):
            return self.narrator.narrate(result, context)
        return None


def build_default_pipeline() -> IntentPipeline:
    actions = ActionRegistry()
    actions.register("system_reminder", lambda _: SystemReminderAction())
    extremities = ExtremityRegistry()
    extremities.register(NotificationExtremity())
    return IntentPipeline(
        actions=actions,
        extremities=extremities,
        narration_policy=NarrationPolicy(),
        narrator=Narrator(),
    )
