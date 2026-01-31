from __future__ import annotations

from dataclasses import dataclass

from alphonse.agent.actions.registry import ActionRegistry
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.extremities.notification import NotificationExtremity
from alphonse.agent.extremities.telegram_notification import TelegramNotificationExtremity
from alphonse.agent.actions.system_reminder import SystemReminderAction
from alphonse.agent.extremities.registry import ExtremityRegistry
from alphonse.agent.cognition.skills.mediation.narrator import Narrator
from alphonse.agent.cognition.skills.mediation.policy import NarrationPolicy


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
    extremities.register(TelegramNotificationExtremity())
    return IntentPipeline(
        actions=actions,
        extremities=extremities,
        narration_policy=NarrationPolicy(),
        narrator=Narrator(),
    )
