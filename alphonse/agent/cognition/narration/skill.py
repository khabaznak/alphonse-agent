from __future__ import annotations

from alphonse.agent.cognition.narration.models import MessageDraft, NarrationIntent, PresentationSpec


class NarrationSkill:
    def compose(
        self,
        *,
        message: str,
        intent: NarrationIntent,
        presentation: PresentationSpec,
        correlation_id: str,
        metadata: dict | None = None,
    ) -> MessageDraft:
        return MessageDraft(
            correlation_id=correlation_id,
            audience=intent.audience,
            channel_type=intent.channel_type,
            format=intent.format,
            content=message,
            metadata=metadata or {},
        )
