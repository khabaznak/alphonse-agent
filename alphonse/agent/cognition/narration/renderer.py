from __future__ import annotations

from alphonse.agent.cognition.narration.models import MessageDraft, PresentationSpec, RenderedMessage


def render_message(draft: MessageDraft, spec: PresentationSpec) -> RenderedMessage:
    content = draft.content
    payload = {
        "content": content,
        "format": draft.format,
        "language": spec.language,
    }
    return RenderedMessage(channel_type=draft.channel_type, payload=payload)
