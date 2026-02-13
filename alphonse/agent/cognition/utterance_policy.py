from __future__ import annotations


def render_utterance_policy_block(
    *,
    locale: str | None,
    tone: str | None,
    address_style: str | None,
    channel_type: str | None,
) -> str:
    return (
        "UTTERANCE POLICY CONTEXT:\n"
        f"- locale: {str(locale or 'en-US')}\n"
        f"- tone: {str(tone or 'neutral')}\n"
        f"- address_style: {str(address_style or 'neutral')}\n"
        f"- channel_type: {str(channel_type or 'unknown')}\n"
        "Rules:\n"
        "- Follow locale and tone naturally in your wording.\n"
        "- Use address_style as a preference, not as a rigid template.\n"
        "- Keep output suitable for the channel_type.\n"
    )
