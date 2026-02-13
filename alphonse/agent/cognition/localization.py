from __future__ import annotations

from typing import Any

_MESSAGES: dict[str, str] = {
    "generic.unknown": "I could not determine the request clearly. Please rephrase it.",
    "clarify.intent": "Could you clarify what you want me to do?",
    "clarify.repeat_input": "Please share your request again so I can continue.",
    "system.unavailable.catalog": "The system is temporarily unavailable.",
    "core.greeting": "Hello. I am here and ready to help.",
    "help": "I can help with reminders, locations, preferences, and status checks.",
}


def _resolve_template(key: str, locale: str) -> str | None:
    _ = locale
    template = _MESSAGES.get(key)
    if isinstance(template, str) and template:
        return template
    return None


def render_message(
    key: str, locale: str, variables: dict[str, Any] | None = None
) -> str:
    template = _resolve_template(key, locale)
    if template:
        return template.format(**(variables or {}))
    return key
