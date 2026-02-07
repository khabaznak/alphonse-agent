from __future__ import annotations

from typing import Any

from alphonse.agent.cognition.safe_fallbacks import get_safe_fallback


def render_message(
    key: str, locale: str, variables: dict[str, Any] | None = None
) -> str:
    template = get_safe_fallback(key, locale)
    if not variables:
        return template
    try:
        return template.format(**variables)
    except Exception:
        return template

