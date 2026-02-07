from __future__ import annotations

from typing import Any

from alphonse.agent.cognition.safe_fallbacks import render_safe_message


def render_message(
    key: str, locale: str, variables: dict[str, Any] | None = None
) -> str:
    return render_safe_message(key, locale, variables)
