from __future__ import annotations

from typing import Any

from alphonse.agent.cognition.prompt_store import PromptContext, SqlitePromptStore


def render_message(
    key: str, locale: str, variables: dict[str, Any] | None = None
) -> str:
    store = SqlitePromptStore()
    context = PromptContext(
        locale=locale or "en-US",
        address_style="any",
        tone="any",
        channel="any",
        variant="default",
        policy_tier="safe",
    )
    match = store.get_template(key, context)
    if match:
        return match.template.format(**(variables or {}))
    return key
