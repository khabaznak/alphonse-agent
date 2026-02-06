from __future__ import annotations

from typing import Any

from alphonse.agent.cognition.localization import render_message
from alphonse.agent.cognition.prompt_store import (
    NullPromptStore,
    PromptContext,
    PromptStore,
    SqlitePromptStore,
)
from alphonse.agent.cognition.response_spec import ResponseSpec


class ResponseComposer:
    def __init__(self, prompt_store: PromptStore | None = None) -> None:
        self._prompt_store = prompt_store or SqlitePromptStore()

    def compose(self, spec: ResponseSpec) -> str:
        policy_tier = spec.policy_tier
        if _is_sensitive_key(spec.key) and not policy_tier:
            policy_tier = "strict"
        context = PromptContext(
            locale=spec.locale,
            address_style=spec.address_style,
            tone=spec.tone,
            channel=spec.channel,
            variant=spec.variant,
            policy_tier=policy_tier,
        )
        match = self._prompt_store.get_template(spec.key, context)
        variables = self._merge_variables(spec)
        if match:
            return match.template.format(**variables)
        if _is_sensitive_key(spec.key):
            return _safe_fallback(spec, variables)
        return render_message(spec.key, spec.locale or "en-US", variables)

    def _merge_variables(self, spec: ResponseSpec) -> dict[str, Any]:
        vars: dict[str, Any] = {}
        if spec.tone:
            vars["tone"] = spec.tone
        if spec.address_style:
            vars["address_style"] = spec.address_style
        vars.update(spec.variables or {})
        if spec.options:
            vars["options"] = spec.options
        if spec.next_prompt:
            vars["next_prompt"] = spec.next_prompt
        return vars


def _is_sensitive_key(key: str) -> bool:
    return key.startswith("policy.") or key.startswith("security.")


def _safe_fallback(spec: ResponseSpec, variables: dict[str, Any]) -> str:
    fallback = render_message(spec.key, spec.locale or "en-US", variables)
    if fallback and "remind" not in fallback.lower():
        return fallback
    return "I can't do that right now."
