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
        context = PromptContext(
            locale=spec.locale,
            address_style=spec.address_style,
            tone=spec.tone,
            channel=spec.channel,
            variant=spec.variant,
            policy_tier=spec.policy_tier,
        )
        match = self._prompt_store.get_template(spec.key, context)
        variables = self._merge_variables(spec)
        if match:
            return match.template.format(**variables)
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
