from __future__ import annotations

from typing import Any

from alphonse.agent.cognition.localization import render_message
from alphonse.agent.cognition.prompt_store import PromptContext
from alphonse.agent.cognition.response_spec import ResponseSpec


class ResponseComposer:
    def __init__(self, prompt_store: Any | None = None) -> None:
        # Legacy compatibility: callers can still inject an object that has get_template().
        self._prompt_store = prompt_store

    def compose(self, spec: ResponseSpec) -> str:
        variables = self._merge_variables(spec)
        if self._prompt_store is not None and hasattr(self._prompt_store, "get_template"):
            context = PromptContext(
                locale=spec.locale,
                address_style=spec.address_style,
                tone=spec.tone,
                channel=spec.channel,
                variant=spec.variant,
                policy_tier=spec.policy_tier,
            )
            try:
                match = self._prompt_store.get_template(spec.key, context)
            except Exception:
                match = None
            template = getattr(match, "template", None) if match is not None else None
            if isinstance(template, str) and template:
                return template.format(**variables)
            return spec.key
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
