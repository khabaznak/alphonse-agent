from __future__ import annotations

from typing import Any

from alphonse.agent.cognition.localization import render_message
from alphonse.agent.cognition.prompt_store import NullPromptStore, PromptStore
from alphonse.agent.cognition.response_spec import ResponseSpec


class ResponseComposer:
    def __init__(self, prompt_store: PromptStore | None = None) -> None:
        self._prompt_store = prompt_store or NullPromptStore()

    def compose(self, spec: ResponseSpec) -> str:
        template = self._prompt_store.get_template(
            spec.key, spec.locale, spec.address_style, spec.tone
        )
        variables = self._merge_variables(spec)
        if template:
            return template.format(**variables)
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
