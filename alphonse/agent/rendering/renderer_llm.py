from __future__ import annotations

import json
from typing import Any

from alphonse.agent.cognition.prompt_templates_runtime import RENDERER_UTTERANCE_SYSTEM_PROMPT


def render_text_from_utterance(utterance: dict[str, Any], llm_client: Any) -> str:
    if llm_client is None or not callable(getattr(llm_client, "complete", None)):
        raise RuntimeError("renderer_unavailable")
    user_prompt = "## Utterance (JSON)\n```json\n" + json.dumps(
        utterance,
        ensure_ascii=False,
        indent=2,
    ) + "\n```"
    raw = llm_client.complete(RENDERER_UTTERANCE_SYSTEM_PROMPT, user_prompt)
    if not isinstance(raw, str):
        raise RuntimeError("renderer_empty")
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("text"):
            cleaned = cleaned[4:].strip()
    if not cleaned:
        raise RuntimeError("renderer_empty")
    return cleaned
