from __future__ import annotations

import json
from typing import Any

_UTTERANCE_RENDERER_PROMPT = (
    "## Role\n"
    "You are Alphonse's Utterance Renderer.\n\n"
    "## Instructions\n"
    "- Produce the final user-facing message.\n"
    "- Output ONLY plain text. No JSON. No markdown.\n"
    "- Respect prefs.locale, prefs.tone, prefs.address_style, and prefs.verbosity.\n"
    "- Do not include internal ids or raw timestamps unless prefs.verbosity is debug.\n"
    "- Avoid parroting user phrasing; paraphrase naturally.\n"
    "- Never mention chat_id, tool names, or DB fields.\n"
    "- Keep it concise by default.\n"
)


def render_text_from_utterance(utterance: dict[str, Any], llm_client: Any) -> str:
    if llm_client is None or not callable(getattr(llm_client, "complete", None)):
        raise RuntimeError("renderer_unavailable")
    user_prompt = "## Utterance (JSON)\n```json\n" + json.dumps(
        utterance,
        ensure_ascii=False,
        indent=2,
    ) + "\n```"
    raw = llm_client.complete(_UTTERANCE_RENDERER_PROMPT, user_prompt)
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
