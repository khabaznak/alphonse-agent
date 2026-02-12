from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_PROMPT_SEED_PATH = (
    Path(__file__).resolve().parent.parent
    / "nervous_system"
    / "resources"
    / "prompt_templates.seed.json"
)


@lru_cache(maxsize=1)
def _load_prompt_seed_rows() -> list[dict[str, Any]]:
    try:
        payload = json.loads(_PROMPT_SEED_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _resolve_seed_template(key: str, locale: str) -> str | None:
    rows = _load_prompt_seed_rows()
    normalized_locale = str(locale or "en-US").strip().lower()
    locale_lang = normalized_locale.split("-", 1)[0]
    best: str | None = None
    best_score = -1
    for row in rows:
        if str(row.get("key") or "") != key:
            continue
        row_locale = str(row.get("locale") or "").strip().lower()
        score = 0
        if row_locale == normalized_locale:
            score = 3
        elif row_locale and row_locale.split("-", 1)[0] == locale_lang:
            score = 2
        elif row_locale in {"any", "*", ""}:
            score = 1
        if score > best_score:
            template = row.get("template")
            if isinstance(template, str) and template.strip():
                best = template
                best_score = score
    return best


def render_message(
    key: str, locale: str, variables: dict[str, Any] | None = None
) -> str:
    template = _resolve_seed_template(key, locale)
    if template:
        return template.format(**(variables or {}))
    return key
