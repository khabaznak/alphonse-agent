from __future__ import annotations

from pathlib import Path

def load_template_or_fallback(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Failed to load prompt template: {path}") from exc
