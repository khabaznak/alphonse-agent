from __future__ import annotations

from pathlib import Path

TEMPLATE_SUBSYSTEM_ERROR_MESSAGE = (
    "Template subsystem error detected. Please contact the administrator."
)


def load_template_or_fallback(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return TEMPLATE_SUBSYSTEM_ERROR_MESSAGE

