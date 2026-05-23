from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent


def load_boot_prompt_context(config_dir: Path | None = None) -> dict[str, str]:
    root = Path(config_dir) if config_dir is not None else CONFIG_DIR
    return {
        "PHILOSOPHY_SECTION": _load_markdown_section(root / "Philosophy.md", "Philosophy"),
        "CORE_CONTEXT_SECTION": _load_markdown_section(root / "CoreContext.md", "Core Context"),
    }


def _load_markdown_section(path: Path, heading: str) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    text = text.strip()
    if not text:
        return ""
    return _strip_matching_first_heading(text, heading).strip()


def _strip_matching_first_heading(text: str, heading: str) -> str:
    lines = text.splitlines()
    first_content_index = next(
        (index for index, line in enumerate(lines) if line.strip()),
        None,
    )
    if first_content_index is None:
        return ""

    first_line = lines[first_content_index].strip()
    expected = f"# {heading}".casefold()
    if first_line.casefold() != expected:
        return text

    remaining = lines[:first_content_index] + lines[first_content_index + 1 :]
    return "\n".join(remaining).strip()
