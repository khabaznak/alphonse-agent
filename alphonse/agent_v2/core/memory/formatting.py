"""Bound knowledge only at complete record boundaries, including downstream prompts."""
from __future__ import annotations

import re

_RECORD_START = re.compile(r"(?m)^- \[memory:")
OMITTED = "- (memory omitted to preserve complete claims and source/status labels)"


def bound_memory_context(value: object, max_chars: int) -> str:
    text = str(value or "").strip() or "- (none)"
    if len(text) <= max_chars:
        return text
    if max_chars < len(OMITTED):
        return ""
    # The current request has its own prompt section. Never interpret a user's
    # goal text as a record boundary in the memory section during shortening.
    text = text.split("\n\n# Current Request\n", 1)[0].rstrip()
    if len(text) <= max_chars:
        return text
    starts = list(_RECORD_START.finditer(text))
    # Unstructured legacy memory cannot be safely sliced into an apparent claim.
    if not starts:
        return OMITTED
    header = text[:starts[0].start()].rstrip()
    if len(header) + len(OMITTED) + 2 > max_chars:
        return OMITTED
    selected = [header] if header else []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[start.start():end].rstrip()
        if len("\n".join(selected + [block, OMITTED])) <= max_chars:
            selected.append(block)
    return "\n".join(selected + [OMITTED])


def join_context_sections(sections: list[str], max_chars: int) -> str:
    """Preserve the memory section if it fits; otherwise select complete records."""
    result = ""
    for section in sections:
        separator = "\n\n" if result else ""
        remaining = max_chars - len(result) - len(separator)
        if remaining <= 0:
            break
        if len(section) > remaining:
            if section.startswith("Durable household memory"):
                section = bound_memory_context(section, remaining)
            else:
                section = section[:remaining]
        if section:
            result += separator + section
    return result
