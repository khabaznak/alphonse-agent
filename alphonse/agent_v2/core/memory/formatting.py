"""Bound knowledge only at complete record boundaries, including downstream prompts."""
from __future__ import annotations

import re

_RECORD_START = re.compile(r"(?m)^- \[memory:")
OMITTED = "- (memory omitted to preserve complete claims and source/status labels)"


def _bound_context_record_text(value: object, max_chars: int) -> str:
    """An opaque message/record either fits whole or carries an omission label."""
    text = str(value or "").strip()
    marker = "(complete record omitted: context budget exceeded; recall required)"
    return text if len(text) <= max_chars else marker if len(marker) <= max_chars else ""


def _bound_memory_context_text(value: object, max_chars: int) -> str:
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


def bound_context_record(value: object, max_chars: int) -> str:
    from alphonse.agent_v2.core.context_inventory import replaced
    return replaced(value, _bound_context_record_text(value, max_chars))


def bound_memory_context(value: object, max_chars: int) -> str:
    from alphonse.agent_v2.core.context_inventory import replaced
    result = replaced(value, _bound_memory_context_text(value, max_chars))
    if len(str(value).strip()) > max_chars and '\n\n# Current Request\n' in str(value):
        for record in result.records:
            if record['source_type'] == 'current_request' and record['status'] == 'omitted':
                record['reason'] = 'duplicate'
    return result


def join_context_sections(sections: list[str], max_chars: int) -> str:
    """Preserve existing selection bytes and carry its call-time inventory."""
    from alphonse.agent_v2.core.context_inventory import combined, omitted
    selected = []
    used = 0
    for index, section in enumerate(sections):
        separator = "\n\n" if used else ""
        remaining = max_chars - used - len(separator)
        if remaining <= 0:
            selected.extend(omitted(part) for part in sections[index:])
            break
        if len(section) > remaining:
            if section.startswith("Durable household memory"):
                section = bound_memory_context(section, remaining)
            else:
                selected.append(omitted(section))
                continue
        selected.append(section)
        if section:
            used += len(separator) + len(section)
    return combined(selected, max_chars=max_chars)
