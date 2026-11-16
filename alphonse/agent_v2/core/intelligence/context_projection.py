"""Shared atomic evidence projection and bounded assembly."""
from __future__ import annotations

from dataclasses import dataclass
from alphonse.agent_v2.system_one import SystemOneUnavailableError


@dataclass(frozen=True)
class Record:
    source_id: str
    title: str
    text: str
    required: bool = False
    group: str = ""

    def render(self) -> str:
        return f"[{self.source_id}] {self.title}:\n{self.text}"


class ContextBudgetExceeded(SystemOneUnavailableError):
    """Required context cannot safely fit: stop rather than lose constraints."""


def token_count(text: str) -> int:
    # Conservative fallback: one token per UTF-8 byte, including labels/separators.
    # No assumption of four characters/token, which undercounts non-Latin text.
    return len(text.encode("utf-8"))


def assemble(records: list[Record], *, budget: int) -> tuple[str, dict]:
    unique: dict[str, Record] = {}
    omissions = []
    for record in records:
        if not record.text.strip():
            continue
        previous = unique.get(record.source_id)
        if previous is not None:
            omissions.append({"source_id": record.source_id, "source_type": record.source_id.split(":", 1)[0],
                              "status": "omitted", "reason": "duplicate", "tokens": 0})
        # Required state wins over optional representations of the same source.
        if previous is None or (record.required and not previous.required):
            unique[record.source_id] = record
    required = [r for r in unique.values() if r.required]
    optional = [r for r in unique.values() if not r.required]
    selected = list(required)

    def render(rows):
        return "\n\n".join(r.render() for r in rows)

    if token_count(render(selected)) > budget:
        raise ContextBudgetExceeded("v3_required_context_exceeds_token_budget")
    omitted = []
    groups: dict[str, list[Record]] = {}
    for record in optional:
        groups.setdefault(record.group or record.source_id, []).append(record)
    for group in groups.values():
        if token_count(render([*selected, *group])) <= budget:
            selected.extend(group)
        else:
            omitted.extend(record.source_id for record in group)
    text = render(selected)
    selected_ids = {r.source_id for r in selected}
    inventory = [{"source_id": r.source_id, "source_type": r.source_id.split(":", 1)[0],
                  "status": "included" if r.source_id in selected_ids else "omitted",
                  "reason": ("required" if r.required else "selected_relevant") if r.source_id in selected_ids else "over_budget",
                  "tokens": token_count(r.render()) if r.source_id in selected_ids else 0,
                  "truncated": False} for r in unique.values()]
    offset = 0
    for record in selected:
        row = next(r for r in inventory if r['source_id'] == record.source_id)
        row['input_range'] = {'start_char': offset, 'end_char': offset + len(record.render())}
        offset += len(record.render()) + 2
    return text, {"source_ids": [r.source_id for r in selected], "omitted_source_ids": omitted,
                  "records": inventory + omissions, "separator_tokens": max(0, len(selected) - 1) * 2,
                  "tokens": token_count(text), "budget": budget, "tokenizer": "conservative_utf8_bytes"}

