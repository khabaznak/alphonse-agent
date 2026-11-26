"""Normalize private Gmail synthesis before projecting it to Desktop."""
from dataclasses import dataclass
from datetime import date
import json
import re


@dataclass(frozen=True)
class GmailDigestAction:
    date: str
    when: str
    title: str
    detail: str
    sources: tuple[int, ...]


@dataclass(frozen=True)
class GmailDigest:
    summary: str
    actions: tuple[GmailDigestAction, ...] = ()

    def cited_sources(self, count: int) -> set[int]:
        citations = {number for action in self.actions for number in action.sources}
        for text in (self.summary, *(action.detail for action in self.actions)):
            citations.update(int(number) for number in re.findall(r"\[(\d{1,2})\]\(#source-\1\)", text)
                             if 1 <= int(number) <= count)
        return citations

    def as_text(self) -> str:
        lines = [
            f"- {item.when.replace(chr(10), ' · ') or 'No date given'} · {item.title}"
            f"{' — ' + item.detail if item.detail else ''} "
            + ' '.join(f'[{number}]' for number in item.sources)
            for item in self.actions
        ]
        return '\n\n'.join(filter(None, (self.summary, '\n'.join(lines))))


def parse_gmail_digest(text: str, count: int) -> GmailDigest:
    """Keep supported rows and exact source numbers. Older replies are Markdown."""
    candidate = text.strip()
    if candidate.startswith('```json\n') and candidate.endswith('\n```'):
        candidate = candidate[len('```json\n'):-len('\n```')].strip()
    if not candidate.startswith('{'):
        return GmailDigest(text)
    try:
        value = json.loads(candidate)
    except (ValueError, RecursionError):
        return GmailDigest('')
    if (not isinstance(value, dict) or set(value) != {'summary', 'actions'}
            or not isinstance(value['summary'], str) or len(value['summary']) > 2000
            or not isinstance(value['actions'], list) or len(value['actions']) > 12):
        return GmailDigest('')
    items = []
    for row in value['actions']:
        if (not isinstance(row, dict) or set(row) != {'date', 'when', 'title', 'detail', 'sources'}
                or not isinstance(row['title'], str) or not row['title'].strip() or len(row['title']) > 240
                or not isinstance(row['when'], str) or len(row['when']) > 140
                or not isinstance(row['detail'], str) or len(row['detail']) > 1000
                or not isinstance(row['sources'], list) or not 1 <= len(row['sources']) <= 6
                or any(type(number) is not int or not 1 <= number <= count for number in row['sources'])):
            continue
        due = row['date']
        if due is not None:
            try:
                if date.fromisoformat(due).isoformat() != due or not row['when'].strip():
                    continue
            except (TypeError, ValueError):
                continue
        items.append(GmailDigestAction(
            due or '', row['when'].strip(), ' '.join(row['title'].split()), row['detail'].strip(),
            tuple(dict.fromkeys(row['sources'])),
        ))
    items.sort(key=lambda item: (not item.date, not item.when, item.date))
    return GmailDigest(value['summary'].strip(), tuple(items))
