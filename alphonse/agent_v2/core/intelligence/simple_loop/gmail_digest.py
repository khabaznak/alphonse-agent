"""Normalize private Gmail synthesis before projecting it to Desktop."""
from dataclasses import dataclass
from datetime import date
import json
import re

_CITATION = re.compile(r"(?<![\\\[])\[([1-9]|1[0-9]|20)\](?:\(#source-\1\)|(?![\[(]))")


@dataclass(frozen=True)
class GmailDigestAction:
    date: str
    when: str
    title: str
    detail: str
    sources: tuple[int, ...]
    superseded_sources: tuple[int, ...] = ()


@dataclass(frozen=True)
class GmailDigest:
    summary: str
    actions: tuple[GmailDigestAction, ...] = ()
    title: str = ''
    replaced_sources: tuple[int, ...] = ()

    def cited_sources(self, count: int) -> set[int]:
        return set(self.source_order(count))

    def source_order(self, count: int) -> tuple[int, ...]:
        citations = []
        for action in self.actions:
            citations.extend(action.sources)
            citations.extend(int(number) for number in _CITATION.findall(action.detail))
        citations.extend(int(number) for number in _CITATION.findall(self.summary))
        return tuple(dict.fromkeys(number for number in citations if 1 <= number <= count))

    def superseded_sources(self) -> set[int]:
        return set(self.replaced_sources).union(number for action in self.actions for number in action.superseded_sources)

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
    if (not isinstance(value, dict) or set(value) - {'title', 'superseded_sources'} != {'summary', 'actions'}
            or not isinstance(value['summary'], str) or len(value['summary']) > 2000
            or not isinstance(value['actions'], list) or len(value['actions']) > 12):
        return GmailDigest('')
    items = []
    for row in value['actions']:
        if (not isinstance(row, dict) or set(row) - {'superseded_sources'} != {'date', 'when', 'title', 'detail', 'sources'}
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
        sources = tuple(dict.fromkeys(row['sources']))
        marked = row.get('superseded_sources', [])
        superseded = ()
        if (isinstance(marked, list) and len(marked) < len(sources)
                and all(type(number) is int and number in sources for number in marked)):
            superseded = tuple(dict.fromkeys(marked))
        items.append(GmailDigestAction(
            due or '', row['when'].strip(), ' '.join(row['title'].split()), row['detail'].strip(),
            sources, superseded,
        ))
    items.sort(key=lambda item: (not item.date, not item.when, item.date))
    summary = value['summary'].strip()
    title = value.get('title')
    title = ' '.join(title.split()) if isinstance(title, str) and len(title) <= 100 else ''
    cited = GmailDigest(summary, tuple(items)).cited_sources(count)
    marked = value.get('superseded_sources')
    replaced = ()
    if isinstance(marked, list) and len(marked) < len(cited):
        replaced = tuple(dict.fromkeys(number for number in marked if type(number) is int and number in cited))
    return GmailDigest(summary, tuple(items), title, replaced)
