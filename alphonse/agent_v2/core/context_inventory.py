"""Provenance carried with rendered strings, without changing their bytes.

Receipts contain IDs, ranges and counts only. The string exists for the model call;
its metadata can be checkpointed without retaining its contents. All selection and
clipping is recorded while it happens, rather than reconstructed from TaskState.
"""
from __future__ import annotations

from copy import deepcopy
import json


def tokens(text):
    return len(str(text).encode('utf-8'))


class ContextText(str):
    def __new__(cls, text, records=(), **metadata):
        obj = super().__new__(cls, text)
        obj.records = deepcopy(list(records))
        obj.details = deepcopy(metadata)
        return obj

    def strip(self, chars=None):
        start = len(self) - len(super().lstrip(chars))
        end = len(super().rstrip(chars))
        return clipped(self, start, max(start, end))

    def lstrip(self, chars=None):
        start = len(self) - len(super().lstrip(chars))
        return clipped(self, start, len(self))

    def rstrip(self, chars=None):
        return clipped(self, 0, len(super().rstrip(chars)))

    def __getitem__(self, key):
        if isinstance(key, slice) and key.step in (None, 1):
            start, end, _ = key.indices(len(self))
            return clipped(self, start, end)
        return super().__getitem__(key)

    def __add__(self, other):
        return combined([self, other], '')

    def __radd__(self, other):
        return combined([other, self], '')


def source(text, source_id, source_type, *, reason='decision_input', **metadata):
    text = str(text or "")
    return ContextText(text, [{
        'source_id': source_id, 'source_type': source_type, 'status': 'included',
        'reason': reason, 'tokens': tokens(text), 'truncated': False,
        'input_range': {'start_char': 0, 'end_char': len(text)},
        'source_range': {'start_char': 0, 'end_char': len(text), 'total_chars': len(text)},
    }], **metadata)


def omitted(value, *, reason='over_budget'):
    records = deepcopy(getattr(value, 'records', []))
    for record in records:
        if record['status'] == 'included':
            record.update(status='omitted', reason=reason, tokens=0, input_range=None)
    return ContextText('', records, **getattr(value, 'details', {}))


def clipped(value, start=0, end=None):
    end = len(value) if end is None else max(start, end)
    rendered = str(value)[start:end]
    records = deepcopy(getattr(value, 'records', []))
    for record in records:
        span = record.get('input_range')
        if record['status'] != 'included' or span is None:
            continue
        left, right = max(start, span['start_char']), min(end, span['end_char'])
        if left >= right:
            record.update(status='omitted', reason='over_budget', tokens=0, input_range=None)
            continue
        if left != span['start_char'] or right != span['end_char']:
            record['truncated'] = True
            record['reason'] = 'source_clip'
            original = record.get('source_range', {})
            origin = original.get('start_char', 0)
            record['source_range'] = {
                'start_char': origin + left - span['start_char'],
                'end_char': origin + right - span['start_char'],
                'total_chars': original.get('total_chars', span['end_char'] - span['start_char']),
            }
        record['tokens'] = tokens(str(value)[left:right])
        record['input_range'] = {'start_char': left - start, 'end_char': right - start}
    return ContextText(rendered, records, **getattr(value, 'details', {}))


def combined(parts, separator='\n\n', **metadata):
    parts = list(parts)
    text = separator.join(str(part) for part in parts if part)
    records, offset = [], 0
    details = {}
    present = False
    for part in parts:
        if part:
            if present:
                offset += len(separator)
            present = True
        for record in deepcopy(getattr(part, 'records', [])):
            if record.get('input_range'):
                record['input_range']['start_char'] += offset
                record['input_range']['end_char'] += offset
            records.append(record)
        details.update(getattr(part, 'details', {}))
        offset += len(part)
    details.update(metadata)
    return ContextText(text, records, **details)


def replaced(value, result):
    """Preserve whole known records found in a boundary-preserving formatter."""
    if str(value) == str(result):
        return ContextText(str(result), getattr(value, 'records', []), **getattr(value, 'details', {}))
    records = []
    cursor = 0
    for record in deepcopy(getattr(value, 'records', [])):
        span = record.get('input_range')
        if record['status'] != 'included' or span is None:
            records.append(record)
            continue
        block = str(value)[span['start_char']:span['end_char']]
        position = str(result).find(block, cursor) if block else -1
        if position < 0:
            record.update(status='omitted', reason='over_budget', tokens=0, input_range=None)
        else:
            record['input_range'] = {'start_char': position, 'end_char': position + len(block)}
            cursor = position + len(block)
        records.append(record)
    return ContextText(str(result), records, **getattr(value, 'details', {}))


def receipt(value, *, purpose='context', **metadata):
    records = deepcopy(getattr(value, 'records', []))
    details = deepcopy(getattr(value, 'details', {}))
    details.update(metadata)
    return {
        **details,
        'schema_version': 1, 'builder_version': 'rendered-context-v2', 'purpose': purpose,
        'records': records, 'source_ids': [r['source_id'] for r in records if r['status'] == 'included'],
        'omitted_source_ids': [r['source_id'] for r in records if r['status'] == 'omitted'],
        'tokens': tokens(value), 'overhead_tokens': tokens(value) - sum(r['tokens'] for r in records if r['status'] == 'included'), 'tokenizer': 'conservative_utf8_bytes',
        'boundary_policy': details.get('boundary_policy', 'whole_records_and_explicit_source_clips'),
    }



def prompt_receipt(prompt, parts, *, purpose):
    """Bind inventories to the exact rendered parts used to compose this prompt."""
    records = []
    covered = []
    part_ranges = []
    details = {}
    budgets = []
    for part in parts:
        part_details = getattr(part, "details", {})
        details.update(part_details)
        if "budget" in part_details or "max_chars" in part_details:
            budgets.append({k: part_details[k] for k in ("budget", "max_chars", "purpose") if k in part_details})
        if not part:
            records.extend(deepcopy(getattr(part, 'records', [])))
            continue
        position = str(prompt).find(str(part))
        while position >= 0 and any(position < right and position + len(part) > left for left, right in part_ranges):
            position = str(prompt).find(str(part), position + 1)
        if position < 0:
            raise ValueError('context_inventory_part_missing_from_prompt')
        part_ranges.append((position, position + len(part)))
        details.update(getattr(part, 'details', {}))
        for record in deepcopy(getattr(part, 'records', [])):
            if record.get('input_range'):
                record['input_range']['start_char'] += position
                record['input_range']['end_char'] += position
                covered.append((record['input_range']['start_char'], record['input_range']['end_char']))
            records.append(record)
    # Everything else is explicitly identified as prompt framing/instructions.
    cursor = 0
    for start, end in sorted(covered) + [(len(prompt), len(prompt))]:
        if start > cursor:
            framing = source(str(prompt)[cursor:start], f'instructions:{purpose}:{cursor}', 'prompt_framing').records[0]
            framing['input_range'] = {'start_char': cursor, 'end_char': start}
            records.append(framing)
        cursor = max(cursor, end)
    return (receipt(ContextText(str(prompt), records, **{k: v for k, v in details.items() if k not in {'purpose', 'tokens', 'budget', 'records', 'source_ids', 'omitted_source_ids', 'max_chars'}}), purpose=purpose, input_section='prompt', budgets=budgets),)


def section_receipts(sections, *, purpose):
    """Inventory exact application fields, including nested selection candidates."""
    result = []
    for name, value in sections.items():
        if isinstance(value, ContextText):
            inventory = receipt(value, purpose=purpose, input_section=name)
        else:
            rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            inventory = receipt(source(rendered, f'input:{purpose}:{name}', 'decision_' + name), purpose=purpose, input_section=name)
            if name in {'context_candidates', 'skill_candidates', 'acceptance_criteria', 'verified_evidence'} and isinstance(value, dict):
                rows = []
                for identity, item in value.items():
                    block = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
                    row = source(block, str(identity), {'context_candidates': 'context_candidate', 'skill_candidates': 'skill_candidate', 'acceptance_criteria': 'acceptance_criterion', 'verified_evidence': 'verified_evidence'}[name], reason='selection_candidate' if 'candidates' in name else 'completion_review').records[0]
                    row.pop('input_range', None)
                    if isinstance(item, dict):
                        row['field_ranges'] = {field: [r.get('source_range') for r in val.records if r.get('source_range')] for field, val in item.items() if isinstance(val, ContextText)}
                        row['truncated'] = any(r.get('truncated') for val in item.values() if isinstance(val, ContextText) for r in val.records)
                    # A bounded evidence summary can carry its original extent.
                    if isinstance(item, dict) and isinstance(item.get('summary'), ContextText):
                        summary_rows = item['summary'].records
                        if summary_rows:
                            row['truncated'] = any(r.get('truncated') for r in summary_rows)
                            row['source_range'] = summary_rows[0].get('source_range')
                    rows.append(row)
                inventory['records'] = rows
                inventory['source_ids'] = list(value.keys())
        result.append(inventory)
    return result


class EvidenceEntries(list):
    def __init__(self, entries, omitted_records):
        super().__init__(entries)
        self.omitted_records = omitted_records
