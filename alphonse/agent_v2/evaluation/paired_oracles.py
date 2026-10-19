"""Source-qualified, call-local delivery and fixture-state oracles for DC-2.

Offsets are Python Unicode character offsets, half open. A source ID in a search
result is not delivery: its content must reach the specified inference boundary.
"""
from __future__ import annotations

import json
from collections import Counter


def delivered_ranges(sections, manifest, sources):
    """Inspect actual call fields, never a union of retrievals or later task state.

    Structured recall envelopes bind IDs and offsets to exact source slices. Plain
    legacy renderings require a call-time receipt binding the ID to an input span.
    Clipped/escaped JSON that cannot be decoded yields no credit.
    """
    found = set()
    by_id = {value['runtime_id']: (key, value['content']) for key, value in sources.items()}

    def inspect(value):
        if isinstance(value, dict):
            reference = value.get('source')
            identity = value.get('source_id') or (reference.get('source_id') if isinstance(reference, dict) else None)
            if isinstance(identity, str) and identity in by_id:
                key, original = by_id[identity]
                content = value.get('content', value.get('excerpt'))
                start = value.get('start_char', 0)
                end = value.get('end_char', len(original))
                if (type(start) is int and type(end) is int and isinstance(content, str)
                        and 0 <= start < end <= len(original) and content == original[start:end]):
                    found.add((key, start, end))
            for child in value.values():
                inspect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                inspect(child)
        elif isinstance(value, str):
            decoder = json.JSONDecoder()
            cursor = 0
            while cursor < len(value):
                positions = [p for c in '{[' if (p := value.find(c, cursor)) >= 0]
                if not positions:
                    break
                position = min(positions)
                try:
                    decoded, length = decoder.raw_decode(value[position:])
                except ValueError:
                    cursor = position + 1
                else:
                    inspect(decoded)
                    cursor = position + length

    for value in sections.values():
        inspect(value)
    for projection in manifest.get('projections', []):
        value = sections.get(projection.get('input_section'))
        if not isinstance(value, str):
            continue
        for record in projection.get('records', []):
            if record.get('status') != 'included' or record.get('source_id') not in by_id:
                continue
            key, original = by_id[record['source_id']]
            span, extent = record.get('input_range'), record.get('source_range')
            if not span:
                continue
            extent = extent or {}
            start, end = extent.get('start_char'), extent.get('end_char')
            block = value[span['start_char']:span['end_char']]
            if not record.get('truncated') and (original in block or original.replace('"', '\\"') in block):
                found.add((key, 0, len(original)))
            elif type(start) is int and type(end) is int and 0 <= start < end <= len(original) and block == original[start:end]:
                found.add((key, start, end))
    return [{'source_id': key, 'start_char': start, 'end_char': end} for key, start, end in sorted(found)]


def covers(delivered, source_id, start, end):
    cursor = start
    for row in sorted((r for r in delivered if r['source_id'] == source_id), key=lambda r: r['start_char']):
        if row['start_char'] > cursor:
            break
        cursor = max(cursor, row['end_char'])
    return cursor >= end


def score_delivery(case, decisions):
    checks = []
    for oracle in case['delivery_oracles']:
        selected = [d for d in decisions if set(d.get('stages', [d['purpose']])) & set(oracle['purposes'])]
        checks.append({'oracle': oracle, 'decision_count': len(selected), 'passed': bool(selected) and all(
            covers(d['delivered'], oracle['source_id'], oracle['start_char'], oracle['end_char']) for d in selected)})
    instances = irrelevant = 0
    forbidden = set()
    duplicates = []
    for decision in decisions:
        ids = {r['source_id'] for r in decision['delivered']}
        if 'historical_source_ids' in case:
            ids &= set(case['historical_source_ids'])
        forbidden.update(ids & set(case.get('expected', {}).get('forbidden_sources', [])))
        for projection in decision.get('manifest', {}).get('projections', []):
            counts = Counter(r['source_id'] for r in projection.get('records', []) if r.get('status') == 'included')
            duplicates.extend({'decision_id': decision['decision_id'], 'source_id': identity, 'count': count}
                              for identity, count in counts.items() if count > 1)
        instances += len(ids)
        irrelevant += len(ids - set(case['necessary_by_purpose'].get(decision['purpose'], [])))
    necessary = all(c['passed'] for c in checks)
    zero_history = not case.get('zero_history') or instances == 0
    return {'checks': checks, 'necessary_delivery_pass': necessary,
            'delivery_pass': necessary and zero_history and not forbidden,
            'zero_history_pass': zero_history, 'forbidden_source_delivery': sorted(forbidden),
            'duplicate_source_ids': duplicates,
            'historical_instances': instances, 'irrelevant_instances': irrelevant,
            'irrelevant_fraction': irrelevant / instances if instances else 0}


def score_outcome(case, state, answer, status):
    """Exact fixture read-back is separate from diagnostic response fragments."""
    oracle = case['state_oracle']
    checks = {'terminal': status in oracle['statuses'],
              'state': all(state.get(k) == v for k, v in oracle['equals'].items()),
              'no_forbidden_effects': not state['forbidden_attempts'],
              'no_duplicate_writes': state['label_writes'] == oracle.get('label_writes', 0),
              'no_unplanned_schedules': bool(oracle.get('schedule')) or not state.get('schedules')}
    for operation in oracle.get('required_reads', []):
        checks['read:' + operation] = operation in state['reads']
    if oracle.get('safe_failure_reason'):
        checks['safe_failure_reason'] = state.get('terminal_reason') == oracle['safe_failure_reason']
        checks['safe_failure_before_inference'] = state.get('inference_calls', 0) <= oracle['max_inference_calls']
    folded = answer.casefold()
    response = case['expected']
    return {'checks': checks, 'effect_pass': all(checks.values()),
            'response_diagnostic_pass': bool(oracle.get('safe_failure_reason') and all(checks.values())) or bool(answer) and all(x.casefold() in folded for x in response['required_answer_fragments'])
                and not any(x.casefold() in folded for x in response['forbidden_answer_fragments']),
            'human_review': 'unassessed'}
