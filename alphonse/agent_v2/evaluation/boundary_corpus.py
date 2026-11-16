"""Immutable synthetic boundary corpus and its independent source oracles."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/fixtures/memory_replay"


def load_boundaries():
    base_bytes = (FIXTURES / 'v1.json').read_bytes()
    base = json.loads(base_bytes)
    extension = json.loads((FIXTURES / 'paired_v3.json').read_text())
    if extension['synthetic'] is not True or extension['base_sha256'] != hashlib.sha256(base_bytes).hexdigest():
        raise ValueError('frozen_v1_corpus_changed')
    if len(base['cases']) != 22 or set(extension['original']) != {c['id'] for c in base['cases']}:
        raise ValueError('original_boundary_cases_changed')
    cases = []
    for original in base['cases']:
        case = deepcopy(original)
        supplement = extension['original'][case['id']]
        required_keys = {'delivery_oracles', 'necessary_by_purpose', 'state_oracle'}
        if not required_keys <= set(supplement) <= required_keys | {'zero_history'}:
            raise ValueError('original_expectations_cannot_be_overridden')
        for identity in original['expected']['necessary_sources']:
            content = next(s['content'] for s in original['sources'] if s['id'] == identity)
            if not any(o['source_id'] == identity and o['start_char'] == 0 and o['end_char'] == len(content)
                       and 'response_composition' in o['purposes'] for o in supplement['delivery_oracles']):
                raise ValueError('original_delivery_expectation_weakened')
        case.update(supplement)
        cases.append(case)
    cases.extend(deepcopy(extension['extensions']))
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('duplicate_case_id')
    for case in cases:
        sources = {s['id']: s for s in case['sources']}
        if len(sources) != len(case['sources']):
            raise ValueError('duplicate_source_id')
        for oracle in case['delivery_oracles']:
            if not 0 <= oracle['start_char'] < oracle['end_char'] <= len(sources[oracle['source_id']]['content']):
                raise ValueError('invalid_source_range')
        if not set(case['expected']['necessary_sources']) <= {o['source_id'] for o in case['delivery_oracles']}:
            raise ValueError('missing_frozen_source_oracle')
    return {**{k: extension[k] for k in ('clock', 'timezone', 'rubric', 'version')}, 'synthetic': True, 'cases': cases}


