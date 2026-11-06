"""DC-5 frozen inputs and mechanical diagnostics; never rollout approval."""
from __future__ import annotations

import hashlib
import json
import math
from statistics import median

from .paired_memory import FIXTURES
from .paired_oracles import covers

CATEGORIES = {'current_evidence', 'explicit_recall', 'implicit_recall', 'english',
              'spanish', 'continuity', 'conflicting_evidence', 'stale_evidence',
              'access_boundaries', 'backend_failure'}


def load_heldout():
    raw = (FIXTURES / 'heldout_v1.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != (FIXTURES / 'heldout_v1.sha256').read_text().strip():
        raise ValueError('frozen_heldout_corpus_changed')
    corpus = json.loads(raw)
    cases = corpus['cases']
    if (corpus['synthetic'] is not True or len(cases) < 100 or
            len({c['id'] for c in cases}) != len(cases) or
            len({json.dumps([c['message'], c['sources'], c.get('active_turns', [])], sort_keys=True) for c in cases}) != len(cases) or
            {c['category'] for c in cases} != CATEGORIES):
        raise ValueError('invalid_heldout_coverage')
    for case in cases:
        sources = {s['id']: s for s in case['sources']}
        if len(sources) != len(case['sources']):
            raise ValueError('duplicate_heldout_source')
        for oracle in case['delivery_oracles']:
            if not (oracle['purposes'] and 0 <= oracle['start_char'] < oracle['end_char'] <= len(sources[oracle['source_id']]['content'])):
                raise ValueError('invalid_heldout_source_oracle')
        if not set(case['expected']['necessary_sources']) <= {o['source_id'] for o in case['delivery_oracles']}:
            raise ValueError('missing_heldout_source_oracle')
        if not case['state_oracle'] or case['outcome_kind'] not in {'verified_effect', 'safe_limitation'}:
            raise ValueError('missing_heldout_state_oracle')
    return corpus


def fraction(passed, total):
    """Wilson 95% interval, descriptive only; repeated trials are dependent."""
    if not total:
        return {'passed': passed, 'total': total, 'rate': None, 'wilson95': None}
    p, z = passed / total, 1.959963984540054
    divisor = 1 + z*z/total
    center = (p + z*z/(2*total))/divisor
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))/divisor
    return {'passed': passed, 'total': total, 'rate': p, 'wilson95': [max(0, center-radius), min(1, center+radius)]}


def assess(manifest, rows):
    cases = manifest['corpus']['cases']
    repetitions = manifest['repetitions']
    arms, breaches = {}, []
    for arm in ('legacy', 'candidate'):
        selected = [r for r in rows if r['arm'] == arm]
        recalls = {category: [0, 0] for category in CATEGORIES}
        passed_tasks, effect_tasks, safe_tasks = 0, 0, 0
        for case in cases:
            trials = [r for r in selected if r['case_id'] == case['id']]
            passed = len(trials) == repetitions and all(
                r.get('status') == 'returned' and not r.get('violations') and
                r.get('outcome', {}).get('effect_pass') and
                r.get('outcome', {}).get('response_diagnostic_pass') and
                r.get('delivery', {}).get('delivery_pass') and
                not r.get('delivery', {}).get('duplicate_source_ids') and
                (not r.get('runtime_error') or r.get('expected_safe_failure'))
                for r in trials)
            passed_tasks += passed
            if case['outcome_kind'] == 'safe_limitation':
                safe_tasks += passed
            else:
                effect_tasks += passed
            for trial in trials:
                for oracle in case['delivery_oracles']:
                    for purpose in oracle['purposes']:
                        decisions = [d for d in trial.get('decisions', []) if purpose in d.get('stages', [d['purpose']])]
                        # A missing required stage is a missed source instance.
                        for decision in decisions or [None]:
                            recalls[case['category']][1] += 1
                            recalls[case['category']][0] += bool(decision and covers(decision['delivered'], oracle['source_id'], oracle['start_char'], oracle['end_char']))
        samples = [r['latency_ms'] for r in selected if isinstance(r.get('latency_ms'), (float, int))]
        per_case = {}
        for case in cases:
            values = sorted(r['latency_ms'] for r in selected if r['case_id'] == case['id'] and isinstance(r.get('latency_ms'), (float, int)))
            per_case[case['id']] = values[math.ceil(.95*len(values))-1] if values else None
        instances = sum(r.get('delivery', {}).get('historical_instances', 0) for r in selected)
        irrelevant = sum(r.get('delivery', {}).get('irrelevant_instances', 0) for r in selected)
        arms[arm] = {'distinct_tasks': len(cases), 'measured_trials': len(selected),
            'task_outcome_diagnostic': fraction(passed_tasks, len(cases)),
            'verified_effect_tasks': fraction(effect_tasks, sum(c['outcome_kind'] == 'verified_effect' for c in cases)),
            'safe_outcome_tasks': fraction(safe_tasks, sum(c['outcome_kind'] == 'safe_limitation' for c in cases)),
            'source_recall': fraction(sum(v[0] for v in recalls.values()), sum(v[1] for v in recalls.values())),
            'recall_by_category': {k: fraction(*v) for k, v in sorted(recalls.items())},
            'historical_instances': instances, 'irrelevant_instances': irrelevant,
            'irrelevant_fraction': irrelevant / instances if instances else 0,
            'calls': {key: sum(len(r[key]) for r in selected) if selected and all(isinstance(r.get(key), list) for r in selected) else None
                      for key in ('model_calls','jev_calls','tool_calls')},
            'latency_samples_ms': samples,
            'instrumentation_complete': all(isinstance(r.get('latency_ms'), (int, float)) and all(
                isinstance(r.get(key), list) for key in ('model_calls', 'jev_calls', 'tool_calls')) for r in selected),
            'p95_ms': sorted(samples)[math.ceil(.95*len(samples))-1] if samples else None,
            'median_ms': median(samples) if samples else None,
            'per_case_p95_ms': per_case}
    complete = all(a['measured_trials'] == len(cases)*repetitions for a in arms.values()) and not manifest.get('smoke_cases')
    if complete:
        a, b = arms['candidate'], arms['legacy']
        for label, minimum in (('source_recall', .98), ('task_outcome_diagnostic', .95)):
            candidate, baseline = a[label]['rate'], b[label]['rate']
            if candidate is not None and (candidate < minimum or baseline is not None and candidate < baseline-.01-1e-12):
                breaches.append(label)
        for category in CATEGORIES:
            candidate = a['recall_by_category'][category]['rate']
            baseline = b['recall_by_category'][category]['rate']
            if candidate is not None and (candidate < .98 or baseline is not None and candidate < baseline-.01-1e-12):
                breaches.append('recall:' + category)
        if a['irrelevant_fraction'] > .05:
            breaches.append('irrelevant_context')
        if repetitions >= 20 and all(v['instrumentation_complete'] for v in arms.values()):
            for key in ('tool_calls', 'total_model_calls'):
                totals = [v['calls']['model_calls']+v['calls']['jev_calls'] if key == 'total_model_calls' else v['calls'][key] for v in (a,b)]
                if totals[0] > 1.10*totals[1]:
                    breaches.append(key)
            for case_id in [None] + [c['id'] for c in cases]:
                candidate = a['p95_ms'] if case_id is None else a['per_case_p95_ms'][case_id]
                baseline = b['p95_ms'] if case_id is None else b['per_case_p95_ms'][case_id]
                if candidate is not None and baseline is not None and candidate > 1.10*baseline:
                    breaches.append('latency:' + (case_id or 'pooled'))
    return {'arms': arms, 'template_families': len(CATEGORIES), 'equal_repetitions': repetitions,
            'threshold_breaches': sorted(breaches),
            'performance_assessment': 'diagnostic' if repetitions >= 20 and complete and all(v['instrumentation_complete'] for v in arms.values()) else 'unassessed',
            'historical_v1_25_percent_target': 'unassessed', 'actual_provider_usage_comparison': 'unassessed',
            'semantic_review': 'unassessed', 'owner_decision': 'unassessed', 'rollback': 'separate evidence required',
            'qualification': 'unassessed',
            'limitations': ['Task scores require every declared repetition and fixture effects, but semantic human review is pending.',
                           'Wilson intervals describe observed counts; trials and template families are not independent population samples.',
                           'No recall denominator in self-contained/fault categories means not applicable, never perfect recall.',
                           'Frozen-v1 savings, DC-0 through DC-4, actual usage, rollback and owner acceptance remain separate requirements.']}
