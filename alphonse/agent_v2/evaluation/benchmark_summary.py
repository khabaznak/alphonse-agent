"""Summarize a predeclared synthetic benchmark without hiding missing attempts.

This offline accounting helper does not execute agents, verify an oracle's
independence, attest receipt provenance, or establish delivery or cost.
"""
from __future__ import annotations

import math
from typing import Any


def summarize(schedule: list[dict[str, str]], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Account for each explicitly identified scheduled attempt.

    A missing or interrupted attempt remains unknown. The caller supplies the
    independently evaluated grade; self-reported agent success is not a grade.
    No average latency is computed over a selected subset of successful runs.
    """
    if not isinstance(schedule, list) or not schedule or not isinstance(attempts, list):
        raise ValueError('invalid_benchmark_inputs')
    schedule_by_id = {}
    for row in schedule:
        if (not isinstance(row, dict) or set(row) != {'attempt_id', 'case_id', 'arm'} or
                any(not isinstance(row[k], str) or not row[k].strip()
                    for k in ('attempt_id', 'case_id', 'arm'))):
            raise ValueError('invalid_schedule')
        if row['attempt_id'] in schedule_by_id:
            raise ValueError('duplicate_schedule')
        schedule_by_id[row['attempt_id']] = row
    indexed = {}
    for row in attempts:
        if (not isinstance(row, dict) or
                set(row) != {'attempt_id', 'case_id', 'arm', 'state', 'grade', 'elapsed_ms'} or
                any(not isinstance(row[k], str) or not row[k].strip()
                    for k in ('attempt_id', 'case_id', 'arm'))):
            raise ValueError('invalid_attempt')
        identity = row['attempt_id']
        expected = schedule_by_id.get(identity)
        if (expected is None or identity in indexed or
                row['case_id'] != expected['case_id'] or row['arm'] != expected['arm']):
            raise ValueError('unexpected_or_duplicate_attempt')
        state, grade, elapsed = row['state'], row['grade'], row['elapsed_ms']
        if state not in {'finished', 'interrupted', 'unavailable'}:
            raise ValueError('invalid_attempt_state')
        if grade not in {'pass', 'fail', 'unknown'} or (state != 'finished' and grade != 'unknown'):
            raise ValueError('invalid_grade')
        if elapsed is not None and (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                                    or not math.isfinite(elapsed) or elapsed < 0):
            raise ValueError('invalid_elapsed')
        indexed[identity] = row
    arms = {}
    for identity, scheduled in schedule_by_id.items():
        case_id, arm = scheduled['case_id'], scheduled['arm']
        result = arms.setdefault(arm, {'scheduled': 0, 'pass': 0, 'fail': 0, 'unknown': 0,
                                       'missing': 0, 'cases': {}, 'attempts': []})
        case_result = result['cases'].setdefault(
            case_id, {'scheduled': 0, 'pass': 0, 'fail': 0, 'unknown': 0, 'missing': 0})
        result['scheduled'] += 1
        case_result['scheduled'] += 1
        row = indexed.get(identity)
        if row is None:
            result['missing'] += 1
            result['unknown'] += 1
            case_result['missing'] += 1
            case_result['unknown'] += 1
            result['attempts'].append({'attempt_id': identity, 'case_id': case_id, 'state': 'not_started',
                                       'grade': 'unknown', 'elapsed_ms': None})
        else:
            result[row['grade']] += 1
            case_result[row['grade']] += 1
            result['attempts'].append({k: row[k] for k in ('attempt_id', 'case_id', 'state', 'grade', 'elapsed_ms')})
    all_cases = {row['case_id'] for row in schedule_by_id.values()}
    unbalanced = {}
    if len(arms) > 1:
        for case_id in sorted(all_cases):
            counts = {arm: result['cases'].get(case_id, {}).get('scheduled', 0)
                      for arm, result in arms.items()}
            if len(set(counts.values())) > 1:
                unbalanced[case_id] = counts
    balance = ('unassessed' if len(arms) < 2 else 'unbalanced' if unbalanced else 'balanced')
    return {'arms': arms, 'case_schedule_balance': {'status': balance, 'unbalanced_cases': unbalanced},
            'qualification': 'unassessed',
            'limitations': ['Grades and provenance are caller-supplied and require independent audit.',
                            'Unknown and missing attempts are not passes.',
                            'Equal case counts alone do not establish fair or paired comparisons.',
                            'Elapsed time alone does not establish useful response, delivery, or cost.']}
