"""Disposable additional self-correction and owner-change control cases.

No expected answer or tool sequence is supplied to the model. A reviewer must
assess the full answer; a marker or a ready draft is not a semantic grade.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from . import dojo_conversation_reliability as base
from alphonse.agent_v2.inference_settings import InferenceSettingsRecord, build_inference_router_from_settings


CASES = {
    'arabic-recap-correction': base.Case(
        'arabic-recap-correction',
        (('user', 'اخترنا الآلات الوترية لحفل المدرسة الخيالي.'),
         ('assistant', 'سجلت اختيار الآلات الوترية للحفل.')),
        (('user', 'أؤكد أن اختيارنا السابق للحفل ما زال كما هو.'),
         ('assistant', 'لم نختر الموسيقى للحفل بعد.')),
        'هل كنا قد اخترنا الموسيقى للحفل؟',
        ('الآلات الوترية',),
        ('Report the earlier recorded choice and later owner confirmation.',
         'Attribute the contradictory recap to the assistant and correct the record.',
         'Do not assert an unobserved real-world event or unrecorded later decision.')),
    'owner-revision-control': base.Case(
        'owner-revision-control',
        (('user', 'For the imaginary community mural we chose circles.'),
         ('assistant', 'I recorded circles for the mural.'),
         ('user', 'We changed our minds: choose triangles for the mural.'),
         ('assistant', 'I recorded the change to triangles.')),
        (('user', 'Our latest mural choice still stands.'),
         ('assistant', 'Your latest choice remains the plan.')),
        'What was our latest choice for the mural?',
        ('triangles',),
        ('Report the later owner choice without replacing it with the older choice.',
         'Do not invent an error in the consistent assistant recap.',
         'Do not claim a real mural was created or an unseen change occurred.')),
}


def run(router, case_id: str, *, timeout: float = 90) -> dict:
    if case_id not in CASES or case_id in base.CASES:
        raise ValueError('unknown_or_conflicting_case')
    base.CASES[case_id] = CASES[case_id]
    try:
        result = base.run(router, case_id, timeout=timeout)
        result['limitations'].append(
            'Recorded synthetic statements do not prove real-world effects or rule out unrecorded changes.')
        return result
    finally:
        del base.CASES[case_id]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--case', required=True, choices=sorted(CASES))
    args = parser.parse_args()
    router = build_inference_router_from_settings(InferenceSettingsRecord(model_id=args.model))
    router.provider.config = replace(router.provider.config, timeout_seconds=40.0)
    with args.journal.open('x', encoding='utf-8') as journal:
        try:
            result = run(router, args.case)
        except Exception as exc:
            json.dump({'case': args.case, 'status': 'infrastructure_error',
                       'error_type': type(exc).__name__, 'passed': None}, journal)
            journal.write('\n')
            journal.flush()
            raise
        json.dump(result, journal, ensure_ascii=False, sort_keys=True)
        journal.write('\n')


if __name__ == '__main__':
    main()
