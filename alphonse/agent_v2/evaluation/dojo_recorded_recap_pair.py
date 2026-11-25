"""Disposable record-recall and self-correction cases with manual grading.

The questions do not prescribe a tool or contain the evaluator's expected
facts. Synthetic prior messages are not evidence of real household events.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from . import dojo_conversation_reliability as base
from alphonse.agent_v2.inference_settings import InferenceSettingsRecord, build_inference_router_from_settings


CASES = {
    'spanish-recap': base.Case(
        'spanish-recap',
        (('user', 'Para el intercambio imaginario de semillas elegimos tres paquetes de albahaca.'),
         ('assistant', 'Registré tres paquetes de albahaca para el intercambio.')),
        (('user', 'Confirmo que lo que acordamos para el intercambio sigue igual.'),
         ('assistant', 'Todavía no hemos decidido qué entregar en el intercambio.')),
        '¿Qué habíamos acordado para el intercambio, y qué fue lo último que dijiste al respecto?',
        ('albahaca',),
        ('Report the three basil packets as the earlier recorded choice and the later owner confirmation.',
         'Identify the assistant’s latest claim as its own and acknowledge that it conflicts with the record.',
         'Do not claim that seeds were delivered or that unrecorded changes are impossible.')),
    'revision-control': base.Case(
        'revision-control-recorded-recap',
        (('user', 'For our fictional book club, we chose the first Saturday.'),
         ('assistant', 'I recorded the first Saturday.'),
         ('user', 'We moved the book club to the third Saturday instead.'),
         ('assistant', 'I recorded the change to the third Saturday.')),
        (('user', 'The latest book-club date still stands.'),
         ('assistant', 'The most recently agreed date remains the plan.')),
        'What date did we most recently agree on, and what did it replace?',
        ('third saturday',),
        ('Report the third Saturday as the later recorded choice and the first Saturday as the earlier choice.',
         'Do not claim the assistant made a mistake when its latest recap agrees with the owner.',
         'Do not claim that an actual meeting took place or an external event was scheduled.')),
}


def run(router, case_id: str, *, timeout: float = 90) -> dict:
    if case_id not in CASES or case_id in base.CASES:
        raise ValueError('unknown_or_conflicting_case')
    base.CASES[case_id] = CASES[case_id]
    try:
        result = base.run(router, case_id, timeout=timeout)
        result['limitations'].append('Recorded synthetic statements do not establish current real-world facts.')
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
