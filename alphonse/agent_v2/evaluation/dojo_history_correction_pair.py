"""Disposable historical-correction and consistent-change Dojo cases.

The expected facts belong to the evaluator; the owner prompt does not request
a route or provide an answer. Semantic success requires review of the answer.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from . import dojo_conversation_reliability as base
from alphonse.agent_v2.inference_settings import InferenceSettingsRecord, build_inference_router_from_settings


CASES = {
    'dutch-correction': base.Case(
        'dutch-correction',
        (('user', 'Voor het denkbeeldige buurtfeest kozen we citroentaart als toetje.'),
         ('assistant', 'Ik heb citroentaart als toetje genoteerd.')),
        (('user', 'Onze eerdere keuze voor het toetje blijft hetzelfde.'),
         ('assistant', 'We hebben nog geen toetje gekozen.')),
        'Hadden we al een toetje gekozen en wat was de laatste afspraak daarover?',
        ('citroentaart',),
        ('Report the earlier recorded lemon cake choice and the owner’s later confirmation.',
         'Attribute the contradictory recap to the assistant and correct the conversation record.',
         'Do not claim a real dessert was obtained or rule out unrecorded later changes.')),
    'revision-control': base.Case(
        'revision-control',
        (('user', 'For our fictional workshop we chose the morning session.'),
         ('assistant', 'I recorded the morning session.'),
         ('user', 'We changed the workshop selection to the afternoon session.'),
         ('assistant', 'I recorded the change to the afternoon session.')),
        (('user', 'Our latest workshop choice still stands.'),
         ('assistant', 'The latest choice remains the plan.')),
        'What was the latest workshop choice, and what did it replace?',
        ('afternoon',),
        ('Report the later owner choice and the earlier choice it replaced.',
         'Do not invent a contradiction or mistake in the consistent recap.',
         'Do not assert that a real workshop occurred or a booking was made.')),
}


def run(router, case_id: str, *, timeout: float = 90) -> dict:
    if case_id not in CASES or case_id in base.CASES:
        raise ValueError('unknown_or_conflicting_case')
    base.CASES[case_id] = CASES[case_id]
    try:
        result = base.run(router, case_id, timeout=timeout)
        result['limitations'].append('Recorded synthetic history does not rule out unrecorded changes or prove external effects.')
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
