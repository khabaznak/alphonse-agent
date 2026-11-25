"""Independent synthetic cases for correction and warranted uncertainty.

The answer and its semantic assessment are distinct. The fixture prescribes no
tool route and makes no inference about current real-world state.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from . import dojo_conversation_reliability as base
from alphonse.agent_v2.inference_settings import InferenceSettingsRecord, build_inference_router_from_settings


CASES = {
    'portuguese-correction': base.Case(
        'portuguese-correction',
        (('user', 'Para a festa imaginária, escolhemos toalhas listradas para as mesas.'),
         ('assistant', 'Registrei que as toalhas serão listradas.')),
        (('user', 'Confirmo que a escolha anterior para as mesas continua a mesma.'),
         ('assistant', 'Ainda não escolhemos as toalhas das mesas.')),
        'Já havíamos escolhido as toalhas para a festa?',
        ('listradas',),
        ('Report the recorded striped tablecloth choice and later confirmation.',
         'Identify the subsequent contradictory recap as the assistant’s own and correct it.',
         'Avoid asserting an unobserved real-world effect or unrecorded later decision.')),
    'later-change-control': base.Case(
        'later-change-control',
        (('user', 'We chose the fictional exhibit label in silver.'),
         ('assistant', 'I recorded silver as the exhibit label choice.'),
         ('user', 'We changed the exhibit label to bronze instead.'),
         ('assistant', 'I recorded the new bronze choice.')),
        (('user', 'The latest choice for the exhibit label still stands.'),
         ('assistant', 'The most recent exhibit label choice remains the plan.')),
        'What was our most recent exhibit label choice?',
        ('bronze',),
        ('Report the later owner decision, bronze, rather than the older silver choice.',
         'Do not invent an error in the consistent assistant recap.',
         'Do not assert the label was physically produced.')),
}


def run(router, case_id: str, *, timeout: float = 90) -> dict:
    if case_id not in CASES or case_id in base.CASES:
        raise ValueError('unknown_or_conflicting_case')
    base.CASES[case_id] = CASES[case_id]
    try:
        result = base.run(router, case_id, timeout=timeout)
        result['limitations'].append(
            'A synthetic conversation establishes recorded statements, not unrecorded later changes.')
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
