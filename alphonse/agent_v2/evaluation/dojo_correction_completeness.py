"""Synthetic recall, correction and completeness cases for the owner worker.

These cases do not prescribe a tool route. A literal marker is diagnostic and
the complete response requires independent semantic review.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from . import dojo_conversation_reliability as base
from alphonse.agent_v2.inference_settings import InferenceSettingsRecord, build_inference_router_from_settings


CASES = {
    'german-correction': base.Case(
        'german-correction',
        (('user', 'Für den fiktiven Nachbarschaftsabend haben wir das Gitarrenduo ausgewählt.'),
         ('assistant', 'Ich habe das Gitarrenduo für den Nachbarschaftsabend notiert.')),
        (('user', 'Ich bestätige, dass unsere frühere Auswahl für den Abend weiterhin gilt.'),
         ('assistant', 'Für den Nachbarschaftsabend wurde noch keine Musik ausgewählt.')),
        'Was hatten wir für den Abend ausgewählt, und galt diese Auswahl später noch?',
        ('Gitarrenduo',),
        ('Report the recorded guitar duo choice and the later owner confirmation.',
         'Attribute the contradictory recap to the assistant and correct the recorded conversation.',
         'Avoid asserting an unrecorded change or that any event occurred.')),
    'italian-owner-change': base.Case(
        'italian-owner-change',
        (('user', 'Per la mostra immaginaria abbiamo scelto una cornice rossa.'),
         ('assistant', 'Ho annotato la cornice rossa.'),
         ('user', 'Abbiamo cambiato scelta: preferiamo una cornice blu.'),
         ('assistant', 'Ho annotato il cambio alla cornice blu.')),
        (('user', 'La nostra ultima scelta per la mostra rimane valida.'),
         ('assistant', 'Rimane valida la vostra scelta più recente per la mostra.')),
        'Qual era la prima scelta per la mostra e quale abbiamo poi confermato?',
        ('cornice blu',),
        ('Report the initial red choice, later blue owner change and later confirmation.',
         'Do not invent an error in the consistent assistant recap or reverse the owner change.',
         'Do not claim the frame was acquired or an event occurred.')),
}


def run(router, case_id: str, *, timeout: float = 90) -> dict:
    if case_id not in CASES or case_id in base.CASES:
        raise ValueError('unknown_or_conflicting_case')
    base.CASES[case_id] = CASES[case_id]
    try:
        result = base.run(router, case_id, timeout=timeout)
        result['limitations'].append(
            'Recorded synthetic statements do not rule out unrecorded changes or prove external effects.')
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
