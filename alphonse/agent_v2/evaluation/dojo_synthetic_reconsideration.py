"""Disposable-only experiment for a budgeted System Two reconsideration.

This opt-in is installed only inside the disposable Dojo process. It is not an
independent semantic verifier and does not enable a review in the owner daemon.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from . import dojo_conversation_reliability as base
from alphonse.agent_v2.core.intelligence.simple_loop import owner_read_processor
from alphonse.agent_v2.inference_settings import InferenceSettingsRecord, build_inference_router_from_settings


def run(router, case_id: str, *, timeout: float = 90) -> dict:
    if case_id not in base.CASES:
        raise ValueError('unknown_reconsideration_case')
    # The fixture creates its own disposable database and stops its worker
    # before returning. The factory override is process-local and restored even
    # on failure. No production daemon or household data is touched.
    original_factory = owner_read_processor.CandidateHost

    def synthetic_factory(*args, **kwargs):
        kwargs['synthetic_preanswer_review'] = True
        return original_factory(*args, **kwargs)

    owner_read_processor.CandidateHost = synthetic_factory
    try:
        result = base.run(router, case_id, timeout=timeout)
        result['experimental_reconsideration'] = 'synthetic_only'
        result['limitations'].extend([
            'One additional budgeted model decision is requested before the first answer.',
            'Reconsideration uses the same model and evidence; it is not independent verification.',
            'This is an opt-in fixture override, not a production owner-worker feature.',
        ])
        return result
    finally:
        owner_read_processor.CandidateHost = original_factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--case', required=True, choices=sorted(base.CASES))
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
