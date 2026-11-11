"""Preview, or explicitly run, one V3 variant on the synthetic clock.

Default preview reads no credentials and uses no network. A separately approved
scope must include both OpenAI and TypeSafe, synthetic data, retention, maximum
budget and uncertainty. Flags below are operator assertions, not independent
proof of user approval, isolation, exact provider billing or a hard dollar cap.
Run each variant in its own process; this CLI handles exactly one attempt.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import getpass
import json
import os
from pathlib import Path

from .clock_pilot_setup import (MODEL, INSTANT, TIMEZONE, MAX_CALLS, INPUT_PER_CALL,
                                OUTPUT_PER_CALL, INPUT_RATE, OUTPUT_RATE, _budget_nanos, _estimated_input, preview)
from .jev_attempt_meter import JevAttemptLedger, MeteredJevTransport
from .jev_pilot_http import JevPilotHTTPTransport
from .responses_pilot_transport import PilotLedger, PilotLimits, ResponsesPilotTransport, requests_sender
from .v3_clock_responses_attempt import run_v3_clock_responses_attempt
from .v3_live_clock import V3PilotResponsesProvider
from alphonse.agent_v2.system_one import DEFAULT_SYSTEM_ONE_URL, JevCriterionDecisionProvider, SystemOneSettings

JEV_MODEL = 'jev-latest'
JEV_MAX_ATTEMPTS = 30
JEV_MAX_PAYLOAD_BYTES = 65536
# TypeSafe's published rate is $0.042 per million input tokens; bytes are a
# conservative heuristic, not a verified token ceiling or bill.
JEV_INPUT_RATE_NANODOLLARS = 42


def v3_preview(*, arm: str, openai_budget_usd: str = '0.25', jev_budget_usd: str = '0.10') -> dict:
    if arm not in {'v3', 'targeted_v3'}:
        raise ValueError('invalid_v3_pilot_arm')
    report = preview(budget_usd=openai_budget_usd)
    jev_budget = _budget_nanos(jev_budget_usd)
    estimated_jev = JEV_MAX_ATTEMPTS * JEV_MAX_PAYLOAD_BYTES * JEV_INPUT_RATE_NANODOLLARS
    report.update(kind='one_synthetic_v3_clock_attempt', arm=arm,
                  projection_enabled=arm == 'targeted_v3',
                  jev={'provider': 'TypeSafe System One', 'model_requested': JEV_MODEL,
                       'max_http_attempts_including_retries': JEV_MAX_ATTEMPTS,
                       'max_request_payload_bytes': JEV_MAX_PAYLOAD_BYTES,
                       'estimated_input_reservations_usd': str(Decimal(estimated_jev) / Decimal(1_000_000_000)),
                       'proposed_budget_usd': str(Decimal(jev_budget) / Decimal(1_000_000_000)),
                       'estimate_fits_proposed_budget': estimated_jev <= jev_budget,
                       'pricing_reference': 'https://help.typesafe.ai/articles/5187202169-how-much-does-jev-cost',
                       'data_scope': 'synthetic task, tool descriptions and evidence supplied as decision state',
                       'provider_retention': 'subject_to_actual_TypeSafe_account_policy',
                       'budget_enforcement': 'attempt count and request bytes only; dollar cost is estimated'})
    report['limitations'] = [
        limitation for limitation in report['limitations']
        if limitation != 'This single candidate run cannot compare V3 or an external agent.'
    ]
    report['limitations'].extend([
        'Jev attempt-count and payload limits are not a hard provider dollar cap.',
        'Each Jev retry is metered, but an in-flight request or retry sleep cannot be forcibly interrupted.',
        'V3 schema-in-prompt and local validation differ from provider-constrained decoding.',
        'This one V3 variant is not a complete matched comparison.',
    ])
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['v3', 'targeted_v3'], required=True)
    parser.add_argument('--openai-budget-usd', default='0.25')
    parser.add_argument('--jev-budget-usd', default='0.10')
    parser.add_argument('--execute', action='store_true', help='enable dispatch only after separate approval')
    parser.add_argument('--scope-approved', action='store_true', help='operator asserts both providers and budgets were approved')
    parser.add_argument('--accept-estimate-uncertainty', action='store_true')
    parser.add_argument('--isolation-reviewed', action='store_true')
    parser.add_argument('--journal', type=Path)
    parser.add_argument('--openai-ledger', type=Path)
    parser.add_argument('--jev-ledger', type=Path)
    args = parser.parse_args(argv)
    try:
        report = v3_preview(arm=args.arm, openai_budget_usd=args.openai_budget_usd, jev_budget_usd=args.jev_budget_usd)
    except ValueError:
        parser.error('budgets must be positive dollar amounts with at most nine decimal places')
    if not args.execute:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    parser.error('direct OpenAI API evaluation is disabled; use a Codex subscription pilot')


if __name__ == '__main__':
    raise SystemExit(main())
