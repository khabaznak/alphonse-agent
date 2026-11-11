"""Preview, or explicitly run, one synthetic candidate clock experiment.

Preview is the default and does not prompt for credentials or use the network.
Execution requires a separately approved scope and an operator-provided key.
The local enable flags do not themselves establish authorization or isolation.
This is a development pilot, not a matched comparison or production rollout.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import getpass
import json
from pathlib import Path

from .clock_responses_attempt import run_clock_responses_attempt
from .responses_pilot_transport import PilotLedger, PilotLimits, ResponsesPilotTransport, requests_sender

MODEL = 'gpt-6-luna'
INSTANT = '2026-04-12T14:00:00+00:00'
TIMEZONE = 'America/Mexico_City'
MAX_CALLS = 10
INPUT_PER_CALL = 65536
OUTPUT_PER_CALL = 512
# Nanodollars per token: $0.138/M input rounds up the published $0.125/M
# cache-write rate plus 10%; $0.55/M output includes 10% over $0.50/M.
# These are assumptions, not verified account charges or a guaranteed cap.
INPUT_RATE = 138
OUTPUT_RATE = 550


def _budget_nanos(value: str) -> int:
    try:
        amount = Decimal(value)
        nanos = amount * 1_000_000_000
        if not amount.is_finite() or amount <= 0 or nanos != nanos.to_integral_value():
            raise ValueError
        return int(nanos)
    except (InvalidOperation, TypeError, ValueError, OverflowError):
        raise ValueError('invalid_budget') from None


def preview(*, budget_usd: str = '0.25') -> dict:
    """Describe the proposed one-attempt scope without using a provider."""
    budget = _budget_nanos(budget_usd)
    expected = MAX_CALLS * (INPUT_PER_CALL * INPUT_RATE + OUTPUT_PER_CALL * OUTPUT_RATE)
    return {
        'kind': 'one_synthetic_candidate_clock_attempt', 'preview_only': True,
        'provider': 'OpenAI Responses API', 'model': MODEL, 'attempts': 1,
        'synthetic_instant_utc': INSTANT, 'synthetic_timezone': TIMEZONE,
        'data_scope': 'synthetic clock request and generated synthetic evidence only',
        'tool_scope': 'read-only local synthetic clock source',
        'limits': {'model_calls': MAX_CALLS, 'input_reservation_per_call': INPUT_PER_CALL,
                   'output_tokens_per_call': OUTPUT_PER_CALL, 'task_seconds_cooperative': 100},
        'reservation_estimate_usd': str(Decimal(expected) / Decimal(1_000_000_000)),
        'proposed_budget_usd': str(Decimal(budget) / Decimal(1_000_000_000)),
        'estimated_reservations_fit_budget': expected <= budget,
        'input_rate_assumption_usd_per_million': '0.138',
        'output_rate_assumption_usd_per_million': '0.55',
        'pricing_reference': 'https://developers.openai.com/api/docs/models/gpt-6-luna',
        'pricing_and_model_access': 'unverified_for_account',
        'recording': 'local journal stores hashes, sanitized events and outcome; no raw answer or API key',
        'local_retention': 'journal and ledger remain until the operator deletes them',
        'provider_retention': 'subject_to_actual_OpenAI_API_account_policy',
        'limitations': [
            'The budget is a reservation estimate, not a guaranteed billing cap.',
            'Input counter is UTF-8 request bytes plus 256, not an independently verified token count.',
            'Unverified pricing, provider framing or other fees can change charges.',
            'Task deadline does not interrupt an in-flight HTTP request.',
            'The program does not establish process isolation, actual model entitlement or user approval.',
            'This single candidate run cannot compare V3 or an external agent.',
        ],
    }


def _estimated_input(body: dict) -> int:
    # This conservative heuristic is not a certified upper bound on billed
    # tokens, particularly for provider framing and model-specific behavior.
    return len(json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')) + 256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget-usd', default='0.25', help='proposed total estimated reservation budget')
    parser.add_argument('--execute', action='store_true', help='enable provider dispatch only after separate approval')
    parser.add_argument('--scope-approved', action='store_true', help='operator asserts the specific scope was approved')
    parser.add_argument('--accept-estimate-uncertainty', action='store_true',
                        help='operator accepts estimates are not a guaranteed billing cap')
    parser.add_argument('--isolation-reviewed', action='store_true', help='operator asserts isolation was independently reviewed')
    parser.add_argument('--journal', type=Path, help='new local synthetic journal path (required to execute)')
    parser.add_argument('--ledger', type=Path, help='new local reservation ledger path (required to execute)')
    args = parser.parse_args(argv)
    try:
        scope = preview(budget_usd=args.budget_usd)
    except ValueError:
        parser.error('budget must be a positive dollar amount with at most nine decimal places')
    if not args.execute:
        print(json.dumps(scope, indent=2, ensure_ascii=False))
        return 0
    if not (args.scope_approved and args.accept_estimate_uncertainty and args.isolation_reviewed
            and args.journal and args.ledger and scope['estimated_reservations_fit_budget']):
        parser.error('execution requires approved scope, acknowledged estimated budget, reviewed isolation, fresh journal and ledger')
    if args.journal.resolve() == args.ledger.resolve() or args.journal.exists() or args.ledger.exists():
        parser.error('journal and ledger must be distinct new paths')
    # Only request a credential after the operator has opted in to every
    # prerequisite. Never print it, pass it in argv, or store it in a journal.
    key = getpass.getpass('OpenAI API key: ').strip()
    if not key:
        parser.error('an API key is required')
    try:
        # Exclusive creation avoids silently sharing/resetting a prior budget.
        with args.ledger.open('x'):
            pass
        limits = PilotLimits(max_calls=MAX_CALLS, max_input_tokens=MAX_CALLS * INPUT_PER_CALL,
            max_output_tokens=MAX_CALLS * OUTPUT_PER_CALL, max_cost_units=_budget_nanos(args.budget_usd),
            input_cost_units_per_token=INPUT_RATE, output_cost_units_per_token=OUTPUT_RATE)
        ledger = PilotLedger(str(args.ledger), limits)
        transport = ResponsesPilotTransport(model=MODEL, api_key=key, ledger=ledger,
            count_complete_input=_estimated_input, sender=requests_sender, enabled=True)
        result = run_clock_responses_attempt(args.journal, source_instant=INSTANT, source_timezone=TIMEZONE,
            expected_instant=INSTANT, expected_timezone=TIMEZONE, transport=transport, dispatch_enabled=True)
    except Exception as exc:
        # Never print provider exceptions, credentials, request bodies or URLs.
        print(json.dumps({'status': 'error', 'error_type': type(exc).__name__}))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['passed'] is True else 1


if __name__ == '__main__':
    raise SystemExit(main())
