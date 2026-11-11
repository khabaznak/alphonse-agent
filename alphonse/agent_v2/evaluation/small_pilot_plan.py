"""Plan the five-case investment pilot without dispatching a model.

This public development agenda is not a frozen experiment: task details,
oracles, executable/model settings, isolation, price and authorization are
still missing. Scripted fixture availability is never model-arm availability.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

from .decision_manifest import bytes_digest, canonical_digest

CASES = ('clock', 'gmail', 'prescription', 'traffic', 'recovery')
ALPHONSE_ARMS = ('v3', 'targeted_v3', 'single_loop')
EXTERNAL_CASES = frozenset({'clock', 'prescription', 'traffic'})


def create_agenda(protocol: bytes, *, seed: str) -> dict:
    """Generate a reproducible paired agenda, not authorization or a run."""
    if not isinstance(protocol, bytes) or not protocol:
        raise ValueError('protocol_bytes_required')
    if not isinstance(seed, str) or not seed.strip():
        raise ValueError('schedule_seed_required')
    schedule = []
    for case in CASES:
        arms = list(ALPHONSE_ARMS) + (['external'] if case in EXTERNAL_CASES else [])
        for repetition in (1, 2):
            order = sorted(arms, key=lambda arm: canonical_digest([seed, case, repetition, arm]))
            schedule.extend({'case': case, 'repetition': repetition, 'arm': arm,
                             'position': position} for position, arm in enumerate(order))
    plan = {'schema_version': 1, 'kind': 'development_agenda_not_frozen_experiment',
            'protocol_sha256': bytes_digest(protocol), 'seed': seed, 'schedule': schedule,
            'arms': {arm: {'availability': 'unavailable', 'reason': reason} for arm, reason in {
                'v3': 'model_provider_jev_budget_and_isolation_not_ready',
                'targeted_v3': 'model_provider_jev_budget_and_isolation_not_ready',
                'single_loop': 'live_bridge_metering_and_isolation_not_ready',
                'external': 'external_agent_isolation_and_attribution_not_ready'}.items()},
            'fixture_status': {
                'clock': 'fixed_clock_http_source_and_offline_injected_processors_only',
                'gmail': 'synthetic_mailbox_model_adapter_not_ready',
                'prescription': 'http_source_and_offline_injected_processors_only',
                'traffic': 'loopback_portal_and_scripted_processors_only',
                'recovery': 'scripted_followthrough_processors_only'},
            'blockers': ['unfrozen_case_data_and_independent_oracles',
                         'unmatched_model_backed_arms', 'isolation_and_provider_attribution_unverified',
                         'unknown_provider_pricing_and_spend_boundary', 'scope_and_budget_not_approved'],
            'limitations': ['Agenda is not independently anchored, executed or held out.',
                            'Scripted fixtures do not establish autonomous performance.',
                            'Unavailable and missing observations are not successes.']}
    return {**plan, 'agenda_sha256': canonical_digest(plan)}


def validate_agenda(agenda: dict, protocol: bytes) -> None:
    """Detect agenda changes and protocol drift; not a readiness decision."""
    if not isinstance(agenda, dict):
        raise ValueError('invalid_agenda')
    try:
        expected = create_agenda(protocol, seed=agenda['seed'])
    except (KeyError, TypeError, ValueError):
        raise ValueError('invalid_agenda') from None
    if agenda != expected:
        raise ValueError('agenda_or_protocol_changed')


def readiness_receipt(agenda: dict, protocol: bytes) -> dict:
    """Account for every scheduled slot without asserting any trial was run."""
    validate_agenda(agenda, protocol)
    slots = []
    for row in agenda['schedule']:
        config = agenda['arms'][row['arm']]
        slots.append({**row, 'attempt_state': 'not_started', 'passed': None,
                      'availability': config['availability'], 'reason': config['reason']})
    return {'kind': 'readiness_only_not_run_receipt', 'agenda_sha256': agenda['agenda_sha256'],
            'model_calls': 0, 'attempted': 0, 'slots': slots,
            'qualification': 'unassessed',
            'limitations': ['No result, latency, delivery or provider usage has been measured.',
                            'This receipt is not an independently anchored experiment.']}


def traffic_fixture_smoke() -> dict:
    """Exercise existing scripts only; never confuse these with pilot results."""
    from .traffic_processor_development import compare_traffic
    result = compare_traffic('ticket', actions=[{'operation': 'inspect'},
        {'operation': 'search', 'field': 'plate', 'subject': 'TEST-701'}],
        answer={'status': 'verified', 'subject': 'TEST-701', 'ticket_ids': ['T-71']})
    return {'kind': 'offline_scripted_traffic_wiring_smoke', 'result': deepcopy(result),
            'pilot_executed': False, 'decision_qualified': False}


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--seed', required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='new public development agenda; never overwritten')
    args = parser.parse_args(argv)
    plan = create_agenda(args.protocol.read_bytes(), seed=args.seed)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'agenda_sha256': plan['agenda_sha256'], 'scheduled': len(plan['schedule']),
                      'model_backed_arms_available': 0, 'pilot_executed': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
