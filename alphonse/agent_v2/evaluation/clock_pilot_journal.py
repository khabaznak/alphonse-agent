"""Run one frozen synthetic clock attempt in the offline pilot journal.

This is a plumbing runner: even a passing fixture is not agent qualification,
real delivery or a live benchmark. No provider or credential is created here.
The injected transport must declare itself offline; that declaration does not
enforce network isolation, and actual provider dispatch is not authorized.
"""
from __future__ import annotations

import json
from pathlib import Path

from .candidate_clock_responses import ClockResponsesChooser, run_clock_candidate
from .loopback_clock_pilot import LoopbackClockPilot, evaluate
from .synthetic_suite_journal import run_synthetic_suite


def run_offline_clock_journal(output: str | Path, *, source_instant: str,
                              source_timezone: str, expected_instant: str,
                              expected_timezone: str, transport=None,
                              enable_offline_test_dispatch: bool = False) -> dict:
    """Record one synthetic attempt without enabling a provider by default.

    The journal freezes a digest of the supplied source and separate oracle
    inputs before dispatch and fsyncs its start/finish records. It does not
    independently anchor them or retain complete inputs; the caller must retain
    them separately. An interrupted attempt stays unknown and is never resumed.
    The fixture cannot prove agent identity or exactly when an answer formed.
    A local switch or ``offline`` assertion is not authorization, isolation,
    a metered gateway, or evidence that a callback makes no network requests.
    """
    if type(enable_offline_test_dispatch) is not bool:
        raise ValueError('invalid_offline_dispatch_flag')
    if not all(isinstance(value, str) and value for value in
               (source_instant, source_timezone, expected_instant, expected_timezone)):
        raise ValueError('invalid_synthetic_clock_inputs')
    if enable_offline_test_dispatch and getattr(transport, 'offline', None) is not True:
        raise ValueError('offline_transport_required')
    attempt = {'case_id': 'synthetic-clock-1', 'arm': 'single_loop', 'input': {
        'source_instant': source_instant, 'source_timezone': source_timezone,
        'expected_instant': expected_instant, 'expected_timezone': expected_timezone}}

    def execute(row):
        if not enable_offline_test_dispatch:
            return {'availability': 'unavailable', 'reason': 'offline_test_dispatch_disabled'}
        config, events = row['input'], []
        chooser = ClockResponsesChooser(transport, max_input_tokens=4096,
                                        emit=events.append, enabled=True)
        with LoopbackClockPilot(config['source_instant'], config['source_timezone']) as clock:
            result = run_clock_candidate(base_url=clock.url, chooser=chooser, emit=events.append)
        try:
            response = json.loads(result.response)
        except (TypeError, ValueError):
            response = None
        oracle = evaluate(clock, response=response, expected_instant_utc=config['expected_instant'],
                          expected_timezone=config['expected_timezone'])
        return {'availability': 'available', 'processor_status': result.status,
                'passed': result.status == 'completed' and oracle['passed'],
                'portal_oracle': oracle, 'events': events, 'fixture_audit': clock.audit(),
                'delivery': 'unassessed', 'decision_qualified': False,
                'limitations': ['Source attribution and answer formation time are unassessed.',
                                'Synthetic output and effect checks do not qualify production.']}

    execute.offline = True
    return run_synthetic_suite(output, suite_id='offline-synthetic-clock-v1',
                               attempts=[attempt], execute=execute)
