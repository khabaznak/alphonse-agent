"""One explicitly enabled V3/Responses synthetic clock attempt.

This is not a process launcher, authorization boundary or production runtime.
The caller supplies an independently scoped V3 provider and metered Jev, and
must execute projection-on and projection-off variants in separate processes.
Each attempt must own a fresh provider, Jev object and both ledgers; never share
them concurrently. The local freshness check is not a cross-process lease.
Model and Jev spend limits, credential handling, isolation and provider
retention remain external prerequisites. No real household data is needed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

from .clock_responses_attempt import _reason_category
from .loopback_clock_pilot import LoopbackClockPilot, evaluate
from .offline_run_journal import _append, _fsync_parent, _timestamp
from .v3_live_clock import V3PilotResponsesProvider, run_v3_live_clock


def run_v3_clock_responses_attempt(output: str | Path, *, source_instant: str,
                                   source_timezone: str, expected_instant: str,
                                   expected_timezone: str, provider: V3PilotResponsesProvider,
                                   system_one: Any, projection_enabled: bool,
                                   dispatch_enabled: bool = False) -> dict:
    """Run one V3 clock attempt, retaining sanitized events and Jev counts.

    The switch does not establish approval or isolation. Only the same proposed
    OpenAI model as the candidate preview is accepted. The separate Jev ledger
    must be new and is checked by ``run_v3_live_clock``. This function preserves
    no raw model output or credentials. An interrupted attempt stays unknown;
    an in-flight provider request cannot be interrupted by the task deadline.
    """
    if type(dispatch_enabled) is not bool or not dispatch_enabled:
        raise ValueError('pilot_v3_dispatch_disabled')
    if (not isinstance(provider, V3PilotResponsesProvider) or
            getattr(provider.transport, '_model', None) != 'gpt-6-luna' or
            type(projection_enabled) is not bool):
        raise ValueError('pilot_v3_clock_configuration_invalid')
    values = (source_instant, source_timezone, expected_instant, expected_timezone)
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError('pilot_v3_clock_inputs_required')
    config = {'source_instant': source_instant, 'source_timezone': source_timezone,
              'expected_instant': expected_instant, 'expected_timezone': expected_timezone}
    raw = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    run_id, output_path, events = str(uuid4()), Path(output), []
    started = monotonic()
    with output_path.open('x', encoding='utf-8') as stream:
        _append(stream, {'event': 'run_started', 'run_id': run_id,
                         'recorded_at': _timestamp(), 'synthetic_config_sha256': hashlib.sha256(raw).hexdigest(),
                         'scope': 'single_v3_clock', 'projection_enabled': projection_enabled,
                         'model': 'gpt-6-luna', 'model_identity': 'configured_not_independently_verified',
                         'limitations': ['Input digest is not an independent anchor.',
                             'Isolation, provider authorization and billing are not established.',
                             'Fixture audit does not prove reader identity or delivery.',
                             'Jev attempt cap is not a hard cost or output-token cap.',
                             'In-flight calls are not forcibly interrupted.']})
        _fsync_parent(output_path)
        _append(stream, {'event': 'attempt_started', 'run_id': run_id, 'recorded_at': _timestamp(),
                         'arm': 'targeted_v3' if projection_enabled else 'v3', 'case': 'clock'})
        previous_emit = provider.emit
        provider.emit = events.append
        try:
            with LoopbackClockPilot(source_instant, source_timezone) as clock:
                result = run_v3_live_clock(base_url=clock.url, provider=provider, system_one=system_one,
                    projection_enabled=projection_enabled, emit=events.append, dispatch_enabled=True)
            try:
                parsed = json.loads(result.response)
            except (TypeError, ValueError):
                parsed = None
            oracle = evaluate(clock, response=parsed, expected_instant_utc=expected_instant,
                              expected_timezone=expected_timezone)
            summary = {'status': result.status, 'reason_category': _reason_category(result.reason),
                       'passed': bool(result.status == 'completed' and oracle['passed']),
                       'answer_sha256': hashlib.sha256(result.response.encode()).hexdigest(),
                       'oracle': oracle, 'events': events, 'delivery': 'unassessed',
                       'decision_qualified': False}
        except Exception as exc:
            summary = {'status': 'error', 'error_type': type(exc).__name__, 'passed': None,
                       'events': events, 'delivery': 'unassessed', 'decision_qualified': False}
        finally:
            provider.emit = previous_emit
        # Only the fixed-layout Jev meter's sanitized snapshot is retained. If
        # the caller did not supply it or reading it fails, usage is unknown.
        try:
            from .jev_attempt_meter import JevAttemptLedger, MeteredJevTransport
            meter = system_one.client.transport
            summary['jev_attempts'] = (meter.ledger.snapshot() if type(meter) is MeteredJevTransport
                                       and type(meter.ledger) is JevAttemptLedger else None)
        except Exception:
            summary['jev_attempts'] = None
        _append(stream, {'event': 'attempt_finished', 'run_id': run_id,
                         'recorded_at': _timestamp(), 'elapsed_seconds': monotonic() - started,
                         'outcome': summary})
        _append(stream, {'event': 'run_finished', 'run_id': run_id, 'recorded_at': _timestamp()})
    return summary
