"""One explicitly enabled, synthetic candidate clock attempt.

This runner can consume the real Responses transport; it does not construct a
provider, select a model, read credentials, or authorize a live call. Its switch
is not proof of authorization, process isolation, pricing, token-count accuracy,
or a non-bypassable spending limit. Supply a fresh synthetic source and separate
oracle, a configured transport and an exclusive output path. Do not send real
household data. A running provider request cannot be interrupted by the task
deadline, and no crash-recovery or production delivery guarantee is implied.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

from .candidate_clock_responses import ClockResponsesChooser, run_clock_candidate
from .loopback_clock_pilot import LoopbackClockPilot, evaluate
from .offline_run_journal import _append, _fsync_parent, _timestamp


def _reason_category(reason: Any) -> str | None:
    """Keep a useful fixed diagnostic without persisting arbitrary error text."""
    if not reason:
        return None
    categories = {
        'cancelled': 'cancelled',
        'candidate_deadline_exceeded': 'deadline',
        'pilot_v3_deadline_or_cancelled': 'deadline_or_cancelled',
        'candidate_model_unavailable': 'model_unavailable',
        'candidate_model_usage_unknown_or_overrun': 'model_usage_unknown_or_overrun',
        'candidate_model_settlement_unavailable': 'model_settlement_unavailable',
        'candidate_evidence_changed_or_revoked': 'evidence_unavailable',
        'candidate_answer_unsupported': 'evidence_unavailable',
        'candidate_read_denied': 'access_denied',
        'candidate_read_revoked': 'access_denied',
        'candidate_input_reservation_too_small': 'input_limit',
        'candidate_chooser_input_cap_exceeds_reservation': 'input_limit',
        'candidate_tool_failed_or_waiting': 'tool_unavailable',
        'question_delivery_not_integrated': 'delivery_unavailable',
    }
    return categories.get(reason, 'other') if isinstance(reason, str) else 'other'


def run_clock_responses_attempt(output: str | Path, *, source_instant: str,
                                source_timezone: str, expected_instant: str,
                                expected_timezone: str, transport: Any,
                                dispatch_enabled: bool = False) -> dict:
    """Run one candidate attempt or refuse before creating a journal.

    ``transport`` is deliberately injected and is not assumed to be offline.
    Enabling the switch is not itself approval or proof of external safeguards.
    The journal holds only synthetic fixture metadata, event measurements and
    the answer hash; source content, provider text and credentials are omitted.
    The outcome is independently graded only after the portal is closed.
    """
    if type(dispatch_enabled) is not bool or not dispatch_enabled:
        raise ValueError('clock_attempt_dispatch_disabled')
    if not callable(getattr(transport, 'generate_json', None)):
        raise ValueError('clock_attempt_transport_required')
    inputs = (source_instant, source_timezone, expected_instant, expected_timezone)
    if not all(isinstance(value, str) and value for value in inputs):
        raise ValueError('clock_attempt_synthetic_inputs_required')

    config = {'source_instant': source_instant, 'source_timezone': source_timezone,
              'expected_instant': expected_instant, 'expected_timezone': expected_timezone}
    raw = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    run_id = str(uuid4())
    # Only the fixed proposed model can be named in a public journal. Do not
    # serialize arbitrary caller-supplied model strings, which could be secret.
    configured_model = 'gpt-6-luna' if getattr(transport, '_model', None) == 'gpt-6-luna' else None
    output_path = Path(output)
    events: list[dict] = []
    started = monotonic()
    with output_path.open('x', encoding='utf-8') as stream:
        _append(stream, {'event': 'run_started', 'run_id': run_id,
                         'recorded_at': _timestamp(), 'synthetic_config_sha256': hashlib.sha256(raw).hexdigest(),
                         'scope': 'single_candidate_clock', 'configured_model': configured_model,
                         'model_identity': 'unverified_by_runner',
                         'per_call_input_reservation': 65536,
                         'per_call_output_reservation': 512,
                         'limitations': ['Input digest is not an independent anchor.',
                             'Isolation, provider authorization and billing are not established.',
                             'Fixture audit does not prove reader identity or delivery.',
                             'In-flight calls are not forcibly interrupted.']})
        _fsync_parent(output_path)
        _append(stream, {'event': 'attempt_started', 'run_id': run_id,
                         'recorded_at': _timestamp(), 'arm': 'single_loop', 'case': 'clock'})
        try:
            chooser = ClockResponsesChooser(transport, max_input_tokens=65536,
                                            emit=events.append, enabled=True)
            with LoopbackClockPilot(source_instant, source_timezone) as clock:
                result = run_clock_candidate(base_url=clock.url, chooser=chooser, emit=events.append)
            try:
                parsed = json.loads(result.response)
            except (TypeError, ValueError):
                parsed = None
            oracle = evaluate(clock, response=parsed, expected_instant_utc=expected_instant,
                              expected_timezone=expected_timezone)
            # Runtime reasons can include arbitrary exception text; persist
            # only a closed-set category instead of copying that text.
            summary = {'status': result.status, 'reason_category': _reason_category(result.reason),
                       'passed': bool(result.status == 'completed' and oracle['passed']),
                       'answer_sha256': hashlib.sha256(result.response.encode()).hexdigest(),
                       'oracle': oracle, 'events': events, 'delivery': 'unassessed',
                       'decision_qualified': False}
        except Exception as exc:
            # Preserve emitted telemetry but never serialize exception text,
            # transport objects, authorization headers or arbitrary output.
            summary = {'status': 'error', 'error_type': type(exc).__name__, 'passed': None,
                       'events': events, 'delivery': 'unassessed', 'decision_qualified': False}
        _append(stream, {'event': 'attempt_finished', 'run_id': run_id,
                         'recorded_at': _timestamp(), 'elapsed_seconds': monotonic() - started,
                         'outcome': summary})
        _append(stream, {'event': 'run_finished', 'run_id': run_id, 'recorded_at': _timestamp()})
    return summary
