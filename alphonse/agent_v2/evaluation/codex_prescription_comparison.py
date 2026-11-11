"""Synthetic prescription-memory V3/simple comparison using Codex subscription.

This test-level switch exercises the actual processors with fresh independent
synthetic source services. Nothing dispatches at import. It is not a production
engine switch, clinical evaluator, real delivery check, or billing sandbox.
"""
from __future__ import annotations

import json
import math
from time import monotonic
from typing import Any, Callable

from .codex_clock_comparison import (_run_v3 as _run_shared_v3,
                                     _run_simple as _run_shared_simple, _summarize_events)
from .codex_subscription_provider import CodexSubscriptionProvider
from .clock_responses_attempt import _reason_category
from .loopback_prescription_pilot import LoopbackPrescriptionPilot, PrescriptionSources, evaluate

_ACTOR = 'synthetic-evaluation-user'
_NAMES = ('index', 'conversation', 'prescription', 'treatment-status')
_REQUIRED = ('conversation', 'prescription', 'treatment-status')
_GOAL = ('It is 2026-04-12 at 07:30 local time. What medicine and prescribed dose '
         'are scheduled for this morning? Return one JSON '
         'object with status and medications; each medication has name, dose and time.')


def _sources():
    return PrescriptionSources(
        conversation={'earlier_message': 'Started the prescribed course on 2026-04-10.'},
        prescription={'name': 'Asterin', 'dose': '10 mg', 'time': '08:00', 'days': 5},
        treatment_status={'state': 'active'},
    )


def _expected():
    # Independently authored expectations: never passed to a model/processor.
    return (
        {'status': 'schedule', 'medications': [{'name': 'Asterin', 'dose': '10 mg', 'time': '08:00'}]},
        {'index': {'sources': ['conversation', 'prescription', 'treatment-status']},
         'conversation': {'earlier_message': 'Started the prescribed course on 2026-04-10.'},
         'prescription': {'name': 'Asterin', 'dose': '10 mg', 'time': '08:00', 'days': 5},
         'treatment-status': {'state': 'active'}},
    )


def _run_v3(*, url, provider, jev, model, emit, cancelled, trace=None):
    return _run_shared_v3(url=url, provider=provider, jev=jev, model=model, emit=emit,
        cancelled=cancelled, trace=trace, goal=_GOAL, source_description='Read a named synthetic household record.',
        allowed_source_names=_NAMES, required_source_names=_REQUIRED)


def _run_simple(*, url, provider, model, emit, cancelled, trace=None):
    return _run_shared_simple(url=url, provider=provider, model=model, emit=emit,
        cancelled=cancelled, trace=trace, goal=_GOAL, source_description='Read a named synthetic household record.',
        allowed_source_names=_NAMES, required_source_names=_REQUIRED)


def run_prescription_arm(*, arm: str, provider: Any, model: str, jev: Any = None,
                         timeout_seconds: float = 150,
                         private_trace: Callable[[dict], None] | None = None) -> dict:
    """Execute a single synthetic arm; caller controls authorization and output custody."""
    if arm not in {'v3', 'simple'}:
        raise ValueError('unsupported_prescription_arm')
    if type(provider) is not CodexSubscriptionProvider or not isinstance(model, str) or not model.strip() or provider.model != model:
        raise ValueError('matching_codex_subscription_provider_required')
    if private_trace is not None and not callable(private_trace):
        raise ValueError('invalid_trace_callback')
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('invalid_timeout')
    started = monotonic()
    deadline = started + timeout_seconds
    events = []
    def cancelled():
        return monotonic() >= deadline
    def safe_trace(event):
        if private_trace is not None:
            try:
                private_trace(event)
            except Exception:
                pass
    with LoopbackPrescriptionPilot(_sources()) as fixture:
        try:
            result = (_run_v3(url=fixture.url, provider=provider, jev=jev, model=model, emit=events.append, cancelled=cancelled, trace=safe_trace if private_trace else None)
                      if arm == 'v3' else
                      _run_simple(url=fixture.url, provider=provider, model=model, emit=events.append, cancelled=cancelled, trace=safe_trace if private_trace else None))
            error = None
        except Exception as exc:
            result = None
            error = type(exc).__name__
    try:
        answer = json.loads(result.response) if result else None
    except (ValueError, TypeError):
        answer = None
    expected_answer, expected_sources = _expected()
    oracle = evaluate(fixture, response=answer, expected_response=expected_answer, expected_sources=expected_sources)
    # Index is optional because the host tool already exposes available names.
    # The three substantive observations are independently verified; any
    # observed source mismatch or incomplete read remains a failure.
    oracle['required_sources_for_task'] = list(_REQUIRED)
    oracle['passed'] = bool(oracle['answer_matches_key']
        and set(_REQUIRED) <= set(oracle['observed_required_sources'])
        and not oracle['source_mismatch'] and not oracle['incomplete_http_request'])
    return {'arm': arm, 'model': model, 'status': result.status if result else 'error',
            'passed': bool(result and result.status == 'completed' and oracle['passed']),
            'error_type': error, 'reason_category': _reason_category(result.reason) if result else None,
            'elapsed_seconds': monotonic() - started,
            'oracle': oracle, 'events': events, 'metrics': _summarize_events(events),
            'delivery': 'unassessed', 'qualification': 'unassessed',
            'limitations': ['Synthetic fixture, not the full Alphonse desktop conversation path.',
                            'Exact structured result and three substantive source reads required; discovery index optional.',
                            'Codex provider usage may be estimated, not reported; hidden tokens are absent.',
                            'Candidate local visible output bound differs from V3.',
                            'Jev is separate; actual user delivery and hard timeout are unassessed.']}


def compare_prescription(*, provider: Any, model: str, jev_factory: Callable[[], Any],
                         repetitions: int = 1, timeout_seconds: float = 150,
                         private_trace: Callable[[dict], None] | None = None) -> dict:
    if type(provider) is not CodexSubscriptionProvider or provider.model != model:
        raise ValueError('matching_codex_subscription_provider_required')
    if not callable(jev_factory) or type(repetitions) is not int or not 1 <= repetitions <= 10:
        raise ValueError('invalid_comparison_configuration')
    rows=[]
    for repetition in range(1, repetitions + 1):
        order = ('v3', 'simple') if repetition % 2 else ('simple', 'v3')
        for arm in order:
            try:
                jev = jev_factory() if arm == 'v3' else None
                result = run_prescription_arm(arm=arm, provider=provider, model=model, jev=jev,
                                              timeout_seconds=timeout_seconds, private_trace=private_trace)
            except Exception as exc:
                result = {'arm': arm, 'model': model, 'status': 'error', 'passed': None,
                          'error_type': type(exc).__name__, 'reason_category': None,
                          'elapsed_seconds': None, 'oracle': None,
                          'events': [], 'delivery': 'unassessed', 'qualification': 'unassessed'}
            rows.append({'repetition': repetition, **result})
    return {'kind': 'synthetic_prescription_v3_vs_simple', 'model': model, 'runs': rows,
            'qualification': 'unassessed',
            'limitations': ['Fresh synthetic sources; no live medical data or full desktop flow.',
                            'TypeSafe Jev and actual user delivery are separate boundaries.',
                            'This is a structured synthetic task, not medical advice or clinical validation.']}
