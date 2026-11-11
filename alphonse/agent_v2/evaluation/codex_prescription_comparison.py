"""Synthetic prescription-memory V3/simple comparison using Codex subscription.

This test-level switch exercises the actual processors with fresh independent
synthetic source services. Nothing dispatches at import. It is not a production
engine switch, clinical evaluator, real delivery check, or billing sandbox.

The default ``fixture_diagnostic`` candidate receives a hidden-source coverage
signal on answer attempts and bounded feedback. ``matched_provenance`` instead
keeps the hidden required-source list out of the candidate entirely, disables
that retry feedback, and uses only basic recorded-read provenance. The latter
is an evaluation convenience, not semantic grounding or a production medical
safeguard. Both modes are scored by the independent post-run oracle; V3's
completion mechanism and the candidate's provenance check still differ.
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
_GOAL_V2 = ('It is 2026-04-12 at 07:30 in America/Mexico_City. What medicine and prescribed dose '
            'are scheduled for this morning? Return a JSON object with status and medications. '
            'Status must be "schedule" if the schedule is established or "unknown" if it is not. '
            'Medications is an array of objects with name, dose and time as strings; '
            'use an empty array when the schedule is unknown.')
_VERSIONS = {'v1', 'v2'}


def _sources(version='v1'):
    if version not in _VERSIONS:
        raise ValueError('invalid_fixture_version')
    if version == 'v2':
        return PrescriptionSources(
            conversation={'earlier_message': 'Started the prescribed course on 2026-04-10.'},
            prescription={'name': 'Asterin', 'dose': '10 mg', 'time': '08:00',
                          'frequency': 'daily', 'days': 5, 'timezone': 'America/Mexico_City'},
            treatment_status={'state': 'active', 'as_of': '2026-04-12T07:30:00-06:00'},
        )
    return PrescriptionSources(
        conversation={'earlier_message': 'Started the prescribed course on 2026-04-10.'},
        prescription={'name': 'Asterin', 'dose': '10 mg', 'time': '08:00', 'days': 5},
        treatment_status={'state': 'active'},
    )


def _expected(version='v1'):
    if version not in _VERSIONS:
        raise ValueError('invalid_fixture_version')
    # Independently authored expectations: never passed to a model/processor.
    if version == 'v2':
        return (
            {'status': 'schedule', 'medications': [{'name': 'Asterin', 'dose': '10 mg', 'time': '08:00'}]},
            {'index': {'sources': ['conversation', 'prescription', 'treatment-status']},
             'conversation': {'earlier_message': 'Started the prescribed course on 2026-04-10.'},
             'prescription': {'name': 'Asterin', 'dose': '10 mg', 'time': '08:00', 'frequency': 'daily',
                              'days': 5, 'timezone': 'America/Mexico_City'},
             'treatment-status': {'state': 'active', 'as_of': '2026-04-12T07:30:00-06:00'}},
        )
    return (
        {'status': 'schedule', 'medications': [{'name': 'Asterin', 'dose': '10 mg', 'time': '08:00'}]},
        {'index': {'sources': ['conversation', 'prescription', 'treatment-status']},
         'conversation': {'earlier_message': 'Started the prescribed course on 2026-04-10.'},
         'prescription': {'name': 'Asterin', 'dose': '10 mg', 'time': '08:00', 'days': 5},
         'treatment-status': {'state': 'active'}},
    )


def _run_v3(*, url, provider, jev, model, emit, cancelled, trace=None, goal=_GOAL):
    return _run_shared_v3(url=url, provider=provider, jev=jev, model=model, emit=emit,
        cancelled=cancelled, trace=trace, goal=goal, source_description='Read a named synthetic household record.',
        allowed_source_names=_NAMES, required_source_names=_REQUIRED)


def _run_simple(*, url, provider, model, emit, cancelled, trace=None,
                verification_mode='fixture_diagnostic', goal=_GOAL):
    return _run_shared_simple(url=url, provider=provider, model=model, emit=emit,
        cancelled=cancelled, trace=trace, goal=goal, source_description='Read a named synthetic household record.',
        allowed_source_names=_NAMES,
        required_source_names=_REQUIRED if verification_mode == 'fixture_diagnostic' else None,
        verification_mode=verification_mode)


def _answer_diagnostics(response, expected):
    """Closed-set evaluation after the run; never retain model-supplied text."""
    if not isinstance(response, str) or not response.strip():
        return {'format': 'no_answer'}
    try:
        answer = json.loads(response)
    except (TypeError, ValueError):
        return {'format': 'invalid_json'}
    if not isinstance(answer, dict):
        return {'format': 'not_object'}
    status = answer.get('status')
    meds = answer.get('medications')
    status_category = ('missing' if 'status' not in answer else
                       'invalid_type' if not isinstance(status, str) else
                       'expected' if status == expected['status'] else
                       'unknown' if status == 'unknown' else 'other')
    meds_category = ('missing' if 'medications' not in answer else
                     'invalid_type' if not isinstance(meds, list) else
                     'expected' if meds == expected['medications'] else 'other')
    diagnosis = {'format': 'object', 'status': status_category, 'medications': meds_category,
                 'exact_answer': answer == expected}
    if isinstance(meds, list) and len(meds) == 1 and isinstance(meds[0], dict):
        for key in ('name', 'dose', 'time'):
            value = meds[0].get(key)
            diagnosis[key] = ('missing' if key not in meds[0] else
                              'invalid_type' if not isinstance(value, str) else
                              'expected' if value == expected['medications'][0][key] else 'other')
    return diagnosis


def run_prescription_arm(*, arm: str, provider: Any, model: str, jev: Any = None,
                         timeout_seconds: float = 150,
                         private_trace: Callable[[dict], None] | None = None,
                         verification_mode: str = 'fixture_diagnostic', fixture_version: str = 'v1') -> dict:
    """Execute a single synthetic arm; caller controls authorization and output custody."""
    if arm not in {'v3', 'simple'}:
        raise ValueError('unsupported_prescription_arm')
    if verification_mode not in {'fixture_diagnostic', 'matched_provenance'}:
        raise ValueError('invalid_verification_mode')
    if fixture_version not in _VERSIONS:
        raise ValueError('invalid_fixture_version')
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
    goal = _GOAL if fixture_version == 'v1' else _GOAL_V2
    with LoopbackPrescriptionPilot(_sources(fixture_version)) as fixture:
        try:
            result = (_run_v3(url=fixture.url, provider=provider, jev=jev, model=model, emit=events.append, cancelled=cancelled, trace=safe_trace if private_trace else None, goal=goal)
                      if arm == 'v3' else
                      _run_simple(url=fixture.url, provider=provider, model=model, emit=events.append, cancelled=cancelled, trace=safe_trace if private_trace else None, verification_mode=verification_mode, goal=goal))
            error = None
        except Exception as exc:
            result = None
            error = type(exc).__name__
    try:
        answer = json.loads(result.response) if result else None
    except (ValueError, TypeError):
        answer = None
    expected_answer, expected_sources = _expected(fixture_version)
    oracle = evaluate(fixture, response=answer, expected_response=expected_answer, expected_sources=expected_sources)
    # Index is optional because the host tool already exposes available names.
    # The three substantive observations are independently verified; any
    # observed source mismatch or incomplete read remains a failure.
    oracle['required_sources_for_task'] = list(_REQUIRED)
    oracle['passed'] = bool(oracle['answer_matches_key']
        and set(_REQUIRED) <= set(oracle['observed_required_sources'])
        and not oracle['source_mismatch'] and not oracle['incomplete_http_request'])
    return {'arm': arm, 'model': model, 'verification_mode': verification_mode, 'fixture_version': fixture_version,
            'status': result.status if result else 'error',
            'passed': bool(result and result.status == 'completed' and oracle['passed']),
            'error_type': error, 'reason_category': _reason_category(result.reason) if result else None,
            'elapsed_seconds': monotonic() - started,
            'oracle': oracle, 'answer_diagnostics': _answer_diagnostics(result.response if result else None, expected_answer),
            'events': events, 'metrics': _summarize_events(events),
            'delivery': 'unassessed', 'qualification': 'unassessed',
            'limitations': ['Synthetic fixture, not the full Alphonse desktop conversation path.',
                            'Exact structured result and three substantive source reads required; discovery index optional.',
                            'Codex provider usage may be estimated, not reported; hidden tokens are absent.',
                            'Candidate local visible output bound differs from V3.',
                            'Jev is separate; actual user delivery and hard timeout are unassessed.',
                            'Matched provenance checks actor, nonempty output and recorded reads, not semantic grounding; V3 completion differs.']}


def compare_prescription(*, provider: Any, model: str, jev_factory: Callable[[], Any],
                         repetitions: int = 1, timeout_seconds: float = 150,
                         private_trace: Callable[[dict], None] | None = None,
                         verification_mode: str = 'fixture_diagnostic', fixture_version: str = 'v1') -> dict:
    if type(provider) is not CodexSubscriptionProvider or provider.model != model:
        raise ValueError('matching_codex_subscription_provider_required')
    if not callable(jev_factory) or type(repetitions) is not int or not 1 <= repetitions <= 10:
        raise ValueError('invalid_comparison_configuration')
    if verification_mode not in {'fixture_diagnostic', 'matched_provenance'}:
        raise ValueError('invalid_verification_mode')
    if fixture_version not in _VERSIONS:
        raise ValueError('invalid_fixture_version')
    rows=[]
    for repetition in range(1, repetitions + 1):
        order = ('v3', 'simple') if repetition % 2 else ('simple', 'v3')
        for arm in order:
            try:
                jev = jev_factory() if arm == 'v3' else None
                result = run_prescription_arm(arm=arm, provider=provider, model=model, jev=jev,
                                              timeout_seconds=timeout_seconds, private_trace=private_trace,
                                              verification_mode=verification_mode, fixture_version=fixture_version)
            except Exception as exc:
                result = {'arm': arm, 'model': model, 'status': 'error', 'passed': None,
                          'error_type': type(exc).__name__, 'reason_category': None,
                          'elapsed_seconds': None, 'oracle': None,
                          'events': [], 'delivery': 'unassessed', 'qualification': 'unassessed'}
            rows.append({'repetition': repetition, **result})
    return {'kind': 'synthetic_prescription_v3_vs_simple', 'model': model,
            'verification_mode': verification_mode, 'fixture_version': fixture_version, 'runs': rows,
            'qualification': 'unassessed',
            'limitations': ['Fresh synthetic sources; no live medical data or full desktop flow.',
                            'TypeSafe Jev and actual user delivery are separate boundaries.',
                            'This is a structured synthetic task, not medical advice or clinical validation.']}
