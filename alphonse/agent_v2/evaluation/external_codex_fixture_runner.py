"""Offline-only fixture wiring for the planned external Codex reference.

No subprocess or provider is launched by this module. A test executor must
explicitly declare itself offline, but that assertion is not a network sandbox
or an authorization boundary. A future live runner still needs a separately
approved scope, independently verified isolation, budget and attribution.
"""
from __future__ import annotations

from time import monotonic
from typing import Any

from .external_codex_pilot import inspect_trace, reference_response
from .loopback_clock_pilot import LoopbackClockPilot, evaluate as evaluate_clock
from .loopback_prescription_pilot import LoopbackPrescriptionPilot, PrescriptionSources, evaluate as evaluate_prescription
from .loopback_traffic_pilot import LoopbackTrafficPortal, PortalConfig, evaluate as evaluate_traffic


def _invoke(executor: Any, *, case: str, url: str, request: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Call one explicitly offline executor without leaking raw trace/URL."""
    failed = False
    result = None
    try:
        # Only the active fixture URL and public task are given to the executor;
        # expected answers, other cases and evaluator audit remain outside it.
        result = executor(case=case, fixture_url=url, request=request)
    except Exception:
        failed = True
    if failed or not isinstance(result, dict) or set(result) != {'jsonl', 'final_text', 'exit_code'}:
        return None, {'status': 'executor_error', 'reported_usage': {'input_tokens': None, 'output_tokens': None},
                      'usage_source': 'unknown', 'decision_qualified': False}
    try:
        summary = inspect_trace(result['jsonl'], final_text=result['final_text'], exit_code=result['exit_code'])
    except (TypeError, ValueError):
        return None, {'status': 'invalid_trace', 'reported_usage': {'input_tokens': None, 'output_tokens': None},
                      'usage_source': 'unknown', 'decision_qualified': False}
    parsed = (reference_response(case, result['final_text'])
              if summary['status'] == 'completed' and isinstance(result['final_text'], str) else None)
    return parsed, summary


def run_fixture(*, case: str, request: str, fixture: dict[str, Any],
                expected: dict[str, Any], executor: Any) -> dict[str, Any]:
    """Run one fresh synthetic fixture with a test executor and separate key.

    The executor receives neither expected values nor other cases. A fresh
    loopback service exists only for this attempt; it is finalized before the
    independent evaluator is invoked. Passing remains *fixture-only*: bearer
    possession is not process attribution, and response ordering is unassessed.
    No attempt is automatically retried. The caller owns safe receipt storage.
    """
    if case not in {'clock', 'prescription', 'traffic'}:
        raise ValueError('unsupported_reference_case')
    if not isinstance(request, str) or not request.strip():
        raise ValueError('synthetic_request_required')
    if getattr(executor, 'offline', None) is not True or not callable(executor):
        raise ValueError('offline_executor_required')
    if not isinstance(fixture, dict) or not isinstance(expected, dict):
        raise ValueError('invalid_synthetic_case')
    started = monotonic()
    if case == 'clock':
        if set(fixture) != {'instant_utc', 'timezone'} or set(expected) != {'instant_utc', 'timezone'}:
            raise ValueError('invalid_clock_case')
        service = LoopbackClockPilot(fixture['instant_utc'], fixture['timezone'])
        with service:
            answer, trace = _invoke(executor, case=case, url=service.url, request=request)
        grade = evaluate_clock(service, response=answer, expected_instant_utc=expected['instant_utc'],
                               expected_timezone=expected['timezone'])
    elif case == 'prescription':
        if set(fixture) != {'sources'} or set(expected) != {'response', 'sources'}:
            raise ValueError('invalid_prescription_case')
        if not isinstance(fixture['sources'], PrescriptionSources):
            raise ValueError('invalid_prescription_case')
        service = LoopbackPrescriptionPilot(fixture['sources'])
        with service:
            answer, trace = _invoke(executor, case=case, url=service.url, request=request)
        grade = evaluate_prescription(service, response=answer, expected_response=expected['response'],
                                      expected_sources=expected['sources'])
    else:
        if set(fixture) != {'config'} or set(expected) != {'subject', 'ticket_ids'}:
            raise ValueError('invalid_traffic_case')
        if not isinstance(fixture['config'], PortalConfig) or not fixture['config'].require_capability:
            raise ValueError('bearer_traffic_fixture_required')
        service = LoopbackTrafficPortal(fixture['config'])
        with service:
            answer, trace = _invoke(executor, case=case, url=service.url, request=request)
        grade = evaluate_traffic(service, response=answer, expected_subject=expected['subject'],
                                 expected_ticket_ids=expected['ticket_ids'])
    return {'kind': 'offline_external_reference_fixture', 'case': case,
            'fixture_passed': grade.get('passed') is True, 'fixture_grade': grade,
            'trace': trace, 'elapsed_ms': round((monotonic() - started) * 1000, 3),
            'decision_qualified': False,
            'limitations': ['Offline executor declaration is not a sandbox or proof of no network.',
                            'No model call, effect provenance, delivery or spending control is established.',
                            'Source observations do not identify the reading process or when an answer was formed.']}
