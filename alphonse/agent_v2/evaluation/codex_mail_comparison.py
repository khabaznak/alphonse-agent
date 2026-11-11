"""Compare real processors on an invented unread-count source.

This is a test-level, read-only fixture: it never connects to Gmail, a mailbox,
or a household account. Model/provider and actor authorization remain separate
from fixture access. Caller must authorize any live provider and Jev use.
"""
from __future__ import annotations

import json
from time import monotonic
from typing import Any, Callable

from .codex_clock_comparison import _run_simple, _run_v3, _summarize_events
from .codex_subscription_provider import CodexSubscriptionProvider
from .loopback_mail_pilot import LoopbackMailPilot, evaluate


GOAL = ('Hola Alphonse, ¿puedes ver mi Gmail? Por ejemplo, ¿cuántos correos '
        'nuevos tengo sin revisar? Usa el buzón de prueba autorizado y cuenta '
        'los mensajes no leídos de INBOX. Responde solo con un objeto JSON '
        'con la clave unread_count y un número entero.')
DESCRIPTION = 'Read the authorized synthetic inbox and its unread message count.'


def run_mail_arm(*, arm: str, provider: Any, model: str, jev: Any = None,
                 unread_flags: tuple[bool, ...] = (True, False, True, True),
                 expected_unread_count: int = 3, timeout_seconds: float = 100,
                 private_trace: Callable[[dict], None] | None = None) -> dict:
    """Run one arm on fresh synthetic data with a separately supplied key.

    A model-backed call occurs only if the explicitly supplied provider is
    enabled. The evaluator cannot establish process attribution, natural-language
    response quality, real delivery or production authorization from this test.
    """
    if arm not in {'v3', 'simple'}:
        raise ValueError('unsupported_mail_arm')
    if type(provider) is not CodexSubscriptionProvider or provider.model != model:
        raise ValueError('matching_codex_subscription_provider_required')
    if type(expected_unread_count) is not int or expected_unread_count < 0:
        raise ValueError('invalid_synthetic_answer_key')
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds < float('inf'):
        raise ValueError('invalid_timeout')
    if private_trace is not None and not callable(private_trace):
        raise ValueError('invalid_trace_callback')
    started = monotonic()
    deadline = started + timeout_seconds
    events: list[dict] = []

    def cancelled():
        return monotonic() >= deadline

    def safe_trace(event):
        if private_trace is not None:
            try:
                private_trace(event)
            except Exception:
                pass

    with LoopbackMailPilot(unread_flags) as fixture:
        try:
            common = dict(url=fixture.url, provider=provider, model=model, emit=events.append,
                          cancelled=cancelled, trace=safe_trace if private_trace else None,
                          goal=GOAL, source_description=DESCRIPTION,
                          allowed_source_names=('index',), required_source_names=('index',))
            run = _run_v3(jev=jev, **common) if arm == 'v3' else _run_simple(**common)
            error = None
        except Exception as exc:
            run, error = None, type(exc).__name__
    try:
        answer = json.loads(run.response) if run else None
    except (TypeError, ValueError):
        answer = None
    oracle = evaluate(fixture, response=answer, expected_unread_count=expected_unread_count)
    return {'arm': arm, 'model': model, 'status': run.status if run else 'error',
            'passed': bool(run and run.status == 'completed' and oracle['passed']),
            'error_type': error, 'elapsed_seconds': monotonic() - started,
            'oracle': oracle, 'events': events, 'metrics': _summarize_events(events),
            'delivery': 'unassessed', 'qualification': 'unassessed',
            'limitations': ['Synthetic count source, not Gmail or the desktop conversation path.',
                            'Subscription login checks do not prove billing mode during dispatch.',
                            'JSON response is a test constraint, not natural-language grading.',
                            'Jev is separate and required by the V3 arm.',
                            'Timeout is cooperative; in-flight work may overrun.']}
