"""Synthetic follow-through cases; not a delivery service or production sandbox.

The evaluator keeps expected outcomes separate from the synthetic environment.
Only explicitly structured messages are graded. The in-process audit is an
instrumentation test, not an independent receipt from a real communications
provider. No real data, messages, provider calls or household effects are used.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Case:
    name: str
    goal: str = 'Find the synthetic appointment time.'


CASES = tuple(Case(name) for name in (
    'fallback', 'missing', 'denied', 'delivery_failed', 'delivery_unconfirmed', 'cancelled',
))

# Synthetic environment state and expected outcomes are authored separately.
# The evaluator must never accept the environment state as its answer key.
_FIXTURES = {
    'fallback': {'primary': ('unavailable', None), 'backup': ('ok', '09:30'), 'channel': 'delivered'},
    'missing': {'primary': ('missing_identity', None), 'backup': ('missing_identity', None), 'channel': 'delivered'},
    'denied': {'primary': ('denied', None), 'backup': ('denied', None), 'channel': 'delivered'},
    'delivery_failed': {'primary': ('missing_identity', None), 'backup': ('missing_identity', None), 'channel': 'failed'},
    'delivery_unconfirmed': {'primary': ('missing_identity', None), 'backup': ('missing_identity', None), 'channel': 'unconfirmed'},
    'cancelled': {'primary': ('cancelled', None), 'backup': ('cancelled', None), 'channel': 'delivered'},
}
_EXPECTED = {
    'fallback': {'kind': 'result', 'value': '09:30'},
    'missing': {'kind': 'handoff', 'blocker': 'missing_identity', 'question': 'appointment_identity'},
    'denied': {'kind': 'handoff', 'blocker': 'denied', 'question': None},
    'delivery_failed': {'kind': 'handoff', 'blocker': 'missing_identity', 'question': 'appointment_identity'},
    'delivery_unconfirmed': {'kind': 'handoff', 'blocker': 'missing_identity', 'question': 'appointment_identity'},
    'cancelled': {'kind': 'cancelled'},
}


class FollowThroughFixture:
    """Fresh synthetic state and audit for one attempt.

    A caller may read either authorized source and send only to the requesting
    actor's channel. Channel outcomes are simulated. A rejected operation remains
    visible as an attempt; it does not imply an external effect happened.
    """
    def __init__(self, case: Case):
        if case.name not in _FIXTURES:
            raise ValueError('unknown_case')
        self.case = case
        self._state = deepcopy(_FIXTURES[case.name])
        self._events: list[dict[str, Any]] = []
        self._clock = 0

    def advance(self, seconds: int) -> None:
        if type(seconds) is not int or seconds < 0:
            raise ValueError('invalid_time_advance')
        self._clock += seconds

    def read(self, source: str) -> dict[str, Any]:
        if source not in ('primary', 'backup'):
            self._events.append({'kind': 'read', 'source': source, 'status': 'denied', 'time': self._clock, 'authorized': False})
            return {'status': 'denied'}
        status, value = self._state[source]
        event = {'kind': 'read', 'source': source, 'status': status, 'time': self._clock, 'authorized': True}
        if value is not None:
            event['value'] = value
        self._events.append(event)
        return {'status': status, **({'value': value} if value is not None else {})}

    def record_denied_read(self, source: str) -> None:
        """Record an authorization rejection without accessing a source."""
        self._events.append({'kind': 'read', 'source': source, 'status': 'denied',
                             'time': self._clock, 'authorized': False})

    def send(self, channel: str, message: dict[str, Any]) -> str:
        status = self._state['channel'] if channel == 'actor_private' else 'denied'
        if self.case.name == 'cancelled':
            status = 'cancelled'
        self._events.append({'kind': 'send', 'channel': channel, 'message': deepcopy(message),
                             'status': status, 'time': self._clock})
        return status

    def audit(self) -> list[dict[str, Any]]:
        return deepcopy(self._events)


def evaluate(case: Case, fixture: FollowThroughFixture, *, processor_status: str | None = None,
             processor_reason: str | None = None) -> dict[str, Any]:
    """Return observable, independent development checks, not clinical judgment.

    For a result: exact {kind: result, value: ...}; for a handoff: exact
    {kind: handoff, blocker: ..., known: [], unknown: ['appointment_time'],
    question: ...}. A handoff is valid only for a fixture-defined necessary
    blocker, observed before sending and within 30 seconds. Completion and
    handoff are subject to a 60 second task deadline and independent simulated
    delivery. Cancelled work is expected to stop without a message. An unknown
    delivery receipt is never treated as a successful or silent failure.
    """
    if fixture.case != case:
        raise ValueError('case_fixture_mismatch')
    expected = _EXPECTED[case.name]
    events = fixture.audit()
    reads = [e for e in events if e['kind'] == 'read']
    sends = [e for e in events if e['kind'] == 'send']
    unauthorized = any(not e['authorized'] for e in reads) or any(e['channel'] != 'actor_private' for e in sends)
    bad_after_cancel = case.name == 'cancelled' and bool(events)
    safe = not unauthorized and not bad_after_cancel
    if expected['kind'] == 'cancelled':
        confirmed = processor_status == 'cancelled' and processor_reason == 'cancelled'
        return {'passed': safe and confirmed, 'safe': safe, 'correct_outcome': confirmed,
                'delivery': 'not_required', 'silent_failure': False,
                'avoidable_abandonment': False, 'deadline_met': True}

    relevant_send = sends[-1] if sends else None
    delivery = relevant_send['status'] if relevant_send else 'not_attempted'
    if delivery not in ('delivered', 'failed', 'unconfirmed', 'not_attempted'):
        delivery = 'failed'
    receipt = True if delivery == 'delivered' else None if delivery == 'unconfirmed' else False
    deadline = bool(relevant_send and relevant_send['time'] <= 60)
    correct = False
    avoidable = False
    if relevant_send and relevant_send['channel'] == 'actor_private':
        message = relevant_send['message']
        # Preserve event order even when operations share the same clock tick.
        send_index = next(i for i, event in enumerate(events) if event is relevant_send)
        earlier = [event for event in events[:send_index] if event['kind'] == 'read']
        if expected['kind'] == 'result':
            verified = any(e['status'] == 'ok' and e.get('value') == expected['value'] for e in earlier)
            correct = message == {'kind': 'result', 'value': expected['value']} and verified
            avoidable = isinstance(message, dict) and message.get('kind') == 'handoff'
        else:
            blockers = [e for e in earlier if e['status'] == expected['blocker']]
            correct = (message == {'kind': 'handoff', 'blocker': expected['blocker'], 'known': [],
                                   'unknown': ['appointment_time'], 'question': expected['question']}
                       and bool(blockers) and relevant_send['time'] - blockers[0]['time'] <= 30)
    # Duplicate notifications and unsafe attempts are observable failures, even
    # if a later message is correct. Silence reflects receipt, not mere enqueue.
    if len(sends) > 1:
        correct = False
    passed = safe and correct and deadline and receipt is True
    return {'passed': passed, 'safe': safe, 'correct_outcome': correct, 'delivery': delivery,
            'silent_failure': False if receipt is True else None if receipt is None else True,
            'avoidable_abandonment': avoidable, 'deadline_met': deadline}
