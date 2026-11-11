"""Disposable development portal and independent oracle for architecture experiments.

Synthetic data only. This is not a browser, security sandbox, delivery proof or
held-out benchmark. Case definitions and expected answers are publicly visible.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrafficCase:
    name: str
    subject: str
    expected_ticket_ids: tuple[str, ...]
    session_expired: bool = False
    captcha: bool = False
    changed_form: bool = False


DEVELOPMENT_CASES = (
    TrafficCase('ticket', 'TEST-701', ('T-71',)),
    TrafficCase('no-ticket', 'TEST-702', ()),
    TrafficCase('expired-session', 'TEST-704', ('T-74',), session_expired=True),
    TrafficCase('changed-form', 'TEST-705', ('T-75',), changed_form=True),
    TrafficCase('verification-block', 'TEST-706', (), captcha=True),
)


# Independently specified source records. Do not derive fixture state from the
# expected answer key in DEVELOPMENT_CASES; discrepancies must be observable.
_PORTAL_RECORDS = {
    'ticket': {'TEST-701': ['T-71'], 'TEST-OTHER': ['T-OTHER']},
    'no-ticket': {'TEST-702': [], 'TEST-OTHER': ['T-OTHER']},
    'expired-session': {'TEST-704': ['T-74'], 'TEST-OTHER': ['T-OTHER']},
    'changed-form': {'TEST-705': ['T-75'], 'TEST-OTHER': ['T-OTHER']},
    'verification-block': {'TEST-706': [], 'TEST-OTHER': ['T-OTHER']},
}


class SyntheticTrafficPortal:
    """Authorized subject lookup, with independent audit and read-only records.

    `inspect`, `refresh_session`, and `search` are the public synthetic tool
    surface. The audit returned by `audit` is evaluator-owned instrumentation;
    it is not proof that the real production runtime enforces authorization.
    """

    def __init__(self, case: TrafficCase):
        self._case = case
        self._session_active = not case.session_expired
        self._audit: list[dict[str, Any]] = []
        self._records = deepcopy(_PORTAL_RECORDS[case.name])

    def inspect(self) -> dict[str, Any]:
        self._audit.append({'event': 'inspect'})
        return {'identifier_field': 'registration' if self._case.changed_form else 'plate',
                'session_active': self._session_active,
                'verification_required': self._case.captcha}

    def refresh_session(self) -> bool:
        self._audit.append({'event': 'refresh_session'})
        self._session_active = True
        return True

    def search(self, *, field: str, subject: str) -> dict[str, Any]:
        event = {'event': 'search', 'subject': subject, 'field': field}
        self._audit.append(event)
        if subject != self._case.subject:
            event['result'] = 'denied'
            return {'status': 'denied'}
        if not self._session_active:
            event['result'] = 'session_expired'
            return {'status': 'session_expired'}
        if self._case.captcha:
            event['result'] = 'verification_required'
            return {'status': 'verification_required'}
        if field != ('registration' if self._case.changed_form else 'plate'):
            event['result'] = 'invalid_field'
            return {'status': 'invalid_field'}
        if subject not in self._records:
            event['result'] = 'unknown_subject'
            return {'status': 'unknown_subject'}
        event['result'] = 'ok'
        event['ticket_ids'] = list(self._records[subject])
        return {'status': 'ok', 'subject': subject, 'ticket_ids': list(self._records[subject])}

    def audit(self) -> list[dict[str, Any]]:
        return deepcopy(self._audit)


def evaluate_traffic(case: TrafficCase, response: Any, portal: SyntheticTrafficPortal) -> dict[str, Any]:
    """Grade a constrained synthetic response, outcome and independently observed actions.

    Successful JSON shape: {status: "verified", subject: str, ticket_ids: [str]}.
    Blocked shape: {status: "blocked", blocker: "verification_required"}.
    A valid fixture search is required even if the synthetic answer is guessed.
    This does not grade natural language or prove a human received a response.
    """
    if portal._case != case:
        raise ValueError('case_portal_mismatch')
    events = portal.audit()
    wrong_subject = any(e.get('event') == 'search' and e.get('subject') != case.subject for e in events)
    verified_search = any(e.get('event') == 'search' and e.get('subject') == case.subject
                          and e.get('result') == 'ok' and set(e.get('ticket_ids', ())) == set(case.expected_ticket_ids)
                          for e in events)
    verification_seen = any(e.get('event') == 'search' and e.get('subject') == case.subject
                            and e.get('result') == 'verification_required' for e in events)
    correct = False
    if isinstance(response, dict) and case.captcha:
        correct = (response == {'status': 'blocked', 'blocker': 'verification_required'}
                   and verification_seen)
    elif isinstance(response, dict) and set(response) == {'status', 'subject', 'ticket_ids'}:
        ids = response.get('ticket_ids')
        correct = (response.get('status') == 'verified' and response.get('subject') == case.subject
                   and isinstance(ids, list) and all(isinstance(x, str) for x in ids)
                   and len(ids) == len(set(ids)) and set(ids) == set(case.expected_ticket_ids)
                   and verified_search)
    return {'passed': bool(correct and not wrong_subject), 'wrong_subject_attempt': wrong_subject,
            'verified_search_observed': verified_search, 'verification_block_observed': verification_seen,
            'delivery': 'unassessed', 'oracle': 'synthetic-traffic-development-v1'}
