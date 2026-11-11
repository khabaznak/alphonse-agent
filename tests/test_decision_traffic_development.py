"""Synthetic portal contracts; these are not model-based performance tests."""
from dataclasses import replace

import pytest
from alphonse.agent_v2.evaluation.decision_traffic_development import DEVELOPMENT_CASES, SyntheticTrafficPortal, evaluate_traffic


@pytest.mark.parametrize('case', DEVELOPMENT_CASES, ids=lambda c: c.name)
def test_synthetic_lookup_and_independent_oracle(case):
    portal = SyntheticTrafficPortal(case)
    page = portal.inspect()
    if not page['session_active']:
        portal.refresh_session()
    result = portal.search(field=page['identifier_field'], subject=case.subject)
    response = ({'status': 'blocked', 'blocker': 'verification_required'} if case.captcha
                else {'status': 'verified', 'subject': case.subject, 'ticket_ids': result['ticket_ids']})
    oracle = evaluate_traffic(case, response, portal)
    assert oracle['passed']
    assert oracle['delivery'] == 'unassessed'


def test_claim_without_search_or_with_wrong_subject_fails():
    case = DEVELOPMENT_CASES[0]
    portal = SyntheticTrafficPortal(case)
    answer = {'status': 'verified', 'subject': case.subject, 'ticket_ids': ['T-71']}
    assert not evaluate_traffic(case, answer, portal)['passed']
    assert portal.search(field='plate', subject='TEST-OTHER') == {'status': 'denied'}
    portal.search(field='plate', subject=case.subject)
    assert not evaluate_traffic(case, answer, portal)['passed']
    assert evaluate_traffic(case, answer, portal)['wrong_subject_attempt']


def test_bad_ticket_claims_fail():
    case = DEVELOPMENT_CASES[0]
    portal = SyntheticTrafficPortal(case)
    portal.search(field='plate', subject=case.subject)
    for ids in ([], ['T-71', 'T-71'], ['T-71', 'T-X'], ['T-X']):
        assert not evaluate_traffic(case, {'status': 'verified', 'subject': case.subject, 'ticket_ids': ids}, portal)['passed']


def test_captcha_requires_observed_block_and_no_false_success():
    case = DEVELOPMENT_CASES[-1]
    portal = SyntheticTrafficPortal(case)
    blocked = {'status': 'blocked', 'blocker': 'verification_required'}
    assert not evaluate_traffic(case, blocked, portal)['passed']
    assert portal.search(field='plate', subject=case.subject)['status'] == 'verification_required'
    assert evaluate_traffic(case, blocked, portal)['passed']
    assert not evaluate_traffic(case, {'status': 'verified', 'subject': case.subject, 'ticket_ids': []}, portal)['passed']


def test_changed_form_and_expired_session_require_recovery():
    changed = DEVELOPMENT_CASES[3]
    p = SyntheticTrafficPortal(changed)
    assert p.search(field='plate', subject=changed.subject)['status'] == 'invalid_field'
    assert p.search(field='registration', subject=changed.subject)['status'] == 'ok'
    expired = DEVELOPMENT_CASES[2]
    p = SyntheticTrafficPortal(expired)
    assert p.search(field='plate', subject=expired.subject)['status'] == 'session_expired'
    p.refresh_session()
    assert p.search(field='plate', subject=expired.subject)['status'] == 'ok'


def test_mismatched_case_and_portal_are_rejected():
    with pytest.raises(ValueError, match='case_portal_mismatch'):
        evaluate_traffic(DEVELOPMENT_CASES[0], {}, SyntheticTrafficPortal(DEVELOPMENT_CASES[1]))


def test_refresh_does_not_bypass_verification():
    case = DEVELOPMENT_CASES[-1]
    portal = SyntheticTrafficPortal(case)
    assert portal.refresh_session() is True
    assert portal.inspect()['verification_required'] is True
    assert portal.search(field='plate', subject=case.subject)['status'] == 'verification_required'
    assert not evaluate_traffic(case, {'status': 'verified', 'subject': case.subject, 'ticket_ids': []}, portal)['passed']


def test_independent_source_and_expected_answer_disagreement_fails():
    case = replace(DEVELOPMENT_CASES[0], expected_ticket_ids=('T-DIFFERENT',))
    portal = SyntheticTrafficPortal(case)
    result = portal.search(field='plate', subject=case.subject)
    assert result['ticket_ids'] == ['T-71']  # source did not inherit altered answer key
    response = {'status': 'verified', 'subject': case.subject, 'ticket_ids': result['ticket_ids']}
    assert not evaluate_traffic(case, response, portal)['passed']


def test_guessing_expected_answer_does_not_override_conflicting_source():
    case = replace(DEVELOPMENT_CASES[0], expected_ticket_ids=('T-DIFFERENT',))
    portal = SyntheticTrafficPortal(case)
    portal.search(field='plate', subject=case.subject)
    response = {'status': 'verified', 'subject': case.subject, 'ticket_ids': ['T-DIFFERENT']}
    assert not evaluate_traffic(case, response, portal)['passed']
