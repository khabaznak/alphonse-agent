"""Real processor wiring with prerecorded choices, never a performance benchmark."""
import pytest

from alphonse.agent_v2.evaluation.traffic_processor_development import compare_traffic


@pytest.mark.parametrize(('case', 'actions', 'answer'), [
    ('ticket', [{'operation': 'inspect'}, {'operation': 'search', 'field': 'plate', 'subject': 'TEST-701'}],
     {'status': 'verified', 'subject': 'TEST-701', 'ticket_ids': ['T-71']}),
    ('no-ticket', [{'operation': 'inspect'}, {'operation': 'search', 'field': 'plate', 'subject': 'TEST-702'}],
     {'status': 'verified', 'subject': 'TEST-702', 'ticket_ids': []}),
    ('changed-form', [{'operation': 'inspect'}, {'operation': 'search', 'field': 'registration', 'subject': 'TEST-705'}],
     {'status': 'verified', 'subject': 'TEST-705', 'ticket_ids': ['T-75']}),
    ('verification-block', [{'operation': 'inspect'}, {'operation': 'search', 'field': 'plate', 'subject': 'TEST-706'}],
     {'status': 'blocked', 'blocker': 'verification_required'}),
])
def test_three_processors_share_portal_observations(case, actions, answer):
    report = compare_traffic(case, actions=actions, answer=answer)
    assert report['decision_qualified'] is False
    for arm, result in report['arms'].items():
        assert result['availability'] == 'available', (arm, result)
        assert result['passed'] is True, (arm, result)
        assert result['delivery'] == 'unassessed'
        assert result['projection_enabled'] is (arm == 'targeted_v3_projection')
        assert result['portal_audit'][-1]['event'] == 'search'


def test_guessed_answer_without_search_fails_each_oracle():
    report = compare_traffic('ticket', actions=[{'operation': 'inspect'}],
        answer={'status': 'verified', 'subject': 'TEST-701', 'ticket_ids': ['T-71']})
    for result in report['arms'].values():
        assert result['availability'] == 'available'
        assert result['passed'] is False
        assert result['portal_oracle']['verified_search_observed'] is False


def test_wrong_subject_attempt_is_not_hidden_by_later_correct_lookup():
    report = compare_traffic('ticket', actions=[
        {'operation': 'search', 'field': 'plate', 'subject': 'TEST-OTHER'},
        {'operation': 'search', 'field': 'plate', 'subject': 'TEST-701'}],
        answer={'status': 'verified', 'subject': 'TEST-701', 'ticket_ids': ['T-71']})
    for result in report['arms'].values():
        assert result['availability'] == 'available'
        assert result['passed'] is False
        assert result['portal_oracle']['wrong_subject_attempt'] is True


def test_missing_case_is_unavailable_not_a_pass():
    report = compare_traffic('not-implemented', actions=[{'operation': 'inspect'}], answer={})
    for result in report['arms'].values():
        assert result['availability'] == 'unavailable'
        assert result['passed'] is None


def test_session_refresh_requires_a_stateful_tool():
    report = compare_traffic('expired-session', actions=[{'operation': 'refresh_session'}],
        answer={'status': 'verified', 'subject': 'TEST-704', 'ticket_ids': ['T-74']})
    for result in report['arms'].values():
        assert result['availability'] == 'unavailable'
        assert result['reason'] == 'session_refresh_requires_stateful_tool'
        assert result['passed'] is None


def test_processor_failure_cannot_pass_even_with_matching_portal_observation(monkeypatch):
    from alphonse.agent_v2.evaluation import traffic_processor_development as module
    import json
    def failed_processor(case, portal, actions, answer, emit):
        portal.search(field='plate', subject='TEST-701')
        return 'failed', json.dumps(answer)
    monkeypatch.setattr(module, '_v3', failed_processor)
    result = module._worker({'case': 'ticket', 'arm': 'current_v3',
        'actions': [{'operation': 'search', 'field': 'plate', 'subject': 'TEST-701'}],
        'answer': {'status': 'verified', 'subject': 'TEST-701', 'ticket_ids': ['T-71']}})
    assert result['portal_oracle']['passed'] is True
    assert result['processor_status'] == 'failed'
    assert result['passed'] is False


def test_wrong_answer_after_valid_lookup_fails_each_oracle():
    report = compare_traffic('ticket', actions=[{'operation': 'search', 'field': 'plate', 'subject': 'TEST-701'}],
        answer={'status': 'verified', 'subject': 'TEST-701', 'ticket_ids': []})
    for result in report['arms'].values():
        assert result['availability'] == 'available'
        assert result['portal_oracle']['verified_search_observed'] is True
        assert result['passed'] is False
