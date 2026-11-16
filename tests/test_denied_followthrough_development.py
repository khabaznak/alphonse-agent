"""Development-only denial replay with no real communications."""
import pytest

from alphonse.agent_v2.evaluation.denied_followthrough_development import compare_denied


@pytest.mark.parametrize('bridge', [False, True])
def test_denial_replay_reports_observed_outcomes_without_claiming_delivery(bridge):
    report = compare_denied(simulate_delivery=bridge)
    assert report['decision_qualified'] is False
    assert report['delivery'] == ('synthetic_bridge' if bridge else 'unassessed')
    for name, arm in report['arms'].items():
        assert arm['decision_qualified'] is False
        assert arm['product_delivery'] in {'unassessed', 'not_integrated'}
        assert any(event.get('status') == 'denied' for event in arm['fixture_audit']), name
        if not bridge:
            assert arm['fixture_outcome']['passed'] is False
        if arm['fixture_outcome']['passed']:
            assert arm['processor_status'] == 'completed'
            assert arm['fixture_outcome']['delivery'] == 'delivered'
            assert bridge


def test_explicit_boolean_bridge_required():
    with pytest.raises(ValueError):
        compare_denied(simulate_delivery=1)
