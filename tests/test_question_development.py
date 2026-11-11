"""Cross-processor synthetic question checks, not delivery qualification."""
import pytest
from alphonse.agent_v2.evaluation.question_development import compare_missing_identity


@pytest.mark.parametrize('bridge', [False, True])
def test_matched_question_requires_independently_simulated_delivery(bridge):
    report = compare_missing_identity(simulate_delivery=bridge)
    assert report['decision_qualified'] is False
    for result in report['arms'].values():
        assert result['processor_status'] == 'waiting_user'
        assert result['fixture_outcome']['passed'] is bridge
        assert result['product_delivery'] in {'unassessed', 'not_integrated'}
        assert result['decision_qualified'] is False
        assert result['fixture_audit'][0]['status'] == 'missing_identity'


def test_bridge_must_be_explicit_boolean():
    with pytest.raises(ValueError):
        compare_missing_identity(simulate_delivery='yes')
