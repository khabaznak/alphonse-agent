"""Development-only denial replay with no real communications."""
import json

import pytest

from alphonse.agent_v2.evaluation.denied_followthrough_development import compare_denied
from alphonse.agent_v2.evaluation.followthrough_v3_adapter import _terminal_matches


def test_terminal_nonquestion_handoff_is_distinct_from_question_or_success():
    handoff = json.dumps({'kind': 'handoff', 'blocker': 'denied', 'question': None})
    assert _terminal_matches('completed', handoff)
    assert not _terminal_matches('waiting_user', handoff)
    assert not _terminal_matches('completed', json.dumps({'kind': 'handoff'}))


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
