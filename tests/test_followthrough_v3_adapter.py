"""Offline scripted wiring checks, not measurements of model capability."""
import json
import pytest

from alphonse.agent_v2.evaluation.decision_followthrough_development import Case
from alphonse.agent_v2.evaluation.followthrough_v3_adapter import run_v3_fixture, _terminal_matches


RESULT = json.dumps({'kind': 'result', 'value': '09:30'})
HANDOFF = json.dumps({'kind': 'handoff', 'blocker': 'missing_identity',
                      'known': [], 'unknown': ['appointment_time'],
                      'question': 'appointment_identity'})


@pytest.mark.parametrize('projection', [False, True])
def test_actual_v3_scripted_fallback_and_simulated_bridge(projection):
    result = run_v3_fixture(Case('fallback'), projection_enabled=projection,
        scripted_sources=['primary', 'backup'], scripted_response=RESULT, simulate_delivery=True)
    assert result['availability'] == 'available'
    assert result['processor_status'] == 'completed'
    assert [e['status'] for e in result['fixture_audit'] if e['kind'] == 'read'] == ['unavailable', 'ok']
    assert result['fixture_outcome']['passed'] is True
    assert result['simulated_bridge'] == 'delivered'
    assert result['product_delivery'] == 'unassessed'
    assert result['real_time_deadline'] == 'unassessed'
    assert result['decision_qualified'] is False


@pytest.mark.parametrize('projection', [False, True])
def test_v3_missing_information_and_cancellation_retain_real_outcomes(projection):
    result = run_v3_fixture(Case('missing'), projection_enabled=projection,
        scripted_sources=['primary'], scripted_response=HANDOFF, simulate_delivery=True)
    # The replay currently stops before final response after negative evidence;
    # never manufacture a question just because the script contains one.
    assert result['availability'] == 'available'
    assert result['processor_status'] == 'failed'
    assert result['fixture_outcome']['passed'] is False
    assert result['simulated_bridge'] == 'not_requested'
    cancelled = run_v3_fixture(Case('cancelled'), projection_enabled=projection,
        scripted_sources=['primary'], scripted_response=HANDOFF, simulate_delivery=True)
    assert cancelled['processor_status'] == 'cancelled'
    assert cancelled['processor_reason'] == 'Task cancelled.'
    assert cancelled['fixture_audit'] == []
    assert cancelled['fixture_outcome']['passed'] is True
    assert cancelled['decision_qualified'] is False


def test_fabricated_or_premature_answer_cannot_pass_without_evidence():
    result = run_v3_fixture(Case('fallback'), projection_enabled=False,
        scripted_sources=['primary'], scripted_response=RESULT, simulate_delivery=True)
    assert result['fixture_outcome']['passed'] is False
    assert not any(e.get('source') == 'backup' for e in result['fixture_audit'])


def test_empty_replay_is_explicitly_unavailable():
    result = run_v3_fixture(Case('missing'), projection_enabled=False,
        scripted_sources=[], scripted_response=HANDOFF)
    assert result['availability'] == 'unavailable'
    assert result['decision_qualified'] is False


def test_terminal_status_matches_answer_or_question():
    assert _terminal_matches('completed', RESULT)
    assert _terminal_matches('waiting_user', HANDOFF)
    assert not _terminal_matches('waiting_user', RESULT)
    assert not _terminal_matches('completed', HANDOFF)
    assert not _terminal_matches('completed', 'malformed')


def test_worker_error_never_claims_empty_audit(monkeypatch, capsys):
    from io import StringIO
    from alphonse.agent_v2.evaluation import followthrough_v3_adapter as module
    def fail(_):
        raise RuntimeError('secret detail')
    monkeypatch.setattr(module, '_execute', fail)
    monkeypatch.setattr('sys.stdin', StringIO('{}'))
    module._worker()
    result = json.loads(capsys.readouterr().out)
    assert result['availability'] == 'error'
    assert result['fixture_audit'] is None and result['events'] is None
    assert result['observation_status'] == 'unknown_after_worker_error'
    assert 'secret detail' not in str(result)


def test_invalid_source_is_recorded_as_denied_attempt():
    result = run_v3_fixture(Case('fallback'), projection_enabled=False,
        scripted_sources=['other'], scripted_response=RESULT, simulate_delivery=True)
    assert result['availability'] == 'available'
    assert any(e.get('source') == 'other' and e.get('authorized') is False
               for e in result['fixture_audit'])
    assert result['fixture_outcome']['safe'] is False
