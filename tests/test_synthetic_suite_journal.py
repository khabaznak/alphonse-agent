"""Synthetic journal preserves incomplete work and does not qualify fixtures."""
import json

import pytest

from alphonse.agent_v2.evaluation.offline_run_journal import inspect_journal
from alphonse.agent_v2.evaluation.synthetic_suite_journal import normalize_suite_result, run_synthetic_suite
from alphonse.agent_v2.evaluation.traffic_processor_development import run_traffic_arm


def offline(callback):
    callback.offline = True
    return callback


def row(case, arm, input):
    return {'case_id': case, 'arm': arm, 'input': input}


def test_real_traffic_arms_are_individual_journal_attempts(tmp_path):
    path = tmp_path / 'traffic.jsonl'
    payload = {'actions': [{'operation': 'inspect'}, {'operation': 'search', 'field': 'plate', 'subject': 'TEST-701'}],
               'answer': {'status': 'verified', 'subject': 'TEST-701', 'ticket_ids': ['T-71']}}
    attempts = [row('ticket', arm, payload) for arm in ('current_v3', 'targeted_v3_projection', 'single_loop')]
    result = run_synthetic_suite(path, suite_id='traffic-development', attempts=attempts,
        execute=offline(lambda item: run_traffic_arm(item['case_id'], item['arm'], **item['input'])))
    assert result['complete'] and result['valid'] and result['qualification'] == 'unassessed'
    assert len(result['outcomes']) == 3
    for outcome in result['outcomes'].values():
        assert outcome['passed'] is True
        assert outcome['oracle']['checks'] == {'fixture_only': True}
        assert outcome['details']['delivery'] == 'unassessed'
        assert outcome['details']['portal_audit'][-1]['result'] == 'ok'
    with pytest.raises(FileExistsError):
        run_synthetic_suite(path, suite_id='traffic-development', attempts=attempts, execute=offline(lambda _: None))


def test_followthrough_simulated_delivery_and_unknowns_preserved(tmp_path):
    synthetic = {'processor_status': 'waiting_user', 'events': [{'kind': 'system_two'}],
                 'fixture_outcome': {'passed': False, 'delivery': 'unconfirmed', 'silent_failure': None, 'safe': True, 'correct_outcome': True, 'deadline_met': True},
                 'product_delivery': 'not_integrated', 'decision_qualified': False,
                 'real_time_deadline': 'unassessed', 'processor_response': '{}', 'fixture_audit': []}
    result = run_synthetic_suite(tmp_path / 'follow.jsonl', suite_id='followthrough-development',
        attempts=[row('missing', 'candidate', {})], execute=offline(lambda _: synthetic))
    outcome = next(iter(result['outcomes'].values()))
    assert outcome['passed'] is False and outcome['details'] == synthetic
    assert outcome['details']['fixture_outcome']['silent_failure'] is None
    assert outcome['details']['product_delivery'] == 'not_integrated'
    assert result['qualification'] == 'unassessed'


def test_interruption_records_attempt_as_unknown_and_does_not_retry(tmp_path):
    path = tmp_path / 'partial.jsonl'
    attempts = [row('first', 'a', {}), row('second', 'a', {})]
    def interrupt(_):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run_synthetic_suite(path, suite_id='synthetic', attempts=attempts, execute=offline(interrupt))
    receipt = inspect_journal(path)
    assert receipt['valid'] and not receipt['complete']
    assert len(receipt['unknown']) == 1 and len(receipt['not_started']) == 1
    with pytest.raises(FileExistsError):
        run_synthetic_suite(path, suite_id='synthetic', attempts=attempts, execute=offline(lambda _: None))


def test_unavailable_and_malformed_results_are_not_passes(tmp_path):
    path = tmp_path / 'mixed.jsonl'
    attempts = [row('missing', 'a', {}), row('bad', 'a', {}), row('after', 'a', {})]
    def execute(attempt):
        if attempt['case_id'] == 'missing':
            return {'availability': 'unavailable', 'reason': 'unsupported'}
        if attempt['case_id'] == 'bad':
            return {'availability': 'unavailable', 'reason': 'unsupported', 'events': [{'kind': 'tool'}]}
        return {'processor_status': 'failed', 'processor_response': '', 'events': [], 'fixture_audit': [], 'product_delivery': 'unassessed', 'decision_qualified': False, 'real_time_deadline': 'unassessed', 'fixture_outcome': {'passed': False, 'safe': True, 'correct_outcome': False, 'delivery': 'not_attempted', 'deadline_met': False}}
    result = run_synthetic_suite(path, suite_id='synthetic', attempts=attempts, execute=offline(execute))
    outcomes = list(result['outcomes'].values())
    assert [outcome['availability'] for outcome in outcomes] == ['unavailable', 'error', 'available']
    assert [outcome['passed'] for outcome in outcomes] == [None, None, False]
    assert result['complete'] and result['qualification'] == 'unassessed'


def test_reject_inconsistent_traffic_processor_grade():
    with pytest.raises(ValueError, match='inconsistent_processor_grade'):
        normalize_suite_result({'availability': 'available', 'processor_status': 'failed', 'passed': True,
                                'portal_oracle': {'passed': True}, 'events': []})


def test_duplicate_schedule_rejected_before_file_creation(tmp_path):
    path = tmp_path / 'duplicate.jsonl'
    with pytest.raises(ValueError, match='duplicate_attempt'):
        run_synthetic_suite(path, suite_id='synthetic', attempts=[row('a', 'b', {}), row('a', 'b', {})], execute=offline(lambda _: None))
    assert not path.exists()


def test_followthrough_failed_processor_or_incomplete_output_cannot_pass():
    outcome = {'processor_status': 'failed', 'processor_reason': None, 'processor_response': '{"kind":"result","value":"09:30"}',
               'events': [], 'fixture_audit': [], 'product_delivery': 'not_integrated',
               'decision_qualified': False, 'real_time_deadline': 'unassessed',
               'fixture_outcome': {'passed': True, 'safe': True, 'correct_outcome': True,
                                   'delivery': 'delivered', 'deadline_met': True}}
    with pytest.raises(ValueError, match='inconsistent_followthrough_grade'):
        normalize_suite_result(outcome)
    with pytest.raises(ValueError, match='incomplete_followthrough_result'):
        normalize_suite_result({'processor_status': 'completed', 'events': [],
                                'fixture_outcome': {'passed': True}})


def test_missing_offline_declaration_is_rejected(tmp_path):
    path = tmp_path / 'unsafe.jsonl'
    with pytest.raises(ValueError, match='executor_must_explicitly_declare_offline'):
        run_synthetic_suite(path, suite_id='synthetic', attempts=[row('x', 'y', {})], execute=lambda _: None)
    assert not path.exists()
