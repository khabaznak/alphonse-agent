"""Accounting of disposable synthetic receipts, not external performance."""
import json
import subprocess
import sys

import pytest

from alphonse.agent_v2.evaluation.synthetic_journal_summary import summarize_synthetic_journal
from alphonse.agent_v2.evaluation.synthetic_suite_journal import run_synthetic_suite
from alphonse.agent_v2.evaluation.clock_pilot_journal import run_offline_clock_journal


def offline(fn):
    fn.offline = True
    return fn


def item(case, arm='candidate'):
    return {'case_id': case, 'arm': arm, 'input': {}}


def fixture_result(passed):
    return {'processor_status': 'completed' if passed else 'failed', 'passed': passed,
            'portal_oracle': {'passed': passed}, 'events': []}


def test_complete_real_journal_preserves_fixture_grades_and_unknowns(tmp_path):
    path = tmp_path / 'suite.jsonl'
    def execute(attempt):
        case = attempt['case_id']
        if case == 'unavailable':
            return {'availability': 'unavailable', 'reason': 'unsupported'}
        if case == 'error':
            raise RuntimeError('private detail')
        return fixture_result(case == 'pass')
    run_synthetic_suite(path, suite_id='synthetic',
        attempts=[item('pass'), item('fail'), item('unavailable'), item('error')], execute=offline(execute))
    report = summarize_synthetic_journal(path)
    arm = report['arms']['candidate']
    assert report['journal']['complete'] and report['qualification'] == 'unassessed'
    assert (arm['scheduled'], arm['pass'], arm['fail'], arm['unknown'], arm['missing']) == (4, 1, 1, 2, 0)
    assert [a['state'] for a in arm['attempts']] == ['finished', 'finished', 'unavailable', 'finished']
    assert all(a['elapsed_ms'] is not None for a in arm['attempts'])
    assert all(a['delivery_observed'] is None for a in arm['attempts'])
    assert arm['observation_coverage']['delivery_observed'] == {'known': 0, 'unknown': 4}
    assert any('not been independently audited' in s for s in report['limitations'])
    assert report['journal']['input_digest_verification'] == 'unverified_header'
    assert 'private detail' not in str(report)


def test_interruption_and_unstarted_attempts_remain_distinct(tmp_path):
    path = tmp_path / 'partial.jsonl'
    def execute(attempt):
        if attempt['case_id'] == 'interrupted':
            raise KeyboardInterrupt()
        return fixture_result(True)
    with pytest.raises(KeyboardInterrupt):
        run_synthetic_suite(path, suite_id='synthetic',
            attempts=[item('done'), item('interrupted'), item('not_started')], execute=offline(execute))
    report = summarize_synthetic_journal(path)
    arm = report['arms']['candidate']
    assert not report['journal']['complete']
    assert (arm['pass'], arm['unknown'], arm['missing']) == (1, 2, 1)
    assert [a['state'] for a in arm['attempts']] == ['finished', 'interrupted', 'not_started']
    assert arm['attempts'][1]['elapsed_ms'] is None and arm['attempts'][2]['elapsed_ms'] is None


@pytest.mark.parametrize(('mutation', 'error'), [
    ('damaged', 'invalid_synthetic_journal'), ('duplicate', 'invalid_synthetic_journal'),
    ('misbound', 'invalid_synthetic_journal'),
    ('wrong_kind', 'synthetic_journal_kind_required'), ('wrong_grade', 'synthetic_fixture_grade_required'),
    ('failed_pass', 'synthetic_fixture_grade_required'), ('wrong_schema', 'invalid_synthetic_journal_schema'),
    ('boolean_schema', 'invalid_synthetic_journal_schema'), ('boolean_count', 'invalid_synthetic_journal_attempt_count'),
    ('bad_digest', 'invalid_synthetic_journal_digest'),
    ('negative_time', 'invalid_synthetic_journal_elapsed'), ('boolean_time', 'invalid_synthetic_journal_elapsed'),
    ('infinite_time', 'invalid_synthetic_journal_elapsed'), ('overflow_time', 'invalid_synthetic_journal_elapsed'),
])
def test_rejects_damaged_mismatched_or_invalid_input(tmp_path, mutation, error):
    path = tmp_path / 'suite.jsonl'
    run_synthetic_suite(path, suite_id='synthetic', attempts=[item('pass')],
                        execute=offline(lambda _: fixture_result(True)))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == 'damaged':
        path.write_text(path.read_text() + '{')
    else:
        if mutation == 'duplicate':
            rows.insert(2, dict(rows[1]))
        elif mutation == 'misbound':
            rows[1]['case_id'] = 'other'
        elif mutation == 'wrong_kind':
            rows[0]['experiment_kind'] = 'other'
        elif mutation == 'wrong_grade':
            rows[2]['outcome']['oracle']['checks'] = {'other': True}
        elif mutation == 'failed_pass':
            rows[2]['outcome']['status'] = 'failed'
            rows[2]['outcome']['details']['processor_status'] = 'failed'
        elif mutation == 'wrong_schema':
            rows[0]['schema_version'] = 2
        elif mutation == 'boolean_schema':
            rows[0]['schema_version'] = True
        elif mutation == 'boolean_count':
            rows[-1]['attempt_count'] = True
        elif mutation == 'bad_digest':
            rows[0]['corpus_sha256'] = 'g' * 64
        elif mutation == 'negative_time':
            rows[2]['elapsed_seconds'] = -1
        elif mutation == 'boolean_time':
            rows[2]['elapsed_seconds'] = True
        elif mutation == 'infinite_time':
            rows[2]['elapsed_seconds'] = float('inf')
        elif mutation == 'overflow_time':
            rows[2]['elapsed_seconds'] = 10 ** 309
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match=error):
        summarize_synthetic_journal(path)


@pytest.mark.parametrize('status', ['waiting_user', 'cancelled'])
def test_valid_fixture_question_and_cancellation_remain_usable(tmp_path, status):
    path = tmp_path / 'suite.jsonl'
    is_cancelled = status == 'cancelled'
    synthetic = {'processor_status': status, 'processor_reason': 'cancelled' if is_cancelled else None,
                 'processor_response': '' if is_cancelled else json.dumps({'kind': 'handoff'}),
                 'events': [], 'fixture_audit': [], 'product_delivery': 'not_integrated',
                 'decision_qualified': False, 'real_time_deadline': 'unassessed',
                 'fixture_outcome': {'passed': True, 'safe': True, 'correct_outcome': True,
                                     'deadline_met': True,
                                     'delivery': 'not_required' if is_cancelled else 'delivered'}}
    run_synthetic_suite(path, suite_id='synthetic', attempts=[item('case')],
                        execute=offline(lambda _: synthetic))
    report = summarize_synthetic_journal(path)
    assert report['arms']['candidate']['pass'] == 1
    assert report['arms']['candidate']['attempts'][0]['delivery_observed'] is None


def test_cli_summarizes_existing_clock_journal_without_running_a_provider(tmp_path):
    journal = tmp_path / 'clock.jsonl'
    run_offline_clock_journal(journal, source_instant='2026-04-12T14:00:00+00:00',
                              source_timezone='America/Mexico_City',
                              expected_instant='2026-04-12T14:00:00+00:00',
                              expected_timezone='America/Mexico_City')
    before = journal.read_bytes()
    result = subprocess.run([sys.executable, '-m', 'alphonse.agent_v2.evaluation.synthetic_journal_summary',
                             str(journal)], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    arm = report['arms']['single_loop']
    assert (arm['scheduled'], arm['pass'], arm['unknown']) == (1, 0, 1)
    assert arm['attempts'][0]['state'] == 'unavailable'
    assert arm['attempts'][0]['delivery_observed'] is None
    assert report['qualification'] == 'unassessed'
    assert journal.read_bytes() == before and not result.stderr


def test_cli_rejects_damaged_journal_without_exposing_its_content(tmp_path):
    journal = tmp_path / 'damaged.jsonl'
    journal.write_text('{sensitive synthetic content')
    result = subprocess.run([sys.executable, '-m', 'alphonse.agent_v2.evaluation.synthetic_journal_summary',
                             str(journal)], capture_output=True, text=True)
    assert result.returncode == 2 and not result.stdout
    assert 'sensitive synthetic content' not in result.stderr
