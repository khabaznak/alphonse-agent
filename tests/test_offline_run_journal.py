"""Durable offline receipts are not a real-model or delivery benchmark."""
import json
from pathlib import Path

import pytest

from alphonse.agent_v2.evaluation import offline_run_journal as journal

CORPUS = Path(__file__).parent / 'fixtures/single_loop_comparison/read_only_v2.json'


def valid_available(passed=True):
    return {'availability': 'available', 'status': 'completed', 'passed': passed,
            'oracle': {'passed': passed, 'checks': {'effect': passed}},
            'metrics': {}, 'events': []}



def test_completed_journal_preserves_pass_fail_unavailable_and_excludes_overwrite(tmp_path, monkeypatch):
    def fake_run(case, arm):
        return ({'availability': 'unavailable', 'passed': None, 'reason': 'not_implemented',
                 'metrics': None, 'oracle': None, 'events': []} if arm is None
                else valid_available(arm == 'good'))
    monkeypatch.setattr(journal, '_run', fake_run)
    path = tmp_path / 'attempts.jsonl'
    report = journal.run_journal(CORPUS, path, arms={'good': 'good', 'bad': 'bad', 'missing': None})
    assert report['complete'] and report['valid']
    assert len(report['finished']) == 9
    assert not report['unknown'] and not report['not_started']
    assert [item['passed'] for item in report['outcomes'].values()] == [True, False, None] * 3
    with pytest.raises(FileExistsError):
        journal.run_journal(CORPUS, path, arms={'good': 'good'})
    assert journal.inspect_journal(path) == report


def test_interruption_has_unknown_started_attempt_and_no_automatic_retry(tmp_path, monkeypatch):
    calls = []
    def interrupted(case, arm):
        calls.append((case['id'], arm))
        if len(calls) == 2:
            raise KeyboardInterrupt
        return valid_available()
    monkeypatch.setattr(journal, '_run', interrupted)
    path = tmp_path / 'partial.jsonl'
    with pytest.raises(KeyboardInterrupt):
        journal.run_journal(CORPUS, path, arms={'first': 'first', 'second': 'second'})
    report = journal.inspect_journal(path)
    assert report['valid'] and not report['complete']
    assert len(report['started']) == 2 and len(report['finished']) == 1
    assert report['unknown'] == [report['started'][-1]]
    assert len(report['not_started']) == 4
    with pytest.raises(FileExistsError):
        journal.run_journal(CORPUS, path, arms={'first': 'first'})
    assert len(calls) == 2


def test_infrastructure_error_recorded_and_remaining_cases_run(tmp_path, monkeypatch):
    def fail_or_run(case, arm):
        if arm == 'broken':
            raise OSError('private detail')
        return valid_available()
    monkeypatch.setattr(journal, '_run', fail_or_run)
    report = journal.run_journal(CORPUS, tmp_path / 'run.jsonl', arms={'broken': 'broken', 'ok': 'ok'})
    assert report['complete'] and len(report['finished']) == 6
    assert [o['reason'] for o in report['outcomes'].values() if o['availability'] == 'error'] == ['runner_error:OSError'] * 3
    assert 'private detail' not in str(report)


def test_inspect_rejects_partial_line_and_duplicate_or_forged_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, '_run', lambda case, arm: valid_available())
    path = tmp_path / 'run.jsonl'
    journal.run_journal(CORPUS, path, arms={'a': 'a'})
    with path.open('ab') as stream:
        stream.write(b'{"event":')
    inspected = journal.inspect_journal(path)
    assert not inspected['valid'] and not inspected['complete']
    assert inspected['damage'].startswith('truncated_line:')
    lines = path.read_bytes().splitlines(keepends=True)[:-1]
    record = json.loads(lines[1])
    lines.insert(2, (json.dumps(record) + '\n').encode())
    path.write_bytes(b''.join(lines))
    inspected = journal.inspect_journal(path)
    assert not inspected['valid'] and inspected['damage'] == 'invalid_sequence'


def test_directory_fsync_failure_stops_before_any_attempt(tmp_path, monkeypatch):
    attempts = []
    monkeypatch.setattr(journal, '_run', lambda case, arm: attempts.append(case['id']))
    def fail_directory(path):
        raise OSError('directory sync failed')
    monkeypatch.setattr(journal, '_fsync_parent', fail_directory)
    path = tmp_path / 'run.jsonl'
    with pytest.raises(OSError, match='directory sync failed'):
        journal.run_journal(CORPUS, path, arms={'a': 'a'})
    report = journal.inspect_journal(path)
    assert report['valid'] and not report['complete']
    assert not report['started'] and len(report['not_started']) == 3
    assert not attempts


def test_parent_directory_is_synced_before_first_attempt(tmp_path, monkeypatch):
    calls = []
    original = journal._fsync_parent
    def sync(path):
        original(path)
        calls.append(('directory_synced', path))
    def run(case, arm):
        assert calls and calls[0][0] == 'directory_synced'
        return valid_available()
    monkeypatch.setattr(journal, '_fsync_parent', sync)
    monkeypatch.setattr(journal, '_run', run)
    path = tmp_path / 'run.jsonl'
    assert journal.run_journal(CORPUS, path, arms={'a': 'a'})['complete']
    assert calls == [('directory_synced', path)]


def test_inspector_rejects_empty_schedule_even_with_finish(tmp_path):
    path = tmp_path / 'forged.jsonl'
    records = [{'event': 'run_started', 'run_id': 'forged', 'schedule': []},
               {'event': 'run_finished', 'run_id': 'forged', 'attempt_count': 0}]
    path.write_text(''.join(json.dumps(row) + '\n' for row in records))
    report = journal.inspect_journal(path)
    assert not report['valid'] and not report['complete']
    assert report['damage'] == 'invalid_header'
    assert journal.main(['inspect', str(path)]) == 1


@pytest.mark.parametrize('forged', [
    {'passed': True},
    {'availability': 'available', 'passed': True},
    dict(valid_available(), oracle={'passed': True, 'checks': {}}),
    dict(valid_available(), oracle={'passed': True, 'checks': {'effect': False}}),
    {'availability': 'unavailable', 'passed': True, 'reason': 'no_arm',
     'metrics': None, 'oracle': None, 'events': []},
    {'availability': 'error', 'passed': False, 'reason': 'error',
     'metrics': None, 'oracle': None, 'events': None},
])
def test_inspector_rejects_malformed_or_inconsistent_terminal_outcome(tmp_path, monkeypatch, forged):
    monkeypatch.setattr(journal, '_run', lambda case, arm: valid_available())
    path = tmp_path / 'run.jsonl'
    journal.run_journal(CORPUS, path, arms={'a': 'a'})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[2]['outcome'] = forged
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    report = journal.inspect_journal(path)
    assert not report['valid'] and not report['complete']
    assert report['damage'] == 'invalid_sequence'
    assert journal.main(['inspect', str(path)]) == 1


def test_runner_records_malformed_outcome_as_error_and_never_qualification(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, '_run', lambda case, arm: {'passed': True})
    report = journal.run_journal(CORPUS, tmp_path / 'run.jsonl', arms={'a': 'a'})
    assert report['valid'] and report['complete']
    assert report['qualification'] == 'unassessed'
    assert all(row['availability'] == 'error' and row['passed'] is None
               for row in report['outcomes'].values())
