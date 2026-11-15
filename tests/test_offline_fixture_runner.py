"""Scripted fixture plumbing; never a live provider comparison."""
from pathlib import Path

from alphonse.agent_v2.evaluation.offline_fixture_runner import compare_fixtures
from alphonse.agent_v2.evaluation.simplified_loop import Run

CORPUS = Path(__file__).parent / 'fixtures/single_loop_comparison/read_only_v2.json'
PRESCRIPTION = Path(__file__).parent / 'fixtures/single_loop_comparison/prescription_schedule_development_v1.json'


def test_simple_loop_fixture_preserve_success_denial_and_cancellation(monkeypatch):
    report = compare_fixtures(CORPUS)
    assert report['arm_names'] == ['single_loop']
    assert report['simulation'] is True
    read, denied, cancelled = report['cases']
    for arm in read['arms'].values():
        assert arm['availability'] == 'available'
        assert arm['passed'] is True
        assert arm['metrics']['provider_input_tokens'] is None
    assert all(arm['oracle']['checks']['forbidden_accesses'] for arm in denied['arms'].values())
    assert all(arm['passed'] is True for arm in cancelled['arms'].values())


def test_unsupported_cases_remain_in_each_denominator():
    report = compare_fixtures(PRESCRIPTION)
    for row in report['cases']:
        assert len(row['arms']) == 1
        for arm in row['arms'].values():
            assert arm['availability'] == 'unavailable'
            assert arm['passed'] is None
            assert arm['oracle'] is None


def test_separate_fixture_roots_and_failed_oracle_remain_visible():
    class WritesFixture:
        offline = True
        revision = 'test-write'
        def __call__(self, case, root, emit):
            (root / 'Receta.md').write_text('changed')
            return Run(status='completed', response='amoxicillin', measured_kinds=('question',))

    class ReadsOriginal:
        offline = True
        revision = 'test-read'
        def __call__(self, case, root, emit):
            assert (root / 'Receta.md').read_text() == 'Medicine: amoxicillin\n'
            emit({'kind': 'source_read', 'source': 'Receta.md'})
            return Run(status='completed', response='amoxicillin', measured_kinds=('question', 'source_read'))

    report = compare_fixtures(CORPUS, arms={'writing': WritesFixture(), 'reading': ReadsOriginal(), 'missing': None})
    first = report['cases'][0]['arms']
    assert first['writing']['passed'] is False
    assert first['writing']['oracle']['checks']['effects'] is False
    assert first['writing']['oracle']['false_success'] is True
    assert first['reading']['passed'] is True
    assert first['missing']['availability'] == 'unavailable'
    assert first['missing']['passed'] is None


def test_infrastructure_failure_is_inconclusive_and_other_arms_remain(monkeypatch):
    from alphonse.agent_v2.evaluation import offline_fixture_runner

    def injected_run(case, arm):
        if arm == 'bad':
            raise OSError('sensitive text must not leak')
        return {'availability': 'available', 'passed': False, 'oracle': {'passed': False}}

    monkeypatch.setattr(offline_fixture_runner, '_run', injected_run)
    report = compare_fixtures(CORPUS, arms={'broken': 'bad', 'observed': 'good'})
    assert len(report['cases']) == 3
    for row in report['cases']:
        failed = row['arms']['broken']
        assert failed['availability'] == 'error'
        assert failed['passed'] is None and failed['oracle'] is None and failed['events'] is None
        assert failed['reason'] == 'runner_error:OSError'
        assert row['arms']['observed']['passed'] is False
    assert 'sensitive text' not in str(report)
    assert any('interruption' in item for item in report['limitations'])
