"""Three-arm scripted plumbing; never a live provider comparison."""
from pathlib import Path

from alphonse.agent_v2.evaluation.offline_three_arm import compare_three
from alphonse.agent_v2.evaluation.simplified_loop import Run

CORPUS = Path(__file__).parent / 'fixtures/single_loop_comparison/read_only_v2.json'
PRESCRIPTION = Path(__file__).parent / 'fixtures/single_loop_comparison/prescription_schedule_development_v1.json'


def test_three_actual_offline_arms_preserve_success_denial_and_cancellation(monkeypatch):
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', 'unexpected-parent-value')
    report = compare_three(CORPUS)
    assert report['arm_names'] == ['current_v3', 'targeted_v3_projection', 'single_loop']
    assert report['simulation'] is True
    read, denied, cancelled = report['cases']
    for arm in read['arms'].values():
        assert arm['availability'] == 'available'
        assert arm['passed'] is True
        assert arm['metrics']['provider_input_tokens'] is None
    assert 'projection-off' in read['arms']['current_v3']['revision']
    assert 'projection-on' in read['arms']['targeted_v3_projection']['revision']
    assert denied['arms']['current_v3']['passed'] is False
    assert all(arm['oracle']['checks']['forbidden_accesses'] for arm in denied['arms'].values())
    assert all(arm['passed'] is True for arm in cancelled['arms'].values())


def test_unsupported_cases_remain_in_each_denominator():
    report = compare_three(PRESCRIPTION)
    for row in report['cases']:
        assert len(row['arms']) == 3
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

    report = compare_three(CORPUS, arms={'writing': WritesFixture(), 'reading': ReadsOriginal(), 'missing': None})
    first = report['cases'][0]['arms']
    assert first['writing']['passed'] is False
    assert first['writing']['oracle']['checks']['effects'] is False
    assert first['writing']['oracle']['false_success'] is True
    assert first['reading']['passed'] is True
    assert first['missing']['availability'] == 'unavailable'
    assert first['missing']['passed'] is None


def test_infrastructure_failure_is_inconclusive_and_other_arms_remain(monkeypatch):
    from alphonse.agent_v2.evaluation import offline_three_arm

    def injected_run(case, arm):
        if arm == 'bad':
            raise OSError('sensitive text must not leak')
        return {'availability': 'available', 'passed': False, 'oracle': {'passed': False}}

    monkeypatch.setattr(offline_three_arm, '_run', injected_run)
    report = compare_three(CORPUS, arms={'broken': 'bad', 'observed': 'good'})
    assert len(report['cases']) == 3
    for row in report['cases']:
        failed = row['arms']['broken']
        assert failed['availability'] == 'error'
        assert failed['passed'] is None and failed['oracle'] is None and failed['events'] is None
        assert failed['reason'] == 'runner_error:OSError'
        assert row['arms']['observed']['passed'] is False
    assert 'sensitive text' not in str(report)
    assert any('interruption' in item for item in report['limitations'])
