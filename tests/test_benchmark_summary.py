import pytest

from alphonse.agent_v2.evaluation.benchmark_summary import summarize


def scheduled(identity, case, arm):
    return dict(attempt_id=identity, case_id=case, arm=arm)


def attempt(identity, case, arm, state, grade, elapsed):
    return dict(attempt_id=identity, case_id=case, arm=arm, state=state, grade=grade, elapsed_ms=elapsed)


def test_preserves_repetitions_and_failed_interrupted_unavailable_missing_attempts():
    schedule = [scheduled('t1', 'traffic', 'candidate'), scheduled('t2', 'traffic', 'candidate'),
                scheduled('t3', 'traffic', 'reference'), scheduled('p1', 'prescription', 'candidate'),
                scheduled('p2', 'prescription', 'reference')]
    report = summarize(schedule, [attempt('t2', 'traffic', 'candidate', 'finished', 'pass', 500),
                                 attempt('t1', 'traffic', 'candidate', 'finished', 'fail', 400),
                                 attempt('t3', 'traffic', 'reference', 'interrupted', 'unknown', 200),
                                 attempt('p1', 'prescription', 'candidate', 'unavailable', 'unknown', None)])
    candidate = report['arms']['candidate']
    assert (candidate['scheduled'], candidate['pass'], candidate['fail'], candidate['unknown']) == (3, 1, 1, 1)
    assert [row['attempt_id'] for row in candidate['attempts']] == ['t1', 't2', 'p1']
    assert report['arms']['reference']['unknown'] == 2
    assert report['arms']['reference']['missing'] == 1
    assert report['arms']['reference']['attempts'][1]['state'] == 'not_started'
    assert candidate['cases']['traffic'] == {'scheduled': 2, 'pass': 1, 'fail': 1, 'unknown': 0, 'missing': 0}
    assert candidate['cases']['prescription'] == {'scheduled': 1, 'pass': 0, 'fail': 0, 'unknown': 1, 'missing': 0}
    assert report['arms']['reference']['cases']['prescription']['missing'] == 1
    assert report['case_schedule_balance'] == {'status': 'unbalanced', 'unbalanced_cases': {
        'traffic': {'candidate': 2, 'reference': 1}}}
    assert report['qualification'] == 'unassessed'


def test_schedule_balance_counts_absent_cases_and_never_assumes_single_arm_is_a_comparison():
    schedule = [scheduled('a', 'inbox', 'candidate'), scheduled('b', 'inbox', 'reference'),
                scheduled('c', 'prescription', 'candidate')]
    report = summarize(schedule, [])
    assert report['case_schedule_balance'] == {'status': 'unbalanced', 'unbalanced_cases': {
        'prescription': {'candidate': 1, 'reference': 0}}}
    assert report['arms']['candidate']['cases']['inbox']['unknown'] == 1
    assert report['arms']['candidate']['cases']['inbox']['missing'] == 1
    assert summarize(schedule[:2], [])['case_schedule_balance']['status'] == 'balanced'
    assert summarize(schedule[:1], [])['case_schedule_balance']['status'] == 'unassessed'


def test_rejects_duplicate_unknown_and_misbound_attempt_ids():
    schedule = [scheduled('one', 'traffic', 'candidate'), scheduled('two', 'traffic', 'candidate')]
    with pytest.raises(ValueError, match='duplicate_schedule'):
        summarize(schedule + [scheduled('one', 'prescription', 'reference')], [])
    with pytest.raises(ValueError, match='unexpected_or_duplicate_attempt'):
        summarize(schedule, [attempt('three', 'traffic', 'candidate', 'finished', 'pass', 1)])
    with pytest.raises(ValueError, match='unexpected_or_duplicate_attempt'):
        summarize(schedule, [attempt('one', 'prescription', 'candidate', 'finished', 'pass', 1)])
    with pytest.raises(ValueError, match='unexpected_or_duplicate_attempt'):
        summarize(schedule, [attempt('one', 'traffic', 'reference', 'finished', 'pass', 1)])
    with pytest.raises(ValueError, match='unexpected_or_duplicate_attempt'):
        summarize(schedule, [attempt('one', 'traffic', 'candidate', 'finished', 'pass', 1)] * 2)


def test_cannot_promote_interruption_to_pass():
    with pytest.raises(ValueError, match='invalid_grade'):
        summarize([scheduled('one', 'traffic', 'candidate')],
                  [attempt('one', 'traffic', 'candidate', 'interrupted', 'pass', 1)])


@pytest.mark.parametrize('elapsed', [-1, float('nan'), float('inf'), True, '10'])
def test_invalid_timing_does_not_enter_report(elapsed):
    with pytest.raises(ValueError, match='invalid_elapsed'):
        summarize([scheduled('one', 'x', 'a')], [attempt('one', 'x', 'a', 'finished', 'pass', elapsed)])
