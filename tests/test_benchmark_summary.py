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


def test_optional_observations_preserve_zero_false_and_missing_in_full_denominator():
    schedule = [scheduled('one', 'traffic', 'candidate'), scheduled('two', 'traffic', 'candidate'),
                scheduled('three', 'traffic', 'candidate')]
    completed = attempt('one', 'traffic', 'candidate', 'finished', 'pass', 100)
    completed.update(first_useful_progress_ms=0, delivery_observed=False, model_calls=2,
                     tool_calls=0, human_interventions=0)
    interrupted = attempt('two', 'traffic', 'candidate', 'interrupted', 'unknown', 50)
    interrupted.update(first_useful_progress_ms=20, delivery_observed=True)

    report = summarize(schedule, [completed, interrupted])['arms']['candidate']

    assert report['observation_coverage']['delivery_observed'] == {'known': 2, 'unknown': 1}
    assert report['observation_coverage']['tool_calls'] == {'known': 1, 'unknown': 2}
    assert report['attempts'][0]['first_useful_progress_ms'] == 0
    assert report['attempts'][0]['delivery_observed'] is False
    assert report['attempts'][1]['delivery_observed'] is True
    assert report['attempts'][2]['delivery_observed'] is None
    assert report['attempts'][2]['tool_calls'] is None
    assert report['unknown'] == 2 and report['missing'] == 1


@pytest.mark.parametrize(('field', 'value'), [
    ('first_useful_progress_ms', float('nan')), ('first_useful_progress_ms', True),
    ('delivery_observed', 1), ('delivery_observed', 'yes'), ('model_calls', True),
    ('tool_calls', -1), ('human_interventions', 1.5),
])
def test_invalid_optional_observations_are_rejected(field, value):
    row = attempt('one', 'traffic', 'candidate', 'finished', 'pass', 1)
    row[field] = value
    with pytest.raises(ValueError, match='invalid_'):
        summarize([scheduled('one', 'traffic', 'candidate')], [row])


@pytest.mark.parametrize('elapsed', [-1, float('nan'), float('inf'), True, '10'])
def test_invalid_timing_does_not_enter_report(elapsed):
    with pytest.raises(ValueError, match='invalid_elapsed'):
        summarize([scheduled('one', 'x', 'a')], [attempt('one', 'x', 'a', 'finished', 'pass', elapsed)])
