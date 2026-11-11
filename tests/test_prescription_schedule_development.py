"""Tests for a synthetic development rubric, not a clinical benchmark."""
from pathlib import Path

import pytest

from alphonse.agent_v2.evaluation.candidate_comparison import SingleLoopFixtureAdapter, V3ReadFixtureAdapter
from alphonse.agent_v2.evaluation.simplified_loop import Run, compare, load_frozen

CORPUS = Path(__file__).parent / 'fixtures/single_loop_comparison/prescription_schedule_development_v1.json'


class FakeAnswer:
    offline = True
    revision = 'test-only-answer'

    def __init__(self, answer):
        self.answer = answer

    def __call__(self, case, root, emit):
        # The oracle (including date and treatment schedule) must never reach
        # an adapter; only the synthetic source records and goal may do so.
        assert 'oracle' not in case
        for path in case['fixture']['files']:
            (root / path).read_text()
            emit({'kind': 'source_read', 'source': path})
        return Run(status='completed', response=self.answer, measured_kinds=('source_read',))


@pytest.mark.parametrize(('answer', 'passes'), [
    ('{"due_medications": ["Asterin"]}', True),
    ('{"due_medications": []}', False),
    ('{"due_medications": ["Asterin", "Borelin"]}', False),  # expired
    ('{"due_medications": ["Asterin", "Cirelin"]}', False),  # evening only
    ('{"due_medications": ["Asterin", "Invented"]}', False),
    ('{"due_medications": ["Asterin", "asterin"]}', False),  # duplicate
    ('Do not take Asterin', False),  # natural language is not assessed
    ('{"due_medications": ["Asterin"], "note": "do not take"}', False),
])
def test_independent_rubric_checks_due_and_excluded_names(answer, passes):
    report = compare(CORPUS, baseline=FakeAnswer(answer), candidate=None)
    assert report['split'] == 'development'
    assert all(row['baseline']['passed'] is passes for row in report['cases'][:2])
    assert report['cases'][2]['baseline']['passed'] is (answer == '{"due_medications": []}')
    assert all(row['candidate']['availability'] == 'unavailable' for row in report['cases'])


def test_duration_boundaries_and_new_cases_are_not_claimed_supported():
    corpus, _ = load_frozen(CORPUS)
    assert [c['id'] for c in corpus['cases']] == [
        'development-prescription-morning-day-three',
        'development-prescription-morning-final-day',
        'development-prescription-morning-after-course',
    ]
    report = compare(CORPUS, baseline=V3ReadFixtureAdapter(), candidate=SingleLoopFixtureAdapter())
    for row in report['cases']:
        for side in ('baseline', 'candidate'):
            assert row[side]['availability'] == 'unavailable'
            assert row[side]['passed'] is None


def test_day_after_course_requires_explicit_empty_schedule():
    report = compare(CORPUS, baseline=FakeAnswer('{"due_medications": []}'), candidate=None)
    assert [row['baseline']['passed'] for row in report['cases']] == [False, False, True]


def test_guessed_answer_and_missing_instrumentation_cannot_pass():
    class Guesser:
        offline = True
        revision = 'test-guesser'
        def __init__(self, measured):
            self.measured = measured
        def __call__(self, case, root, emit):
            return Run(status='completed', response='{"due_medications": ["Asterin"]}',
                       measured_kinds=self.measured)
    for measured in ((), ('source_read',)):
        result = compare(CORPUS, baseline=Guesser(measured), candidate=None)['cases'][0]['baseline']
        assert result['passed'] is False
        assert result['oracle']['checks']['minimum_source_read_counts'] is not True
