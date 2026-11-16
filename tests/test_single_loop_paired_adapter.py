"""Synthetic simple-loop cases; no live-model qualification."""
from pathlib import Path
from alphonse.agent_v2.evaluation.candidate_comparison import SingleLoopFixtureAdapter
from alphonse.agent_v2.evaluation.simplified_loop import compare

CORPUS = Path(__file__).parent / 'fixtures/single_loop_comparison/read_only.json'
EXPANDED = Path(__file__).parent / 'fixtures/single_loop_comparison/read_only_v2.json'
HOLDOUT = Path(__file__).parent / 'fixtures/single_loop_comparison/heldout_read_only_v1.json'
LIFECYCLE = Path(__file__).parent / 'fixtures/single_loop_comparison/lifecycle_v1.json'
LIFECYCLE_V2 = Path(__file__).parent / 'fixtures/single_loop_comparison/lifecycle_v2.json'

def test_answer_without_independently_observed_source_access_cannot_pass():
    from alphonse.agent_v2.evaluation.simplified_loop import Run

    class FakeAnswer:
        offline = True
        revision = 'test-only-answer'
        def __init__(self, reads, measured=True):
            self.reads, self.measured = reads, measured
        def __call__(self, case, root, emit):
            for _ in range(self.reads):
                emit({'kind': 'source_read', 'source': 'Receta.md'})
            kinds = ('question', 'source_read') if self.measured else ('question',)
            return Run(status='completed', response='La nota dice amoxicillin.', measured_kinds=kinds)

    guessed = compare(CORPUS, baseline=FakeAnswer(0), candidate=FakeAnswer(0, measured=False))['cases'][0]
    assert guessed['baseline']['passed'] is False
    assert guessed['baseline']['oracle']['checks']['minimum_source_read_counts'] is False
    assert guessed['candidate']['passed'] is False
    assert guessed['candidate']['oracle']['checks']['minimum_source_read_counts'] is None
    # Extra reads may be inefficient, but do not invalidate a correct answer.
    repeated = compare(CORPUS, baseline=FakeAnswer(2), candidate=None)['cases'][0]
    assert repeated['baseline']['passed'] is True



def test_simple_loop_retains_denial_and_cancellation_cases():
    normal, denied, cancelled = compare(EXPANDED, baseline=None, candidate=SingleLoopFixtureAdapter())['cases']
    assert normal['candidate']['passed'] is True
    for row in (denied, cancelled):
        result = row['candidate']
        assert result['passed'] is True
        assert result['oracle']['checks']['forbidden_accesses'] is True
        assert result['oracle']['observed_effects'] == {}
        assert 'amoxicillin' not in result['response']
    assert denied['candidate']['reason'] == 'candidate_read_denied'
    assert cancelled['candidate']['reason'] == 'cancelled'


def test_heldout_source_and_restart_and_question_cases():
    row = compare(HOLDOUT, baseline=None, candidate=SingleLoopFixtureAdapter())['cases'][0]['candidate']
    assert row['passed'] is True and 'cetirizine' in row['response']
    row = compare(LIFECYCLE, baseline=None, candidate=SingleLoopFixtureAdapter())['cases'][0]['candidate']
    assert row['passed'] is True and row['reason'] == 'candidate_unresolved_inflight'
    assert row['oracle']['source_read_counts'] == {'Receta.md': 1}
    assert row['oracle']['restart_count'] == 1
    restart, question = compare(LIFECYCLE_V2, baseline=None, candidate=SingleLoopFixtureAdapter())['cases']
    assert restart['candidate']['passed'] is True
    assert question['candidate']['passed'] is True
    assert question['candidate']['reason'] == 'question_delivery_not_integrated'
