"""Offline adapter contract, not evidence of autonomous clinical reasoning."""
import json
from pathlib import Path

from alphonse.agent_v2.evaluation.prescription_candidate_adapter import PrescriptionCandidateAdapter
from alphonse.agent_v2.evaluation.simplified_loop import _run, load_frozen

CORPUS = Path(__file__).parent / 'fixtures/single_loop_comparison/prescription_schedule_development_v1.json'
PATHS = ['household/index.md', 'household/prescription.md', 'household/treatment-start.md']


def scripted(answer, observed):
    actions = iter([*({'kind': 'tool', 'tool_id': 'native.fixture_read',
                       'arguments': {'path': path}} for path in PATHS),
                    {'kind': 'answer', 'text': json.dumps({'due_medications': answer})}])
    def chooser(text):
        observed.append(text)
        return next(actions)
    return chooser


def test_candidate_executes_fixture_sources_and_independent_oracle():
    corpus, _ = load_frozen(CORPUS)
    for case, expected in zip(corpus['cases'], [['Asterin'], ['Asterin'], []]):
        observed = []
        result = _run(case, PrescriptionCandidateAdapter(scripted(expected, observed)))
        assert result['status'] == 'completed' and result['passed'] is True
        assert result['oracle']['source_read_counts'] == {path: 1 for path in PATHS}
        assert len(observed) == 4
        assert 'Asterin' not in observed[0]
        assert 'Asterin' in observed[-1] and '2026-04-10' in observed[-1]
        assert 'medication_schedule' not in ''.join(observed)
        assert result['metrics']['provider_input_tokens'] is None


def test_wrong_scripted_answer_does_not_pass_independent_oracle():
    corpus, _ = load_frozen(CORPUS)
    result = _run(corpus['cases'][0], PrescriptionCandidateAdapter(scripted(['Borelin'], [])))
    assert result['status'] == 'completed'
    assert result['passed'] is False
    assert result['oracle']['checks']['medication_schedule'] is False


def test_unread_source_prevents_completion_and_path_traversal_is_denied():
    corpus, _ = load_frozen(CORPUS)
    result = _run(corpus['cases'][0], PrescriptionCandidateAdapter(
        lambda text: {'kind': 'answer', 'text': '{"due_medications": ["Asterin"]}'}))
    assert result['status'] == 'blocked' and result['reason'] == 'candidate_answer_unsupported'
    result = _run(corpus['cases'][0], PrescriptionCandidateAdapter(
        lambda text: {'kind': 'tool', 'tool_id': 'native.fixture_read',
                      'arguments': {'path': '../outside'}}))
    assert result['status'] == 'blocked' and result['reason'] == 'candidate_read_denied'
    assert not any(event.get('kind') == 'source_read' for event in result['events'])


def test_unrecognized_case_is_unavailable():
    corpus, _ = load_frozen(CORPUS)
    case = dict(corpus['cases'][0], id='unknown')
    result = _run(case, PrescriptionCandidateAdapter(lambda text: None))
    assert result['availability'] == 'unavailable' and result['passed'] is None
