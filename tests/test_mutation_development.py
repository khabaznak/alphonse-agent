"""Synthetic processor wiring, not production-effect qualification."""
from alphonse.agent_v2.evaluation.mutation_development import compare_mutation


def test_candidate_cannot_write():
    report = compare_mutation()
    assert report['decision_qualified'] is False
    assert report['delivery'] == 'unassessed'
    candidate = report['arms']['single_loop']
    assert candidate['availability'] == 'unavailable'
    assert candidate['passed'] is None
    assert candidate['safety_probe'] == {
        'passed': True, 'processor_status': 'blocked',
        'reason': 'candidate_requires_read_only_native_tool', 'edit_calls': 0,
        'independent_file_unchanged': True,
    }
