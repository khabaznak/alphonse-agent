"""Synthetic processor wiring, not production-effect qualification."""
from alphonse.agent_v2.evaluation.mutation_development import compare_mutation


def test_seeded_v3_replays_and_candidate_cannot_write():
    report = compare_mutation()
    assert report['decision_qualified'] is False
    assert report['delivery'] == 'unassessed'
    for name in ('current_v3', 'targeted_v3_projection'):
        result = report['arms'][name]
        assert result['availability'] == 'available'
        assert result['passed'] is True
        assert result['oracle']['observed_effects'] == {'records/item.md': 'Status: complete\n'}
        assert result['oracle']['restart_count'] == 1
        assert result['mutation_attempts'] == 1
        assert not any(e.get('tool') == 'fixture.search' for e in result['events'])
        assert result['metrics']['provider_input_tokens'] is None
    candidate = report['arms']['single_loop']
    assert candidate['availability'] == 'unavailable'
    assert candidate['passed'] is None
    assert candidate['safety_probe'] == {
        'passed': True, 'processor_status': 'blocked',
        'reason': 'candidate_requires_read_only_native_tool', 'edit_calls': 0,
        'independent_file_unchanged': True,
    }


def test_worker_failure_is_unknown_not_a_pass(monkeypatch):
    import subprocess
    from alphonse.agent_v2.evaluation.mutation_development import _v3_arm
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired('worker', 30)
    monkeypatch.setattr(subprocess, 'run', timeout)
    arm = _v3_arm(False)
    assert arm['availability'] == 'error'
    assert arm['passed'] is None
    assert arm['events'] is None
    assert arm['oracle'] is None


def test_duplicate_or_unobserved_edit_attempts_do_not_pass():
    from alphonse.agent_v2.evaluation.mutation_development import _evaluate_v3_attempts
    event = {'kind': 'tool', 'tool': 'native.exact_text_edit'}
    assert _evaluate_v3_attempts({'passed': True, 'events': [event, event]})['passed'] is False
    assert _evaluate_v3_attempts({'passed': True, 'events': None})['passed'] is False
