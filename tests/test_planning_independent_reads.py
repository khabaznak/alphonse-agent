import pytest

from alphonse.agent_v2.evaluation.planning_independent_reads import report, run_case


@pytest.mark.parametrize('scenario', ['normal', 'uncertain', 'revoked', 'cancelled'])
@pytest.mark.parametrize('sequence', [False, True])
def test_both_shapes_preserve_independent_outcomes(scenario, sequence):
    assert run_case(scenario, sequence)['oracle_pass']


def test_independent_sequence_reduces_only_unambiguous_composition_calls():
    baseline = run_case('normal', False)
    simplified = run_case('normal', True)
    assert baseline['observed_keys'] == simplified['observed_keys'] == ['a', 'b', 'c']
    assert baseline['composition_calls'] == 3
    assert simplified['composition_calls'] == 1
    assert baseline['tool_calls'] == simplified['tool_calls'] == 3
    uncertain = run_case('uncertain', True)
    assert uncertain['oracle_pass'] and uncertain['composition_calls'] == 3


def test_report_retains_all_trials_and_failure_flags(monkeypatch):
    def failed(scenario, sequence):
        return {'scenario': scenario, 'variant': str(sequence), 'oracle_pass': scenario != 'revoked'}
    monkeypatch.setattr('alphonse.agent_v2.evaluation.planning_independent_reads.run_case', failed)
    data = report(repeats=2)
    assert len(data['results']) == 16
    assert sum(not row['oracle_pass'] for row in data['results']) == 4
