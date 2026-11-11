"""The comparison reports missing variants and preserves every failed trial."""
from alphonse.agent_v2.evaluation import planning_comparison


def test_frozen_comparison_runs_every_variant_and_preserves_failure(monkeypatch):
    calls = []
    def run(case, variant):
        calls.append((case['id'], variant))
        return {'case_id': case['id'], 'variant': variant,
                'independent_oracle_pass': case['id'] != 'changed-form'}
    monkeypatch.setattr(planning_comparison, '_run_case', run)
    monkeypatch.setattr(planning_comparison, '_run_tactical_case', run)
    monkeypatch.setattr(planning_comparison, '_run_independent_case', run)
    report = planning_comparison.run_comparison()
    assert len(calls) == 13 * 6 * 2
    assert len(report['results']) == len(calls)
    assert sum(not row['independent_oracle_pass'] for row in report['results']) == 12
    assert 'flat_phase_prototype' in report['prototype_limitations']
    assert report['mode'] == 'scripted-models-real-processor'
    assert not report['comparison_complete']  # Missing measurements cannot qualify a run.
    assert len(report['corpus_sha256']) == 64


def test_scripted_uncertain_fragments_expose_false_stall_without_extra_composition():
    case = {'id': 'fragments', 'split': 'development', 'fragments': ['one', 'two', 'three', 'four']}
    legacy = planning_comparison._run_tactical_case(case, 'historical_progress_emulation')
    aware = planning_comparison._run_tactical_case(case, 'uncertainty_aware')
    targeted = planning_comparison._run_tactical_case(case, 'targeted_recovery')
    assert legacy['false_stall'] and not legacy['independent_oracle_pass']
    assert aware['independent_oracle_pass'] and targeted['independent_oracle_pass']
    assert aware['inference_calls'] == targeted['inference_calls'] == 4
    assert targeted['targeted_recovery_prompted'] and not aware['targeted_recovery_prompted']


def test_flat_prototype_retains_verified_completion_and_missing_input_failure():
    import json
    cases = {case['id']: case for case in json.loads(planning_comparison.fixtures.CORPUS.read_text())['cases']}
    success = planning_comparison._run_case(cases['unfamiliar-form'], 'flat_phase_prototype')
    missing = planning_comparison._run_case(cases['essential-input'], 'flat_phase_prototype')
    assert success['independent_oracle_pass'] and success['verified_completion']
    assert success['submission_attempts'] == 1
    assert not missing['independent_oracle_pass']
    assert missing['outcome'] == 'waiting_user'
    assert missing['submission_attempts'] == 0


def test_completed_comparison_retains_failures_without_treating_them_as_missing(monkeypatch):
    def run(case, variant):
        return {'case_id': case['id'], 'variant': variant, 'outcome': 'blocked',
                'independent_oracle_pass': False, 'false_stall': True, 'repeated_observations': 0,
                'unnecessary_questions': 0, 'inference_calls': 1, 'tool_calls': 1,
                'provider_cost_usd': 0, 'elapsed_seconds': .01}
    monkeypatch.setattr(planning_comparison, '_run_case', run)
    monkeypatch.setattr(planning_comparison, '_run_tactical_case', run)
    monkeypatch.setattr(planning_comparison, '_run_independent_case', run)
    result = planning_comparison.run_comparison()
    assert result['comparison_complete']
    assert len(result['results']) == 156
    assert all(not row['independent_oracle_pass'] for row in result['results'])
    assert 'Retain the hierarchical loop' in result['architecture_decision']


def test_scripted_check_uses_evidence_without_accessing_fixture_oracle():
    from types import SimpleNamespace
    from alphonse.agent_v2.evaluation.planning_comparison import _use_evidence_only_review
    def forbidden():
        raise AssertionError('fixture oracle must not be consulted by scripted Check')
    system_one = SimpleNamespace()
    trial = SimpleNamespace(context=SimpleNamespace(system_one=system_one), oracle_effect=forbidden)
    _use_evidence_only_review(trial)
    result = system_one.evaluate(contract={'criteria': [{'id': 'ac-1'}]}, evidence={'entries': [
        {'evidence_ref': 'readback', 'status': 'success', 'result': {'verification': {'status': 'verified'}}}]})
    assert result.updates[0]['evidence_refs'] == ['readback']


def test_architecture_decision_does_not_invent_flat_prototype_failures(monkeypatch):
    def run(case, variant):
        return {'case_id': case['id'], 'variant': variant, 'outcome': 'complete',
                'independent_oracle_pass': True, 'false_stall': False, 'repeated_observations': 0,
                'unnecessary_questions': 0, 'inference_calls': 1, 'tool_calls': 1,
                'provider_cost_usd': 0, 'elapsed_seconds': .01}
    monkeypatch.setattr(planning_comparison, '_run_case', run)
    monkeypatch.setattr(planning_comparison, '_run_tactical_case', run)
    monkeypatch.setattr(planning_comparison, '_run_independent_case', run)
    result = planning_comparison.run_comparison()
    assert result['comparison_complete']
    assert 'no flat failures' in result['architecture_decision']
    assert 'pending' in result['architecture_decision']
