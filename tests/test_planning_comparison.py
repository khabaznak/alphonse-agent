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
    report = planning_comparison.run_comparison()
    assert len(calls) == 9 * 4 * 2
    assert len(report['results']) == len(calls)
    assert sum(not row['independent_oracle_pass'] for row in report['results']) == 8
    assert 'simpler_loop' in report['unavailable_variants']
    assert report['mode'] == 'scripted-models-real-processor'
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
