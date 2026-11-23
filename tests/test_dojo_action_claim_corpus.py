from alphonse.agent_v2.evaluation.dojo_action_claim_corpus import CASES, evaluate_corpus


def test_scorecard_keeps_expected_labels_out_of_evaluator_input():
    seen = []

    def fake_evaluate(*, state, questions):
        assert set(state) == {'proposed_response', 'recorded_actions'}
        assert 'unsupported_action_claim' in questions
        seen.append(state)
        return {'answers': {'unsupported_action_claim': {'type': 'noul', 'noul': 0.1}}}

    report = evaluate_corpus(fake_evaluate)
    assert len(seen) == len(CASES)
    assert report['counts']['false_negative'] == sum(case.unsupported for case in CASES)
    assert report['counts']['true_negative'] == sum(not case.unsupported for case in CASES)
    assert sum(report['counts'].values()) == len(CASES)


def test_scorecard_preserves_ambiguous_and_unavailable_outcomes():
    calls = 0

    def fake_evaluate(*, state, questions):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError('provider unavailable')
        return {'answers': {'unsupported_action_claim': {'type': 'noul', 'noul': 0.5}}}

    report = evaluate_corpus(fake_evaluate)
    assert report['counts']['unavailable'] == 1
    assert report['counts']['ambiguous'] == len(CASES) - 1
    assert report['cases'][0]['bucket'] == 'unavailable'


def test_corpus_contains_both_outcomes_and_unique_case_ids():
    assert {case.unsupported for case in CASES} == {True, False}
    assert len({case.case_id for case in CASES}) == len(CASES)
