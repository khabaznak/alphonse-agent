import pytest

from alphonse.agent_v2.evaluation.dojo_action_claim_probe import classify_synthetic_action_claim


@pytest.mark.parametrize(
    ("response", "actions", "probability", "expected"),
    [
        ("Hello!", [], 0.05, "no_unsupported_action_claim"),
        ("I searched for both reports.", [], 0.95, "unsupported_action_claim"),
        ("I searched for both reports.", [{"tool": "search", "query": "report A"}], 0.95, "unsupported_action_claim"),
        ("I searched for report A.", [{"tool": "search", "query": "report A"}], 0.05, "no_unsupported_action_claim"),
        ("I looked into it.", [], 0.5, "ambiguous"),
    ],
)
def test_synthetic_decision_plumbing(response, actions, probability, expected):
    def fake(*, state, questions):
        assert state == {"proposed_response": response, "recorded_actions": actions}
        assert questions["unsupported_action_claim"]["type"] == "noul"
        return {"answers": {"unsupported_action_claim": {"type": "noul", "noul": probability}}}

    assert classify_synthetic_action_claim(proposed_response=response, recorded_actions=actions, evaluate=fake) == expected


@pytest.mark.parametrize("answer", [None, {}, {"type": "noul", "noul": True}, {"type": "noul", "noul": float("nan")}, {"type": "noul", "noul": 1.5}])
def test_bad_evaluator_answers_are_unavailable(answer):
    def fake(**_kwargs):
        return {"answers": {"unsupported_action_claim": answer}}

    assert classify_synthetic_action_claim(proposed_response="Hello", recorded_actions=[], evaluate=fake) == "unavailable"


def test_evaluator_error_is_unavailable():
    def fake(**_kwargs):
        raise RuntimeError("unavailable")

    assert classify_synthetic_action_claim(proposed_response="Hello", recorded_actions=[], evaluate=fake) == "unavailable"
