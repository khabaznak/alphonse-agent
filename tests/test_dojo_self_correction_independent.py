"""Fixture plumbing checks; these do not assess model behavior."""
import pytest

from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.evaluation import dojo_conversation_reliability as base
from alphonse.agent_v2.evaluation.dojo_self_correction_independent import CASES, run


@pytest.mark.parametrize('case_id', sorted(CASES))
def test_independent_case_records_manual_rubric_and_restores_catalog(case_id):
    class Provider:
        supports_image_input = False

        def generate_json(self, request):
            return InferenceResult(json_value={'kind': 'answer', 'tool_id': None,
                                              'arguments': None, 'text': 'Scripted answer.'},
                                   usage={'input_tokens': 100, 'output_tokens': 10})

    original = dict(base.CASES)
    router = InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'scripted', 'fixture'))
    result = run(router, case_id, timeout=10)
    assert result['status'] == 'ready'
    assert result['acknowledgement'] == 'client_acknowledged'
    assert result['passed'] is None and result['response_semantics'] == 'requires_manual_review'
    assert result['source_completeness'] == 'not independently observed'
    assert result['manual_rubric'] == list(CASES[case_id].rubric)
    assert result['decisions'][0]['answer_marker_visible_in_prompt'] is False
    assert base.CASES == original


def test_catalog_restored_on_error(monkeypatch):
    original = dict(base.CASES)
    def fail(*args, **kwargs):
        raise RuntimeError('synthetic infrastructure error')
    monkeypatch.setattr(base, 'run', fail)
    with pytest.raises(RuntimeError):
        run(None, 'portuguese-correction')
    assert base.CASES == original
