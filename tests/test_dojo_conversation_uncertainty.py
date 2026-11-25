"""Plumbing checks for disposable observation and missing-record cases."""
import pytest

from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.evaluation import dojo_conversation_reliability as base
from alphonse.agent_v2.evaluation.dojo_conversation_uncertainty import CASES, run


@pytest.mark.parametrize('case_id', sorted(CASES))
def test_countercases_retain_manual_review_and_restore_catalog(case_id):
    class Provider:
        supports_image_input = False

        def generate_json(self, request):
            return InferenceResult(json_value={'kind': 'answer', 'tool_id': None,
                                              'arguments': None, 'text': 'Scripted answer.'},
                                   usage={'input_tokens': 100, 'output_tokens': 10})

    original_cases = dict(base.CASES)
    router = InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'scripted', 'fixture'))
    report = run(router, case_id, timeout=10)
    assert report['status'] == 'ready' and report['acknowledgement'] == 'client_acknowledged'
    assert report['passed'] is None and report['response_semantics'] == 'requires_manual_review'
    assert report['source_completeness'] == 'not independently observed'
    assert report['manual_rubric'] == list(CASES[case_id].rubric)
    assert base.CASES == original_cases


def test_catalog_is_restored_when_harness_raises(monkeypatch):
    original_cases = dict(base.CASES)

    def fail(*args, **kwargs):
        raise RuntimeError('synthetic infrastructure error')

    monkeypatch.setattr(base, 'run', fail)
    with pytest.raises(RuntimeError):
        run(None, 'missing-historical-record')
    assert base.CASES == original_cases
