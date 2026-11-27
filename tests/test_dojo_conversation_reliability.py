"""Plumbing checks for independent synthetic recall cases, not model grading."""
import pytest

from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.evaluation.dojo_conversation_reliability import CASES, FILLER, run


@pytest.mark.parametrize('case_id', sorted(CASES))
def test_reliability_cases_keep_older_answers_out_of_initial_prompt(case_id):
    class Provider:
        supports_image_input = False
        def __init__(self):
            self.prompts = []
        def generate_json(self, request):
            self.prompts.append(request.prompt)
            return InferenceResult(json_value={'kind': 'answer', 'tool_id': None, 'arguments': None,
                                               'text': 'Scripted response.'},
                                   usage={'input_tokens': 100, 'output_tokens': 10})

    provider = Provider()
    router = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'scripted', 'fixture'))
    original = router.generate_json
    result = run(router, case_id, timeout=10)
    assert len(FILLER) > 16
    assert result['status'] == 'ready' and result['acknowledgement'] == 'client_acknowledged'
    assert result['decisions'][0]['answer_marker_visible_in_prompt'] is False
    assert all(marker not in provider.prompts[0] for marker in CASES[case_id].markers)
    assert CASES[case_id].recent[-1][1] in provider.prompts[0]
    assert result['passed'] is None and result['response_semantics'] == 'requires_manual_review'
    assert result['source_completeness'] == 'not independently observed'
    assert router.generate_json == original
