"""Scripted IPC plumbing; semantic diagnostic quality is unassessed."""
import pytest

from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.evaluation.dojo_failure_diagnosis import run


@pytest.mark.parametrize('case,blocker', [('elapsed', 'BudgetExhausted:budget_elapsed_exhausted'),
                                          ('model_failure', 'candidate_model_unavailable')])
def test_injected_failure_and_followup_are_observed_without_semantic_pass(case, blocker):
    class Provider:
        supports_image_input = False
        def generate_json(self, request):
            return InferenceResult(json_value={'kind': 'answer', 'tool_id': None, 'arguments': None,
                                              'text': 'I cannot determine that from the available information.'},
                                   usage={'input_tokens': 100, 'output_tokens': 20})
    router = InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'scripted', 'fixture'))
    original = router.generate_json
    report = run(router, case, timeout=10)
    assert report['initial_blocker'] == blocker
    assert report['initial_ipc']['status'] == 'failed'
    assert report['initial_ipc']['failure_code']
    assert report['followup_ipc']['status'] == 'ready'
    assert report['acknowledgement'] == 'client_acknowledged'
    assert report['passed'] is None and report['response_semantics'] == 'requires_manual_review'
    assert router.generate_json == original
