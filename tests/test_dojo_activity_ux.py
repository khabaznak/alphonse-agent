"""Plumbing checks only; do not grade the model or rendered experience."""
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.evaluation.dojo_activity_ux import run


def test_two_selected_steps_then_failure_recorded_without_ux_pass():
    class Provider:
        supports_image_input = False
        def generate_json(self, request):
            return InferenceResult(json_value={'kind': 'tool', 'tool_id': 'native.search_memory',
                'arguments': {'query': 'picnic', 'source_types': ['conversation']}, 'text': None},
                usage={'input_tokens': 100, 'output_tokens': 20})
    router = InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'scripted', 'fixture'))
    original = router.generate_json
    report = run(router, timeout=10)
    assert report['fault'] == 'injected_after_two_selected_tools'
    assert report['initial_ipc']['status'] == 'failed'
    assert report['live_work_log']['status'] == 'running'
    assert len(report['live_work_log']['events']) >= 2
    assert report['terminal_work_log']['status'] == 'blocked'
    assert isinstance(report['recovery']['items'], list)
    assert report['critic_verdict'] == 'not_assessed' and report['passed'] is None
    assert router.generate_json == original


def test_early_answer_does_not_claim_fault_or_success():
    class Provider:
        supports_image_input = False
        def generate_json(self, request):
            return InferenceResult(json_value={'kind': 'answer', 'tool_id': None,
                'arguments': None, 'text': 'I do not know.'}, usage={})
    router = InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'scripted', 'fixture'))
    report = run(router, timeout=10)
    assert report['fault'] == 'not_triggered'
    assert report['live_work_log'] is None
    assert report['critic_verdict'] == 'not_assessed' and report['passed'] is None
