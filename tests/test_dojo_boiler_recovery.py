"""Disposable harness plumbing only; no semantic model-performance claim."""
import pytest

from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.simple_loop import owner_read_processor
from alphonse.agent_v2.evaluation.dojo_boiler_recovery import CASES, FRESH_CASE, MODEL, SYNTHETIC_URL, run


@pytest.mark.parametrize('case', CASES)
def test_injected_verification_failure_followed_by_separate_owner_turn(case):
    class Provider:
        supports_image_input = False
        def __init__(self):
            self.prompts = []
        def generate_json(self, request):
            self.prompts.append(request.prompt)
            return InferenceResult(json_value={'kind': 'answer', 'tool_id': None,
                'arguments': None, 'text': 'I cannot determine the next step from this information.'},
                usage={'input_tokens': 100, 'output_tokens': 20})

    provider = Provider()
    router = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'scripted', 'fixture'))
    original_factory = owner_read_processor.CandidateHost
    original_generate = router.generate_json
    report = run(router, case, timeout=10)
    assert report['initial_status']['status'] == 'failed'
    assert report['initial_status']['failure_code'] == 'answer_unverified'
    assert report['followup_status']['status'] == 'ready'
    assert report['acknowledgement'] == 'client_acknowledged'
    assert report['passed'] is None and report['response_semantics'] == 'requires_manual_review'
    assert isinstance(report['recovery']['items'], list)
    assert ('Northstar H10' in provider.prompts[0]) == (case == 'available-evidence')
    assert owner_read_processor.CandidateHost is original_factory
    assert router.generate_json == original_generate


@pytest.mark.parametrize('case', CASES)
def test_same_request_reconsiders_after_one_injected_rejection(case):
    class Provider:
        supports_image_input = False
        def generate_json(self, request):
            return InferenceResult(json_value={'kind': 'answer', 'tool_id': None,
                'arguments': None, 'text': 'I cannot determine the next step from this information.'},
                usage={'input_tokens': 100, 'output_tokens': 20})

    router = InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'scripted', 'fixture'))
    report = run(router, case, timeout=10, mode='same-request')
    assert report['initial_status']['status'] == 'ready'
    assert report['followup_status']['status'] == 'not_applicable'
    assert report['verification_attempts'] == ['injected_rejection', 'accepted']
    assert [entry['reconsideration_feedback_visible'] for entry in report['calls']] == [False, True]
    assert report['acknowledgement'] == 'client_acknowledged'
    assert report['passed'] is None


def test_simulated_clarification_review_records_question_and_does_not_claim_safety():
    class Provider:
        supports_image_input = False
        def __init__(self):
            self.calls = 0
        def generate_json(self, request):
            self.calls += 1
            decision = ({'kind': 'answer', 'text': 'I cannot determine the next step.'}
                        if self.calls == 1 else
                        {'kind': 'question', 'text': 'What control options are available?'})
            return InferenceResult(json_value={**decision, 'tool_id': None, 'arguments': None},
                usage={'input_tokens': 100, 'output_tokens': 20})
    router = InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'scripted', 'fixture'))
    report = run(router, 'available-evidence', timeout=10, mode='same-request',
                 simulated_clarification_review=True)
    assert report['clarification_review'] == 'simulated_permissive_noop'
    assert report['clarification_reviews'] == [{'question': 'What control options are available?', 'allowed': True}]
    assert report['calls'][-1]['proposed_question'] == 'What control options are available?'
    assert report['answer_kind'] == 'question' and report['initial_status']['status'] == 'ready'
    assert report['passed'] is None


@pytest.mark.parametrize('case', CASES)
def test_offline_read_only_web_evidence_and_unavailable_control(case):
    class Provider:
        supports_image_input = False
        def __init__(self):
            self.calls = 0
            self.prompts = []
        def generate_json(self, request):
            self.calls += 1
            self.prompts.append(request.prompt)
            if self.calls == 2:
                decision = {'kind': 'tool', 'tool_id': 'native.web_search',
                            'arguments': {'query': 'heater model'}, 'text': None}
            else:
                decision = {'kind': 'answer', 'tool_id': None, 'arguments': None,
                            'text': 'I cannot establish the device from the available information.'}
            return InferenceResult(json_value=decision, usage={'input_tokens': 100, 'output_tokens': 20})
    provider = Provider()
    router = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'scripted', 'fixture'))
    report = run(router, case, timeout=10, mode='same-request', evidence_fixture=True)
    assert report['initial_status']['status'] == 'ready'
    assert report['web_fixture_calls'] == [{'tool': 'native.web_search', 'query': 'heater model',
        'fixture': 'available' if case == 'available-evidence' else 'unavailable'}]
    assert ('Inventory snapshot:' in provider.prompts[-1]) == (case == 'available-evidence')
    assert 'Northstar H10' in provider.prompts[0]
    if case == 'missing-evidence':
        assert 'searxng_http_error' in provider.prompts[-1]
    assert report['passed'] is None


def test_offline_fetch_observes_only_disposable_fixture():
    class Provider:
        supports_image_input = False
        def __init__(self):
            self.calls = 0
            self.prompts = []
        def generate_json(self, request):
            self.calls += 1
            self.prompts.append(request.prompt)
            if self.calls == 2:
                decision = {'kind': 'tool', 'tool_id': 'native.web_fetch',
                            'arguments': {'url': SYNTHETIC_URL}, 'text': None}
            else:
                decision = {'kind': 'answer', 'tool_id': None, 'arguments': None,
                            'text': 'I cannot establish the device from the available information.'}
            return InferenceResult(json_value=decision, usage={'input_tokens': 100, 'output_tokens': 20})
    provider = Provider()
    router = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'scripted', 'fixture'))
    report = run(router, 'available-evidence', timeout=10, mode='same-request', evidence_fixture=True)
    assert report['web_fixture_calls'] == [{'tool': 'native.web_fetch', 'url': SYNTHETIC_URL, 'fixture': 'available'}]
    assert 'Inventory snapshot:' in provider.prompts[-1]
    assert report['initial_status']['status'] == 'ready'
    assert report['passed'] is None


def test_irrigation_evidence_is_available_but_not_in_seeded_conversation():
    class Provider:
        supports_image_input = False
        def __init__(self):
            self.prompts = []
        def generate_json(self, request):
            self.prompts.append(request.prompt)
            if len(self.prompts) == 2:
                decision = {'kind': 'tool', 'tool_id': 'native.web_search',
                            'arguments': {'query': 'controller guide'}, 'text': None}
            else:
                decision = {'kind': 'answer', 'tool_id': None, 'arguments': None,
                            'text': 'I cannot determine when watering will resume.'}
            return InferenceResult(json_value=decision, usage={'input_tokens': 100, 'output_tokens': 20})
    provider = Provider()
    router = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'scripted', 'fixture'))
    report = run(router, FRESH_CASE, timeout=10, mode='same-request', evidence_fixture=True)
    assert report['initial_status']['status'] == 'ready'
    assert report['web_fixture_calls'][0]['tool'] == 'native.web_search'
    assert 'View Schedule screen' not in provider.prompts[0]
    assert 'View Schedule screen' in provider.prompts[-1]
    assert report['manual_rubric'][0].startswith('Carry forward the reported active rain delay')
    assert report['passed'] is None
