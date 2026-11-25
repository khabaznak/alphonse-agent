"""Exercise the historical-correction harness with a scripted provider."""
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.evaluation.dojo_conversation_context import run_historical_correction


class Provider:
    supports_image_input = False

    def generate_json(self, request):
        decision = {'kind': 'answer', 'tool_id': None, 'arguments': None,
                    'text': 'Yes, Harbor Blue. My earlier recap was mistaken.'}
        return InferenceResult(json_value=decision, usage={'input_tokens': 100, 'output_tokens': 20})


def test_historical_correction_records_delivery_and_does_not_claim_semantic_pass():
    router = InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'scripted', 'fixture'))
    original = router.generate_json
    result = run_historical_correction(router, timeout=10)
    assert result['status'] == 'ready'
    assert result['expected_fact_mentioned'] is True
    assert result['acknowledgement'] == 'client_acknowledged'
    assert result['passed'] is None and result['response_semantics'] == 'requires_manual_review'
    assert result['source_completeness'] == 'not independently observed'
    assert result['decisions'][0]['usage']['input_tokens'] == 100
    assert router.generate_json == original


def test_provider_failure_is_retained_without_a_success_claim():
    class FailingProvider:
        supports_image_input = False

        def generate_json(self, request):
            raise ValueError('synthetic failure')

    router = InferenceRouter(provider=FailingProvider(), default_profile=ModelProfile('test', 'scripted', 'fixture'))
    result = run_historical_correction(router, timeout=10)
    assert result['status'] != 'ready'
    assert result['failure_code'] is not None
    assert result['answer'] is None and result['acknowledgement'] is None
    assert result['passed'] is None
    assert result['decisions'] == [{'status': 'provider_error'}]


def test_owner_change_variant_preserves_later_owner_choice_without_claiming_semantic_pass():
    class ChangedProvider:
        supports_image_input = False

        def generate_json(self, request):
            decision = {'kind': 'answer', 'tool_id': None, 'arguments': None,
                        'text': 'You later changed it to Moss Green in a satin finish; my recap was wrong.'}
            return InferenceResult(json_value=decision, usage={'input_tokens': 100, 'output_tokens': 20})

    router = InferenceRouter(provider=ChangedProvider(), default_profile=ModelProfile('test', 'scripted', 'fixture'))
    result = run_historical_correction(router, timeout=10, owner_changed_choice=True)
    assert result['case'] == 'historical-owner-change'
    assert result['status'] == 'ready' and result['acknowledgement'] == 'client_acknowledged'
    assert result['expected_fact_mentioned'] is True
    assert result['decisions'][0]['historical_fact_visible_in_prompt'] is True
    assert result['passed'] is None and result['response_semantics'] == 'requires_manual_review'
