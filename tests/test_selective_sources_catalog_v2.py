"""The catalog describes capabilities without exposing a case's oracle."""
from alphonse.agent_v2.core.inference import InferenceResult
from alphonse.agent_v2.evaluation import selective_sources_catalog_v2 as catalog
from alphonse.agent_v2.evaluation import selective_sources_variants as original
from alphonse.agent_v2.evaluation.codex_candidate_chooser import CodexSubscriptionChooser
from alphonse.agent_v2.evaluation.codex_subscription_provider import CodexSubscriptionProvider


def test_catalog_is_separate_from_original_and_discloses_no_fixture_values():
    assert catalog.CASE_IDS == original.CASE_IDS
    assert catalog.GOAL == original.GOAL
    assert catalog.SOURCE_DESCRIPTION != original.SOURCE_DESCRIPTION
    requests = []
    provider = CodexSubscriptionProvider(model='gpt-6-luna')
    def generate(request):
        requests.append(request)
        return InferenceResult(json_value={'kind': 'question', 'tool_id': None,
                                          'arguments': None, 'text': 'Need more information'})
    provider.generate_json = generate
    chooser = CodexSubscriptionChooser(provider=provider, model='gpt-6-luna',
        max_input_tokens=65536, emit=lambda _: None,
        allowed_source_names=catalog.ALLOWED_SOURCES, source_description=catalog.SOURCE_DESCRIPTION)
    chooser(catalog.GOAL, 1024)
    prompt = requests[0].prompt
    assert catalog.SOURCE_DESCRIPTION in prompt
    for hidden in ('rx-41', 'rx-72', 'Cedar Pharmacy', 'Maple Pharmacy',
                   'required_sources', 'expected_answer', 'resolution_source'):
        assert hidden not in prompt
