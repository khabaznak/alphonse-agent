"""Exercise a unified recent and historical context through the owner worker."""
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.evaluation.dojo_conversation_reliability import run


def test_recent_confirmation_reaches_search_timeline():
    class Provider:
        supports_image_input = False

        def __init__(self):
            self.prompts = []

        def generate_json(self, request):
            self.prompts.append(request.prompt)
            decision = ({'kind': 'tool', 'tool_id': 'native.search_memory',
                         'arguments': {'query': 'dessert pique-nique', 'source_types': ['conversation']}, 'text': None}
                        if len(self.prompts) == 1 else
                        {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic answer.'})
            return InferenceResult(json_value=decision, usage={'input_tokens': 100, 'output_tokens': 20})

    provider = Provider()
    router = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'scripted', 'fixture'))
    result = run(router, 'french-recap', timeout=10)
    assert result['status'] == 'ready'
    assert '"earlier_turns_omitted": true' in provider.prompts[0]
    assert 'tartes aux pommes' not in provider.prompts[0]
    prompt = provider.prompts[1]
    search_start = prompt.index('"results": [')
    timeline = prompt[search_start:]
    assert 'tartes aux pommes' in timeline
    assert 'Je confirme que notre choix' in timeline
    assert '"origin": "recent_context"' in timeline
    assert 'not the complete conversation' in prompt


def test_multiple_searches_include_recent_unicode_context_only_once(monkeypatch):
    from alphonse.agent_v2.evaluation import dojo_conversation_reliability as fixture

    monkeypatch.setattr(fixture, 'FILLER', tuple(
        ('user' if index % 2 == 0 else 'assistant', f'日常の記録 {index}: ' + '家' * 280)
        for index in range(18)))

    class Provider:
        supports_image_input = False

        def __init__(self):
            self.prompts = []

        def generate_json(self, request):
            self.prompts.append(request.prompt)
            queries = ('dessert', 'pique', 'tartes', 'pommes')
            index = len(self.prompts) - 1
            decision = ({'kind': 'tool', 'tool_id': 'native.search_memory',
                         'arguments': {'query': queries[index], 'source_types': ['conversation']}, 'text': None}
                        if index < len(queries) else
                        {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic answer.'})
            return InferenceResult(json_value=decision, usage={'input_tokens': 100, 'output_tokens': 20})

    provider = Provider()
    router = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'scripted', 'fixture'))
    result = fixture.run(router, 'french-recap', timeout=15)
    assert result['status'] == 'ready', result.get('failure_code')
    assert len(provider.prompts) == 5
    assert provider.prompts[-1].count('"origin": "recent_context"') <= 16
    assert len(provider.prompts[-1].encode('utf-8')) < 65536
