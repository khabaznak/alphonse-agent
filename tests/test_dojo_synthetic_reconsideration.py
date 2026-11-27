"""The extra decision is confined to a disposable owner worker."""
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.simple_loop import owner_read_processor
from alphonse.agent_v2.evaluation import dojo_synthetic_reconsideration as exercise
import pytest


class ScriptedProvider:
    supports_image_input = False

    def __init__(self):
        self.calls = 0

    def generate_json(self, request):
        self.calls += 1
        text = 'First proposed answer.' if self.calls == 1 else 'Second proposed answer.'
        return InferenceResult(json_value={'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': text},
                               usage={'input_tokens': 100, 'output_tokens': 20})


def test_reconsideration_is_one_extra_decision_and_restores_factory():
    provider = ScriptedProvider()
    router = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'scripted', 'fixture'))
    original_factory = owner_read_processor.CandidateHost
    result = exercise.run(router, 'english-consistent-control', timeout=10)
    assert result['status'] == 'ready'
    assert result['answer'] == 'Second proposed answer.'
    assert provider.calls == 2
    assert result['experimental_reconsideration'] == 'synthetic_only'
    assert result['passed'] is None
    assert owner_read_processor.CandidateHost is original_factory


def test_fixture_error_still_restores_factory(monkeypatch):
    original_factory = owner_read_processor.CandidateHost
    def fail(*_args, **_kwargs):
        raise RuntimeError('synthetic fixture failure')
    monkeypatch.setattr(exercise.base, 'run', fail)
    with pytest.raises(RuntimeError, match='synthetic fixture failure'):
        exercise.run(None, 'english-consistent-control')
    assert owner_read_processor.CandidateHost is original_factory
