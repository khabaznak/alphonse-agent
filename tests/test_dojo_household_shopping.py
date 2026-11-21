import hashlib
import json

from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.evaluation.dojo_household_shopping import FILES, fixture_bytes, grade, run


def readback(output):
    return {'status': 'observed', 'output': output,
            'source_hashes': {name: hashlib.sha256(fixture_bytes(value)).hexdigest()
                              for name, value in FILES.items()}}


def test_shopping_oracle_checks_quantities_and_preserves_sources():
    good = readback({'rice': 2, 'tomatoes': 3})
    assert grade(good)['effect_passed'] is True
    assert grade(readback({'rice': 2, 'tomatoes': 3, 'beans': 0}))['effect_passed'] is False
    assert grade(readback({'rice': True, 'tomatoes': 3}))['effect_passed'] is False
    good['source_hashes']['pantry.json'] = 'changed'
    assert grade(good)['checks']['source_files_unchanged'] is False
    assert grade({'status': 'unavailable'})['effect_passed'] is None


def test_wrong_path_is_diagnosed_without_passing_the_requested_effect():
    observed = readback(None)
    observed.update(target_status='missing', alternative_scan_complete=True,
                    other_json_files={'/workspace/elsewhere/list.json': {'rice': 2, 'tomatoes': 3}})
    outcome = grade(observed)
    assert outcome['effect_passed'] is False
    assert outcome['requested_target_status'] == 'missing'
    assert outcome['matching_alternative_paths'] == ['/workspace/elsewhere/list.json']
    assert outcome['alternative_scan_complete'] is True


def test_harness_does_not_treat_claimed_success_as_file_effect(monkeypatch):
    class Shell:
        cleanup_verified = None
        def __init__(self, **_): pass
        def __enter__(self): return self
        def __exit__(self, *_): self.cleanup_verified = True
        def execute(self, _command, **_kwargs):
            return {'status': 'completed', 'exit_code': 0, 'stdout': '', 'stderr': '', 'truncated': False}
    class Provider:
        supports_image_input = False
        def generate_json(self, request):
            value = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Done.'}
            return InferenceResult(json_value=value, raw_response=json.dumps(value),
                                   usage={'input_tokens': 100, 'output_tokens': 20})
    monkeypatch.setattr('alphonse.agent_v2.evaluation.dojo_household_shopping.DojoBash', Shell)
    monkeypatch.setattr('alphonse.agent_v2.evaluation.dojo_household_shopping._seed', lambda _: None)
    monkeypatch.setattr('alphonse.agent_v2.evaluation.dojo_household_shopping._readback', lambda _: readback(None))
    router = InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'fixture', 'offline'))
    result = run(image_id='sha256:' + 'a' * 64, socket_path='/tmp/fixture', router=router, emit=lambda _: None)
    assert result['status'] == 'completed'
    assert result['effect_passed'] is False
    assert result['task_passed'] is None and result['response_semantics'] == 'unassessed'


def test_blocked_candidate_retains_observed_partial_effect(monkeypatch):
    class Shell:
        cleanup_verified = None
        def __init__(self, **_): pass
        def __enter__(self): return self
        def __exit__(self, *_): self.cleanup_verified = True
        def execute(self, _command, **_kwargs):
            return {'status': 'completed', 'exit_code': 0, 'stdout': '', 'stderr': '', 'truncated': False}
    class Provider:
        supports_image_input = False
        def __init__(self): self.calls = 0
        def generate_json(self, request):
            self.calls += 1
            if self.calls > 1:
                raise ValueError('offline')
            value = {'kind': 'tool', 'tool_id': 'native.bash', 'arguments': {'command': 'fixture'}, 'text': None}
            return InferenceResult(json_value=value, raw_response=json.dumps(value),
                                   usage={'input_tokens': 100, 'output_tokens': 20})
    monkeypatch.setattr('alphonse.agent_v2.evaluation.dojo_household_shopping.DojoBash', Shell)
    monkeypatch.setattr('alphonse.agent_v2.evaluation.dojo_household_shopping._seed', lambda _: None)
    monkeypatch.setattr('alphonse.agent_v2.evaluation.dojo_household_shopping._readback', lambda _: readback({'rice': 99}))
    router = InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'fixture', 'offline'))
    result = run(image_id='sha256:' + 'a' * 64, socket_path='/tmp/fixture', router=router, emit=lambda _: None)
    assert result['status'] == 'blocked'
    assert result['observed_readback']['output'] == {'rice': 99}
    assert result['effect_passed'] is False
