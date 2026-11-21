import json

from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.evaluation.dojo_bash_candidate import run


def test_candidate_uses_disposable_bash_and_independent_answer_check(monkeypatch):
    calls = []
    class Shell:
        cleanup_verified = None
        def __init__(self, **_kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *_args): self.cleanup_verified = True
        def execute(self, command, **_kwargs):
            calls.append(command)
            return {'status': 'completed', 'exit_code': 0, 'stdout': '3\n5\n8\n13\n',
                    'stderr': '', 'truncated': False, 'cleanup_verified': None}
    class Provider:
        supports_image_input = False
        def __init__(self): self.calls = 0
        def generate_json(self, request):
            self.calls += 1
            value = ({'kind': 'tool', 'tool_id': 'native.bash', 'arguments': {'command': 'cat notes/numbers.txt'}, 'text': None}
                     if self.calls == 1 else {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': '29'})
            return InferenceResult(json_value=value, raw_response=json.dumps(value),
                                   usage={'input_tokens': 100, 'output_tokens': 20})
    monkeypatch.setattr('alphonse.agent_v2.evaluation.dojo_bash_candidate.DojoBash', Shell)
    provider = Provider()
    router = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline'))
    records = []
    result = run(image_id='sha256:' + 'a' * 64, socket_path='/tmp/fixture', router=router, emit=records.append)
    assert result['passed'] and result['task_answer_correct'] and result['cleanup_verified']
    assert result['successful_bash_observed'] and result['grounding'] == 'unassessed'
    assert calls[-1] == 'cat notes/numbers.txt' and len(calls) == 2
    assert [r['event'] for r in records if r['event'].startswith('tool_')] == ['tool_started', 'tool_finished']
