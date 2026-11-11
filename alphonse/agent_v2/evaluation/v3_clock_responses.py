"""Offline V3/Responses wiring against the shared synthetic clock.

Each call launches a fresh V3 worker, pins the context-projection flag there,
and injects a fake Responses sender and fake Jev. The fake script is explicitly
provided by the caller. This is not a live model, authorization, cost or
delivery test; ``offline`` declarations are not an OS/network boundary.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def run_v3_clock_offline(*, base_url: str, goal: str, scripted_answer: dict,
                         projection_enabled: bool, timeout_seconds: float = 30) -> dict:
    """Execute one scripted worker; independently grade the clock separately.

    No provider credentials or surrounding environment are forwarded. Partial
    child traces after a worker failure or timeout are unknown. A completed
    processor is not evidence of a correct answer or delivered notification.
    """
    if type(projection_enabled) is not bool or not isinstance(scripted_answer, dict) or not isinstance(goal, str):
        raise ValueError('invalid_v3_clock_configuration')
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= 120:
        raise ValueError('invalid_v3_clock_timeout')
    env = {k: v for k, v in os.environ.items() if k in {'PATH', 'LANG', 'LC_ALL', 'TMPDIR', 'SYSTEMROOT'}}
    env['ALPHONSE_V3_CONTEXT_PROJECTIONS'] = '1' if projection_enabled else '0'
    payload = {'base_url': base_url, 'goal': goal, 'scripted_answer': scripted_answer}
    try:
        completed = subprocess.run([sys.executable, '-m', __name__, '--worker'],
            cwd=str(Path(__file__).resolve().parents[3]), env=env,
            input=json.dumps(payload, ensure_ascii=False), capture_output=True, text=True,
            timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        return {'availability': 'error', 'reason': 'worker_timeout', 'observations': None,
                'projection_enabled': projection_enabled, 'decision_qualified': False}
    if completed.returncode != 0:
        return {'availability': 'error', 'reason': 'worker_failed', 'observations': None,
                'projection_enabled': projection_enabled, 'decision_qualified': False}
    try:
        result = json.loads(completed.stdout)
        if not isinstance(result, dict) or result.get('projection_enabled') is not projection_enabled:
            raise ValueError('worker_flag_mismatch')
        return result
    except (TypeError, ValueError):
        return {'availability': 'error', 'reason': 'worker_response_invalid', 'observations': None,
                'projection_enabled': projection_enabled, 'decision_qualified': False}


def _worker() -> None:
    from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest
    from alphonse.agent_v2.core.intelligence.v3.projections import enabled
    from .deterministic_adapters import _Action, _DeterministicProvider, _ReplayJev
    from .prescription_http_processors import run_v3_http
    from .responses_pilot_transport import PilotLedger, PilotLimits, ResponsesPilotTransport
    from .v3_responses_adapter import OfflineV3ResponsesProvider

    projection = enabled()
    observations = []
    try:
        supplied = json.load(sys.stdin)
        scripted_answer = json.dumps(supplied['scripted_answer'], ensure_ascii=False)
        with TemporaryDirectory(prefix='v3-clock-responses-') as directory:
            ledger = PilotLedger(str(Path(directory) / 'budget.db'), PilotLimits(
                max_calls=20, max_input_tokens=2000000, max_output_tokens=20000,
                max_cost_units=3000000, input_cost_units_per_token=1, output_cost_units_per_token=1))

            class Judge(_ReplayJev):
                offline = True

            def factory(state):
                state.scenario = replace(state.scenario,
                    actions=(_Action('native.fixture_read', 'project_record_search',
                             {'source': 'index'}, role='read_clock', behavior='read'),),
                    final_response=scripted_answer)
                replay = _DeterministicProvider(state)

                def fake_http(_url, body, _key):
                    envelope = json.loads(body['input'])
                    purpose = InferencePurpose(envelope['purpose'])
                    original = InferenceRequest(prompt=envelope['prompt'], purpose=purpose)
                    if purpose == InferencePurpose.TASK_UNDERSTANDING:
                        output = json.dumps({'requested_outcome': supplied['goal'], 'supplied_parameters': [],
                            'constraints': [], 'unresolved_dependencies': [], 'superseded_constraints': []})
                    elif 'text' in body:
                        output = json.dumps(replay.generate_json(original).json_value, ensure_ascii=False)
                    else:
                        output = replay.generate_markdown(original).content
                    return {'status': 'completed', 'model': 'synthetic',
                        'usage': {'input_tokens': 10, 'output_tokens': 10},
                        'output': [{'type': 'message', 'role': 'assistant',
                                    'content': [{'type': 'output_text', 'text': output}]}]}

                transport = ResponsesPilotTransport(model='synthetic', api_key='synthetic', ledger=ledger,
                    count_complete_input=lambda body: len(json.dumps(body).encode()),
                    sender=fake_http, enabled=True)

                class FakeOfflineTransport:
                    # This declaration is not a network sandbox. The sender
                    # above is a local function returning scripted fixture data.
                    offline = True
                    def generate_json(self, request, **limits):
                        return transport.generate_json(request, **limits)
                    def generate_markdown(self, request, **limits):
                        return transport.generate_markdown(request, **limits)

                return OfflineV3ResponsesProvider(FakeOfflineTransport(),
                    max_input_tokens=100000, max_output_tokens=1000, emit=observations.append)

            factory.offline = True
            result = run_v3_http(base_url=supplied['base_url'], goal=supplied['goal'],
                provider_factory=factory, jev=Judge(), emit=observations.append,
                allowed_source_names=('index',))
            output = {'availability': 'available', 'processor': asdict(result),
                      'projection_enabled': projection, 'observations': observations,
                      'ledger': ledger.snapshot(), 'decision_qualified': False,
                      'delivery': 'unassessed', 'model': 'scripted_fake', 'jev': 'scripted_fake'}
    except Exception as exc:
        # Do not expose bearer URLs, provider output or exception text.
        output = {'availability': 'error', 'reason': type(exc).__name__,
                  'projection_enabled': projection, 'observations': None,
                  'decision_qualified': False}
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__' and sys.argv[1:] == ['--worker']:
    _worker()
