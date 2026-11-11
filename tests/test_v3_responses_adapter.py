"""Synthetic V3 envelope compatibility, not a live model qualification."""
import json
from dataclasses import replace

import pytest

from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest, InferenceResult
from alphonse.agent_v2.evaluation.v3_responses_adapter import OfflineV3ResponsesProvider


class FakeTransport:
    offline = True

    def __init__(self, result):
        self.result = result
        self.calls = []

    def generate_json(self, request, **limits):
        self.calls.append((request, limits))
        return self.result

    generate_markdown = generate_json


def request(**kwargs):
    return InferenceRequest(prompt='Find the synthetic record.',
                            purpose=InferencePurpose.TASK_UNDERSTANDING, **kwargs)


def test_v3_schema_is_supplied_in_envelope_and_checked_independently():
    schema = {'type': 'object', 'required': ['answer'], 'additionalProperties': False,
              'properties': {'answer': {'type': 'string'}}}
    transport = FakeTransport(InferenceResult(json_value={'answer': 'ok'}, usage={'input_tokens': 30}))
    events = []
    provider = OfflineV3ResponsesProvider(transport, max_input_tokens=4096, max_output_tokens=100, emit=events.append)
    result = provider.generate_json(request(output_schema=schema))
    sent, limits = transport.calls[0]
    assert sent.output_schema is None and not sent.tools
    assert json.loads(sent.prompt)['output_schema'] == schema
    assert limits == {'max_input_tokens': 4096, 'max_output_tokens': 100}
    assert result.json_value == {'answer': 'ok'}
    assert events[0]['provider_usage'] == {'input_tokens': 30, 'output_tokens': None}
    transport.result = InferenceResult(json_value={'answer': 5})
    with pytest.raises(ValueError, match='pilot_schema_validation_failed'):
        provider.generate_json(request(output_schema=schema))


def test_tool_planning_passes_exact_authorized_descriptors_not_live_tools():
    descriptor = ToolDescriptor('native.fixture_read', 'fixture_read', ToolKind.NATIVE,
                                argument_schema={'type': 'object'}, read_only=True)
    transport = FakeTransport(InferenceResult(json_value={'tool_id': descriptor.tool_id, 'arguments': {}}))
    provider = OfflineV3ResponsesProvider(transport, max_input_tokens=4096, max_output_tokens=100)
    result = provider.plan_tool_call(request(tools=(descriptor,)))
    sent = transport.calls[0][0]
    assert json.loads(sent.prompt)['tools'][0]['tool_id'] == descriptor.tool_id
    assert sent.tools == () and result.tool_call == result.json_value


def test_invalid_schema_and_envelope_limit_fail_before_transport():
    transport = FakeTransport(InferenceResult(json_value={}))
    provider = OfflineV3ResponsesProvider(transport, max_input_tokens=128, max_output_tokens=100)
    with pytest.raises(ValueError, match='references_unsupported'):
        provider.generate_json(request(output_schema={'$ref': 'https://example.test/secret'}))
    with pytest.raises(ValueError, match='envelope_exceeds'):
        provider.generate_json(request())
    assert not transport.calls


def test_live_transport_rejected_usage_overrun_and_cancellation():
    with pytest.raises(ValueError, match='live_transport_not_authorized'):
        OfflineV3ResponsesProvider(type('Live', (), {'offline': False})(), max_input_tokens=10, max_output_tokens=10)
    transport = FakeTransport(InferenceResult(json_value={}, usage={'output_tokens': 101}))
    provider = OfflineV3ResponsesProvider(transport, max_input_tokens=4096, max_output_tokens=100)
    with pytest.raises(ValueError, match='usage_invalid_or_over_limit'):
        provider.generate_json(request())
    assert len(transport.calls) == 1
    with pytest.raises(ValueError, match='inference_cancelled'):
        provider.generate_json(replace(request(), cancel_checker=lambda: True))
    assert len(transport.calls) == 1


def test_failed_dispatched_calls_are_observed_without_leaking_exception_text():
    class Failure(FakeTransport):
        def generate_json(self, request, **limits):
            raise RuntimeError('sensitive synthetic text')
    events = []
    provider = OfflineV3ResponsesProvider(Failure(None), max_input_tokens=4096,
                                         max_output_tokens=100, emit=events.append)
    with pytest.raises(RuntimeError):
        provider.generate_json(request())
    assert events[0]['status'] == 'failed'
    assert events[0]['provider_usage'] == {'input_tokens': None, 'output_tokens': None}
    assert events[0]['usage_diagnostics']['input']['source'] == 'estimated'
    assert events[0]['usage_diagnostics']['output']['source'] == 'unknown'
    assert 'sensitive synthetic text' not in str(events)

    events.clear()
    transport = FakeTransport(InferenceResult(json_value={}, usage={'input_tokens': 33, 'output_tokens': 101}))
    provider = OfflineV3ResponsesProvider(transport, max_input_tokens=4096, max_output_tokens=100,
                                         emit=events.append)
    with pytest.raises(ValueError, match='usage_invalid_or_over_limit'):
        provider.generate_json(request())
    assert events[0]['status'] == 'failed' and events[0]['provider_usage']['output_tokens'] == 101


def test_incomplete_provider_result_retains_known_usage_without_response_text():
    from alphonse.agent_v2.evaluation.responses_pilot_transport import PilotProviderResultError
    class Incomplete(FakeTransport):
        def generate_json(self, request, **limits):
            raise PilotProviderResultError('pilot_response_not_completed',
                {'input_tokens': 123, 'output_tokens': 9, 'private': 'sensitive text'})
    events = []
    provider = OfflineV3ResponsesProvider(Incomplete(None), max_input_tokens=4096,
                                         max_output_tokens=100, emit=events.append)
    with pytest.raises(PilotProviderResultError):
        provider.generate_json(request())
    assert events[0]['status'] == 'failed'
    assert events[0]['provider_usage'] == {'input_tokens': 123, 'output_tokens': 9}
    assert 'sensitive text' not in str(events)


def test_real_v3_processor_uses_adapter_for_synthetic_read(tmp_path):
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.inference import InferenceRouter, ModelProfile
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.v3 import HierarchicalCAPDProcessor
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.evaluation.deterministic_adapters import (
        _DeterministicProvider, _ReplayJev, _RunState, _registry, _scenario,
    )
    from alphonse.agent_v2.evaluation.responses_pilot_transport import (
        PilotLedger, PilotLimits, ResponsesPilotTransport,
    )

    (tmp_path / 'Receta.md').write_text('Medicine: amoxicillin', encoding='utf-8')
    state = _RunState(scenario=_scenario('markdown-prescription-no-ocr'), root=tmp_path, engine='hierarchical_v3')
    script = _DeterministicProvider(state)

    calls = []
    def fake_http(_url, body, _key):
        envelope = json.loads(body['input'])
        purpose = InferencePurpose(envelope['purpose'])
        calls.append(purpose)
        original = InferenceRequest(prompt=envelope['prompt'], purpose=purpose)
        if purpose == InferencePurpose.TASK_UNDERSTANDING:
            value = {'requested_outcome': 'Read the synthetic note', 'supplied_parameters': [],
                     'constraints': [], 'unresolved_dependencies': [], 'superseded_constraints': []}
            response_text = json.dumps(value)
        elif 'text' in body:
            response_text = json.dumps(script.generate_json(original).json_value)
        else:
            response_text = script.generate_markdown(original).content
        return {'status': 'completed', 'model': 'synthetic',
                'usage': {'input_tokens': 10, 'output_tokens': 10},
                'output': [{'type': 'message', 'role': 'assistant',
                            'content': [{'type': 'output_text', 'text': response_text}]}]}

    ledger = PilotLedger(str(tmp_path / 'pilot.sqlite'), PilotLimits(
        max_calls=20, max_input_tokens=2000000, max_output_tokens=20000,
        max_cost_units=3000000, input_cost_units_per_token=1, output_cost_units_per_token=1))
    real_transport = ResponsesPilotTransport(model='synthetic', api_key='synthetic', ledger=ledger,
        count_complete_input=lambda body: len(json.dumps(body).encode()), sender=fake_http, enabled=True)
    # Self-declared fake; this does not establish OS/network isolation. The
    # injected sender above returns fixture data and never performs networking.
    class FakeOfflineTransport:
        offline = True
        def generate_json(self, wrapped, **limits):
            return real_transport.generate_json(wrapped, **limits)
        def generate_markdown(self, wrapped, **limits):
            return real_transport.generate_markdown(wrapped, **limits)

    transport = FakeOfflineTransport()
    provider = OfflineV3ResponsesProvider(transport, max_input_tokens=100000, max_output_tokens=1000)
    router = InferenceRouter(provider=provider, default_profile=ModelProfile(
        provider='offline', model='injected', profile_id='v3-responses-test'))
    task = TaskState(task_id='fixture-task', message_id='fixture-message', user='fixture-user',
                     project_id='fixture-project', goal='Read the synthetic note',
                     intelligence_engine='hierarchical_v3', intelligence_schema_version=3)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=_registry(state),
                              inference=router, system_one=_ReplayJev())
    outcome = HierarchicalCAPDProcessor().process(task, context)
    final = TaskState.from_dict(outcome.snapshot.metadata.get('task_state') or task.to_dict())
    assert final.status == 'completed', final.outcome
    assert 'fixture.read' in state.tool_calls
    assert len(calls) > 1
    assert ledger.snapshot()['calls'] == len(calls)
