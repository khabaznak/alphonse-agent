"""Synthetic mailbox comparison: scripted providers only, no Gmail."""
import json

import pytest

from alphonse.agent_v2.core.inference import InferencePurpose, InferenceResult
from alphonse.agent_v2.evaluation import codex_mail_comparison as comparison
from alphonse.agent_v2.evaluation.codex_subscription_provider import CodexSubscriptionProvider

MODEL = 'gpt-6-luna'


def test_candidate_reads_synthetic_mail_and_unknown_usage_stays_unknown(monkeypatch):
    decisions = iter([
        {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': 'index'}, 'text': None},
        {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': json.dumps({'unread_count': 3})},
        {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': json.dumps({'unread_count': 3})},
    ])
    prompts = []
    def fake(self, request):
        prompts.append(request.prompt)
        data = next(decisions)
        return InferenceResult(json_value=data, raw_response=json.dumps(data))
    monkeypatch.setattr(CodexSubscriptionProvider, 'generate_json', fake)
    result = comparison.run_mail_arm(arm='simple', provider=CodexSubscriptionProvider(model=MODEL), model=MODEL)
    assert result['status'] == 'completed' and result['passed'] is True
    assert result['metrics']['system_two_calls'] == 3 and result['metrics']['tool_calls'] == 1
    assert result['metrics']['provider_input_tokens'] is None
    assert result['delivery'] == 'unassessed' and result['qualification'] == 'unassessed'
    assert 'unread_count' in prompts[0] and 'unread_count": 3' not in prompts[0]
    assert comparison.DESCRIPTION in prompts[0]
    assert 'unread_count' in prompts[1]


def test_candidate_cannot_guess_without_read(monkeypatch):
    monkeypatch.setattr(CodexSubscriptionProvider, 'generate_json',
        lambda self, request: InferenceResult(json_value={'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': '{"unread_count": 3}'}))
    result = comparison.run_mail_arm(arm='simple', provider=CodexSubscriptionProvider(model=MODEL), model=MODEL)
    assert not result['passed'] and not result['oracle']['source_matches_key']


def test_v3_real_processor_with_fake_provider_and_jev(monkeypatch, tmp_path):
    from alphonse.agent_v2.evaluation.deterministic_adapters import _Action, _DeterministicProvider, _ReplayJev, _RunState, _Scenario
    action = _Action('native.fixture_read', 'project_record_search', {'source': 'index'}, role='read_mail', behavior='read')
    replay = _RunState(_Scenario((action,), 'Read the synthetic inbox.', '{"unread_count": 3}'), tmp_path, 'hierarchical_v3')
    delegate = _DeterministicProvider(replay)
    descriptions = []
    def fake_json(self, request):
        descriptions.extend(tool.description for tool in request.tools)
        descriptions.append(request.prompt)
        if request.purpose == InferencePurpose.TASK_UNDERSTANDING:
            return InferenceResult(json_value={'requested_outcome': comparison.GOAL, 'supplied_parameters': [],
                'constraints': [], 'unresolved_dependencies': [], 'superseded_constraints': []})
        return delegate.generate_json(request)
    monkeypatch.setattr(CodexSubscriptionProvider, 'generate_json', fake_json)
    monkeypatch.setattr(CodexSubscriptionProvider, 'generate_markdown', lambda self, request: delegate.generate_markdown(request))
    def fake_plan(self, request):
        descriptions.extend(tool.description for tool in request.tools)
        return delegate.plan_tool_call(request)
    monkeypatch.setattr(CodexSubscriptionProvider, 'plan_tool_call', fake_plan)
    result = comparison.run_mail_arm(arm='v3', provider=CodexSubscriptionProvider(model=MODEL), model=MODEL, jev=_ReplayJev())
    assert result['oracle']['source_matches_key'] is True
    assert any(comparison.DESCRIPTION in item for item in descriptions)
    # Native respond behavior can expose a scripted-provider incompatibility;
    # never call this a real model or production-delivery result.
    assert result['qualification'] == 'unassessed' and result['delivery'] == 'unassessed'


def test_wrong_key_and_provider_are_not_success(monkeypatch):
    decisions = iter([
        {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': 'index'}, 'text': None},
        {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': '{"unread_count": 3}'},
        {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': '{"unread_count": 3}'},
    ])
    monkeypatch.setattr(CodexSubscriptionProvider, 'generate_json', lambda self, request:
                        InferenceResult(json_value=next(decisions)))
    result = comparison.run_mail_arm(arm='simple', provider=CodexSubscriptionProvider(model=MODEL),
                                     model=MODEL, expected_unread_count=2)
    assert result['status'] == 'completed' and result['passed'] is False and result['oracle']['source_matches_key'] is False
    with pytest.raises(ValueError, match='matching_codex_subscription_provider_required'):
        comparison.run_mail_arm(arm='simple', provider=object(), model=MODEL)
