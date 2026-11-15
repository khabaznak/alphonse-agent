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
