"""Candidate context path invariants; no live model calls."""
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_search_memory_tool_definition, build_read_memory_source_tool_definition
from alphonse.agent_v2.system_one import JevCriterionDecisionProvider, SystemOneSettings


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHONSE_V3_CONTEXT_PROJECTIONS", "1")
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    store = SQLiteConversationStore(tmp_path / 'history.sqlite')
    task = TaskState(user='alex', task_id='task', message_id='message', goal='Why did thermostat fail yesterday?')
    jev = SimpleNamespace(judge_recall=lambda **kw: True if not kw['candidates'] else [h['source_id'] for h in kw['candidates']])
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, conversation_store=store, system_one=jev)
    return task, context


def record(context, content, *, owner='alex', message=None):
    return context.conversation_store.record(owner_user_id=owner, content=content, role='user', source='test', source_message_id=message or content)


















@pytest.mark.parametrize('answer', [None, float('nan'), True, -1, 2, 'yes'])
def test_jev_malformed_recall_answers_are_ambiguous(answer):
    client = SimpleNamespace(evaluate=lambda **kw: {'answers': {'recall_needed': {'type': 'noul', 'noul': answer}}})
    decider = JevCriterionDecisionProvider(settings=SystemOneSettings())
    decider.client = client
    assert decider.judge_recall(request='Recall failure', candidates=[]) is None


def test_jev_relevance_retains_ambiguity_and_never_composes_queries():
    captured = []
    def evaluate(**kw):
        captured.append(kw)
        return {'answers': {f'candidate_{i}': {'type': 'noul', 'noul': p} for i, p in enumerate([.9, .5, .1])}}
    decider = JevCriterionDecisionProvider(SystemOneSettings())
    decider.client = SimpleNamespace(evaluate=evaluate)
    hits = [{'source_id': str(i), 'excerpt': 'original evidence'} for i in range(3)]
    assert decider.judge_recall(request='Explain earlier timeout', candidates=hits) == ['0', '1']
    assert all(q['type'] == 'noul' for q in captured[0]['questions'].values())
    assert captured[0]['state']['candidates'] == hits












def test_admission_rebuilds_projection_after_context_selection():
    captured = []
    calls = []
    def evaluate(**kw):
        captured.append(kw['state'])
        if 'context_relevant_0' in kw['questions']:
            return {'answers': {'context_relevant_0': {'type': 'noul', 'noul': .99}}}
        return {'answers': {'requires_task': {'type': 'noul', 'noul': .99}}}
    def projection(ids):
        calls.append(list(ids))
        return 'REQUEST; selected context: ' + ','.join(ids)
    decider = JevCriterionDecisionProvider(SystemOneSettings())
    decider.client = SimpleNamespace(evaluate=evaluate)
    decider.classify_task_admission(message='REQUEST', context_candidates=[{'id': 'project:test', 'content': 'workflow'}], decision_context=projection)
    assert calls == [[], ['project:test']]
    assert captured[0]['invariant_context'].count('REQUEST') == 1
    assert captured[-1] == {'decision_context': 'REQUEST; selected context: project:test'}




























# Existing scripted providers delegate the shared interpretation call explicitly.
