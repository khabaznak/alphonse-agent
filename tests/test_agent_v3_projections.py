"""Candidate context path invariants; no live model calls."""
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.context import task_context_for_decision
from alphonse.agent_v2.core.intelligence.v3.projections import (
    Record, assemble, token_count, ContextBudgetExceeded, project, retrieve,
)
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


def test_atomic_budget_dedup_and_required_priority():
    records = [Record('same', 'Optional', 'old'), Record('large', 'Optional', 'x' * 900),
               Record('same', 'Required', 'Policy', True), Record('request', 'Request', '温度🌡', True)]
    text, manifest = assemble(records, budget=100)
    assert 'Policy' in text and '温度🌡' in text and 'old' not in text
    assert manifest['source_ids'] == ['same', 'request']
    assert manifest['omitted_source_ids'] == ['large']
    assert manifest['tokens'] == len(text.encode('utf-8')) <= 100
    assert token_count('温度') == 6
    with pytest.raises(ContextBudgetExceeded):
        assemble(records, budget=5)


def test_request_once_and_canonical_recall_deduplicated(setup):
    task, context = setup
    record(context, task.goal, message=task.message_id)
    task.append_conversation_message('alex', task.goal)
    record(context, 'The thermostat failed with timeout code E-42 yesterday.')
    rendered = project(task, context, purpose='planning')
    assert rendered.count(task.goal) == 1
    manifest = task.metadata['v3_context_projections']['planning']
    assert len(manifest['source_ids']) == len(set(manifest['source_ids']))
    assert 'E-42' in rendered


def test_review_excludes_ledger_and_unrelated_history(setup):
    task, context = setup
    task.acceptance_contract = {'criteria': [{'id': 'ac-1', 'statement': 'Explain timeout'}]}
    task.memory_context_md = 'UNRELATED_LEDGER'
    record(context, 'UNRELATED_HISTORY')
    rendered = project(task, context, purpose='review')
    assert 'Explain timeout' in rendered and task.goal in rendered
    assert 'UNRELATED' not in rendered
    assert 'recall:status' not in rendered


def test_private_recall_and_revocation_rechecked(setup):
    task, context = setup
    private = record(context, 'thermostat private PIN', owner='other')
    visible = record(context, 'thermostat diagnostic timeout')
    result = retrieve(task, context)
    assert 'timeout' in str(result) and 'private PIN' not in str(result)
    context.conversation_store.delete(event_id=visible.event_id, owner_user_id='alex')
    assert 'timeout' not in str(retrieve(task, context))
    assert private


def test_fresh_request_skips_past_measurement_but_history_overrides_negative(setup):
    task, context = setup
    context.system_one.judge_recall = lambda **kw: False
    task.goal = 'What is the temperature now?'
    assert retrieve(task, context)['status'] == 'not_needed'
    task.goal = 'Why did the thermostat fail yesterday?'
    record(context, 'thermostat timeout yesterday')
    assert retrieve(task, context)['results']


def test_missing_backend_and_failed_judge_are_explicit(setup):
    task, context = setup
    context.system_one.judge_recall = lambda **kw: (_ for _ in ()).throw(ValueError('secret'))
    context.tools = None
    result = retrieve(task, context)
    assert result['status'] == 'unavailable' and result['further_search_required']
    rendered = project(task, context, purpose='execution')
    assert 'honest limitation' in rendered and 'unavailable' in rendered and 'secret' not in rendered


def test_required_steering_policy_and_skill_instructions_not_clipped(setup, monkeypatch):
    task, context = setup
    task.append_conversation_message('alex', 'Do not contact anyone. ' * 800)
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_TOKEN_BUDGET', '1000')
    with pytest.raises(ContextBudgetExceeded):
        project(task, context, purpose='planning')


def test_flag_off_keeps_legacy_ledger(setup, monkeypatch):
    task, context = setup
    task.memory_context_md = 'LEGACY_TAIL'
    monkeypatch.delenv('ALPHONSE_V3_CONTEXT_PROJECTIONS')
    assert 'LEGACY_TAIL' in task_context_for_decision(task, context)


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


def test_request_curation_cannot_hide_recall_tools(setup):
    from alphonse.agent_v2.core.intelligence.v3.planner import _curate_request_tools, _curate_phase_tools
    from alphonse.agent_v2.core.intelligence.v3 import PhasePlan, PhaseSubgoal
    from alphonse.agent_v2.system_one import SystemOneToolRegistrySelection
    task, context = setup
    context.system_one.curate_request_tools = lambda **kw: SystemOneToolRegistrySelection()
    context.system_one.select_plan_tools = lambda **kw: SystemOneToolRegistrySelection()
    tools = context.tools.list()
    curated = _curate_request_tools(task, tools, system_prompt='', session_history='', project_context='', durable_memory='', attachment_manifest=[], context=context)
    assert {t.tool_id for t in curated} == {'native.search_memory', 'native.read_memory_source'}
    phase = PhasePlan('p', 'Recall', (PhaseSubgoal('s', 'Recall', 'text', allowed_capabilities=('memory_recall',)),), authorized_capabilities=('memory_recall',))
    selected = _curate_phase_tools(task, phase, curated, context)
    assert set(selected.curated_tool_ids) == {t.tool_id for t in curated}


def test_admission_uses_atomic_projection_without_reslicing(setup):
    task, context = setup
    captured = []
    decider = JevCriterionDecisionProvider(SystemOneSettings())
    def evaluate(**kw):
        captured.append(kw)
        return {'answers': {'requires_task': {'type': 'noul', 'noul': .99}}}
    decider.client = SimpleNamespace(evaluate=evaluate)
    text = project(task, context, purpose='admission')
    decider.classify_task_admission(message=task.goal, recent_conversation=task.goal, decision_context=text)
    assert captured[-1]['state'] == {'decision_context': text}
    assert text.count(task.goal) == 1


def test_direct_response_prompt_has_one_request(setup):
    from alphonse.agent_v2.core.inference import InferenceResult
    from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
    task, context = setup
    prompts = []
    def generate(request):
        prompts.append(request.prompt)
        return InferenceResult(content='The past diagnostic evidence is unavailable.')
    context.inference = SimpleNamespace(generate_markdown=generate)
    HierarchicalCAPDProcessor._complete_direct_response(task, context)
    assert prompts[0].count(task.goal) == 1
    assert task.metadata['v3_context_projections']['response']['tokenizer'] == 'conservative_utf8_bytes'


def test_whole_continuity_exchange_is_atomic():
    records = [Record('request', 'Request', 'Do work', True),
               Record('user', 'Prior user', 'Long antecedent ' * 100, group='exchange'),
               Record('assistant', 'Prior reply', 'Yes', group='exchange')]
    text, manifest = assemble(records, budget=100)
    assert 'Yes' not in text
    assert manifest['omitted_source_ids'] == ['user', 'assistant']


def test_enabled_planning_and_execution_receive_one_request(setup):
    from alphonse.agent_v2.core.intelligence.v3.planner import plan_phase
    from alphonse.agent_v2.core.intelligence.v3.executor import PhaseExecutor
    from alphonse.agent_v2.core.intelligence.v3 import PhasePlan, PhaseSubgoal, new_tactical_state
    from alphonse.agent_v2.system_one import SystemOneToolRegistrySelection
    task, context = setup
    phase = PhasePlan('p', 'Recall prior diagnostic', (PhaseSubgoal('s', 'Recall', 'text', allowed_capabilities=('memory_recall',)),), authorized_capabilities=('memory_recall',))
    task.acceptance_contract = {'criteria': [{'id': 'ac-1', 'statement': 'Explain failure'}]}
    task.memory_context_md = 'UNRELATED_LEDGER'
    context.system_one.curate_request_tools = lambda **kw: SystemOneToolRegistrySelection()
    context.system_one.select_plan_tools = lambda **kw: SystemOneToolRegistrySelection()
    prompts = []
    def generate(request):
        prompts.append(request.prompt)
        return SimpleNamespace(json_value=phase.to_dict())
    context.inference = SimpleNamespace(generate_json=generate)
    planned = plan_phase(task, context)
    state = new_tactical_state(planned)
    PhaseExecutor()._select_action(task, state, phase.subgoals[0], context.tools.list(), context)
    assert all(prompt.count(task.goal) == 1 for prompt in prompts)
    assert all('UNRELATED_LEDGER' not in prompt for prompt in prompts)
    assert {'planning', 'curation', 'execution'} <= task.metadata['v3_context_projections'].keys()


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


def test_fresh_request_filters_optional_stale_history_without_losing_steering(setup):
    task, context = setup
    task.goal = 'What is the temperature now?'
    task.append_conversation_message('alex', 'Use the guest room instead.')
    record(context, 'Yesterday the temperature was 31 C.')
    context.system_one.judge_recall = lambda **kw: [] if kw['candidates'] else False
    text = project(task, context, purpose='execution')
    assert 'Yesterday the temperature was 31 C.' not in text
    assert 'Use the guest room instead.' in text
    assert task.metadata['v3_history_selection']['status'] == 'used'


def test_ambiguous_history_selection_retains_whole_exchange(setup):
    task, context = setup
    record(context, 'Required antecedent: guest room sensor.')
    context.conversation_store.record(owner_user_id='alex', content='It reports percent humidity.', role='assistant', source='test', source_message_id='reply')
    context.system_one.judge_recall = lambda **kw: None
    text = project(task, context, purpose='execution')
    assert 'Required antecedent: guest room sensor.' in text
    assert 'It reports percent humidity.' in text
    assert task.metadata['v3_history_selection']['status'] == 'ambiguous_or_unavailable'


@pytest.mark.parametrize('cancel_during_judgment', [False, True])
def test_history_selection_respects_cancellation_before_and_after_jev(setup, cancel_during_judgment):
    from alphonse.agent_v2.system_one import SystemOneUnavailableError
    task, context = setup
    record(context, 'Prior device diagnostic.')
    state = {'cancelled': not cancel_during_judgment, 'calls': 0}
    def judge(**kwargs):
        state['calls'] += 1
        state['cancelled'] = True
        return []
    context.system_one.judge_recall = judge
    context.cancellation_checker = lambda: state['cancelled']
    with pytest.raises(SystemOneUnavailableError, match='cancelled'):
        project(task, context, purpose='execution')
    assert state['calls'] == int(cancel_during_judgment)
