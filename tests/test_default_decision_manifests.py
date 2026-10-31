"""Exercise public decision paths with the default context builder, no live models."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext, PromptFile, _task_progress_snapshot, CoreActivityEvent, ImprovementPhase
from alphonse.agent_v2.core.context_manifests import manifest_scope
from alphonse.agent_v2.core.context_inventory import source
from alphonse.agent_v2.core.inference import InferenceRouter, InferenceResult, ModelProfile
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import PhasePlan, PhaseSubgoal, new_tactical_state
from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
from alphonse.agent_v2.core.intelligence.v3.planner import plan_phase
from alphonse.agent_v2.core.intelligence.v3.outer import _system_two_acceptance_review
from alphonse.agent_v2.core.intelligence.v3.context import task_context_for_decision, selected_skill_guidance
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_search_memory_tool_definition
from alphonse.agent_v2.interfaces.a2ui import A2UiAdapter
from alphonse.agent_v2.system_one import JevCriterionDecisionProvider, SystemOneSettings


def all_records(manifest):
    return [record for projection in manifest['projections'] for record in projection['records']]


def assert_ranges(manifest, sections):
    for projection in manifest['projections']:
        value = sections[projection['input_section']]
        expected = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        assert projection['tokens'] == len(expected.encode('utf-8'))
        if projection['input_section'] == 'prompt' and len(projection['records']) > 1:
            assert sum(r['tokens'] for r in projection['records'] if r['status'] == 'included') == projection['tokens']
        for record in projection['records']:
            if record.get('input_range') and record['status'] == 'included':
                left = record['input_range']['start_char']
                right = record['input_range']['end_char']
                assert 0 <= left <= right <= len(expected)
                assert record['tokens'] == len(expected[left:right].encode('utf-8'))


@pytest.fixture
def default_calls(tmp_path, monkeypatch):
    monkeypatch.delenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', raising=False)
    task = TaskState(user='alex', task_id='request-1', message_id='request-1', goal='Explain the earlier thermostat timeout.', metadata={'routing_disposition': 'pdca_task'})
    task.acceptance_contract = {'criteria': [{'id': 'ac-1', 'statement': 'Explain the timeout', 'status': 'pending', 'required': True}]}
    store = SQLiteConversationStore(tmp_path / 'conversation.sqlite')
    event = store.record(owner_user_id='alex', role='user', content='Prior timeout E-42.', source='test', source_message_id='prior')
    private = store.record(owner_user_id='other', role='user', content='PRIVATE SECRET 12345', source='test', source_message_id='private')
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    phase = PhasePlan('phase-1', 'Recall diagnostic', (PhaseSubgoal('read', 'Read diagnostic', 'text', allowed_capabilities=('memory_recall',)),), authorized_capabilities=('memory_recall',), criterion_ids=('ac-1',))
    calls, inference = [], []
    jev = JevCriterionDecisionProvider(SystemOneSettings())
    def evaluate(**kwargs):
        calls.append(deepcopy(kwargs))
        return {'answers': {key: {'type': 'noul', 'noul': .99 if key != 'context_relevant_1' else .01} for key in kwargs['questions']}}
    jev.client = SimpleNamespace(evaluate=evaluate)
    def generate(request):
        inference.append(request)
        return InferenceResult(json_value=phase.to_dict() if request.purpose.value == 'phase_planning' else {'updates': []}, content='I will check the diagnostic.')
    router = InferenceRouter(provider=SimpleNamespace(generate_json=generate, generate_markdown=generate), default_profile=ModelProfile('test', 'test', 'test'))
    global_text = '# Global instructions\nHousehold default.\n\n## Family definition\nUse household privacy.\n\n## Diagnostics\nKeep diagnostic codes.\n\n## Unrelated\nGarden workflow.'
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, inference=router,
        system_one=jev, conversation_store=store, question_store=SQLiteQuestionStore(':memory:'),
        prompts=SimpleNamespace(load=lambda name: PromptFile(name, global_text if name == 'GlobalContext.md' else '')))
    return task, context, phase, calls, inference, event, private


def test_default_admission_planning_curation_and_both_reviews_match_call_inputs(default_calls):
    task, context, phase, calls, inference, event, private = default_calls
    with manifest_scope(task, context):
        assert not HierarchicalCAPDProcessor._admit_initial_human_task(task, context)
        planned = plan_phase(task, context)
        state = new_tactical_state(planned)
        for i in range(14):
            state.evidence.append({'evidence_ref': f'evidence-{i}', 'phase_id': phase.phase_id, 'subgoal_id': 'read', 'tool_id': 'native.search_memory', 'status': 'success' if i != 13 else 'failed', 'result': {'diagnostic': 'E-42' + 'x' * 2500}})
        shared = task_context_for_decision(task, context, purpose='review', selected_context_ids=['global:diagnostics'])
        context.system_one.evaluate(contract=task.acceptance_contract, phase=phase.to_dict(), evidence=state.evidence.to_dict(), task_context=shared)
        _system_two_acceptance_review(task, state, context)
    manifests = deepcopy(task.metadata['decision_context_manifests'])
    purposes = {m['purpose'] for m in manifests}
    assert {'admission_curation', 'admission', 'phase_planning', 'tool_curation_request', 'tool_curation_phase', 'completion_review', 'phase_review', 'recall_selection', 'retry_context_selection'} <= purposes
    assert all(m['provenance'] == 'source_manifest' and m['context_builder_version'] == 'call-input-v2' for m in manifests)
    jev_rows = [m for m in manifests if m['purpose'] in {'admission_curation', 'admission', 'task_acknowledgement', 'tool_curation_request', 'tool_curation_phase', 'completion_review', 'recall_selection', 'retry_context_selection'}]
    assert len(jev_rows) == len(calls)
    for row, call in zip(jev_rows, calls):
        assert_ranges(row, {**call['state'], 'questions': call['questions']})
    for request in inference:
        row = next(m for m in manifests if m['decision_id'] == request.metadata['decision_id'])
        sections = {'prompt': request.prompt}
        if request.output_schema is not None:
            sections['output_schema'] = request.output_schema
        assert_ranges(row, sections)
    planning = next(m for m in manifests if m['purpose'] == 'phase_planning')
    assert 'conversation:' + event.event_id in {r['source_id'] for r in all_records(planning)}
    admission = next(m for m in manifests if m['purpose'] == 'admission')
    assert any(r['source_id'] == 'global:unrelated' and r['reason'] == 'irrelevant' for r in all_records(admission))
    review = next(m for m in manifests if m['purpose'] == 'completion_review')
    assert {'evidence-0', 'evidence-1'} <= {r['source_id'] for r in review['omissions'] if r['reason'] == 'over_budget'}
    assert any(r['source_id'] == 'evidence-13' and r['reason'] == 'irrelevant' for r in review['omissions'])
    evidence_row = next(r for r in all_records(review) if r['source_id'] == 'evidence-12')
    assert evidence_row['truncated'] and evidence_row['source_range']['end_char'] == 2000
    assert evidence_row['source_range']['total_chars'] > 2500
    assert 'PRIVATE SECRET' not in json.dumps(manifests) and private.event_id not in json.dumps(manifests)
    # A later task/context rebuild cannot modify the persisted call inventories.
    task.goal = 'LATER REQUEST'
    task.memory_context_md = 'LATER MEMORY'
    task_context_for_decision(task, context)
    assert task.metadata['decision_context_manifests'] == manifests
    snapshot = _task_progress_snapshot(task)
    envelopes = A2UiAdapter().task_progress(task.task_id, snapshot)
    assert envelopes[2]['updateDataModel']['value']['decision_context_manifests'] == manifests
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.ipc import V2DaemonServer
    daemon = object.__new__(V2Daemon)
    daemon.runtime = SimpleNamespace(user_store=SimpleNamespace(is_admin=lambda actor: actor == 'alex'), question_store=context.question_store)
    result = V2DaemonServer(daemon)._dispatch({'method': 'decision_context', 'params': {'actor_user_id': 'alex', 'task_id': task.task_id}})
    assert result['manifests'] == manifests


def test_default_skill_clips_and_unavailable_guidance_are_explicit(default_calls):
    task, context, *_ = default_calls
    skill = SimpleNamespace(skill_id='skill:diagnostics', name='Diagnostics', description='Diagnostics', directory='/synthetic/skills', instructions='Long operating instructions. ' * 900)
    context.skill_store = SimpleNamespace(get=lambda identity: skill if identity in {skill.skill_id, 'skill:later'} else None)
    task.metadata['v3_skill_selection'] = {'selected_skill_ids': ['skill:missing', skill.skill_id, 'skill:later']}
    text, loaded = selected_skill_guidance(task, context)
    assert loaded == [skill.skill_id]
    row = next(r for r in text.records if r['source_id'] == skill.skill_id)
    assert row['truncated'] and row['source_range']['end_char'] < row['source_range']['total_chars']
    assert any(r['source_id'] == 'skill:missing' and r['reason'] == 'unavailable' for r in text.records)
    assert any(r['source_id'] == 'skill:later' and r['reason'] == 'over_budget' for r in text.records)
    with manifest_scope(task, context):
        HierarchicalCAPDProcessor._complete_direct_response(task, context)
    manifest = task.metadata['decision_context_manifests'][-1]
    assert manifest['projections'][0]['selected_skill_ids'] == ['skill:missing', skill.skill_id, 'skill:later']
    assert any(r['source_id'] == skill.skill_id and r['truncated'] for r in all_records(manifest))
    assert 'Long operating instructions' not in json.dumps(manifest)



def test_default_memory_and_conversation_budget_keep_source_omissions(default_calls):
    task, context, *_ = default_calls
    task.facts_md = 'Whole fact too long. ' * 500
    task.memory_context_md = '# Knowledge\n- [memory:short; historical] Small claim.\n- [memory:long; historical] ' + 'Long claim. ' * 1200
    text = task_context_for_decision(task, context)
    assert 'Small claim.' in text and 'Long claim.' not in text
    assert any(r['source_id'] == 'memory:short' and r['status'] == 'included' for r in text.records)
    assert any(r['source_id'] == 'memory:long' and r['reason'] == 'over_budget' for r in text.records)
    assert any(r['source_id'].endswith(':facts') and r['reason'] == 'over_budget' for r in text.records)
    assert 'Whole fact too long.' not in text


def test_tui_displays_the_captured_owner_inventory_after_task_changes(default_calls):
    from alphonse.agent_v2.interfaces.tui import _build_textual_app_class
    from textual.widgets import TextArea
    task, context, *_ = default_calls
    with manifest_scope(task, context):
        plan_phase(task, context)
    snapshot = _task_progress_snapshot(task)
    cls = _build_textual_app_class()
    # Invoke the actual app handlers without initializing any user runtime stores.
    app = SimpleNamespace(runtime=SimpleNamespace(user='alex'), decision_work_log={}, spinner_index=0, reply_message_id='',
        query_one=lambda *args: SimpleNamespace(update=lambda text: None), _is_tui_reply_event=lambda event: False)
    event = CoreActivityEvent(phase=ImprovementPhase.PLAN, label='captured', message='', task_id=task.task_id, user='alex', progress=snapshot)
    cls._append_activity_event(app, event)
    cls._append_activity_event(app, CoreActivityEvent(phase=ImprovementPhase.PLAN, label='private', message='', task_id='other', user='other', progress={'decision_context_manifests': [{'decision_id': 'PRIVATE'}]}))
    assert 'other' not in app.decision_work_log
    screens = []
    app.push_screen = screens.append
    task.goal = 'LATER REQUEST'
    cls.action_decision_work_log(app)
    screen = screens[0]
    import asyncio
    from textual.app import App
    async def inspect_screen():
        async with App().run_test(size=(120, 40)) as pilot:
            await pilot.app.push_screen(screen)
            await pilot.pause()
            text_widget = screen.query_one('#decision-manifest-text', TextArea)
            assert 'LATER REQUEST' not in text_widget.text
            assert all(m['decision_id'] in text_widget.text for m in snapshot['decision_context_manifests'])
            assert 'PRIVATE' not in text_widget.text
    asyncio.run(inspect_screen())


def test_default_admission_candidate_clips_correspond_to_the_actual_fields(default_calls):
    task, context, _, calls, *_ = default_calls
    candidate = source('Authorized workflow. ' * 600, 'project:authorized', 'project_guidance')
    with manifest_scope(task, context):
        context.system_one.classify_task_admission(message=task.goal, context_candidates=[{
            'id': 'project:authorized', 'title': 'Long project name. ' * 20, 'content': candidate,
        }])
    state = calls[0]['state']
    assert len(state['context_candidates']['project:authorized']['content']) == 8000
    manifest = task.metadata['decision_context_manifests'][0]
    candidate_record = next(r for r in all_records(manifest) if r['source_id'] == 'project:authorized')
    assert candidate_record['truncated']
    assert candidate_record['field_ranges']['content'][0] == {'start_char': 0, 'end_char': 8000, 'total_chars': len(candidate)}
    assert candidate_record['field_ranges']['title'][0]['end_char'] == 160
    assert_ranges(manifest, {**state, 'questions': calls[0]['questions']})
    assert 'Authorized workflow.' not in json.dumps(manifest)


def test_default_memory_request_suffix_is_a_duplicate_not_a_partial_claim(default_calls):
    task, context, *_ = default_calls
    task.memory_context_md = '# Knowledge\n- [memory:claim; historical] Original whole claim.\n\n# Current Request\n' + 'Large repeated request. ' * 600
    with manifest_scope(task, context):
        text = task_context_for_decision(task, context)
    claim = next(r for r in text.records if r['source_id'] == 'memory:claim')
    assert claim['status'] == 'included' and not claim['truncated']
    assert any(r['source_type'] == 'current_request' and r['status'] == 'omitted' and r['reason'] == 'duplicate' for r in text.records)
    assert 'Original whole claim.' in text and 'Large repeated request.' not in text
