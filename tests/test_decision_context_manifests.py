"""Call-time receipts, privacy and work-log projection contracts."""
import json
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.context_manifests import manifest_scope, manifest_summary, projection_receipt
from alphonse.agent_v2.core.core import CoreLoopContext, _task_progress_snapshot
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.inference import InferenceRequest, InferencePurpose, InferenceResult, InferenceRouter, ModelProfile
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.projections import project, assemble, Record
from alphonse.agent_v2.interfaces.a2ui import A2UiAdapter
from alphonse.agent_v2.system_one import JevCriterionDecisionProvider, SystemOneSettings


def setup_call(provider=None):
    task = TaskState(user='alex', task_id='task', goal='original request')
    context = CoreLoopContext(messages=InMemoryMessageQueue(), system_one=SimpleNamespace(judge_recall=lambda **kw: False))
    telemetry = []
    router = InferenceRouter(provider=provider or SimpleNamespace(generate_json=lambda r: InferenceResult(json_value={})),
                             default_profile=ModelProfile('test', 'test', 'test'), telemetry_sink=telemetry.append)
    return task, context, router, telemetry


def test_receipt_matches_call_not_later_state_and_never_retains_raw_context():
    captured = []
    def generate(request):
        captured.append(request)
        return InferenceResult(json_value={})
    task, context, router, telemetry = setup_call(SimpleNamespace(generate_json=generate))
    task.facts_md = 'PRIVATE MATERIAL password=12345'
    with manifest_scope(task, context):
        rendered = project(task, context, purpose='planning')
        router.generate_json(InferenceRequest(prompt='Instructions\n' + rendered, context_manifests=projection_receipt(rendered), purpose=InferencePurpose.PHASE_PLANNING, task_id=task.task_id))
        task.goal = 'later state'
        project(task, context, purpose='planning')
    manifest = task.metadata['decision_context_manifests'][0]
    assert manifest['decision_id'] == captured[0].metadata['decision_id']
    assert manifest['sections'][0]['tokens'] == len(captured[0].prompt.encode())
    assert manifest['projections'][0]['tokens'] == len(rendered.encode())
    assert 'request:task' in manifest['projections'][0]['source_ids']
    task.metadata['v3_context_projections']['planning']['source_ids'].clear()
    assert manifest['projections'][0]['source_ids']
    exported = json.dumps(manifest)
    assert 'PRIVATE MATERIAL' not in exported and '12345' not in exported
    assert 'original request' not in json.dumps(telemetry)
    progress = _task_progress_snapshot(task)
    task.metadata['decision_context_manifests'].clear()
    assert progress['decision_context_manifests'] == [manifest]
    envelopes = A2UiAdapter().task_progress('task', progress)
    components = envelopes[1]['updateComponents']['components']
    text = next(c['text'] for c in components if c['id'] == 'decision_context')
    assert manifest['decision_id'] in text and 'PRIVATE MATERIAL' not in text
    assert envelopes[2]['updateDataModel']['value']['decision_context_manifests'] == [manifest]


def test_only_projection_present_in_actual_call_is_associated():
    task, context, router, _ = setup_call()
    with manifest_scope(task, context):
        project(task, context, purpose='review')
        router.generate_json(InferenceRequest(prompt='An unrelated exact prompt', purpose=InferencePurpose.PHASE_REVIEW))
    manifest = task.metadata['decision_context_manifests'][0]
    assert all('request:task' not in p['source_ids'] for p in manifest['projections'])
    assert manifest['provenance'] == 'source_manifest'
    assert manifest['projections'][0]['input_section'] == 'prompt'


def test_failed_calls_repeated_calls_and_scope_isolation():
    def fail(request):
        raise ValueError('PRIVATE FAILURE')
    task, context, router, telemetry = setup_call(SimpleNamespace(generate_json=fail))
    with manifest_scope(task, context):
        for _ in range(2):
            with pytest.raises(ValueError):
                router.generate_json(InferenceRequest(prompt='secret', purpose=InferencePurpose.PHASE_REVIEW))
    rows = task.metadata['decision_context_manifests']
    assert len({m['decision_id'] for m in rows}) == 2
    assert all(m['status'] == 'failed' for m in rows)
    assert 'PRIVATE FAILURE' not in json.dumps(rows)
    other, context, router, _ = setup_call()
    router.generate_json(InferenceRequest(prompt='outside task', purpose=InferencePurpose.PHASE_PLANNING))
    assert 'decision_context_manifests' not in other.metadata


def test_jev_receipts_capture_admission_curation_and_review_inputs():
    task, context, _, _ = setup_call()
    captured = []
    decider = JevCriterionDecisionProvider(SystemOneSettings())
    def evaluate(**kwargs):
        captured.append(kwargs)
        return {'answers': {}}
    decider.client = SimpleNamespace(evaluate=evaluate)
    with manifest_scope(task, context):
        projection = project(task, context, purpose='admission')
        for purpose in ('admission', 'tool_curation', 'completion_review'):
            decider._evaluate(purpose=purpose, state={'decision_context': projection}, questions={'test': {'instructions': 'PRIVATE QUESTION'}})
    rows = task.metadata['decision_context_manifests']
    assert [r['purpose'] for r in rows] == ['admission', 'tool_curation', 'completion_review']
    for row, call in zip(rows, captured):
        assert row['sections'][0]['tokens'] == len(call['state']['decision_context'].encode())
        assert row['projections'][0]['source_ids']
        assert 'PRIVATE QUESTION' not in json.dumps(row)


def test_whole_record_budget_omission_and_duplicate_are_explicit():
    _, manifest = assemble([Record('a', 'Required', 'policy', True), Record('a', 'Duplicate', 'same'), Record('b', 'Large', 'x' * 1000)], budget=100)
    assert {(r['source_id'], r['reason']) for r in manifest['records']} == {('a', 'required'), ('a', 'duplicate'), ('b', 'over_budget')}
    assert sum(r['tokens'] for r in manifest['records']) + manifest['separator_tokens'] == manifest['tokens']


def test_legacy_context_is_explicit_and_readable():
    envelopes = A2UiAdapter().task_progress('old', {'facts': 'old facts'})
    texts = {c['id']: c.get('text', '') for c in envelopes[1]['updateComponents']['components']}
    assert 'Legacy work log' in texts['decision_context']
    assert 'Task state diagnostics' in texts['task_context']
    assert 'old facts' in texts['task_context']
    assert 'no call-time' in manifest_summary([])


def test_ipc_manifest_and_current_source_expansion_are_owner_scoped(tmp_path):
    from alphonse.agent_v2.users import V2UserStore
    from alphonse.agent_v2.runtime import build_runtime_host
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
    users = V2UserStore(':memory:')
    admin = users.onboard(display_name='Admin', users_root=tmp_path / 'users')
    member = users.create_user(display_name='Member', role='member')
    _, _, router, _ = setup_call()
    runtime = build_runtime_host(user_store=users, schedule_store=ScheduledTaskStore(':memory:'), inference=router)
    daemon = V2Daemon(runtime)
    event = runtime.conversation_store.record(owner_user_id=admin.user_id, content='original source', role='user', source='test', source_message_id='prior')
    private = runtime.conversation_store.record(owner_user_id=member.user_id, content='PRIVATE SOURCE', role='user', source='test', source_message_id='private')
    task = TaskState(task_id='captured', user=admin.user_id, goal='Remember prior source')
    context = CoreLoopContext(messages=InMemoryMessageQueue(), conversation_store=runtime.conversation_store,
                              question_store=runtime.question_store, system_one=SimpleNamespace(judge_recall=lambda **kw: None))
    with manifest_scope(task, context):
        rendered = project(task, context, purpose='planning')
        router.generate_json(InferenceRequest(prompt=rendered, context_manifests=projection_receipt(rendered), purpose=InferencePurpose.PHASE_PLANNING))
    manifest = task.metadata['decision_context_manifests'][0]
    assert f'conversation:{private.event_id}' not in json.dumps(manifest)
    assert 'PRIVATE SOURCE' not in json.dumps(manifest)
    params = {'actor_user_id': admin.user_id, 'task_id': task.task_id}
    payload = daemon.ipc._dispatch({'method': 'decision_context', 'params': params})
    assert payload['manifests'] == [manifest]
    params.update(decision_id=manifest['decision_id'], source_id=f'conversation:{event.event_id}')
    result = daemon.ipc._dispatch({'method': 'decision_context', 'params': params})
    assert result['status'] == 'found'
    assert result['source']['items'][0]['content'] == 'original source'
    runtime.conversation_store.delete(event_id=event.event_id, owner_user_id=admin.user_id)
    revoked = daemon.ipc._dispatch({'method': 'decision_context', 'params': params})
    assert revoked['status'] == 'not_found' and 'original source' not in str(revoked)
    params['source_id'] = f'conversation:{private.event_id}'
    forbidden = daemon.ipc._dispatch({'method': 'decision_context', 'params': params})
    assert forbidden == {'status': 'not_found', 'manifests': []}
    params['source_id'] = 'conversation:missing'
    assert daemon.ipc._dispatch({'method': 'decision_context', 'params': params}) == forbidden
    params['actor_user_id'] = member.user_id
    with pytest.raises(PermissionError):
        daemon.ipc._dispatch({'method': 'decision_context', 'params': params})
    params['actor_user_id'] = ''
    with pytest.raises(PermissionError):
        daemon.ipc._dispatch({'method': 'decision_context', 'params': params})
    task.user = member.user_id
    runtime.question_store.save_task_checkpoint(task, status='running')
    params.update(actor_user_id=admin.user_id, source_id='')
    assert daemon.ipc._dispatch({'method': 'decision_context', 'params': params}) == forbidden


def test_production_processing_captures_real_admission_and_response(monkeypatch):
    from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', '1')
    task, context, router, _ = setup_call(SimpleNamespace(generate_markdown=lambda r: InferenceResult(content='Hello.')))
    task.message_id = task.task_id
    task.metadata['routing_disposition'] = 'pdca_task'
    decider = JevCriterionDecisionProvider(SystemOneSettings())
    captured = []
    def evaluate(**kwargs):
        captured.append(kwargs)
        return {'answers': {'recall_needed': {'type': 'noul', 'noul': .01}, 'requires_task': {'type': 'noul', 'noul': .01}}}
    decider.client = SimpleNamespace(evaluate=evaluate)
    context.system_one = decider
    context.inference = router
    HierarchicalCAPDProcessor().process(task, context)
    assert task.status == 'completed'
    rows = task.metadata['decision_context_manifests']
    admission = next(row for row in rows if row['purpose'] == 'admission')
    response = next(row for row in rows if row['purpose'] == 'final_response')
    assert admission['projections'][0]['purpose'] == 'admission'
    assert response['projections'][0]['purpose'] == 'response'
    assert all(row['status'] == 'returned' for row in rows)
    assert len({row['decision_id'] for row in rows}) == len(rows)


def test_projection_omissions_and_source_excerpt_ranges_are_metadata_only(monkeypatch):
    from alphonse.agent_v2.core.intelligence.v3 import projections
    task, context, _, _ = setup_call()
    monkeypatch.setattr(projections, 'retrieve', lambda *args: {
        'status': 'partial', 'coverage': {'knowledge': 'unavailable'}, 'further_search_required': True,
        'omissions': [{'source_id': 'conversation:irrelevant', 'source_type': 'conversation', 'status': 'omitted', 'reason': 'irrelevant', 'tokens': 0}],
        'results': [{'source_id': 'conversation:recalled', 'excerpt': 'PRIVATE EXCERPT', 'truncated': True, 'start_char': 20, 'end_char': 34, 'total_chars': 100}],
    })
    project(task, context, purpose='planning')
    manifest = task.metadata['v3_context_projections']['planning']
    assert 'PRIVATE EXCERPT' not in json.dumps(manifest)
    omissions = {row['source_id']: row['reason'] for row in manifest['records'] if row['status'] == 'omitted'}
    assert omissions['recall:knowledge'] == 'unavailable'
    assert omissions['conversation:irrelevant'] == 'irrelevant'
    recalled = next(row for row in manifest['records'] if row['source_id'] == 'conversation:recalled')
    assert recalled['truncated']
    assert recalled['source_range'] == {'start_char': 20, 'end_char': 34, 'total_chars': 100}
    assert manifest['recall']['result_count'] == 1
