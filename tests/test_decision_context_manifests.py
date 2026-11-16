"""Call-time receipts, privacy and work-log projection contracts."""
import json
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.context_manifests import manifest_scope, manifest_summary, projection_receipt
from alphonse.agent_v2.core.core import CoreLoopContext, _task_progress_snapshot
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.inference import InferenceRequest, InferencePurpose, InferenceResult, InferenceRouter, ModelProfile
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.interfaces.a2ui import A2UiAdapter
from alphonse.agent_v2.system_one import JevCriterionDecisionProvider, SystemOneSettings


def setup_call(provider=None):
    task = TaskState(user='alex', task_id='task', goal='original request')
    context = CoreLoopContext(messages=InMemoryMessageQueue(), system_one=SimpleNamespace(judge_recall=lambda **kw: False))
    telemetry = []
    router = InferenceRouter(provider=provider or SimpleNamespace(generate_json=lambda r: InferenceResult(json_value={})),
                             default_profile=ModelProfile('test', 'test', 'test'), telemetry_sink=telemetry.append)
    return task, context, router, telemetry






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






def test_legacy_context_is_explicit_and_readable():
    envelopes = A2UiAdapter().task_progress('old', {'facts': 'old facts'})
    texts = {c['id']: c.get('text', '') for c in envelopes[1]['updateComponents']['components']}
    assert 'Legacy work log' in texts['decision_context']
    assert 'task_context' not in texts
    assert 'old facts' not in str(envelopes)
    assert 'no call-time' in manifest_summary([])








def test_saved_manifest_without_optional_source_reason_still_renders_work_log():
    from copy import deepcopy
    task, context, router, _ = setup_call()
    with manifest_scope(task, context):
        router.generate_json(InferenceRequest(prompt='Synthetic answer interpretation',
            purpose=InferencePurpose.PHASE_PLANNING,
            input_sources=({'source_id':'question-answer:fixture','source_type':'authenticated_question_answer'},)))
    manifest = task.metadata['decision_context_manifests'][0]
    manifest['omissions'] = [{'source_id':'conversation:omitted','source_type':'conversation'}]
    saved = deepcopy(manifest)
    progress = _task_progress_snapshot(task)
    envelopes = A2UiAdapter().task_progress('fixture', progress)
    texts = {c['id']:c.get('text','') for c in envelopes[1]['updateComponents']['components']}
    assert 'question-answer:fixture · authenticated_question_answer · included · unspecified' in texts['decision_context']
    assert 'conversation:omitted · conversation · omitted · unspecified' in texts['decision_context']
    assert manifest == saved  # Display compatibility must not rewrite stored evidence.
    manifest['sources'][0]['reason'] = 'clarification_answer'
    assert 'included · clarification_answer' in manifest_summary([manifest])
