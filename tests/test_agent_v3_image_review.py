from types import SimpleNamespace

from alphonse.agent_v2.core.core import CoreLoopContext, ProcessingStatus
from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRouter, ModelProfile, StubInferenceProvider
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import PhaseOutcome, PhasePlan, PhaseStatus, PhaseSubgoal, new_tactical_state
from alphonse.agent_v2.core.intelligence.v3.outer import PhaseReviewStatus, _system_two_acceptance_review
from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
from alphonse.agent_v2.core.intelligence.acceptance_contract import apply_status_patch
from alphonse.agent_v2.core.messages import InMemoryMessageQueue


def test_visual_review_cites_successful_action_instead_of_image_delivery_receipt(tmp_path):
    from alphonse.agent_v2.assets import AttachmentDescriptor, SQLiteAssetStore
    assets = SQLiteAssetStore(':memory:', tmp_path / 'assets')
    image = assets.register_bytes(owner_user_id='alex', descriptor=AttachmentDescriptor('receipt.png','image/png',3), content=b'png',source='test')
    task = TaskState(user='alex', metadata={'asset_ids': [image.asset_id]}, acceptance_criteria_md='- [ ] The receipt total is stated correctly.')
    state = new_tactical_state(PhasePlan(phase_id='receipt',objective='Read receipt',subgoals=(PhaseSubgoal("reply", "Read receipt", "text"),)))
    state.evidence.append({'evidence_ref':'inference-images:delivery','kind':'inference_image_input','status':'provided','asset_ids':[image.asset_id]})
    state.evidence.append({'evidence_ref':'tactical-action:response','tool_id':'native.respond','status':'success','result':{'message':'Total $20'}})
    provider = StubInferenceProvider(json_by_purpose={InferencePurpose.PHASE_REVIEW:{'updates':[{'criterion_id':'ac-1','status':'satisfied','evidence_refs':['tactical-action:response']}]}})
    context = CoreLoopContext(messages=InMemoryMessageQueue(),asset_store=assets,inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test','selected','test')))
    result = _system_two_acceptance_review(task,state,context)
    request = provider.requests[0]
    assert request.images[0].asset_id == image.asset_id
    properties = request.output_schema['properties']['updates']['items']['properties']
    assert properties['evidence_refs']['type'] == 'array'
    assert properties['evidence_refs']['items']['enum'] == ['tactical-action:response']
    assert 'evidence_ref' not in properties
    assert 'Valid evidence references: ["tactical-action:response"]' in request.prompt
    assert 'inspect them directly' in request.prompt
    assert 'not valid outcome evidence references' in request.prompt
    contract,rejected=apply_status_patch(task.ensure_acceptance_contract(), result,valid_evidence_refs={'tactical-action:response'})
    assert not rejected
    assert contract['criteria'][0]['status']=='satisfied'
    _, rejected=apply_status_patch(task.ensure_acceptance_contract(), {'updates':[{'criterion_id':'ac-1','status':'satisfied','evidence_refs':['inference-images:delivery']}]},valid_evidence_refs={'tactical-action:response'})
    assert rejected == ['invalid_evidence:ac-1']


def _run_replans(monkeypatch, *, distinct_results):
    task=TaskState(user='alex',task_id='receipt',goal='Read receipt',acceptance_criteria_md='- [ ] Receipt is verified.')
    processor=HierarchicalCAPDProcessor()
    monkeypatch.setattr(processor,'_admit_initial_human_task',lambda *_:False)
    calls=[]
    monkeypatch.setattr(processor,'_new_state',lambda *_:new_tactical_state(PhasePlan(phase_id=f'phase-{len(calls)}',objective='Read receipt',subgoals=(PhaseSubgoal("reply", "Read receipt", "text"),))))
    class Executor:
        def run(self, task, state, context):
            calls.append(len(calls))
            # Distinct source reads are substantive progress; reworded replies
            # alone cannot establish that the receipt has been verified.
            state.evidence.append({'phase_id':state.phase.phase_id,'evidence_ref':f'action:{len(calls)}','tool_id':'native.read_memory_source' if distinct_results else 'native.respond','arguments':{'source_id':'receipt'} if distinct_results else {'message':'receipt summary'},'status':'success','result':{'content':f'step {len(calls)}'} if distinct_results else {'message':'same summary'}})
            state.status=PhaseStatus.PHASE_COMPLETE
            return PhaseOutcome(state.phase.phase_id,PhaseStatus.PHASE_COMPLETE)
    class Outer:
        def review_and_route(self, task, state, outcome, context):
            if len(calls)==6:
                task.status='completed';task.metadata['v3_route']='respond_and_end'
            else:task.metadata['v3_route']='strategic_replan'
            return SimpleNamespace(status=PhaseReviewStatus.PHASE_VERIFIED_TASK_INCOMPLETE),None
    processor.executor=Executor();processor.outer=Outer()
    result=processor.process(task,CoreLoopContext(messages=InMemoryMessageQueue()))
    return task,calls,result


def test_new_phase_and_action_ids_cannot_keep_repeating_the_same_unverified_answer(monkeypatch):
    task,calls,result=_run_replans(monkeypatch,distinct_results=False)
    assert len(calls)==3
    assert result.status==ProcessingStatus.FAILED
    assert task.metadata['v3_no_progress_stop'] is True


def test_distinct_successful_results_still_allow_multiphase_work(monkeypatch):
    task,calls,result=_run_replans(monkeypatch,distinct_results=True)
    assert len(calls)==6
    assert 'v3_no_progress_stop' not in task.metadata
