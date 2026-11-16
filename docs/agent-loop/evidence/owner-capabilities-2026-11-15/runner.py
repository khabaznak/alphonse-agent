import hashlib,json,os,tempfile,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
from unittest.mock import patch
from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.inference.openai_codex import OpenAICodexProvider,OpenAICodexProviderConfig
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_search_memory_tool_definition,build_read_memory_source_tool_definition
from alphonse.agent_v2.core.tools.registry.native.skill_catalog import build_skill_catalog_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.users import V2UserStore
out=Path('docs/agent-loop/evidence/owner-capabilities-2026-11-15')
paths=['alphonse/agent_v2/core/intelligence/simple_loop/owner_read_processor.py','alphonse/agent_v2/core/intelligence/simple_loop/owner_read_recovery.py','alphonse/agent_v2/core/intelligence/simple_loop/owner_schedules.py','alphonse/agent_v2/core/tools/registry/native/skill_catalog.py','alphonse/agent_v2/core/intelligence/simple_loop/candidate_decision_policy.py','alphonse/agent_v2/daemon.py','alphonse/agent_v2/runtime.py']
def hashes():return {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
class Provider:
 supports_image_input=True
 def __init__(self):self.inner=OpenAICodexProvider(OpenAICodexProviderConfig(timeout_seconds=40));self.decisions=[];self.usage=[];self.prompts=[]
 def generate_json(self,request):
  self.prompts.append(request.prompt);r=self.inner.generate_json(request);self.decisions.append(r.json_value);self.usage.append(r.usage);return r
for name in ('schedules','skills'):
 result_path=out/(name+'.json');receipt={'case':name,'status':'started','provider':'Codex CLI ChatGPT subscription','model':'gpt-6-luna','delivery':'fixture pickup and acknowledgement only','start_source_sha256':hashes()};result_path.write_text(json.dumps(receipt,indent=2)+'\n');started=time.monotonic();daemon=None
 try:
  with tempfile.TemporaryDirectory(prefix='owner-capability-gym-') as tmp:
   root=Path(tmp);path=root/'fixture.sqlite';env={k:v for k,v in os.environ.items() if not k.startswith('ALPHONSE_')};env['ALPHONSE_V2_DB_PATH']=str(path)
   with patch.dict(os.environ,env,clear=True),patch.object(Path,'home',classmethod(lambda cls:root)):
    users=V2UserStore(path);owner=users.onboard(display_name='Synthetic owner',users_root=root/'users')
    schedule_store=ScheduledTaskStore(path);future=(datetime.now(timezone.utc)+timedelta(days=3)).isoformat()
    schedule_store.create_task(owner_user_id=owner.user_id,name='Water basil',prompt='Water basil plants',schedule_kind='once',run_at=future)
    paused=schedule_store.create_task(owner_user_id=owner.user_id,name='Call locksmith',prompt='Call locksmith about fixture',schedule_kind='once',run_at=future);schedule_store.pause_task(paused.scheduled_task_id)
    schedule_store.create_task(owner_user_id='other-fixture-owner',name='OTHER_OWNER_PRIVATE_SENTINEL',prompt='OTHER_OWNER_PRIVATE_SENTINEL',schedule_kind='once',run_at=future)
    skills_root=root/'skills';skills_root.mkdir()
    for slug,description in [('garden-guide','Synthetic expertise for garden care.'),('pantry-guide','Synthetic expertise for pantry organization.'),('private-sentinel','PRIVATE_SKILL_DESCRIPTION_SENTINEL')]:
     directory=skills_root/slug;directory.mkdir();(directory/'SKILL.md').write_text(f'---\nname: {slug}\ndescription: {description}\n---\nPRIVATE_SKILL_INSTRUCTION_SENTINEL\n')
    skill_store=SkillStore(skills_root);skill_store._bundled_skills_dir=lambda:root/'absent-bundles';skill_store.can_use=lambda skill_id,actor:actor==owner.user_id and skill_id!='skill:private-sentinel'
    queue=SQLiteMessageQueue(path,experimental_claim_fencing=True);conversation=SQLiteConversationStore(path);registry=InMemoryToolRegistry();registry.register(build_search_memory_tool_definition());registry.register(build_read_memory_source_tool_definition());registry.register(build_skill_catalog_tool_definition())
    provider=Provider();router=InferenceRouter(provider=provider,default_profile=ModelProfile('openai','gpt-6-luna','owner-capability-gym'))
    runtime=build_runtime_host(user_store=users,skill_store=skill_store,messages=queue,question_store=SQLiteQuestionStore(path),conversation_store=conversation,outbox=SQLiteOutboundStore(path),schedule_store=schedule_store,tools=registry,inference=router,coding_skill_service=object(),experimental_owner_read=True,owner_selected_draft_sink=lambda *_:None)
    daemon=V2Daemon(runtime,daemon_id='synthetic-owner-gym',experimental_owner_only=True);daemon._acquire_single_instance_lock=lambda:None;daemon._release_single_instance_lock=lambda:None;daemon.ipc.start=lambda:None;daemon.ipc.stop=lambda:None;daemon.start()
    prompt='What reminders do I have? Include any I paused.' if name=='schedules' else 'Which installed skills can you see? List their names.'
    message_id=queue.enqueue_trusted_local_owner(prompt=prompt,peer_uid=os.getuid());deadline=time.monotonic()+130;status='pending'
    while time.monotonic()<deadline:
     status=daemon.owner_message_status(message_id)
     if status not in {'pending','processing'}:break
     time.sleep(.05)
    draft='';ack=False;history=[]
    if status=='ready':
     selected=daemon.take_owner_desktop_draft(message_id);draft=selected['message'];daemon.acknowledge_owner_desktop_draft(message_id,selected['acknowledgement_token']);ack=True;history=conversation.list(owner_user_id=owner.user_id)
    calls=[d.get('tool_id') for d in provider.decisions if isinstance(d,dict) and d.get('kind')=='tool'];lower=draft.casefold();requests='\n'.join(provider.prompts)
    if name=='schedules':
     checks={'native_tool_called':'native.owner_schedules' in calls,'owner_active_named':'basil' in lower,'owner_paused_named':'locksmith' in lower,'truthful_schedule_notice':'Earlier scheduled work is paused. New Desktop reminders appear only while the Desktop conversation is open.' in draft,'other_owner_not_visible':'OTHER_OWNER_PRIVATE_SENTINEL' not in requests and 'OTHER_OWNER_PRIVATE_SENTINEL' not in draft,'fixture_acknowledged':ack and len(history)>=2}
    else:
     checks={'native_tool_called':'native.skill_catalog' in calls,'visible_garden_named':'garden-guide' in lower,'visible_pantry_named':'pantry-guide' in lower,'denied_skill_not_visible':all(x not in requests and x.casefold() not in lower for x in ('private-sentinel','PRIVATE_SKILL_DESCRIPTION_SENTINEL','PRIVATE_SKILL_INSTRUCTION_SENTINEL')),'fixture_acknowledged':ack and len(history)>=2}
    receipt.update(status=status,answer=draft,checks=checks,oracle_pass=all(checks.values()),decisions=provider.decisions,usage=provider.usage,provider_calls=len(provider.decisions),fixture_acknowledged=ack,history_turn_count=len(history),scheduler_started=daemon.scheduler._thread is not None)
    daemon.stop();daemon=None
 except Exception as exc:receipt.update(status='runner_error',error_type=type(exc).__name__,error_code=str(exc)[:120])
 finally:
  if daemon is not None:
   try:daemon.stop()
   except Exception:pass
  receipt['elapsed_seconds']=round(time.monotonic()-started,3);receipt['end_source_sha256']=hashes();receipt['source_changed_during_attempt']=receipt['start_source_sha256']!=receipt['end_source_sha256'];result_path.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'case':name,'status':receipt['status'],'oracle_pass':receipt.get('oracle_pass'),'elapsed_seconds':receipt['elapsed_seconds'],'calls':receipt.get('provider_calls')}),flush=True)
