import hashlib,json,os,tempfile,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from unittest.mock import patch
from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.inference.openai_codex import OpenAICodexProvider,OpenAICodexProviderConfig
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_search_memory_tool_definition,build_read_memory_source_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.owner_desktop_reminders import OwnerDesktopReminders
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore
folder=Path('docs/agent-loop/evidence/owner-reminder-calendar-2026-11-15')
out=folder/'attempt.json'
paths=['alphonse/agent_v2/core/intelligence/simple_loop/owner_reminder_create.py','alphonse/agent_v2/core/intelligence/simple_loop/owner_read_processor.py','alphonse/agent_v2/core/intelligence/simple_loop/owner_read_recovery.py','alphonse/agent_v2/core/intelligence/simple_loop/owner_atomic_checkpoint.py','alphonse/agent_v2/core/intelligence/simple_loop/owner_clock.py','alphonse/agent_v2/core/intelligence/simple_loop/candidate_decision_policy.py','alphonse/agent_v2/owner_desktop_reminders.py','alphonse/agent_v2/daemon.py','alphonse/agent_v2/runtime.py','tests/test_owner_conversational_reminder.py',str(folder/'runner.py'),str(folder/'plan.json')]
def hashes():return {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
receipt={'status':'started','provider':'Codex CLI subscription','model':'gpt-6-luna','start_source_sha256':hashes()};out.write_text(json.dumps(receipt,indent=2)+'\n');started=time.monotonic();daemon=None
try:
 with tempfile.TemporaryDirectory(prefix='owner-reminder-gym-') as tmp:
  root=Path(tmp);path=root/'fixture.sqlite';env={k:v for k,v in os.environ.items() if not k.startswith('ALPHONSE_')};env['ALPHONSE_V2_DB_PATH']=str(path)
  with patch.dict(os.environ,env,clear=True),patch.object(Path,'home',classmethod(lambda cls:root)):
   users=V2UserStore(path);owner=users.onboard(display_name='Synthetic owner',users_root=root/'users');owner=users.update_user(owner.user_id,role='owner');users.set_timezone('America/Mexico_City',owner.user_id)
   class Provider:
    supports_image_input=True
    def __init__(self):self.inner=OpenAICodexProvider(OpenAICodexProviderConfig(timeout_seconds=40));self.decisions=[];self.usage=[];self.prompts=[]
    def generate_json(self,request):
     self.prompts.append(request.prompt);r=self.inner.generate_json(request);self.decisions.append(r.json_value);self.usage.append(r.usage);return r
   provider=Provider();queue=SQLiteMessageQueue(path,experimental_claim_fencing=True);conversation=SQLiteConversationStore(path);registry=InMemoryToolRegistry();registry.register(build_search_memory_tool_definition());registry.register(build_read_memory_source_tool_definition())
   runtime=build_runtime_host(user_store=users,messages=queue,question_store=SQLiteQuestionStore(path),conversation_store=conversation,outbox=SQLiteOutboundStore(path),tools=registry,inference=InferenceRouter(provider=provider,default_profile=ModelProfile('openai','gpt-6-luna','synthetic-reminder-gym')),coding_skill_service=object(),experimental_owner_read=True,owner_selected_draft_sink=lambda *_:None)
   daemon=V2Daemon(runtime,daemon_id='synthetic-reminder-gym',experimental_owner_only=True);daemon._acquire_single_instance_lock=lambda:None;daemon._release_single_instance_lock=lambda:None;daemon.ipc.start=lambda:None;daemon.ipc.stop=lambda:None;daemon.start()
   local_now=datetime.now(ZoneInfo('America/Mexico_City'));tomorrow=local_now.date()+timedelta(days=1);due=datetime(tomorrow.year,tomorrow.month,tomorrow.day,9,tzinfo=ZoneInfo('America/Mexico_City')).astimezone(timezone.utc).isoformat();goal='Please remind me to call the plumber tomorrow at 9 am';mid=queue.enqueue_trusted_local_owner(prompt=goal,peer_uid=os.getuid());deadline=time.monotonic()+120;status='pending'
   while time.monotonic()<deadline:
    status=daemon.owner_message_status(mid)
    if status not in {'pending','processing'}:break
    time.sleep(.05)
   store=OwnerDesktopReminders(str(path));before=store.list_open(owner.user_id);answer='';kind=None;restored=False;ack=False;history=[];calls_before=len(provider.decisions)
   if status=='ready':
    daemon._lock_file=object();daemon._owner_checkpoints.pop(mid);daemon._owner_statuses.pop(mid);restored=daemon._restore_owner_ready(mid)
    if restored:
     draft=daemon.take_owner_desktop_draft(mid);answer=draft['message'];kind=draft['kind'];daemon.acknowledge_owner_desktop_draft(mid,draft['acknowledgement_token']);ack=True;history=conversation.list(owner_user_id=owner.user_id)
   after=store.list_open(owner.user_id);decisions=provider.decisions
   checks={'clock_selected':any(d.get('tool_id')=='native.owner_clock' for d in decisions if isinstance(d,dict)),'reminder_create_selected':any(d.get('tool_id')=='native.owner_reminder_create' for d in decisions if isinstance(d,dict)),'one_exact_reminder_saved':len(before)==1 and before[0].get('message','').casefold()=='call the plumber' and datetime.fromisoformat(before[0]['due_at'])==datetime.fromisoformat(due),'truthful_confirmation':kind=='answer' and answer.casefold().startswith('reminder saved: call the plumber') and 'Desktop is open' in answer,'recovered_without_duplicate_or_model_call':restored and before==after and len(after)==1 and len(provider.decisions)==calls_before,'confirmation_acknowledged':ack and len(history)>=2 and history[-1].content==answer}
   receipt.update(status=status,goal=goal,answer=answer,reminders_before_recovery=before,reminders_after_recovery=after,checks=checks,oracle_pass=all(checks.values()),decisions=decisions,usage=provider.usage,provider_calls=len(decisions),captured_model_prompts=provider.prompts)
   daemon.stop();daemon=None
except Exception as exc:receipt.update(status='runner_error',error_type=type(exc).__name__,error_code=str(exc)[:200])
finally:
 if daemon is not None:
  try:daemon.stop()
  except Exception:pass
 receipt['elapsed_seconds']=round(time.monotonic()-started,3);receipt['end_source_sha256']=hashes();receipt['source_changed_during_attempt']=receipt['start_source_sha256']!=receipt['end_source_sha256'];out.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'status':receipt['status'],'oracle_pass':receipt.get('oracle_pass'),'elapsed_seconds':receipt['elapsed_seconds'],'calls':receipt.get('provider_calls'),'error_type':receipt.get('error_type')}),flush=True)
