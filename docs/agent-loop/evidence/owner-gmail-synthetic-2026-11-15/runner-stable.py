import hashlib,json,os,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path('tests').resolve()))
from test_owner_gmail import f as account_fixture
from test_oauth_lifecycle import connect
from test_gmail_unread import grant
from test_gmail_adapter import Wire
from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.inference.openai_codex import OpenAICodexProvider,OpenAICodexProviderConfig
from alphonse.agent_v2.core.intelligence.simple_loop.owner_gmail import OwnerGmailService
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_search_memory_tool_definition,build_read_memory_source_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
out=Path('docs/agent-loop/evidence/owner-gmail-synthetic-2026-11-15/attempt-stable.json')
paths=['alphonse/agent_v2/core/intelligence/simple_loop/owner_gmail.py','alphonse/agent_v2/core/intelligence/simple_loop/owner_read_processor.py','alphonse/agent_v2/core/intelligence/simple_loop/owner_read_recovery.py','alphonse/agent_v2/core/intelligence/simple_loop/owner_atomic_checkpoint.py','alphonse/agent_v2/core/intelligence/simple_loop/owner_clock.py','alphonse/agent_v2/core/intelligence/simple_loop/candidate_decision_policy.py','alphonse/agent_v2/daemon.py','alphonse/agent_v2/runtime.py','tests/test_owner_gmail.py','tests/test_gmail_adapter.py','tests/test_gmail_unread.py','tests/test_oauth_lifecycle.py','/private/tmp/alphonse_gmail_gym.py']
def hashes():return {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
receipt={'status':'started','model':'gpt-6-luna','provider':'Codex CLI ChatGPT subscription','gmail':'fake transport only','start_source_sha256':hashes()};out.write_text(json.dumps(receipt,indent=2)+'\n');started=time.monotonic();daemon=None
try:
 with tempfile.TemporaryDirectory(prefix='owner-gmail-gym-') as tmp:
  root=Path(tmp);path=root/'fixture.sqlite';env={k:v for k,v in os.environ.items() if not k.startswith('ALPHONSE_')};env['ALPHONSE_V2_DB_PATH']=str(path)
  with patch.dict(os.environ,env,clear=True),patch.object(Path,'home',classmethod(lambda cls:root)):
   f=account_fixture.__wrapped__(root);connect(f);grant(f,'one','grant');wire=Wire({'id':'INBOX','messagesUnread':7,'unexpected':'FAKE_PRIVATE_MAIL_BODY_SENTINEL'})
   class Provider:
    supports_image_input=True
    def __init__(self):self.inner=OpenAICodexProvider(OpenAICodexProviderConfig(timeout_seconds=40));self.decisions=[];self.usage=[];self.prompts=[]
    def generate_json(self,request):
     self.prompts.append(request.prompt);r=self.inner.generate_json(request);self.decisions.append(r.json_value);self.usage.append(r.usage);return r
   queue=SQLiteMessageQueue(path,experimental_claim_fencing=True);conversation=SQLiteConversationStore(path);registry=InMemoryToolRegistry();registry.register(build_search_memory_tool_definition());registry.register(build_read_memory_source_tool_definition())
   provider=Provider();router=InferenceRouter(provider=provider,default_profile=ModelProfile('openai','gpt-6-luna','fake-gmail-gym'))
   runtime=build_runtime_host(user_store=f.accounts.users,integration_store=f.store,messages=queue,question_store=SQLiteQuestionStore(path),conversation_store=conversation,outbox=SQLiteOutboundStore(path),tools=registry,inference=router,coding_skill_service=object(),experimental_owner_read=True,owner_selected_draft_sink=lambda *_:None)
   runtime.owner_gmail_service=OwnerGmailService(f.accounts,lambda _:f.service,transport=wire)
   daemon=V2Daemon(runtime,daemon_id='synthetic-gmail-gym',experimental_owner_only=True);daemon._acquire_single_instance_lock=lambda:None;daemon._release_single_instance_lock=lambda:None;daemon.ipc.start=lambda:None;daemon.ipc.stop=lambda:None;daemon.start()
   mid=queue.enqueue_trusted_local_owner(prompt='How many unread messages are in my Gmail inbox?',peer_uid=os.getuid());deadline=time.monotonic()+120;status='pending'
   while time.monotonic()<deadline:
    status=daemon.owner_message_status(mid)
    if status not in {'pending','processing'}:break
    time.sleep(.05)
   answer='';observed=None;ack=False;restored=False;history=[];kind=None
   if status=='ready':
    cp,ctx=daemon._owner_checkpoints[mid];saved,bindings=cp.load(expected_version=cp.version)
    for b in bindings.values():
     if b.get('tool_id')=='native.owner_gmail_unread':observed=b.get('gmail_observation')
    daemon._lock_file=object();daemon._owner_checkpoints.pop(mid);daemon._owner_statuses.pop(mid);restored=daemon._restore_owner_ready(mid)
    if restored:
     d=daemon.take_owner_desktop_draft(mid);answer=d['message'];kind=d['kind'];daemon.acknowledge_owner_desktop_draft(mid,d['acknowledgement_token']);ack=True;history=conversation.list(owner_user_id='owner')
   requests='\n'.join(provider.prompts);decisions=provider.decisions;calls=[d.get('tool_id') for d in decisions if isinstance(d,dict) and d.get('kind')=='tool']
   checks={'native_gmail_tool_called':'native.owner_gmail_unread' in calls,'fake_gmail_read_once':len(wire.calls)==1 and wire.calls[0][0]=='labels/INBOX','offline_recovery':restored and len(wire.calls)==1,'count_and_scope_correct':'Gmail INBOX unread messages: 7' in answer and 'covers the inbox count only' in answer,'source_time_included':bool(observed and observed['observed_at'] in answer),'private_fake_data_not_in_prompt_or_answer':all(x not in requests and x not in answer for x in ('FAKE_PRIVATE_MAIL_BODY_SENTINEL','disposable-access','grant_id','connection_id')),'answer_kind_correct':kind=='answer','fixture_acknowledged':ack and len(history)>=2}
   receipt.update(status=status,answer=answer,observed_at=observed['observed_at'] if observed else None,checks=checks,oracle_pass=all(checks.values()),fake_gmail_call_count=len(wire.calls),decisions=decisions,usage=provider.usage,provider_calls=len(decisions),restored_ready=restored,fixture_acknowledged=ack,history_turn_count=len(history),captured_model_prompts=provider.prompts)
   daemon.stop();daemon=None
except Exception as exc:receipt.update(status='runner_error',error_type=type(exc).__name__,error_code=str(exc)[:160])
finally:
 if daemon is not None:
  try:daemon.stop()
  except Exception:pass
 receipt['elapsed_seconds']=round(time.monotonic()-started,3);receipt['end_source_sha256']=hashes();receipt['source_changed_during_attempt']=receipt['start_source_sha256']!=receipt['end_source_sha256'];out.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'status':receipt['status'],'oracle_pass':receipt.get('oracle_pass'),'elapsed_seconds':receipt['elapsed_seconds'],'calls':receipt.get('provider_calls'),'error_type':receipt.get('error_type')}),flush=True)
