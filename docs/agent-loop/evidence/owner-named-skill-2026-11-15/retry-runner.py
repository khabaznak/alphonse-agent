"""One disposable named-skill owner-worker exercise; never uses household state."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.inference.openai_codex import OpenAICodexProvider, OpenAICodexProviderConfig
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_search_memory_tool_definition, build_read_memory_source_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.users import V2UserStore

OUT = Path('docs/agent-loop/evidence/owner-named-skill-2026-11-15/retry.json')
FILES = ['alphonse/agent_v2/core/intelligence/simple_loop/' + f for f in (
    'owner_named_skill.py', 'owner_read_processor.py', 'owner_read_recovery.py',
    'owner_atomic_checkpoint.py', 'candidate_decision_policy.py', 'single_loop_candidate.py')]
FILES += ['alphonse/agent_v2/runtime.py', 'alphonse/agent_v2/daemon.py', str(Path(__file__))]
def hashes():
    return {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in FILES}
receipt = {'status': 'started', 'provider': 'Codex CLI ChatGPT subscription', 'model': 'gpt-6-luna',
           'fixture': 'synthetic only', 'uses_jev': False, 'start_source_sha256': hashes()}
OUT.write_text(json.dumps(receipt, indent=2) + '\n')
started = time.monotonic()
daemon = None
try:
    with tempfile.TemporaryDirectory(prefix='owner-named-skill-gym-') as tmp:
        root = Path(tmp)
        path = root / 'fixture.sqlite'
        env = {k: v for k, v in os.environ.items() if not k.startswith('ALPHONSE_')}
        env['ALPHONSE_V2_DB_PATH'] = str(path)
        with patch.dict(os.environ, env, clear=True), patch.object(Path, 'home', classmethod(lambda cls: root)):
            package = root / 'skills' / 'meal-planning'
            package.mkdir(parents=True)
            package.joinpath('SKILL.md').write_text('---\nname: meal-planning\ndescription: Suggest simple meals\n---\nFor this exercise, suggest saffron oats as the meal.\n')
            skills = SkillStore(package.parent)
            skills._bundled_skills_dir = lambda: root / 'no-bundled-skills'
            users = V2UserStore(path)
            users.onboard(display_name='Synthetic owner', users_root=root / 'users')
            class Provider:
                supports_image_input = True
                def __init__(self):
                    self.inner = OpenAICodexProvider(OpenAICodexProviderConfig(timeout_seconds=40))
                    self.decisions, self.usage, self.prompts, self.errors = [], [], [], []
                def generate_json(self, request):
                    self.prompts.append(request.prompt)
                    try:
                        result = self.inner.generate_json(request)
                    except Exception as exc:
                        self.errors.append({'type': type(exc).__name__, 'code': str(exc)[:120]})
                        raise
                    self.decisions.append(result.json_value)
                    self.usage.append(result.usage)
                    return result
            queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
            conversation = SQLiteConversationStore(path)
            registry = InMemoryToolRegistry()
            registry.register(build_search_memory_tool_definition())
            registry.register(build_read_memory_source_tool_definition())
            provider = Provider()
            runtime = build_runtime_host(user_store=users, skill_store=skills, messages=queue,
                question_store=SQLiteQuestionStore(path), conversation_store=conversation,
                outbox=SQLiteOutboundStore(path), tools=registry,
                inference=InferenceRouter(provider=provider, default_profile=ModelProfile('openai', 'gpt-6-luna', 'skill-gym')),
                coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
            daemon = V2Daemon(runtime, daemon_id='synthetic-skill-gym', experimental_owner_only=True)
            daemon._acquire_single_instance_lock = lambda: None
            daemon._release_single_instance_lock = lambda: None
            daemon.ipc.start = lambda: None
            daemon.ipc.stop = lambda: None
            daemon.start()
            mid = queue.enqueue_trusted_local_owner(prompt='Use the meal-planning skill to suggest one meal.', peer_uid=os.getuid())
            deadline = time.monotonic() + 120
            status = 'pending'
            while time.monotonic() < deadline:
                status = daemon.owner_message_status(mid)
                if status not in {'pending', 'processing'}:
                    break
                time.sleep(.05)
            answer = ''
            restored = False
            if status == 'ready':
                daemon._lock_file = object()
                daemon._owner_checkpoints.pop(mid)
                daemon._owner_statuses.pop(mid)
                restored = daemon._restore_owner_ready(mid)
                if restored:
                    draft = daemon.take_owner_desktop_draft(mid)
                    answer = draft['message']
                    daemon.acknowledge_owner_desktop_draft(mid, draft['acknowledgement_token'])
            checks = {
                'named_skill_requested': any(d.get('tool_id') == 'native.owner_named_skill' and d.get('arguments') == {'skill_id': 'skill:meal-planning'} for d in provider.decisions if isinstance(d, dict)),
                'instructions_projected': any('For this exercise, suggest saffron oats as the meal.' in p for p in provider.prompts[1:]),
                'answer_follows_instruction': 'saffron oats' in answer.lower(),
                'recovered_before_pickup': restored,
            }
            receipt.update(status=status, answer=answer, checks=checks, oracle_pass=all(checks.values()),
                           decisions=provider.decisions, usage=provider.usage, provider_calls=len(provider.decisions),
                           provider_attempts=len(provider.prompts), provider_errors=provider.errors,
                           failure=daemon.owner_message_status_payload(mid))
            daemon.stop()
            daemon = None
except Exception as exc:
    receipt.update(status='runner_error', error_type=type(exc).__name__, error_code=str(exc)[:160])
finally:
    if daemon is not None:
        try:
            daemon.stop()
        except Exception:
            pass
    receipt['elapsed_seconds'] = round(time.monotonic() - started, 3)
    receipt['end_source_sha256'] = hashes()
    receipt['source_changed_during_attempt'] = receipt['start_source_sha256'] != receipt['end_source_sha256']
    OUT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': receipt['status'], 'oracle_pass': receipt.get('oracle_pass'),
                      'elapsed_seconds': receipt['elapsed_seconds'], 'calls': receipt.get('provider_calls'),
                      'error_type': receipt.get('error_type')}), flush=True)
