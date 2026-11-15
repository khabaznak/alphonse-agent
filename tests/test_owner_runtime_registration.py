"""Explicit runtime registration never enables the fenced worker itself."""
import pytest
import json
import os
import subprocess
import sys

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext, LoopStepStatus, ProcessingStatus
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.owner_read_processor import OwnerReadConversationProcessor
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_read_memory_source_tool_definition, build_search_memory_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def test_owner_import_does_not_load_v3_processor_in_a_fresh_interpreter():
    # Other tests may have imported V3 already; a fresh interpreter verifies
    # the actual import boundary rather than relying on test order.
    subprocess.run([sys.executable, '-c', '''
import sys
import alphonse.agent_v2.runtime
import alphonse.agent_v2.core.intelligence.v3.owner_read_processor
for name in ('processor', 'executor', 'planner', 'outer'):
    assert f'alphonse.agent_v2.core.intelligence.v3.{name}' not in sys.modules, name
from alphonse.agent_v2.core.intelligence.v3 import HierarchicalCAPDProcessor
from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor as Direct
assert HierarchicalCAPDProcessor is Direct
'''], check=True)


def test_explicit_registration_uses_same_store_and_preserves_worker_refusal(tmp_path, monkeypatch):
    def forbid_v3():
        pytest.fail('owner-only runtime must not construct the V3 loop')
    monkeypatch.setattr('alphonse.agent_v2.core.intelligence.v3.HierarchicalCAPDProcessor', forbid_v3)
    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    questions = SQLiteQuestionStore(path)
    callback = lambda checkpoint, context: None
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=questions,
        conversation_store=SQLiteConversationStore(path), outbox=SQLiteOutboundStore(path), tools=InMemoryToolRegistry(),
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=callback)
    candidate = runtime.processor.owner_read_candidate
    assert isinstance(candidate, OwnerReadConversationProcessor)
    assert runtime.processor.v3 is None
    with pytest.raises(ValueError, match='intelligence_engine_unregistered'):
        runtime.processor.processor_for(TaskState(intelligence_engine='hierarchical_v3', intelligence_schema_version=3))
    assert candidate.atomic_checkpoints.store is questions
    assert candidate.selected_draft_sink is callback
    result = runtime.core.step()
    assert (result.status, result.error) == (LoopStepStatus.BUSY, 'claim_fencing_runtime_unavailable')


def test_registration_rejects_unfenced_or_split_stores(tmp_path):
    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    other = SQLiteQuestionStore(tmp_path / 'other.sqlite')
    with pytest.raises(ValueError, match='owner_read_runtime_store_mismatch'):
        build_runtime_host(user_store=users, messages=queue, question_store=other,
            outbox=SQLiteOutboundStore(path), coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    with pytest.raises(ValueError, match='owner_read_runtime_store_mismatch'):
        build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(path),
            conversation_store=SQLiteConversationStore(path), outbox=SQLiteOutboundStore(tmp_path / 'other-outbox.sqlite'),
            coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    with pytest.raises(ValueError, match='owner_read_runtime_configuration_invalid'):
        build_runtime_host(owner_selected_draft_sink=lambda *_: None)


@pytest.mark.parametrize('kind', ['answer', 'question'])
def test_registered_processor_hands_server_checkpoint_to_sink_after_selection(tmp_path, kind):
    class Provider:
        supports_image_input = False
        def generate_json(self, request):
            decision = {'kind': kind, 'tool_id': None, 'arguments': None, 'text': 'Synthetic answer.'}
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 100, 'output_tokens': 20})
    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    questions = SQLiteQuestionStore(path)
    conversations = SQLiteConversationStore(path)
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    inference = InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'fixture', 'offline'))
    holder = {}
    def deliver_to_daemon(checkpoint, context):
        holder['daemon'].register_owner_checkpoint(checkpoint, context)
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=questions,
        conversation_store=conversations, outbox=SQLiteOutboundStore(path), tools=registry, inference=inference, coding_skill_service=object(),
        experimental_owner_read=True, owner_selected_draft_sink=deliver_to_daemon)
    daemon = V2Daemon(runtime, daemon_id='fixture-owner-daemon')
    holder['daemon'] = daemon
    message_id = queue.enqueue_trusted_local_owner(prompt='Synthetic question', peer_uid=os.getuid())
    acquired = queue.claim_next()
    task = TaskState.from_message(acquired.message, message_id=message_id)
    task.task_id = message_id
    context = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt, tools=registry,
        inference=inference, conversation_store=conversations, cancellation_checker=lambda: False)
    result = runtime.processor.owner_read_candidate.process(task, context)
    assert result.status == ProcessingStatus.WAITING, result.error
    assert result.snapshot.metadata['draft'] == 'Synthetic answer.'
    received = daemon._owner_checkpoints[message_id]
    assert received[0].receipt == acquired.claim_receipt
    assert received[1] is context and task.user == owner.user_id
    saved, _ = received[0].load(expected_version=received[0].version)
    assert saved.state.status == ('completed' if kind == 'answer' else 'waiting_user')
