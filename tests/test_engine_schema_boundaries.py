"""Direct router and owner-processor entry points reject invalid schemas."""
from datetime import datetime, timezone

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext, CoreMessage, ProcessingStatus
from alphonse.agent_v2.core.intelligence.engine_router import EngineRoutingProcessor
from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_processor import OwnerReadConversationProcessor
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import SQLiteTaskBudgetStore
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.core.messages.queue import InMemoryMessageQueue
from alphonse.agent_v2.users import V2UserStore


@pytest.mark.parametrize("schema", [True, False, 1.0, "1", None, 0, 2])
def test_router_rejects_noninteger_and_wrong_schema_without_dispatch(schema):
    candidate = object()
    router = EngineRoutingProcessor(owner_read_candidate=candidate)
    task = TaskState(intelligence_engine="single_loop_experimental", intelligence_schema_version=schema)
    with pytest.raises(ValueError, match="intelligence_engine_unregistered"):
        router.processor_for(task)


@pytest.mark.parametrize("engine", ["", "tactical_v2", "hierarchical_v3", "unknown"])
def test_router_rejects_legacy_and_unknown_stamps(engine):
    router = EngineRoutingProcessor(owner_read_candidate=object())
    with pytest.raises(ValueError, match="intelligence_engine_unregistered"):
        router.processor_for(TaskState(intelligence_engine=engine, intelligence_schema_version=1))


def test_router_accepts_only_registered_exact_integer_schema():
    candidate = object()
    router = EngineRoutingProcessor(owner_read_candidate=candidate)
    assert router.processor_for(TaskState(intelligence_engine="single_loop_experimental",
                                          intelligence_schema_version=1)) is candidate
    with pytest.raises(ValueError, match="intelligence_engine_unregistered"):
        EngineRoutingProcessor().processor_for(TaskState(intelligence_engine="single_loop_experimental",
                                                        intelligence_schema_version=1))


@pytest.mark.parametrize("schema", [True, False, 1.0, "1", None, 0, -1, 2, [], {}])
@pytest.mark.parametrize("source", ["message", "checkpoint"])
def test_deserialization_does_not_upgrade_invalid_schema_to_active_processor(schema, source):
    stamp = {"intelligence_engine": "single_loop_experimental", "intelligence_schema_version": schema}
    if source == "message":
        task = TaskState.from_message(CoreMessage(timestamp=datetime.now(timezone.utc), prompt="fixture",
                                                  user="owner", metadata=stamp))
    else:
        task = TaskState.from_dict(stamp)
    assert type(task.intelligence_schema_version) is type(schema)
    assert task.intelligence_schema_version == schema
    with pytest.raises(ValueError, match="intelligence_engine_unregistered"):
        EngineRoutingProcessor(owner_read_candidate=object()).processor_for(task)


@pytest.mark.parametrize("schema", [True, False, 1.0, "1", None, 0, -1, 2])
def test_ingress_rejects_invalid_schema_before_queueing(schema):
    queue = InMemoryMessageQueue()
    with pytest.raises(ValueError, match="intelligence_schema_invalid"):
        CommunicationChannel(queue).queue_message(prompt="fixture", user="owner",
                                                   metadata={"intelligence_schema_version": schema})
    assert queue.peek() is None


@pytest.mark.parametrize("schema", [True, 1.0, "1", None])
def test_owner_processor_rejects_invalid_schema_before_claim_check(tmp_path, schema):
    authorized = []
    processor = OwnerReadConversationProcessor(
        users=V2UserStore(":memory:"), budgets=SQLiteTaskBudgetStore(str(tmp_path / "budget.sqlite")),
        authorize_claim=lambda *_: authorized.append(True) or True,
        save_checkpoint=lambda *_: None,
    )
    task = TaskState(task_id="message", message_id="message", user="owner", goal="Synthetic",
                     intelligence_engine="single_loop_experimental", intelligence_schema_version=schema)
    result = processor.process(task, CoreLoopContext(messages=InMemoryMessageQueue()))
    assert result.status == ProcessingStatus.FAILED
    assert result.error == "owner_read_authority_unavailable"
    assert authorized == []
