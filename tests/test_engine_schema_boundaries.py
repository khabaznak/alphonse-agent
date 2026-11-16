"""Direct router and owner-processor entry points reject invalid schemas."""
import pytest

from alphonse.agent_v2.core.core import CoreLoopContext, ProcessingStatus
from alphonse.agent_v2.core.intelligence.engine_router import EngineRoutingProcessor
from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_processor import OwnerReadConversationProcessor
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import SQLiteTaskBudgetStore
from alphonse.agent_v2.core.intelligence.task_state import TaskState
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
