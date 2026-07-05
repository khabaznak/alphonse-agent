from __future__ import annotations

from datetime import datetime

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.core import ProcessingStatus
from alphonse.agent_v2.core.intelligence import BasicIntelligenceProcessor
from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.core.messages import InMemoryMessageQueue


def test_basic_intelligence_returns_response_for_normal_prompt() -> None:
    processor = BasicIntelligenceProcessor()
    message = CoreMessage(timestamp=datetime.now().astimezone(), prompt="hello", user="alex")

    result = processor.process(message, CoreLoopContext(messages=InMemoryMessageQueue()))

    assert result.status == ProcessingStatus.COMPLETED
    assert result.snapshot.task_owner == "alex"
    assert result.snapshot.current_work == "hello"
    assert result.snapshot.metadata["response"] == "I received your message: hello"


def test_basic_intelligence_acknowledges_slash_command_without_execution() -> None:
    queue = InMemoryMessageQueue()
    channel = CommunicationChannel(queue)
    queued = channel.queue_message(prompt="/project new", user="alex")

    result = BasicIntelligenceProcessor().process(
        queued.message,
        CoreLoopContext(messages=queue),
    )

    assert result.snapshot.metadata["response"] == (
        "Command /project new detected. Command execution is not implemented yet."
    )

