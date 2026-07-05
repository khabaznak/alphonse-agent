from __future__ import annotations

from datetime import datetime
from datetime import timezone

import pytest

from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.messages import MessageSelector


def test_queue_message_uses_mandatory_prompt_and_user() -> None:
    queue = InMemoryMessageQueue()
    channel = CommunicationChannel(queue)

    queued = channel.queue_message(prompt="hello", user="alex")

    assert queued.message.prompt == "hello"
    assert queued.message.user == "alex"
    assert queue.size() == 1


def test_queue_message_defaults_project_id_and_tag_to_empty_strings() -> None:
    queued = CommunicationChannel(InMemoryMessageQueue()).queue_message(prompt="hello", user="alex")

    assert queued.message.project_id == ""
    assert queued.message.tag == ""


def test_queue_message_generates_timezone_aware_local_timestamp() -> None:
    queued = CommunicationChannel(InMemoryMessageQueue()).queue_message(prompt="hello", user="alex")

    assert queued.message.timestamp.tzinfo is not None
    assert queued.message.timestamp.utcoffset() is not None


def test_queue_message_preserves_explicit_timestamp() -> None:
    timestamp = datetime(2026, 7, 4, 12, 30, tzinfo=timezone.utc)

    queued = CommunicationChannel(InMemoryMessageQueue()).queue_message(
        prompt="hello",
        user="alex",
        timestamp=timestamp,
    )

    assert queued.message.timestamp == timestamp


def test_selector_dequeues_by_user_project_id_and_tag() -> None:
    queue = InMemoryMessageQueue()
    channel = CommunicationChannel(queue)
    channel.queue_message(prompt="one", user="alex", project_id="alpha", tag="work")
    channel.queue_message(prompt="two", user="gaby", project_id="home", tag="family")

    selected = queue.dequeue(MessageSelector(user="gaby", project_id="home", tag="family"))

    assert selected is not None
    assert selected.message.prompt == "two"


def test_slash_command_detected_only_at_prompt_start() -> None:
    channel = CommunicationChannel(InMemoryMessageQueue())

    command = channel.queue_message(prompt="/project new", user="alex").message
    indented = channel.queue_message(prompt=" /project new", user="alex").message
    sentence = channel.queue_message(prompt="please /project new", user="alex").message

    assert command.metadata == {
        "is_command": True,
        "command": "project",
        "command_args": "new",
    }
    assert indented.metadata == {
        "is_command": True,
        "command": "project",
        "command_args": "new",
    }
    assert sentence.metadata == {
        "is_command": False,
        "command": "",
        "command_args": "",
    }


def test_blank_prompt_or_user_raises_value_error() -> None:
    channel = CommunicationChannel(InMemoryMessageQueue())

    with pytest.raises(ValueError, match="prompt_required"):
        channel.queue_message(prompt=" ", user="alex")
    with pytest.raises(ValueError, match="user_required"):
        channel.queue_message(prompt="hello", user="")
